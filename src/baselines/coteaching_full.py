"""Full Co-Teaching baseline (Han et al., 2018) for tabular noisy-label IDS.

This is the P5 upgrade of the earlier lightweight SGD-probe approximation
(``src/baselines/coteaching.py``, retained as "Co-Teaching-lite"). It follows
the published Co-Teaching protocol faithfully:

- two independently initialized deep networks (small MLPs for tabular flows),
- per-mini-batch small-loss selection,
- co-exchange: each network is updated on the peer network's selected batch,
- forget-rate schedule tracking the true injected noise rate with a gradual
  warmup (``num_gradual`` epochs), as in the authors' reference code,
- final predictions average the two networks' softmax probabilities.

The networks are deliberately compact (two hidden layers) because the protocol
uses fixed tabular flow features and small deterministic audit windows; the
selection mechanism, exchange rule, and schedule are the faithful parts.
"""
from __future__ import annotations

import numpy as np

from src.baselines.base import BaselineResult, array_hash


class CoTeachingFullBaseline:
    method = "Co-Teaching"
    method_family = "co_teaching_full"
    implementation_status = "verified_implementation"

    def __init__(
        self,
        seed: int = 0,
        noise_rate: float = 0.0,
        epochs: int = 40,
        batch_size: int = 512,
        hidden: tuple[int, int] = (256, 128),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_gradual: int = 10,
    ):
        self.seed = int(seed)
        self.noise_rate = float(noise_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.hidden = tuple(int(h) for h in hidden)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.num_gradual = int(num_gradual)

    def _forget_rate(self, epoch: int) -> float:
        target = float(np.clip(self.noise_rate, 0.0, 0.5))
        if target <= 0.0:
            return 0.0
        return target * min(1.0, float(epoch + 1) / float(max(self.num_gradual, 1)))

    def fit_predict(self, X_train, y_noisy, X_test, num_classes: int, **kwargs) -> BaselineResult:
        import torch

        del kwargs
        torch.manual_seed(self.seed * 1000 + 7)
        rng = np.random.default_rng(self.seed * 1000 + 11)

        X = np.asarray(X_train, dtype=np.float32)
        Xt = np.asarray(X_test, dtype=np.float32)
        y = np.asarray(y_noisy, dtype=np.int64)
        n, d = X.shape
        device = torch.device("cpu")

        def _make_net():
            net = torch.nn.Sequential(
                torch.nn.Linear(d, self.hidden[0]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden[0], self.hidden[1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hidden[1], num_classes),
            )
            return net.to(device)

        net_a = _make_net()
        net_b = _make_net()
        opt_a = torch.optim.Adam(net_a.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        opt_b = torch.optim.Adam(net_b.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        x_all = torch.from_numpy(X)
        y_all = torch.from_numpy(y)
        exchanged_fractions: list[float] = []

        for epoch in range(max(self.epochs, 1)):
            forget = self._forget_rate(epoch)
            remember = 1.0 - forget
            net_a.train()
            net_b.train()
            order = rng.permutation(n)
            epoch_selected: list[int] = []
            for start in range(0, n, max(1, self.batch_size)):
                idx = order[start : start + max(1, self.batch_size)]
                if idx.size == 0:
                    continue
                xb = x_all[idx]
                yb = y_all[idx]
                with torch.no_grad():
                    loss_a = loss_fn(net_a(xb), yb)
                    loss_b = loss_fn(net_b(xb), yb)
                keep_n = int(np.floor(remember * idx.size))
                keep_n = int(np.clip(keep_n, 1, idx.size))
                # Peer selection: B's small-loss samples update A and vice versa.
                sel_b = torch.topk(loss_b, keep_n, largest=False).indices
                sel_a = torch.topk(loss_a, keep_n, largest=False).indices
                epoch_selected.append(keep_n / idx.size)

                opt_a.zero_grad()
                loss_fn_mean = torch.nn.CrossEntropyLoss()
                out_a = net_a(xb[sel_b])
                loss_update_a = loss_fn_mean(out_a, yb[sel_b])
                loss_update_a.backward()
                opt_a.step()

                opt_b.zero_grad()
                out_b = net_b(xb[sel_a])
                loss_update_b = loss_fn_mean(out_b, yb[sel_a])
                loss_update_b.backward()
                opt_b.step()
            exchanged_fractions.append(float(np.mean(epoch_selected)))

        net_a.eval()
        net_b.eval()
        with torch.no_grad():
            proba_train = (
                torch.softmax(net_a(x_all), dim=1).numpy() + torch.softmax(net_b(x_all), dim=1).numpy()
            ) / 2.0
            proba_test = (
                torch.softmax(net_a(torch.from_numpy(Xt)), dim=1).numpy()
                + torch.softmax(net_b(torch.from_numpy(Xt)), dim=1).numpy()
            ) / 2.0
        y_pred = np.argmax(proba_test, axis=1).astype(np.int64)

        # Retained mask for audit columns: samples whose final-epoch small-loss
        # confidence (mean peer probability on the noisy label) clears the
        # remember-rate threshold of the last epoch.
        confidence = proba_train[np.arange(n), y]
        final_remember = 1.0 - self._forget_rate(max(self.epochs, 1) - 1)
        if self.noise_rate > 0 and 0.0 < final_remember < 1.0:
            threshold = float(np.quantile(confidence, 1.0 - final_remember))
            retained = confidence >= threshold
        else:
            retained = np.ones(n, dtype=bool)

        return BaselineResult(
            method=self.method,
            method_family=self.method_family,
            implementation_status=self.implementation_status,
            y_pred=y_pred,
            proba=proba_test,
            weights=retained.astype(np.float64),
            retained_mask=retained,
            details={
                "classifier": "two torch MLPs ({} hidden) with small-loss co-exchange".format(self.hidden),
                "epochs": max(self.epochs, 1),
                "batch_size": max(1, self.batch_size),
                "num_gradual": max(self.num_gradual, 1),
                "forget_rate_target": float(np.clip(self.noise_rate, 0.0, 0.5)),
                "mean_remembered_fraction": float(np.mean(exchanged_fractions)) if exchanged_fractions else 1.0,
                "trained_on": "noisy_y_train",
                "train_label_source": "noisy_y_train",
                "eval_label_source": "clean_y_test",
                "training_label_hash": array_hash(y),
                "retained_fraction": float(np.mean(retained)),
            },
        )
