"""OpTC-style enterprise mini-case for D4.

This module intentionally avoids full OpTC ingestion. When real enterprise
feature files are absent, it builds a deterministic SOC/provenance-style event
table with host, process, IP, temporal, and threat-intel semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.graph.build import EdgeIndex, MultiViewGraph
from src.models import graph_cdm
from src.models.evidence import compute as compute_evidence
from src.ranking.prioritize import priority_scores, topk


FEATURE_COLUMNS = [
    "host_code",
    "process_code",
    "parent_process_code",
    "src_ip_code",
    "dst_ip_code",
    "event_type_code",
    "alert_type_code",
    "risk_score",
    "time_offset",
]


@dataclass
class OpTCCaseResult:
    report: dict
    events: pd.DataFrame
    graph: MultiViewGraph
    graph_cdm: np.ndarray
    ranking: np.ndarray


class EnterpriseBaselineAdapter:
    """Flash/Argus-compatible baseline adapter with lightweight fallback."""

    def __init__(self, baseline: str = "flash", backend: str = "xgboost", seed: int = 42):
        if baseline not in {"flash", "argus"}:
            raise ValueError("baseline must be 'flash' or 'argus'.")
        self.baseline = baseline
        self.backend = backend
        self.seed = seed
        self.model = None
        self.backend_used = None

    def fit(self, X, y):
        if self.backend == "xgboost":
            try:
                from xgboost import XGBClassifier

                self.model = XGBClassifier(
                    n_estimators=20,
                    max_depth=2,
                    learning_rate=0.2,
                    random_state=self.seed,
                    eval_metric="logloss",
                )
                self.backend_used = "xgboost"
            except Exception:
                self.model = RandomForestClassifier(n_estimators=50, random_state=self.seed, class_weight="balanced")
                self.backend_used = "sklearn_random_forest_fallback"
        else:
            self.model = RandomForestClassifier(n_estimators=50, random_state=self.seed, class_weight="balanced")
            self.backend_used = "sklearn_random_forest"
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64))
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("EnterpriseBaselineAdapter must be fitted before predict_proba.")
        X = np.asarray(X, dtype=np.float32)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
        else:
            pred = self.model.predict(X)
            n_classes = int(np.max(pred)) + 1
            proba = np.eye(n_classes)[pred]
        if proba.shape[1] == 1:
            proba = np.hstack([1.0 - proba, proba])
        return proba.astype(np.float64)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def run_case(cfg: dict | None = None, out_dir: str | Path = "reports") -> OpTCCaseResult:
    cfg = _default_cfg(cfg)
    seed = int(cfg.get("seed", 42))
    events, mode = _load_or_generate_events(cfg, seed)
    X, y = _feature_matrix(events)
    graph = build_optc_graph(events, X)

    adapter = EnterpriseBaselineAdapter(
        baseline=str(cfg.get("baseline", "flash")),
        backend=str(cfg.get("backend", "xgboost")),
        seed=seed,
    ).fit(X, y)
    soft_labels = adapter.predict_proba(X)
    pred = np.argmax(soft_labels, axis=1)
    view_preds = _view_predictions(pred, events)
    optc_cfg = {
        "graph_cdm": {
            "lambda_pred": 0.35,
            "lambda_neigh": 0.25,
            "lambda_view": 0.30,
            "lambda_chain": float(cfg.get("lambda_chain", 0.1)),
        },
        "evidence_preserving": {
            "freq_protect": "log",
            "gamma_anomaly": 1.0,
            "rho": 0.2,
            "theta": 0.5,
            "kappa": 4.0,
        },
        "ranking": {"alpha1": 1.0, "alpha2": 0.7, "alpha3": 0.4, "benign_class": 0},
    }
    cdm_result = graph_cdm.forward(view_preds, soft_labels, y, optc_cfg, graph=graph, return_components=True)
    evidence = compute_evidence(y, optc_cfg, anomaly=events["risk_score"].to_numpy(dtype=float))
    scores = priority_scores(
        {"graph_cdm": cdm_result.score, "evidence": evidence, "soft_labels": soft_labels},
        {"risk_score": events["risk_score"].to_numpy(dtype=float)},
        optc_cfg,
    )
    ranking = topk(scores, int(cfg.get("top_k", 5)))
    report = _report(events, graph, cdm_result, ranking, mode, adapter, optc_cfg)
    _write_reports(report, out_dir)
    return OpTCCaseResult(report=report, events=events, graph=graph, graph_cdm=cdm_result.score, ranking=ranking)


def build_optc_graph(events: pd.DataFrame, X: np.ndarray) -> MultiViewGraph:
    n = len(events)
    host_edges = _group_edges(events["host_id"].to_numpy())
    ip_edges = _pair_relation_edges(events["src_ip"].to_numpy(), events["dst_ip"].to_numpy())
    process_edges = _process_edges(events)
    temporal_edges, snapshots, temporal_pairs = _temporal_edges(events["timestamp"].to_numpy())
    threat_edges = _threat_edges(events)
    views = {
        "host": _edge(host_edges, n, [0]),
        "ip": _edge(ip_edges, n, [3, 4]),
        "process": _edge(process_edges, n, [1, 2, 5]),
        "temporal": _edge(temporal_edges, n, [8]),
        "threat_intel": _edge(threat_edges, n, [6, 7]),
    }
    return MultiViewGraph(
        views=views,
        node_features=X.astype(np.float32),
        node_index={idx: idx for idx in range(n)},
        snapshots=snapshots,
        view_masks={view: edge.feature_mask for view, edge in views.items()},
        temporal_pairs=temporal_pairs,
    )


def _load_or_generate_events(cfg: dict, seed: int) -> tuple[pd.DataFrame, str]:
    path = Path(str(cfg.get("path", "")))
    event_file = path / "events.csv"
    if event_file.exists():
        return pd.read_csv(event_file), "real"
    return _synthetic_events(seed), "synthetic"


def _synthetic_events(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hosts = ["host-a", "host-b", "host-c", "host-d"]
    rows = []
    base = pd.Timestamp("2026-01-01T00:00:00")
    for idx in range(36):
        attack_chain = 12 <= idx < 22
        host = hosts[(idx // 9) % len(hosts)]
        process = f"proc-{idx % 6}"
        parent = f"proc-{(idx - 1) % 6}" if idx % 6 else "root"
        src_ip = f"10.0.{idx % 3}.{idx % 7 + 10}"
        dst_ip = "203.0.113.50" if attack_chain else f"10.1.{idx % 4}.{idx % 8 + 20}"
        event_type = ["process_start", "dns_query", "net_conn", "file_write"][idx % 4]
        alert_type = "credential_access" if attack_chain and idx % 3 else ("c2_ioc" if attack_chain else "benign_activity")
        risk = float(np.clip((0.25 + 0.55 * attack_chain + 0.1 * rng.random()), 0.0, 1.0))
        label = int(attack_chain or (idx in {29, 31}))
        rows.append(
            {
                "host_id": host,
                "process_id": process,
                "parent_process_id": parent,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "timestamp": (base + pd.Timedelta(minutes=idx * 4)).isoformat(),
                "event_type": event_type,
                "alert_type": alert_type,
                "label": label,
                "risk_score": risk,
            }
        )
    return pd.DataFrame(rows)


def _feature_matrix(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = events.copy()
    for col in ["host_id", "process_id", "parent_process_id", "src_ip", "dst_ip", "event_type", "alert_type"]:
        frame[f"{col.replace('_id', '')}_code"] = LabelEncoder().fit_transform(frame[col].astype(str))
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    frame["time_offset"] = (timestamps - timestamps.min()).dt.total_seconds() / 3600.0
    X = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    return X, y


def _view_predictions(pred: np.ndarray, events: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {view: pred.copy() for view in ["host", "ip", "process", "temporal", "threat_intel"]}
    high_risk = events["risk_score"].to_numpy() > 0.75
    out["threat_intel"][high_risk] = 1
    out["temporal"][events["alert_type"].to_numpy() == "credential_access"] = 1
    return out


def _edge(edge_index: np.ndarray, n_nodes: int, feature_ids: list[int]) -> EdgeIndex:
    mask = np.zeros(len(FEATURE_COLUMNS), dtype=bool)
    mask[feature_ids] = True
    if edge_index.size == 0 and n_nodes > 1:
        src = np.arange(n_nodes - 1)
        edge_index = np.vstack([src, src + 1])
    return EdgeIndex(
        edge_index=edge_index.astype(np.int64),
        edge_weight=np.ones(edge_index.shape[1], dtype=np.float32),
        feature_mask=mask,
        node_mask=np.ones(n_nodes, dtype=bool),
        batches=[np.arange(n_nodes, dtype=np.int64)],
    )


def _group_edges(values: np.ndarray) -> np.ndarray:
    edges = []
    for value in np.unique(values):
        idx = np.flatnonzero(values == value)
        edges.extend(_complete_edges(idx))
    return _edge_array(edges)


def _pair_relation_edges(src_values: np.ndarray, dst_values: np.ndarray) -> np.ndarray:
    edges = []
    for values in (src_values, dst_values):
        for value in np.unique(values):
            idx = np.flatnonzero(values == value)
            edges.extend(_complete_edges(idx))
    return _edge_array(edges)


def _process_edges(events: pd.DataFrame) -> np.ndarray:
    edges = []
    processes = events["process_id"].to_numpy()
    parents = events["parent_process_id"].to_numpy()
    for idx, proc in enumerate(processes):
        related = np.flatnonzero((processes == proc) | (parents == proc) | (processes == parents[idx]))
        for dst in related:
            if idx != dst:
                edges.append((idx, int(dst)))
    return _edge_array(edges)


def _temporal_edges(timestamps: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    times = pd.to_datetime(timestamps, utc=True)
    minutes = ((times - times.min()).total_seconds() / 60.0).to_numpy()
    bins = np.floor(minutes / 20).astype(int)
    snapshots = [np.flatnonzero(bins == value) for value in np.unique(bins)]
    edges = []
    for snapshot in snapshots:
        ordered = np.sort(snapshot)
        for left, right in zip(ordered[:-1], ordered[1:]):
            edges.append((int(left), int(right)))
            edges.append((int(right), int(left)))
    pairs = []
    for left, right in zip(snapshots[:-1], snapshots[1:]):
        width = min(left.size, right.size)
        if width:
            pairs.extend(zip(np.sort(left)[:width], np.sort(right)[:width]))
    return _edge_array(edges), snapshots, _edge_array(pairs)


def _threat_edges(events: pd.DataFrame) -> np.ndarray:
    edges = []
    alerts = events["alert_type"].to_numpy()
    risk_bin = np.floor(events["risk_score"].to_numpy(dtype=float) * 4).astype(int)
    for value in np.unique(alerts):
        edges.extend(_complete_edges(np.flatnonzero(alerts == value)))
    for value in np.unique(risk_bin):
        edges.extend(_complete_edges(np.flatnonzero(risk_bin == value)))
    return _edge_array(edges)


def _complete_edges(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size <= 1:
        return []
    return [(int(src), int(dst)) for src in indices for dst in indices if src != dst]


def _edge_array(edges: list[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(sorted(set(edges)), dtype=np.int64).T


def _report(events, graph, cdm_result, ranking, mode, adapter, cfg):
    return {
        "stage": "D4-OpTC-case",
        "mode": mode,
        "num_events": int(len(events)),
        "num_hosts": int(events["host_id"].nunique()),
        "num_processes": int(events["process_id"].nunique()),
        "num_ips": int(pd.concat([events["src_ip"], events["dst_ip"]]).nunique()),
        "five_views_non_empty": bool(all(edge.edge_index.shape[1] > 0 for edge in graph.views.values())),
        "d_chain_enabled": bool(cfg["graph_cdm"]["lambda_chain"] > 0),
        "lambda4": float(cfg["graph_cdm"]["lambda_chain"]),
        "graph_cdm_computed": bool(np.isfinite(cdm_result.score).all()),
        "ranking_topk_generated": bool(len(ranking) > 0),
        "enterprise_baseline_adapter_ready": True,
        "baseline_backend_used": adapter.backend_used,
        "known_limitations": [
            "synthetic mini-case is used when real OpTC event files are not present",
            "Flash/Argus adapters expose comparison interface but do not reimplement full papers",
        ],
    }


def _write_reports(report: dict, out_dir: str | Path) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "d4_optc_case_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = "\n".join(
        [
            "# D4 OpTC Case Report",
            "",
            f"- Mode: {report['mode']}",
            f"- Events: {report['num_events']}",
            f"- Hosts: {report['num_hosts']}",
            f"- Processes: {report['num_processes']}",
            f"- IPs: {report['num_ips']}",
            f"- Five views non-empty: {report['five_views_non_empty']}",
            f"- D_chain enabled: {report['d_chain_enabled']} (lambda4={report['lambda4']})",
            f"- Graph-CDM computed: {report['graph_cdm_computed']}",
            f"- Top-K generated: {report['ranking_topk_generated']}",
            f"- Baseline backend: {report['baseline_backend_used']}",
            "",
            "Known limitations:",
            *[f"- {item}" for item in report["known_limitations"]],
            "",
        ]
    )
    (path / "d4_optc_case_report.md").write_text(md, encoding="utf-8")


def _default_cfg(cfg: dict | None) -> dict:
    out = {
        "seed": 42,
        "path": "data/optc",
        "baseline": "flash",
        "backend": "xgboost",
        "lambda_chain": 0.1,
        "top_k": 5,
    }
    if cfg:
        out.update(cfg)
    return out
