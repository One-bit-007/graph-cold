# Reproduce the full Graph-CoLD experiment matrix (Windows PowerShell).
# Codex: flesh out once src/train.py and src/eval.py are wired.

$ErrorActionPreference = "Stop"

$datasets = @("cicids2017", "maltls22")
$noises   = @("symmetric", "asymmetric", "graph_consistency")
$ratios   = @(0.1, 0.2, 0.4, 0.6)
$methods  = @("graph_cold", "cold", "ablation_hard")
$seeds    = @(0, 1, 2)

foreach ($d in $datasets) {
  foreach ($n in $noises) {
    foreach ($r in $ratios) {
      foreach ($m in $methods) {
        foreach ($s in $seeds) {
          python -m src.train --dataset $d --noise $n --ratio $r --method $m --seed $s
        }
      }
    }
  }
}

# OpTC enterprise case study (single setting).
python -m src.train --dataset optc --noise symmetric --ratio 0.4 --method graph_cold --seed 0

# Aggregate tables + figures.
python -m src.eval --matrix full
