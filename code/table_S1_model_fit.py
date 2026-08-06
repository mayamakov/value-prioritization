#!/usr/bin/env python3
"""
table_S1_model_fit.py — Supplementary Table S1 (held-out RankNet predictive fit).

Reports test-set AUC and accuracy for each rater's RankNet model under the single
reporting configuration selected by the ablation:

    soft_path_noAuto_syn0_noHier
    (soft confidence labels, recommendations grouped into care-pathway features,
     no auto-included pairs, no synthetic-pair cap, no hierarchical term)

WHY THIS READS A SAVED FILE
---------------------------
RankNet training is stochastic (random weight initialisation and batch shuffling)
and no RNG seed is fixed, so retraining reproduces these metrics only to within
about +/-0.02. The full 48-configuration ablation sweep is therefore shipped as
results/ablation_full_auc.csv (26 raters x 48 configurations), and this script
selects the reporting configuration from it. That makes Table S1 exactly
reproducible without a GPU and without retraining.

To retrain from scratch instead, run run_all_experiments.py (slow, and the
numbers will differ slightly for the reason above).

NOTE
----
results/model_fit_metrics.csv and results/ablation_A3_best_per_rater.csv hold a
DIFFERENT quantity: the best configuration chosen separately for each rater
(mean AUC 0.821 physicians / 0.937 LLMs). Table S1 reports the single shared
reporting configuration, not the per-rater best.

USAGE
-----
    python table_S1_model_fit.py
"""
import os
import pandas as pd

from panel_config import HUMANS, LLMS, DISPLAY

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
RESULTS = os.path.join(HERE, "results")
ABLATION = os.path.join(RESULTS, "ablation_full_auc.csv")

REPORTING_CONFIG = "soft_path_noAuto_syn0_noHier"


def _anonymised_labels():
    """Map rater keys to the anonymised labels used in the manuscript
    (FM-1..FM-5, PH-1..PH-5 for physicians; model names for LLMs).
    Falls back to panel_config.DISPLAY if the keymap has not been built yet."""
    keymap = os.path.join(RESULTS, "rater_keymap.csv")
    labels = dict(DISPLAY)
    if os.path.exists(keymap):
        km = pd.read_csv(keymap)
        labels.update(dict(zip(km.rater_key, km.display_name)))
    return lambda k: labels.get(k, k)


def main():
    if not os.path.exists(ABLATION):
        raise SystemExit(
            f"missing {ABLATION}\n"
            "This file ships with the repository; if it is absent, regenerate the\n"
            "ablation sweep with run_recipe_ablation_v2.py (slow)."
        )

    df = pd.read_csv(ABLATION)
    win = df[df.config == REPORTING_CONFIG].copy()
    if win.empty:
        raise SystemExit(f"configuration {REPORTING_CONFIG!r} not found in {ABLATION}")

    win["rater"] = win.doctor.map(_anonymised_labels())
    win["panel"] = win.doctor.map(
        lambda k: "Physician" if k in HUMANS else ("LLM" if k in LLMS else "other")
    )
    win = win[win.panel != "other"]

    out = win[["rater", "panel", "test_auc", "test_accuracy"]].sort_values(
        ["panel", "test_auc"], ascending=[True, False]
    )
    out.to_csv(os.path.join(RESULTS, "table_S1_model_fit.csv"), index=False)

    print("=" * 70)
    print("Supplementary Table S1 — held-out RankNet predictive fit")
    print(f"  reporting configuration : {REPORTING_CONFIG}")
    print("=" * 70)
    print(out.round(3).to_string(index=False))
    print("-" * 70)
    for panel in ("Physician", "LLM"):
        s = win[win.panel == panel]
        print(
            f"  {panel + 's':12} n={len(s):2}  "
            f"AUC {s.test_auc.mean():.3f} (SD {s.test_auc.std():.3f})   "
            f"accuracy {s.test_accuracy.mean():.3f} (SD {s.test_accuracy.std():.3f})"
        )
    print("-" * 70)
    print("  Reported in the supplement:")
    print("    Physicians  AUC 0.803 (SD 0.124)   accuracy 0.746 (SD 0.102)")
    print("    LLMs        AUC 0.930 (SD 0.043)   accuracy 0.859 (SD 0.047)")
    print("  wrote results/table_S1_model_fit.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
