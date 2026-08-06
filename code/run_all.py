#!/usr/bin/env python3
"""
================================================================================
run_all.py  —  ONE-BUTTON reproduction of all analyses in the manuscript
================================================================================

"LLMs and Physicians Prioritize Preventive-Care Patients by Different Values"

WHAT THIS DOES
--------------
Runs the full pipeline end-to-end and regenerates every figure and table in the
paper, from the raw ranked pairs to the final PNG / PDF / CSV outputs. Each
stage runs as an isolated subprocess, so memory is released between steps and a
single failure never takes down the whole run.

    Stage A  (CPU)  pair-based build : common pairs, Top-N, profiles, agreement
    Stage B  (GPU)  SHAP attribution  [reused from precomputed pkl by default]
    Stage C  (CPU)  z-score normalization of the profile tables
    Stage D  (CPU)  all figures (main + supplementary)
    Stage E  (CPU)  all manuscript tables (printed + CSV)

GPU NOTE
--------
The SHAP step (stage1_ethical_value_mapping.py) needs a GPU and a few minutes.
Its output — results/ethical_shap_per_patient.pkl and the SHAP-derived profile
CSVs — is SHIPPED precomputed, so the default run reuses it and reproduces every
SHAP-based figure exactly. To recompute SHAP from scratch on a GPU box:

    python run_all.py --with-shap

HOW TO RUN
----------
    cd code
    python run_all.py               # full reproduction using precomputed SHAP
    python run_all.py --with-shap   # also recompute SHAP (needs GPU)

OUTPUTS  ->  ../results/  (CSVs, figures/, paper_figures/)
================================================================================
"""
import os, sys, argparse, subprocess, time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results"


def link_data_into_cwd():
    """Expose data + results under the bare names the legacy scripts expect,
    so the original analysis code runs UNCHANGED (exact reproducibility)."""
    links = {
        "synthetic_data (7).xlsx": DATA_DIR / "synthetic_data.xlsx",
        "synthetic_data.xlsx":     DATA_DIR / "synthetic_data.xlsx",
        "rec_lists.xlsx":          DATA_DIR / "rec_lists.xlsx",
        "doctor_rankings.pkl":     DATA_DIR / "doctor_rankings.pkl",
        "doctor_rankings.pkl.gz":  DATA_DIR / "doctor_rankings.pkl.gz",
        "initial_pairs.pkl":       DATA_DIR / "initial_pairs.pkl",
        "physicians_results":      DATA_DIR / "physicians_results",
        "results":                 RESULTS_DIR,
    }
    for name, target in links.items():
        link = CODE_DIR / name
        if not target.exists():
            continue
        if link.is_symlink() or link.exists():
            try:
                link.unlink()
            except (OSError, IsADirectoryError):
                # a real directory shadowing the link (e.g. a stray results/
                # created by running a script directly) -- remove it so the
                # canonical ../data and ../results are always what get used
                import shutil
                if link.is_dir() and not link.is_symlink():
                    shutil.rmtree(link, ignore_errors=True)
                if link.exists() or link.is_symlink():
                    continue
        try:
            link.symlink_to(target)
        except OSError:
            import shutil
            if target.is_dir():
                shutil.copytree(target, link, dirs_exist_ok=True)
            else:
                shutil.copy2(target, link)


# Pipeline definition: (script, label, needs_gpu)
STAGE_A = [
    ("rebuild_pair_csvs.py",   "A1  common pairs + profiles"),
    ("make_topN_frozen.py",    "A2  Top-N frozen rankings"),
    ("stage2_top_and_pairs.py","A3  common-pairs + combined profiles"),
]
STAGE_B = [
    ("stage1_ethical_value_mapping.py", "B1  SHAP attribution (GPU)"),
    ("raw_shap_clustering_FIXED.py",    "B2  raw-SHAP clustering"),
]
STAGE_D = [
    ("make_figS2.py",                    "D1  Figure S2 (three methods)"),
    ("paper_figures_CORRECTED.py",       "D2  main-paper figures"),
    ("generate_all_figures_FINAL_FIXED.py","D3  S1/S4 agreement panels"),
]


def run_script(script, label, manifest, extra_env=None):
    path = CODE_DIR / script
    if not path.exists():
        manifest.append(("MISSING", label, 0.0))
        print(f"  [MISSING] {label}")
        return
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(path)], cwd=str(CODE_DIR),
                       env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode == 0:
        manifest.append(("PASS", label, dt))
        print(f"  [PASS] {label} ({dt:.0f}s)")
    else:
        manifest.append(("FAIL", label, dt))
        tail = (r.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        print(f"  [FAIL] {label} ({dt:.0f}s) -> {tail}")


def run_normalize(manifest):
    """Stage C: z-score normalization, invoked via the original orchestrator's
    normalize block (kept in normalize_tables.py for isolation)."""
    run_script("normalize_tables.py", "C   normalize profile tables (z-score)", manifest)


def main():
    ap = argparse.ArgumentParser(description="Reproduce all manuscript analyses.")
    ap.add_argument("--with-shap", action="store_true",
                    help="Recompute SHAP on GPU (default: reuse precomputed).")
    args = ap.parse_args()

    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    link_data_into_cwd()

    print("=" * 78)
    print("REPRODUCING: LLMs vs Physicians — preventive-care value prioritization")
    print(f"  SHAP recompute : {'ON (GPU)' if args.with_shap else 'OFF (precomputed pkl)'}")
    print(f"  data dir       : {DATA_DIR}")
    print(f"  results dir    : {RESULTS_DIR}")
    print("=" * 78)

    manifest = []

    print("\n[STAGE A] pair-based build (CPU)")
    for s, l in STAGE_A:
        run_script(s, l, manifest)

    print("\n[STAGE B] SHAP attribution")
    if args.with_shap:
        for s, l in STAGE_B:
            run_script(s, l, manifest)
    else:
        print("  [SKIP] SHAP recompute — using shipped results/ethical_shap_per_patient.pkl")

    print("\n[STAGE C] normalize profile tables")
    run_normalize(manifest)

    print("\n[STAGE D] figures")
    for s, l in STAGE_D:
        run_script(s, l, manifest)
    print("\n[STAGE D2] per-rater figR figures")
    for s in sorted(CODE_DIR.glob("figR_*.py")):
        run_script(s.name, f"figR  {s.stem}", manifest)

    print("\n[STAGE E] manuscript tables")
    run_script("run_to_run_agreement.py", "E1  run-to-run agreement (16 LLMs)", manifest)
    run_script("family_distances.py", "E2  developer-family distances", manifest)
    run_script("shap_mapping_correlation.py", "E3  raw-SHAP vs mapped correlation", manifest)
    run_script("table_S1_model_fit.py", "E4  Table S1 model fit (reporting config)", manifest)
    run_script("print_all_tables_v10.py", "E5  print all tables (T3-T13)", manifest)

    # ---- MANIFEST ----
    print("\n" + "=" * 78)
    print("MANIFEST")
    print("=" * 78)
    order = {"FAIL": 0, "MISSING": 1, "PASS": 2}
    for status, label, dt in sorted(manifest, key=lambda r: order.get(r[0], 9)):
        t = f"{dt:5.0f}s" if dt else "      "
        print(f"  {status:8} {t}  {label}")
    n_fail = sum(1 for s, *_ in manifest if s == "FAIL")
    n_miss = sum(1 for s, *_ in manifest if s == "MISSING")
    n_pass = sum(1 for s, *_ in manifest if s == "PASS")
    print("-" * 78)
    print(f"  {n_pass} passed, {n_fail} failed, {n_miss} missing")
    print(f"  Outputs in: {RESULTS_DIR}")
    print("=" * 78)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
