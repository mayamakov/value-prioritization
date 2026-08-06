"""
==========================================================================
llm_helpers.py
==========================================================================
Helpers for incorporating LLM raters alongside the human physicians.

This module:
  1. Lists the existing LLM raters (doctor8, doctor9, doctor10) that were
     already ranked with the older models (kept as "legacy LLM").
  2. Lists slots for future LLM raters (Claude, Gemini, GPT-new, ...) that
     you'll add later — each is just a string key pointing to a folder
     under physicians_results/.
  3. Provides loaders that handle the two ranking-file formats:
       - Human/new format:  <key>_test_part_A_ranked.json and ..._part_B...
       - Legacy LLM format: <key>_test_ranked.json  (single test file)
  4. Provides group labels (family-medicine / public-health / LLM-legacy /
     LLM-new) for plotting and analysis.

If you later want to add a brand-new LLM rater (e.g. Claude-3.5-Sonnet):
   - Run your existing physicians_trial_llm pipeline against it.
   - Save the ranked json files under physicians_results/<your_name>/
     in the SAME naming convention as the human raters
     (i.e. <key>_train_iter_0_ranked.json ... iter_3, and
           <key>_test_part_A_ranked.json + part_B).
   - Add the key to NEW_LLMS below.
==========================================================================
"""
from pathlib import Path
import json


# --------------------------------------------------------------------------
# Group definitions
# --------------------------------------------------------------------------

# Family-medicine physicians (פנימית / משפחה)
FAMILY_MEDICINE = [
    'fm_1',
    'fm_2',
    'fm_3',
    'fm_4',
    'fm_5',
]

# Public-health physicians (בריאות הציבור)
PUBLIC_HEALTH = [
    'ph_1',
    'ph_2',
    'ph_3',
    'ph_4',
    'ph_5',
]

# Legacy LLM raters: ranked back when the old codebase was used.
# Format: <key>_test_ranked.json (single test file, no A/B split)
LEGACY_LLMS = [
    'doctor8',
    'doctor9',
    'doctor10',
]

# New LLM raters: to be added after their rankings are produced.
# Format: same as humans (test_part_A + test_part_B).
# Leave empty for now — append keys here once you have the files.
NEW_LLMS = [
    'claude_opus_4_7',
    'gpt_5',
    'gemini_2_5_pro',
    # previously folded in via EXTRA_LLMS:
    'gemini_3_1_pro',
    'gpt_5_4',
    # newly run (June 2026):
    'claude_opus_4_8',
    'claude_opus_4_5',
    'gpt_5_5',
    'gemini_3_5_flash',
    'llama_3_3_70b',
    'deepseek_v3_2',
    'qwen3_5_27b',
    'mistral_medium_3_5',
]

# Convenience aggregates
ALL_HUMAN_DOCTORS = FAMILY_MEDICINE + PUBLIC_HEALTH
ALL_LLMS = LEGACY_LLMS + NEW_LLMS
ALL_DOCTORS = ALL_HUMAN_DOCTORS + ALL_LLMS


# --------------------------------------------------------------------------
# Group lookup
# --------------------------------------------------------------------------
def get_group(doctor_key):
    """Returns one of: 'family_medicine', 'public_health',
    'llm_legacy', 'llm_new', or 'unknown'."""
    if doctor_key in FAMILY_MEDICINE:
        return 'family_medicine'
    if doctor_key in PUBLIC_HEALTH:
        return 'public_health'
    if doctor_key in LEGACY_LLMS:
        return 'llm_legacy'
    if doctor_key in NEW_LLMS:
        return 'llm_new'
    return 'unknown'


def get_group_color(doctor_key):
    """Consistent color per group, for plotting."""
    g = get_group(doctor_key)
    return {
        'family_medicine': '#1f77b4',   # blue
        'public_health':   '#2ca02c',   # green
        'llm_legacy':      '#ff7f0e',   # orange
        'llm_new':         '#d62728',   # red
        'unknown':         '#7f7f7f',   # grey
    }[g]


def get_group_label(doctor_key):
    """Pretty group label for legends."""
    return {
        'family_medicine': 'Family medicine',
        'public_health':   'Public health',
        'llm_legacy':      'LLM (legacy)',
        'llm_new':         'LLM (current)',
        'unknown':         'Unknown',
    }[get_group(doctor_key)]


# --------------------------------------------------------------------------
# Pair-format normalization
# --------------------------------------------------------------------------
def _norm_pair_item(item):
    """Normalises a ranked-pair item to ((a, b), conf).
    Handles legacy formats: [[a,b], conf], [a, b, conf], etc."""
    try:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            inner, conf = item
            if isinstance(inner, (list, tuple)) and len(inner) == 2:
                a, b = inner
                return ((int(a), int(b)), float(conf))
            elif not isinstance(inner, (list, tuple)):
                a, b = item
                return ((int(a), int(b)), 6.0)
        if isinstance(item, (list, tuple)) and len(item) == 3:
            a, b, conf = item
            return ((int(a), int(b)), float(conf))
    except (TypeError, ValueError):
        return None
    return None


def _load_json_pairs(path):
    """Load a single ranked-pairs JSON file. Returns list of ((a,b), conf)."""
    if not path.exists():
        return []
    with open(path) as f:
        raw = json.load(f)
    out = []
    for item in raw:
        norm = _norm_pair_item(item)
        if norm is not None:
            out.append(norm)
    return out


# --------------------------------------------------------------------------
# Unified loader (handles both human and LLM formats)
# --------------------------------------------------------------------------
def load_ranked_pairs(doctor_key, results_dir, n_iters=4):
    """
    Returns (train_pairs, test_pairs).

    train_pairs : all pairs from <key>_train_iter_0_ranked.json
                  through <key>_train_iter_{n_iters-1}_ranked.json,
                  concatenated.
    test_pairs  : either
                  - <key>_test_part_A_ranked.json + ..._part_B... (humans / NEW_LLMS)
                  - <key>_test_ranked.json                        (LEGACY_LLMS)
    """
    doc_dir = Path(results_dir) / doctor_key

    # Train: same convention for everyone
    train_pairs = []
    for it in range(n_iters):
        train_pairs.extend(
            _load_json_pairs(doc_dir / f"{doctor_key}_train_iter_{it}_ranked.json")
        )

    # Test: different formats
    test_pairs = []
    if doctor_key in LEGACY_LLMS:
        # Single test file
        test_pairs = _load_json_pairs(doc_dir / f"{doctor_key}_test_ranked.json")
    else:
        # Two parts A+B
        for part in ['A', 'B']:
            test_pairs.extend(
                _load_json_pairs(doc_dir / f"{doctor_key}_test_part_{part}_ranked.json")
            )
        # Fallback: if A+B don't exist but a single _test_ranked.json does
        # (this happens for ph_5), use that.
        if len(test_pairs) == 0:
            test_pairs = _load_json_pairs(doc_dir / f"{doctor_key}_test_ranked.json")

    return train_pairs, test_pairs


def load_train_pairs_per_iter(doctor_key, results_dir, n_iters=4):
    """
    Returns a list of length n_iters, where entry i is the cumulative list
    of train pairs labeled through iteration i (inclusive).
    Use this to build the learning-curve datasets (100, 200, 300, 400).
    """
    doc_dir = Path(results_dir) / doctor_key
    cumulative = []
    out = []
    for it in range(n_iters):
        cumulative.extend(
            _load_json_pairs(doc_dir / f"{doctor_key}_train_iter_{it}_ranked.json")
        )
        out.append(list(cumulative))  # snapshot copy
    return out


# --------------------------------------------------------------------------
# Availability scan
# --------------------------------------------------------------------------
def doctors_with_complete_rankings(results_dir, n_iters=4, min_test=50):
    """
    Scans physicians_results/ and returns three lists:
        (humans_ok, legacy_llms_ok, new_llms_ok)
    Each list contains keys with >=n_iters train iters and >=min_test test pairs.
    """
    results_dir = Path(results_dir)
    humans_ok, legacy_ok, new_ok = [], [], []

    for key in ALL_DOCTORS:
        train, test = load_ranked_pairs(key, results_dir, n_iters=n_iters)
        if len(train) < (n_iters * 50) or len(test) < min_test:
            continue
        g = get_group(key)
        if g in ('family_medicine', 'public_health'):
            humans_ok.append(key)
        elif g == 'llm_legacy':
            legacy_ok.append(key)
        elif g == 'llm_new':
            new_ok.append(key)
    return humans_ok, legacy_ok, new_ok


# --------------------------------------------------------------------------
# Pretty-print summary
# --------------------------------------------------------------------------
def print_availability_summary(results_dir):
    """Quick scan and pretty-print what's available."""
    results_dir = Path(results_dir)
    print("=" * 70)
    print("Ranker availability scan")
    print("=" * 70)

    groups = [
        ('Family medicine', FAMILY_MEDICINE),
        ('Public health',   PUBLIC_HEALTH),
        ('LLM (legacy)',    LEGACY_LLMS),
        ('LLM (new)',       NEW_LLMS),
    ]
    for group_name, keys in groups:
        if not keys:
            continue
        print(f"\n--- {group_name} ---")
        for key in keys:
            train, test = load_ranked_pairs(key, results_dir)
            status = "OK" if (len(train) >= 300 and len(test) >= 100) else "MISSING"
            print(f"  {key:<20}  train={len(train):4d}  test={len(test):4d}  [{status}]")
