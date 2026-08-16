# -*- coding: utf-8 -*-
"""
Re-run the 400 failed/fallback pairs (Gemini 2.5 Pro, Gemini 3.1 Pro, Qwen3.5 27B).

Place this file NEXT TO rank_with_llm.py (your original code folder) together with
repair_pairs.json, fill in the keys/model ids below, then:

    python rerun_failed_pairs.py            # everything
    python rerun_failed_pairs.py --primary  # only the 75 primary-analysis pairs

Outputs: repair_output/<task>.json  — same entry format as the ranked files:
[[winner, loser], confidence]. NO fallback is ever written: suspected fallback
responses are retried (batch 5 -> singles x3) and anything still unresolved is
written to repair_output/UNRESOLVED_<task>.json for manual inspection.
"""
import json, sys, time
from pathlib import Path
import pandas as pd
import rank_with_llm as R   # your original module — prompts stay byte-identical

# ----------------------------- EDIT ME ------------------------------------
OPENROUTER_KEY = "PUT-YOUR-KEY-HERE"
MODELS = {
    #  folder name        (provider,     api model id,               api key)
    "gemini_2_5_pro": ("openrouter", "google/gemini-2.5-pro",   OPENROUTER_KEY),
    "gemini_3_1_pro": ("openrouter", "google/gemini-3.1-pro",   OPENROUTER_KEY),  # <-- verify id
    "qwen3_5_27b":    ("openrouter", "qwen/qwen3.5-27b",        OPENROUTER_KEY),  # <-- verify id
}
PATIENT_XLSX = "synthetic_data.xlsx"   # the same cohort file used originally
# ---------------------------------------------------------------------------

def suspected_fallback(entries, presented):
    """A whole-call fallback = every pair answered 'first presented' with conf 1."""
    if not entries: return True
    return all(c == 1 and (w, l) == (a, b)
               for ((w, l), c), (a, b) in zip(entries, presented))

def run_task(name, task, pdf):
    prov, mid, key = MODELS[task["model"]]
    pairs = [tuple(p) for p in task["pairs_presented_order"]]
    got = {}
    todo = list(pairs)
    for attempt, bs in enumerate([5, 1, 1, 1], 1):
        if not todo: break
        print(f"[{name}] attempt {attempt} (batch={bs}) on {len(todo)} pairs")
        res = R.rank_pair_list(todo, pdf, prov, mid, key,
                               batch_size=bs, progress_label=name)
        # walk results call-by-call and drop suspected whole-call fallbacks
        i = 0
        while i < len(res):
            chunk = res[i:i+bs]; pres = todo[i:i+bs]
            if bs > 1 and suspected_fallback(chunk, pres):
                pass                       # drop -> retried next round
            else:
                for ((w, l), c), (a, b) in zip(chunk, pres):
                    got[(a, b)] = ((w, l), c)
            i += bs
        todo = [p for p in pairs if p not in got]
        time.sleep(2)
    out = [[list(got[p][0]), got[p][1]] for p in pairs if p in got]
    Path("repair_output").mkdir(exist_ok=True)
    json.dump(out, open(f"repair_output/{name}.json", "w"))
    if todo:
        json.dump([list(p) for p in todo],
                  open(f"repair_output/UNRESOLVED_{name}.json", "w"))
        print(f"[{name}] WARNING: {len(todo)} pairs unresolved after retries")
    print(f"[{name}] done: {len(out)}/{len(pairs)} pairs")

def main():
    only_primary = "--primary" in sys.argv
    tasks = json.load(open("repair_pairs.json"))
    pdf = pd.read_excel(PATIENT_XLSX)
    for name, task in tasks.items():
        if only_primary and "__primary__" not in name:
            continue
        run_task(name, task, pdf)

if __name__ == "__main__":
    main()
