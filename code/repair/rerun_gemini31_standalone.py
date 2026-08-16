# -*- coding: utf-8 -*-
"""
rerun_gemini31_standalone.py
============================
השלמת 75 הקריאות החסרות של Gemini 3.1 Pro (rollout seed1: 50, seed3: 25).

קובץ עצמאי לחלוטין — לא צריך את rank_with_llm.py ולא שום קובץ אחר.
הפרומפטים, ה-system prompt והפורמט זהים בייט-בייט לריצה המקורית.
אין שום fallback: כישלון נשמר ככישלון (UNRESOLVED), לעולם לא מומצאת החלטה.

איך מריצים (בסביבת ה-Jupyter שלך):
    1) ודאי ש-synthetic_data.xlsx באותה תיקייה (או עדכני את PATIENT_XLSX).
    2) הדביקי את מפתח ה-OpenRouter שלך ב-OPENROUTER_KEY למטה.
    3) בתא: %run rerun_gemini31_standalone.py     (או: python rerun_gemini31_standalone.py)
    4) בסוף נוצרים בתיקיית repair_output:
         gemini_3_1_pro__rollout_seed1__missing.json   (50 החלטות)
         gemini_3_1_pro__rollout_seed3__missing.json   (25 החלטות)
       את שני הקבצים האלה מעלים אליי ואני משחיל, מחשב S11 מחדש ומעדכן את המסמכים.

דרישות: pip install openai pandas openpyxl   (יש לך כבר מהריצה הקודמת)
"""

import json, time, sys
from pathlib import Path
import pandas as pd

# ----------------------------- EDIT ME ------------------------------------
OPENROUTER_KEY = "PUT-YOUR-KEY-HERE"
MODEL_ID       = "google/gemini-3.1-pro-preview"      # המזהה הנכון שמצאת
PATIENT_XLSX   = "synthetic_data.xlsx"
OUT_DIR        = Path("repair_output")
# ---------------------------------------------------------------------------

# ============================================================
# 75 הזוגות החסרים, בסדר ההצגה המקורי (מוטמעים — אין תלות בקבצים)
# ============================================================
TASKS = {
  "gemini_3_1_pro__rollout_seed1__missing":
    [[2,701],[5,476],[16,603],[17,227],[27,570],[47,151],[48,226],[51,587],[112,207],[116,748],
     [139,253],[151,244],[162,667],[181,370],[187,373],[188,269],[188,835],[191,611],[191,642],[201,264],
     [207,985],[208,268],[237,447],[244,294],[285,642],[340,512],[344,851],[350,415],[350,792],[375,547],
     [379,710],[379,887],[404,863],[408,970],[414,611],[462,649],[462,659],[472,836],[504,864],[505,891],
     [506,756],[559,984],[574,861],[587,997],[626,740],[633,862],[673,761],[703,945],[743,842],[845,932]],
  "gemini_3_1_pro__rollout_seed3__missing":
    [[65,920],[88,264],[96,509],[135,642],[139,285],[154,202],[178,721],[181,781],[181,963],[192,848],
     [218,844],[227,341],[272,459],[373,538],[407,739],[416,461],[462,938],[563,910],[601,994],[608,858],
     [615,792],[682,771],[755,822],[770,849],[770,898]],
}
# ============================================================

REC_TEXTS = {
    'rec1':  'Basic laboratory panel (lipid profile, glucose)',
    'rec2':  'Advanced laboratory panel (ApoB, Lp(a))',
    'rec3':  'Pathophysiology labs (inflammation, thyroid)',
    'rec4':  'Routine laboratory monitoring',
    'rec5':  'Diagnostic imaging (carotid ultrasound)',
    'rec6':  'Advanced imaging (coronary CTA)',
    'rec8':  'BP / BMI measurement',
    'rec10': 'First-line treatment (low-dose statin)',
    'rec11': 'Advanced treatment (medium/high statin or PCSK9)',
    'rec12': 'Treatment upgrade (dose increase or new drug)',
    'rec13': 'Treatment replacement (due to contraindication)',
    'rec16': 'Specialist consultation (Lipidologist)',
    'rec17': 'Other consultation (Hepatology / Cardiology)',
    'rec18': 'Nutritional consultation (Dietitian)',
    'rec19': 'Lifestyle improvement (exercise / diet)',
    'rec21': 'Curate medical record',
}
ACTIVE_RECS = list(REC_TEXTS.keys())

def patient_to_text(patient_row):
    age  = int(patient_row['age'])
    risk = float(patient_row['risk'])
    recs = [REC_TEXTS[r] for r in ACTIVE_RECS
            if r in patient_row and patient_row[r] >= 0.5]
    parts = [f"age {age}", f"10-year CV risk {risk:.0f}%"]
    parts.append(f"Care plan: {'; '.join(recs)}" if recs else "Care plan: none")
    return ' | '.join(parts)

SYSTEM_PROMPT = """You are a physician asked to prioritize patients for proactive care plan reviews.
You will be shown pairs of patients. For each pair, decide which patient deserves earlier
prioritization based on their age, 10-year cardiovascular risk, and current care plan
recommendations.

For each pair return:
  - d: "p1" if Patient 1 deserves earlier prioritization, "p2" otherwise
  - c: confidence on a 1-5 scale, where:
       5 = very confident, 4 = confident, 3 = moderate,
       2 = slight preference, 1 = essentially a tie

Be consistent and clinically reasonable across all pairs.
Output ONLY a JSON object with the form:
  {"items": [{"id": "pair_0", "d": "p1", "c": 4}, {"id": "pair_1", "d": "p2", "c": 3}, ...]}
Do not include any other text.
"""

def build_user_prompt(pairs_batch, start_idx=0):
    lines = [
        "Decide for each pair which patient deserves earlier prioritization.",
        "Return strict JSON: {\"items\": [{\"id\":..., \"d\":\"p1\"|\"p2\", \"c\":1-5}, ...]}",
        "",
    ]
    for i, (p1, p2) in enumerate(pairs_batch):
        pair_id = f"pair_{start_idx + i}"
        lines.append(f"--- {pair_id} ---")
        lines.append(f"Patient 1: {patient_to_text(p1)}")
        lines.append(f"Patient 2: {patient_to_text(p2)}")
        lines.append("")
    return '\n'.join(lines)

def call_openrouter(system_prompt, user_prompt):
    from openai import OpenAI
    client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

def parse_items(text, n_expected, start_idx):
    """מחזיר dict: מיקום-בבצ'  ->  ('p1'/'p2', conf). זריקת חריגה = הקריאה נכשלה."""
    txt = (text or "").strip()
    if not txt:
        raise ValueError("empty response")
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:]
    obj = json.loads(txt)
    out = {}
    for it in obj.get("items", []):
        pid = str(it.get("id", ""))
        if not pid.startswith("pair_"):
            continue
        k = int(pid.split("_")[1]) - start_idx
        d = str(it.get("d", "")).strip().lower()
        c = float(it.get("c", 3))
        if d in ("p1", "p2") and 0 <= k < n_expected:
            out[k] = (d, max(1.0, min(5.0, c)))
    if not out:
        raise ValueError("no valid items parsed")
    return out

def main():
    if OPENROUTER_KEY.startswith("PUT-"):
        sys.exit("!! הדביקי את מפתח ה-OpenRouter ב-OPENROUTER_KEY בראש הקובץ")
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_excel(PATIENT_XLSX).set_index('patient_num')
    P = {pid: row for pid, row in df.iterrows()}

    for task_name, pairs in TASKS.items():
        pairs = [tuple(p) for p in pairs]
        print(f"\n===== {task_name}: {len(pairs)} pairs =====")
        results = {}          # pair(tuple) -> [[winner, loser], conf]
        unresolved = []

        def attempt(pair_subset, batch_size, tag):
            todo_after = []
            for bstart in range(0, len(pair_subset), batch_size):
                chunk = pair_subset[bstart:bstart + batch_size]
                pdicts = [(P[a], P[b]) for a, b in chunk]
                prompt = build_user_prompt(pdicts, start_idx=0)
                try:
                    text = call_openrouter(SYSTEM_PROMPT, prompt)
                    got = parse_items(text, len(chunk), 0)
                except Exception as e:
                    print(f"  [{tag}] call failed ({chunk[0]}...): {e}")
                    todo_after.extend(chunk)
                    time.sleep(2)
                    continue
                # חשד ל'תשובה מנוונת' בבצ' גדול: הכל p1 עם ביטחון 1 — לא מקבלים, מנסים שוב
                if batch_size > 1 and len(got) == len(chunk) and \
                   all(d == 'p1' and c == 1 for d, c in got.values()):
                    print(f"  [{tag}] degenerate batch (all p1/c=1) — retrying as singles")
                    todo_after.extend(chunk)
                    continue
                for k, (a, b) in enumerate(chunk):
                    if k in got:
                        d, c = got[k]
                        w, l = (a, b) if d == 'p1' else (b, a)
                        results[(a, b)] = [[int(w), int(l)], float(c)]
                    else:
                        todo_after.append((a, b))
                done = len(results)
                print(f"  [{tag}] {done}/{len(pairs)} resolved")
                time.sleep(0.7)
            return todo_after

        todo = attempt(pairs, 5, "batch5")
        for r in (1, 2, 3):
            if not todo:
                break
            todo = attempt(todo, 1, f"single#{r}")
        unresolved = todo

        # שמירה בסדר ההצגה המקורי, בפורמט של קבצי ה-ranked
        ordered = []
        for p in pairs:
            if p in results:
                ordered.append(results[p])
        with open(OUT_DIR / f"{task_name}.json", "w") as f:
            json.dump(ordered, f)
        print(f"  SAVED {len(ordered)}/{len(pairs)} -> {OUT_DIR / (task_name + '.json')}")
        if unresolved:
            with open(OUT_DIR / f"UNRESOLVED_{task_name}.json", "w") as f:
                json.dump([list(p) for p in unresolved], f)
            print(f"  !! {len(unresolved)} pairs unresolved -> UNRESOLVED_{task_name}.json")

    print("\nסיימנו. אם אין UNRESOLVED — מעלים את שני קבצי ה-JSON מ-repair_output אליי.")

if __name__ == "__main__":
    main()
