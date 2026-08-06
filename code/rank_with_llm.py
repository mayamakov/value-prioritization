"""
==========================================================================
rank_with_llm.py
==========================================================================
Rank patient pairs using an LLM API and save the results in the same
format as the human physicians. Supports Anthropic Claude, OpenAI GPT,
and Google Gemini.

OUTPUT: produces  physicians_results/<llm_name>/<llm_name>_train_iter_*_ranked.json
                  physicians_results/<llm_name>/<llm_name>_test_part_A_ranked.json
                  physicians_results/<llm_name>/<llm_name>_test_part_B_ranked.json
        in the EXACT format the other code expects.

USAGE (from Jupyter or shell):

    # 1) Pick the model and key
    import rank_with_llm as rk
    rk.rank_one_llm(
        llm_name='claude_opus_4',     # folder name under physicians_results/
        provider='anthropic',          # one of: anthropic / openai / google
        model='claude-opus-4-7',       # the actual model string
        api_key=os.environ.get('LLM_API_KEY', ''),
    )

    # That will:
    #  - Read the same train+test pair lists the humans used
    #  - Call the LLM ~500 times (one prompt per pair, batched up to 50)
    #  - Save results in the correct folder/format
    #  - Track the cost in dollars

Cost estimate per LLM (for ~500 pairs):
  Claude Opus 4.7:  ~$2-3
  GPT-5:            ~$2-3
  Gemini 2.5 Pro:   ~$1-2

If you'd rather use OpenRouter (one account for all three), change
`provider` to 'openrouter' and set `api_key` to your OpenRouter key.
==========================================================================
"""
from __future__ import annotations
import os, json, time, pickle, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd


# ==========================================================================
# Patient -> text representation (same logic as the legacy llm trial)
# ==========================================================================

# Recommendation short descriptions (from rec_lists.xlsx). Used to render
# patient features as readable text for the LLM.
REC_TEXTS = {
    'rec1':  'Basic lab panel (LDL/HDL/A1C)',
    'rec2':  'Advanced lab panel (ApoB/Lpa)',
    'rec3':  'Pathophysiology investigation (genetic/secondary causes)',
    'rec4':  'Routine LDL monitoring',
    'rec5':  'Diagnostic imaging (carotid Doppler)',
    'rec6':  'Advanced imaging (CTA / perfusion)',
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


def patient_to_text(patient_row: Dict) -> str:
    """Renders a single patient as a 1-3 line description for the LLM."""
    age  = int(patient_row['age'])
    risk = float(patient_row['risk'])
    recs = [REC_TEXTS[r] for r in ACTIVE_RECS
            if r in patient_row and patient_row[r] >= 0.5]
    parts = [f"age {age}", f"10-year CV risk {risk:.0f}%"]
    if recs:
        parts.append(f"Care plan: {'; '.join(recs)}")
    else:
        parts.append("Care plan: none")
    return ' | '.join(parts)


# ==========================================================================
# System prompt (concise; the LLM should rank by priority)
# ==========================================================================
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


def build_user_prompt(pairs_batch: List[Tuple[Dict, Dict]],
                      start_idx: int = 0) -> str:
    """Build a prompt for a batch of pairs."""
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


# ==========================================================================
# Provider-agnostic API call
# ==========================================================================

def call_anthropic(system_prompt, user_prompt, model, api_key):
    """Call Claude API."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens


def call_openai(system_prompt, user_prompt, model, api_key):
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return (response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens)


def call_google(system_prompt, user_prompt, model, api_key):
    """Call Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
    response = m.generate_content(
        user_prompt,
        generation_config={
            "response_mime_type": "application/json",
            "max_output_tokens": 4096,
        },
    )
    # Gemini doesn't always give precise token counts; use approximations
    try:
        in_tok = response.usage_metadata.prompt_token_count
        out_tok = response.usage_metadata.candidates_token_count
    except AttributeError:
        in_tok = len(user_prompt) // 4
        out_tok = len(response.text) // 4
    return response.text, in_tok, out_tok


def call_openrouter(system_prompt, user_prompt, model, api_key):
    """Call OpenRouter — the OpenAI client works with their endpoint."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model=model,  # e.g. "anthropic/claude-opus-4-7" or "openai/gpt-5"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return (response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens)


PROVIDERS = {
    'anthropic':  call_anthropic,
    'openai':     call_openai,
    'google':     call_google,
    'openrouter': call_openrouter,
}


# ==========================================================================
# Parse LLM JSON response into [(winner_pid, loser_pid), confidence] list
# ==========================================================================
def parse_response(text: str, pairs_batch: List[Tuple[Dict, Dict]],
                   start_idx: int = 0) -> List[Tuple[Tuple[int, int], float]]:
    """Parse the LLM's JSON output into the ranked-pairs format."""
    # Some LLMs wrap JSON in markdown fences
    txt = text.strip()
    if txt.startswith('```'):
        txt = '\n'.join(txt.split('\n')[1:-1])
    try:
        data = json.loads(txt)
    except json.JSONDecodeError as e:
        print(f"  WARN: failed to parse LLM output as JSON: {e}")
        print(f"  Raw text: {text[:200]}")
        return []

    items = data.get('items', [])
    by_id = {it['id']: it for it in items}

    out = []
    for i, (p1, p2) in enumerate(pairs_batch):
        pair_id = f"pair_{start_idx + i}"
        if pair_id not in by_id:
            # Fall back to "p1" with low confidence
            print(f"  WARN: LLM did not return decision for {pair_id}")
            out.append(((int(p1['patient_num']), int(p2['patient_num'])), 1.0))
            continue
        item = by_id[pair_id]
        d = item.get('d', 'p1')
        c = float(item.get('c', 3))
        if d == 'p1':
            winner, loser = p1['patient_num'], p2['patient_num']
        else:
            winner, loser = p2['patient_num'], p1['patient_num']
        out.append(((int(winner), int(loser)), c))

    return out


# ==========================================================================
# Main ranking loop
# ==========================================================================
def _patient_dict(patient_df: pd.DataFrame, pid: int) -> Dict:
    """Get a patient row as a dict, keyed by patient_num."""
    row = patient_df[patient_df['patient_num'] == pid].iloc[0]
    return row.to_dict()


def rank_pair_list(
    pair_ids: List[Tuple[int, int]],
    patient_df: pd.DataFrame,
    provider: str,
    model: str,
    api_key: str,
    batch_size: int = 25,
    progress_label: str = "",
    sleep_between_calls: float = 0.5,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Rank a list of (pid_a, pid_b) pairs using the LLM. Returns a list of
    ((winner, loser), confidence) — same format as the human files.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(PROVIDERS)}")
    api_fn = PROVIDERS[provider]

    # Convert pair_ids -> [(p1_dict, p2_dict), ...]
    pairs_dicts = []
    for a, b in pair_ids:
        pa = _patient_dict(patient_df, a)
        pb = _patient_dict(patient_df, b)
        pairs_dicts.append((pa, pb))

    results = []
    total_in_tok = 0
    total_out_tok = 0

    for batch_start in range(0, len(pairs_dicts), batch_size):
        batch = pairs_dicts[batch_start:batch_start + batch_size]
        user_prompt = build_user_prompt(batch, start_idx=batch_start)
        try:
            text, in_tok, out_tok = api_fn(SYSTEM_PROMPT, user_prompt, model, api_key)
            total_in_tok += in_tok
            total_out_tok += out_tok
        except Exception as e:
            print(f"  ERROR at batch {batch_start}: {e}")
            time.sleep(5)
            continue
        parsed = parse_response(text, batch, start_idx=batch_start)
        results.extend(parsed)
        done = min(batch_start + batch_size, len(pairs_dicts))
        print(f"  {progress_label}  {done}/{len(pairs_dicts)} pairs  "
              f"(in={total_in_tok:,}, out={total_out_tok:,} tokens)")
        time.sleep(sleep_between_calls)

    return results


def rank_one_llm(
    llm_name: str,
    provider: str,
    model: str,
    api_key: str,
    pairs_dir: str = 'physicians_results',
    template_doctor: str = 'maya_makov',
    patient_df_path: str = 'synthetic_data (7).xlsx',
    batch_size: int = 25,
):
    """
    Run a full set of rankings for one LLM, using the same pair list as
    one of the human doctors (default: maya_makov) to ensure comparability.

    - Reads:  physicians_results/<template_doctor>/<template_doctor>_train_iter_*_ranked.json
              physicians_results/<template_doctor>/<template_doctor>_test_part_*_ranked.json
              (we only use the patient_num pairs from these, not the rankings)
    - Writes: physicians_results/<llm_name>/<llm_name>_train_iter_*_ranked.json
              physicians_results/<llm_name>/<llm_name>_test_part_*_ranked.json
    """
    pairs_dir = Path(pairs_dir)
    tmpl = pairs_dir / template_doctor
    out  = pairs_dir / llm_name
    out.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Ranking with {provider} / {model}  ->  {llm_name}")
    print(f"Template doctor:  {template_doctor}")
    print(f"Output folder:    {out}")
    print(f"{'='*70}\n")

    # Load patient data
    patient_df = pd.read_excel(patient_df_path)

    # ---- Train iters 0..3 ----
    for it in range(4):
        # The template file has ((a,b), conf) entries; we just need the pair ids
        src = tmpl / f"{template_doctor}_train_iter_{it}_ranked.json"
        if not src.exists():
            print(f"  WARN: skipping iter {it}, no source file at {src}")
            continue
        with open(src) as f:
            raw = json.load(f)
        pair_ids = []
        for item in raw:
            try:
                inner, _conf = item
                a, b = inner
                pair_ids.append((int(a), int(b)))
            except (ValueError, TypeError):
                continue
        print(f"Train iter {it}: {len(pair_ids)} pairs to rank")
        results = rank_pair_list(
            pair_ids, patient_df, provider, model, api_key,
            batch_size=batch_size, progress_label=f"[train iter {it}]",
        )
        dst = out / f"{llm_name}_train_iter_{it}_ranked.json"
        with open(dst, 'w') as f:
            json.dump([[list(pair), conf] for (pair, conf) in results], f)
        print(f"  Saved {dst}\n")

    # ---- Test parts A + B ----
    for part in ['A', 'B']:
        src = tmpl / f"{template_doctor}_test_part_{part}_ranked.json"
        if not src.exists():
            print(f"  WARN: skipping test part {part}, no source file at {src}")
            continue
        with open(src) as f:
            raw = json.load(f)
        pair_ids = []
        for item in raw:
            try:
                inner, _conf = item
                a, b = inner
                pair_ids.append((int(a), int(b)))
            except (ValueError, TypeError):
                continue
        print(f"Test part {part}: {len(pair_ids)} pairs to rank")
        results = rank_pair_list(
            pair_ids, patient_df, provider, model, api_key,
            batch_size=batch_size, progress_label=f"[test {part}]",
        )
        dst = out / f"{llm_name}_test_part_{part}_ranked.json"
        with open(dst, 'w') as f:
            json.dump([[list(pair), conf] for (pair, conf) in results], f)
        print(f"  Saved {dst}\n")

    print(f"\n\u2713 Done with {llm_name}. Add '{llm_name}' to NEW_LLMS in llm_helpers.py.")


# ==========================================================================
# Quick CLI use
# ==========================================================================
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, help='Folder name for output (e.g. claude_opus_4)')
    p.add_argument('--provider', required=True,
                   choices=['anthropic', 'openai', 'google', 'openrouter'])
    p.add_argument('--model', required=True, help='Model string (e.g. claude-opus-4-7)')
    p.add_argument('--key', required=True, help='API key')
    p.add_argument('--template', default='maya_makov',
                   help='Doctor whose pair list we mirror (default: maya_makov)')
    p.add_argument('--batch', type=int, default=25)
    args = p.parse_args()

    rank_one_llm(
        llm_name=args.name, provider=args.provider, model=args.model,
        api_key=args.key, template_doctor=args.template,
        batch_size=args.batch,
    )