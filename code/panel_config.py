# -*- coding: utf-8 -*-
"""
panel_config.py  —  SINGLE SOURCE OF TRUTH for the rater panel.
Import the lists/colours from here in every analysis & figure script, so adding
a model later means editing only THIS file. Pure-python (no torch/matplotlib),
safe to import anywhere.

To add a model: append its key to the right family in MODELS_BY_FAMILY, add a
DISPLAY name, and (if it uses the legacy single-test-file format) to LEGACY.
"""

# ----- physicians -----------------------------------------------------------
FAMILY = ['fm_1', 'fm_2', 'fm_3', 'fm_4', 'fm_5']
PUBLIC = ['ph_1', 'ph_2', 'ph_3', 'ph_4', 'ph_5']
HUMANS = FAMILY + PUBLIC

# ----- language models, grouped by vendor family (plot order) ---------------
MODELS_BY_FAMILY = {
    'OpenAI':    ['doctor8', 'doctor9', 'gpt_5', 'gpt_5_4', 'gpt_5_5'],
    'Anthropic': ['claude_opus_4_5', 'claude_opus_4_7', 'claude_opus_4_8'],
    'Google':    ['gemini_2_5_pro', 'gemini_3_1_pro', 'gemini_3_5_flash'],
    'Meta':      ['doctor10', 'llama_3_3_70b'],
    'DeepSeek':  ['deepseek_v3_2'],
    'Qwen':      ['qwen3_5_27b'],
    'Mistral':   ['mistral_medium_3_5'],
}
MODELS = [k for fam in MODELS_BY_FAMILY.values() for k in fam]   # 16, family order
MODEL_FAMILY = {k: fam for fam, ks in MODELS_BY_FAMILY.items() for k in ks}

# legacy = single combined test file (<key>_test_ranked.json); the rest use A/B
LEGACY    = ['doctor8', 'doctor9', 'doctor10']
LEGACYSET = set(LEGACY)
NEWER     = [k for k in MODELS if k not in LEGACYSET]

# aliases used by the various scripts
LLMS  = MODELS
PANEL = HUMANS + MODELS

# ----- display names --------------------------------------------------------
DISPLAY = {
    'doctor8': 'GPT-4o-mini', 'doctor9': 'GPT-5-mini', 'doctor10': 'Llama-4 Maverick',
    'gpt_5': 'GPT-5', 'gpt_5_4': 'GPT-5.4', 'gpt_5_5': 'GPT-5.5',
    'claude_opus_4_5': 'Claude Opus 4.5', 'claude_opus_4_7': 'Claude Opus 4.7',
    'claude_opus_4_8': 'Claude Opus 4.8',
    'gemini_2_5_pro': 'Gemini 2.5 Pro', 'gemini_3_1_pro': 'Gemini 3.1 Pro',
    'gemini_3_5_flash': 'Gemini 3.5 Flash',
    'llama_3_3_70b': 'Llama 3.3 70B', 'deepseek_v3_2': 'DeepSeek V3.2',
    'qwen3_5_27b': 'Qwen3.5 27B', 'mistral_medium_3_5': 'Mistral Medium 3.5',
}
# smaller / non-flagship tier (note in captions; comparison fairness)
SMALL_TIER = {'doctor8', 'doctor9', 'gemini_3_5_flash', 'qwen3_5_27b'}

# ----- colours --------------------------------------------------------------
FAM_C = '#6baed6'   # family-medicine physicians (light blue)
PUB_C = '#08306b'   # public-health physicians   (navy)
FAMILY_COLOR = {    # ColorBrewer Dark2 — distinct, print-safe, NOT blue
    'OpenAI':   '#1b9e77', 'Anthropic': '#d95f02', 'Google':  '#e7298a',
    'Meta':     '#7570b3', 'DeepSeek':  '#66a61e', 'Qwen':    '#a6761d',
    'Mistral':  '#e6ab02',
}

def _hex_to_rgb(h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
def _rgb_to_hex(t): return '#%02x%02x%02x' % tuple(int(round(x)) for x in t)

def model_color(k, lighten_within_family=True):
    """Family base colour, lightened slightly per member so same-family lines differ."""
    fam = MODEL_FAMILY[k]; base = _hex_to_rgb(FAMILY_COLOR[fam])
    members = MODELS_BY_FAMILY[fam]
    if not lighten_within_family or len(members) == 1:
        return FAMILY_COLOR[fam]
    f = 0.50 * members.index(k) / (len(members) - 1)        # 0 .. 0.50
    return _rgb_to_hex(tuple(c + (255 - c) * f for c in base))

def display(k):  return DISPLAY.get(k, k)
def family(k):   return MODEL_FAMILY.get(k, 'Other')

if __name__ == '__main__':
    print(f'{len(HUMANS)} physicians, {len(MODELS)} models, {len(PANEL)} raters')
    for fam, ks in MODELS_BY_FAMILY.items():
        print(f'  {fam:10} ({len(ks)}): ' + ', '.join(display(k) for k in ks))
