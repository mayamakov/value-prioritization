import os, sys, math, json, csv, copy, random, pickle, shutil, concurrent.futures
from collections import defaultdict, Counter
from itertools import combinations
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.decomposition import PCA
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import seaborn as sns
import graphviz
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import ColorScaleRule
import re
import pandas as pd
from typing import Dict, Tuple, Optional, Iterable

# ---------- 21 recs total, 16 with positive prevalence ----------
ACTIVE_RECS = [
    'rec1','rec2','rec3','rec4','rec5','rec6','rec8','rec10','rec11',
    'rec12','rec13','rec16','rec17','rec18','rec19','rec21',
]  # rec7, rec9, rec14, rec15, rec20 are always zero in the data

ALL_RECS = [f'rec{i}' for i in range(1, 22)]  # rec1..rec21

# ---------- 6 path features, mapped from active 16 recs ----------
PATH_MAPPING = {
    'path_lab':       ['rec1', 'rec2', 'rec3', 'rec4'],
    'path_referrals': ['rec5', 'rec6', 'rec8'],
    'path_treatment': ['rec10', 'rec11', 'rec12', 'rec13'],
    'path_consult':   ['rec16', 'rec17'],
    'path_lifestyle': ['rec18', 'rec19'],
    'path_other':     ['rec21'],
}
PATH_FEATURE_NAMES = list(PATH_MAPPING.keys())  # 6 names

# ---------- FEATURE_COLS: 23 base + 6 paths = 29 features ----------
FEATURE_COLS = ['age', 'risk'] + ALL_RECS + PATH_FEATURE_NAMES


def add_path_features(df):
    """
    Adds 6 binary path-membership features to df.
    Each path = 1 if patient has at least one rec from that path.
    Returns a COPY.
    """
    df = df.copy()
    for path_name, recs in PATH_MAPPING.items():
        existing = [c for c in recs if c in df.columns]
        if not existing:
            df[path_name] = 0.0
        else:
            df[path_name] = (df[existing].sum(axis=1) > 0).astype(float)
    return df


def align_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has all 29 FEATURE_COLS. Adds path features if missing,
    zero-fills any other missing columns.
    """
    df = df.copy()
    if any(p not in df.columns for p in PATH_FEATURE_NAMES):
        df = add_path_features(df)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    if 'patient_num' not in df.columns:
        raise ValueError("DataFrame must contain 'patient_num'")
    return df[['patient_num'] + FEATURE_COLS].copy()

def get_patient_dict(patient_num, patient_df):
    row = patient_df.loc[patient_df['patient_num'] == patient_num].iloc[0]

    age  = float(row['age'])
    risk = float(row['risk'])  # 0..60

    recommendation_cols = [f'rec{i}' for i in range(1, 22)]

    recommendations_str = []
    rec_sum = 0
    for col in recommendation_cols:
        if col in row:
            val = int(row[col])
            rec_sum += val
            if val == 1:
                recommendations_str.append(col)

    recommendations_num = []
    for col in recommendation_cols:
        val = int(row[col]) if col in row else 0
        recommendations_num.append(val)

    return {
        'age': age,
        'risk': risk,
        'recommendations': recommendations_str,   # old code
        'rec_sum': rec_sum,
        'recommendations_num': recommendations_num # new code
    }

# Maya D - added scaling logic for age and risk  --- BUT THIS NEEDS TO BE ADDED/SWITCHED to split_pairs_train_test in create_ranking_data
def split_and_scale_data(patient_df):
    # Split patient_num into train and test sets
    all_train_patients, all_test_patients = train_test_split(
        patient_df['patient_num'].unique(), test_size=0.5, random_state=42
    )
    # Subset the dataframe
    df_train = patient_df[patient_df['patient_num'].isin(all_train_patients)]
    df_test = patient_df[patient_df['patient_num'].isin(all_test_patients)]
    # Copy for scaling
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    # Scale 'age' and 'risk' based on train data
    scaler = StandardScaler()
    df_train_scaled[['age', 'risk']] = scaler.fit_transform(df_train[['age', 'risk']])
    df_test_scaled[['age', 'risk']] = scaler.transform(df_test[['age', 'risk']])
    patient_df_scaled = patient_df.copy()
    patient_df_scaled[['age', 'risk']] = scaler.transform(patient_df_scaled[['age', 'risk']])
    return all_train_patients, all_test_patients, df_train, df_train_scaled, df_test, df_test_scaled, patient_df_scaled

def store_chosen_first(pair, decision, confidence):
    """
    If decision == "Patient 1", store (p1_num, p2_num), confidence
    If decision == "Patient 2", store (p2_num, p1_num), confidence
    If "No Decision"/"Tie" => keep pair as-is
    """
    p1_num, p2_num = pair
    if decision == "Patient 1":
        return ((p1_num, p2_num), confidence)
    elif decision == "Patient 2":
        return ((p2_num, p1_num), confidence)
    else:
        return (pair, confidence)
    
def print_ranking(ranking_scores, print_text):
    sorted_ranking_scores = sorted(ranking_scores, key=lambda x: x[1], reverse=True)
    final_ranking_df = pd.DataFrame(sorted_ranking_scores, columns=['patient_num', 'score'])
    print(print_text)
    print(final_ranking_df)

def get_weight(col_name):
    weights_dict = {
            'rec1': 1.0,
            'rec2': 2.0,
            'rec3': 1.5,
            'rec4': 1.5,
            'rec5': 3,
            'rec6': 4,
            'rec7': 4,
            'rec8': 4,
            'rec9': 10,
            'rec10': 10,
            'rec11':10,
            'rec12': 10,
            'rec13': 10,
            'rec14': 10,
            'rec15': 9,
            'rec16': 8,
            'rec17': 8,
            'rec18': 3,
            'rec19': 1,
            'rec20': 1,
            'rec21': 1,}
    return weights_dict.get(col_name, 1)

def compute_rec_weight_sum(row):
    rec_cols = [c for c in row.index if c.startswith("rec")]
    total = 0
    for col in rec_cols:
        total += row[col] * get_weight(col)
    return total    

def take_until_quota(src_pairs, quota, exclude_keys):
    """
    מחזיר עד *quota* זוגות שלא קיימים כבר ב-exclude_keys,
    ומעדכן את exclude_keys כדי שלא ייבחרו שוב.
    """
    picked = []
    for p1, p2 in src_pairs:
        k = tuple(sorted((p1, p2)))
        if k in exclude_keys:
            continue
        exclude_keys.add(k)
        picked.append((p1, p2))
        if len(picked) == quota:
            break
    return picked

def load_patient_df(patient_df_path):
    return pd.read_excel(patient_df_path)

def load_rec_translation_df():
    return pd.read_excel("rec_lists.xlsx")

def get_rec_weight_dict(patient_df):
    rec_translation_df=load_rec_translation_df()
    rec_cols = [c for c in patient_df.columns if c.startswith('rec')]
    pop = "DL"
    rec_weight_dict = rec_translation_df.set_index("rec").to_dict()["weight"]
    return rec_weight_dict

def get_rec_translation_dict(patient_df):
    rec_translation_df=load_rec_translation_df()
    rec_cols = [c for c in patient_df.columns if c.startswith('rec')]
    pop = "DL"
    rec_translation_dict = rec_translation_df.set_index("rec").to_dict()[pop]
    return rec_translation_dict

def get_rec_cols(patient_df):
    rec_cols = [c for c in patient_df.columns if c.startswith('rec')]
    return rec_cols

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -----alpha set name

def get_alpha_name_from_row(row):
    """
    row[0] -> המספר (100,200,300...)
    row[1] -> 1 אם kmeans
    row[2] -> 1 אם outlier
    row[3] -> 1 אם conflict
    row[4] -> 1 אם patients_selection
    row[5] -> 0.5 אם child_chain (יחד עם row[6] == 0.5)
    row[6] -> 0.5 אם child_chain
    row[7] -> 1 אם random
    row[8]  -> 1 אם systematic
    """
    number = row[0]

    if row[1] == 1:
        return f"KMEANS{number}"
    elif row[2] == 1:
        return f"OUTLIER{number}"
    elif row[3] == 1:
        return f"CONFLICT{number}"
    elif row[4] == 1:
        return f"PATIENTS_SELECTION{number}"
    elif row[5] == 0.5 and row[6] == 0.5:
        return f"CHILD_CHAIN{number}"
    elif row[7] == 1:
        return f"RANDOM{number}"
    elif row[8] == 1:
        return f"Systematic{number}"
    elif row[9] == 1:
        return f"Influential{number}"
    else:
        return f"UNKNOWN{number}"  # אם לא נמצא שום תנאי תואם
    
def get_rank_dict(ranking_scores):
    sorted_scores = sorted(ranking_scores, key=lambda x: x[1], reverse=True)
    return {patient: rank for rank, (patient, _) in enumerate(sorted_scores, start=1)}

# utils.py  – גרסה חדשה לפונקציה
BASE_FEATURES = ["age", "risk"] + [f"rec{i}" for i in range(1, 22)] + PATH_FEATURE_NAMES

# def _core_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Age, risk, rec1…rec21 בלבד – מומר ל-float32 לפי הצורך."""
#     cols = [c for c in BASE_FEATURES if c in df.columns]
#     return df[cols].astype(float)

def _core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Age, risk, rec1…rec21 בלבד – עם שמירה קשוחה על מבנה קבוע"""
    cols = BASE_FEATURES.copy()

    # ודא שכל עמודה קיימת, ואם לא – הוסף עם ערך 0
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0

    # סדר קבוע + המרה ל-float32
    return df[cols].astype(np.float32)


BASE_FEATURES = ["age", "risk"] + [f"rec{i}" for i in range(1, 22)] + PATH_FEATURE_NAMES

def _core_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = BASE_FEATURES.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[cols].astype(np.float32)


# ----------------------------------------------
# בתחתית utils.py, אחרי ההגדרה של _core_features
# ----------------------------------------------
def predict_ranking_scores(model, patient_df, model_type="ranknet"):
    # שמירה על סדר ופיצ'רים קבוע
    patient_nums = patient_df["patient_num"].values
    feats_df     = _core_features(patient_df)

    if model_type == "ranknet":
        model.eval()
        with torch.no_grad():
            # קוראים את כל השורות בבת אחת
            x = torch.tensor(feats_df.values, dtype=torch.float32)  # [N, 23]
            y = model(x).squeeze().tolist()                         # [N]
        return list(zip(patient_nums, y))

    elif model_type == "lambdamart":
        preds = model.predict(feats_df.values)                    # [N]
        return list(zip(patient_nums, preds))

    else:
        raise ValueError("Unknown model type provided!")

# def predict_ranking_scores(model, patient_df, model_type="ranknet"):
#     patient_nums = patient_df["patient_num"].values
#     feats_df     = _core_features(patient_df)

#     if model_type == "ranknet":
#         model.eval()
#         with torch.no_grad():
#             x = torch.tensor(feats_df.values, dtype=torch.float32)
#             y = model(x).squeeze().tolist()
#         return list(zip(patient_nums, y))

#     elif model_type == "lambdamart":
#         preds = model.predict(feats_df.values)
#         return list(zip(patient_nums, preds))

#     else:
#         raise ValueError("Unknown model type provided!")


# def predict_ranking_scores(model, patient_df, model_type='ranknet'):
#     patient_nums = patient_df['patient_num'].values
#     scores = []

#     if model_type == 'ranknet':
#         model.eval()
#         with torch.no_grad():
#             for patient_num in patient_nums:
#                 features = patient_df.loc[
#                     patient_df['patient_num'] == patient_num
#                 ].drop('patient_num', axis=1).values.flatten()
#                 features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
#                 score = model(features).item()
#                 scores.append((patient_num, score))

#     elif model_type == 'lambdamart':
#         features = patient_df.drop(columns=['patient_num']).values
#         predicted_scores = model.predict(features)
#         scores = list(zip(patient_nums, predicted_scores))

#     else:
#         raise ValueError("Unknown model type provided!")

#     return scores


def filter_patients_with_recs(patient_df):
    "returns the patient_df of only patients with recs"
    rec_cols = [col for col in patient_df.columns if col.startswith("rec") and col[3:].isdigit()]
    return patient_df[patient_df[rec_cols].sum(axis=1) > 0]



def filter_pairs_by_doctor_truth(
    pairs_chosen_by_method, # list of (i, j) pairs selected by the sampler
    all_train_pairs, #true 
    doctor_key
):
    """
        pairs_chosen_by_method: list of (i, j) pairs selected by the sampler.
        all_train_pairs: dict of doctor keys to true pairwise labels, e.g.:
                         {'doctor1': [((i, j), confidence), ...]}
        doctor_key: the doctor to filter with (e.g., 'doctor1')
    """
    if doctor_key not in all_train_pairs:
        raise ValueError(f"Doctor key '{doctor_key}' not found in all_train_pairs")

    # Build lookup: (winner, loser) → confidence
    true_pair_dict = {
        (int(winner), int(loser)): conf
        for (winner, loser), conf in all_train_pairs[doctor_key]
    }

    result = []
    for a, b in pairs_chosen_by_method:
        pair_forward  = (int(a), int(b))
        pair_backward = (int(b), int(a))

        if pair_forward in true_pair_dict:
            result.append((pair_forward, true_pair_dict[pair_forward]))
        elif pair_backward in true_pair_dict:
            result.append((pair_backward, true_pair_dict[pair_backward]))
        # else: skip this pair

    return result


def load_rec_weights_and_patient_map(patient_df, rec_file="rec_lists.xlsx", population_col="DL"):
    dfw = pd.read_excel(rec_file)
    rec_weight_dict = dfw.set_index("rec").to_dict()["weight"]
    rec_translation = dfw.set_index("rec").to_dict().get(population_col, {})

    # patient_rec_weight_dict (סכום משוקלל של כל עמודות rec*)
    rec_cols = [c for c in patient_df.columns if c.startswith("rec")]
    rec_only = patient_df[["patient_num"] + rec_cols].copy()
    for c in rec_cols:
        rec_only[c] *= rec_weight_dict.get(c, 0.0)
    rec_only["rec_weight"] = rec_only[rec_cols].sum(axis=1)
    patient_rec_weight_dict = rec_only.set_index("patient_num")["rec_weight"].to_dict()

    print(f"✓ rec weights loaded from {rec_file} ({len(rec_weight_dict)} רשומות)")
    return rec_weight_dict, rec_translation, patient_rec_weight_dict

def save_pkl_with_meta(obj, path, model_type: str, al_variant: int):
    payload = {
        "meta": {
            "model_type": model_type,
            "al_variant": int(al_variant),
        },
        "data": obj,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl_with_meta(path):
    """
    Returns: (data, meta_dict). If file is old-style (no meta), meta_dict = {}.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "data" in payload and "meta" in payload:
        return payload["data"], payload["meta"]
    # backward compatibility (file was a raw object)
    return payload, {}

