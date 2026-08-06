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
from utils import FEATURE_COLS, align_feature_columns

import numpy as np
import pandas as pd
import sys
import os
import json
import pickle

def load_checkpoint(trial_dir: str, chunk_id: int | None = None):
    """
    Load one checkpoint pair written by your save_checkpoint():
      - checkpoint_alpha_performance[_chunk{chunk_id}].json
      - checkpoint_per_doctor_results[_chunk{chunk_id}].pkl
    Returns:
      alpha_performance (list of dicts),
      per_doctor_results (dict of lists),
      last_alpha_idx_completed (int or None),
      last_doctor_completed (str or None)
    """
    suffix = f"_chunk{chunk_id}" if chunk_id is not None else ""
    json_path = os.path.join(trial_dir, f"checkpoint_alpha_performance{suffix}.json")
    pkl_path  = os.path.join(trial_dir, f"checkpoint_per_doctor_results{suffix}.pkl")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing JSON: {json_path}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing PKL : {pkl_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    with open(pkl_path, "rb") as f:
        per_doc = pickle.load(f)

    alpha_performance = j.get("alpha_performance", [])
    last_alpha_idx_completed = j.get("last_alpha_idx_completed", None)
    last_doctor_completed = j.get("last_doctor_completed", None)
    return alpha_performance, per_doc, last_alpha_idx_completed, last_doctor_completed

def convert_np(obj):
    # dict → מורידים עמוק
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    # list, tuple, set → הופכים לרשימה של ערכים ממומרים
    elif isinstance(obj, (list, tuple, set)):
        return [convert_np(v) for v in obj]
    # כל סוג סקאלרי של NumPy → .item() (int, float, bool, וכו׳) 
    elif isinstance(obj, np.generic):
        return obj.item()
    # מערכים של NumPy → list
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # זמנים או תאריכים של pandas → מחרוזת ISO
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    # כל דבר אחר (str, int רגיל, float רגיל, bool רגיל) → מחזירים כמו שהוא
    else:
        return obj

def save_checkpoint(alpha_performance, per_doctor_results, last_idx_completed, last_doctor_completed=None):
    # Safely access CHUNK_ID and TRIAL_DIR from the main script's globals
    chunk_id = getattr(sys.modules["__main__"], "CHUNK_ID", None)
    save_dir = getattr(sys.modules["__main__"], "TRIAL_DIR", None)

    suffix = f"_chunk{chunk_id}" if chunk_id is not None else ""

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, f"checkpoint_alpha_performance{suffix}.json")
        pkl_path = os.path.join(save_dir, f"checkpoint_per_doctor_results{suffix}.pkl")
    else:
        json_path = f"checkpoint_alpha_performance{suffix}.json"
        pkl_path = f"checkpoint_per_doctor_results{suffix}.pkl"

    data_to_save = {
        "alpha_performance": alpha_performance,
        "last_alpha_idx_completed": last_idx_completed
    }
    if last_doctor_completed is not None:
        data_to_save["last_doctor_completed"] = last_doctor_completed

    # convert_np should be defined/imported in this module
    data_to_save_clean = convert_np(data_to_save)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save_clean, f, indent=2, ensure_ascii=False)

    with open(pkl_path, "wb") as f:
        pickle.dump(per_doctor_results, f)

    print(f"✓ Saved checkpoint for chunk {chunk_id or 'default'} to: {json_path}")

