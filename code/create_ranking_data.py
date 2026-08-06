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
import random
# internal imports
import doctor_heuristics
# from doctor_heuristics import (
#     doctor_decision_tree_1, doctor_decision_tree_2, doctor_decision_tree_3, doctor_decision_tree_4, doctor_decision_tree_5,
#     doctor_decision_tree_6, doctor_decision_tree_7, doctor_decision_tree_8, doctor_decision_tree_9, doctor_decision_tree_10
# )
from utils import get_patient_dict

# FEATURE_COLS and align_feature_columns are now imported from utils
# (see top of this file). They include the 6 path features.


def rank_all_doctors_for_pairs(pairs, patient_df,PAIRS_ONLY):
    """
    Processes patient pairs using either Decision Trees or LLM.
    """

    # טעינה מפורשת של פונקציות החלטה לפי בחירתך
    doctor_decision_funcs = {}

    if PAIRS_ONLY:
        import patient_ranking_llm as _llm
        _llm.activate_as_doctor_heuristics()
        print("[Info] Activated Gemma-based heuristics.")
        # טעינה מהמודול שהוחלף
        import doctor_heuristics
        doctor_decision_funcs = {
            f'doctor{i}': getattr(doctor_heuristics, f'doctor_decision_tree_{i}')
            for i in range(1, 11)
        }
    else:
        print("[Info] Using original Decision Tree heuristics.")
        from doctor_heuristics import (
            doctor_decision_tree_1, doctor_decision_tree_2, doctor_decision_tree_3, 
            doctor_decision_tree_4, doctor_decision_tree_5, 
        doctor_decision_tree_6, 
            doctor_decision_tree_7, doctor_decision_tree_8, doctor_decision_tree_9, 
            doctor_decision_tree_10
        )
        doctor_decision_funcs = {
            'doctor1': doctor_decision_tree_1,
            'doctor2': doctor_decision_tree_2,
            'doctor3': doctor_decision_tree_3,
            'doctor4': doctor_decision_tree_4,
            'doctor5': doctor_decision_tree_5,
            'doctor6': doctor_decision_tree_6,
            'doctor7': doctor_decision_tree_7,
            'doctor8': doctor_decision_tree_8,
            'doctor9': doctor_decision_tree_9,
            'doctor10': doctor_decision_tree_10,
        }

    doctor_confidences = {f'doctor{i}_confidences': [] for i in range(1, 11)}


    # Helper function to determine preferred pair given a decision and the input pair

    def determine_preferred_pair(decision, numA, numB):
        if decision == "Patient 1":
            return (numA, numB)
        elif decision == "Patient 2":
            return (numB, numA)
        else:
            return (numA, numB)
    
    for (numA, numB) in pairs:

        numA, numB = int(numA), int(numB)

        try:
            p1 = get_patient_dict(numA, patient_df)
            p2 = get_patient_dict(numB, patient_df)
        except IndexError:
            print(f"Patient pair ({numA}, {numB}) not found in the DataFrame.")
            continue
    
        for idx in range(1, 11):
            doc_key = f'doctor{idx}'
            decision_func = doctor_decision_funcs[doc_key]
            decision, confidence = decision_func(p1, p2)
            preferred_pair = determine_preferred_pair(decision, numA, numB)
            doctor_confidences[f'{doc_key}_confidences'].append((preferred_pair, confidence))
    
    return doctor_confidences

        # # DOC1
        # dec1, conf1 = doctor_decision_tree_1(p1, p2)
        # preferred_pair = determine_preferred_pair(dec1, numA, numB)
        # doctor_confidences['doctor1_confidences'].append((preferred_pair, conf1))

        # # DOC2
        # dec2, conf2 = doctor_decision_tree_2(p1, p2)
        # preferred_pair = determine_preferred_pair(dec2, numA, numB)
        # doctor_confidences['doctor2_confidences'].append((preferred_pair, conf2))

        # # DOC3
        # dec3, conf3 = doctor_decision_tree_3(p1, p2)
        # preferred_pair = determine_preferred_pair(dec3, numA, numB)
        # doctor_confidences['doctor3_confidences'].append((preferred_pair, conf3))

        # # DOC4
        # dec4, conf4 = doctor_decision_tree_4(p1, p2)
        # preferred_pair = determine_preferred_pair(dec4, numA, numB)
        # doctor_confidences['doctor4_confidences'].append((preferred_pair, conf4))

        # # DOC5
        # dec5, conf5 = doctor_decision_tree_5(p1, p2)
        # preferred_pair = determine_preferred_pair(dec5, numA, numB)
        # doctor_confidences['doctor5_confidences'].append((preferred_pair, conf5))

        # # DOC6
        # dec6, conf6 = doctor_decision_tree_6(p1, p2)
        # preferred_pair = determine_preferred_pair(dec6, numA, numB)
        # doctor_confidences['doctor6_confidences'].append((preferred_pair, conf6))

        # # DOC7
        # dec7, conf7 = doctor_decision_tree_7(p1, p2)
        # preferred_pair = determine_preferred_pair(dec7, numA, numB)
        # doctor_confidences['doctor7_confidences'].append((preferred_pair, conf7))

        # # DOC8
        # dec8, conf8 = doctor_decision_tree_8(p1, p2)
        # preferred_pair = determine_preferred_pair(dec8, numA, numB)
        # doctor_confidences['doctor8_confidences'].append((preferred_pair, conf8))

        # # DOC9
        # dec9, conf9 = doctor_decision_tree_9(p1, p2)
        # preferred_pair = determine_preferred_pair(dec9, numA, numB)
        # doctor_confidences['doctor9_confidences'].append((preferred_pair, conf9))

        # # DOC10
        # dec10, conf10 = doctor_decision_tree_10(p1, p2)
        # preferred_pair = determine_preferred_pair(dec10, numA, numB)
        # doctor_confidences['doctor10_confidences'].append((preferred_pair, conf10))


def rank_single_doctor_for_pairs(pairs, patient_df, doctor_name):
    """
    Runs a single doctor's decision tree for a list of patient pairs.
    Returns the pair ranked higher by the doctor, and the confidence.
    """
    from doctor_heuristics import (
        doctor_decision_tree_1, doctor_decision_tree_2, doctor_decision_tree_3, doctor_decision_tree_4, doctor_decision_tree_5, doctor_decision_tree_6, doctor_decision_tree_7, doctor_decision_tree_8, doctor_decision_tree_9, doctor_decision_tree_10)
    pairs = [(int(a), int(b)) for (a, b) in pairs]

    print('rank_single_doctor_for_pairs', doctor_name, len(pairs), sorted(pairs))
    results = []

    # New: simpler map without global lists
    doctor_map = {
        'doctor1': doctor_decision_tree_1,
        'doctor2': doctor_decision_tree_2,
        'doctor3': doctor_decision_tree_3,
        'doctor4': doctor_decision_tree_4,
        'doctor5': doctor_decision_tree_5,
        'doctor6': doctor_decision_tree_6,
        'doctor7': doctor_decision_tree_7,
        'doctor8': doctor_decision_tree_8,
        'doctor9': doctor_decision_tree_9,
        'doctor10': doctor_decision_tree_10,
    }

    if doctor_name not in doctor_map:
        print(f"Doctor {doctor_name} not found.")
        return results

    decision_func = doctor_map[doctor_name]

    for (numA, numB) in pairs:
        numA, numB = int(numA), int(numB)

        try:
            p1 = get_patient_dict(numA, patient_df)
            p2 = get_patient_dict(numB, patient_df)
        except IndexError:
            print(f"Patient pair ({numA}, {numB}) not found in the DataFrame.")
            continue

        decision, confidence = decision_func(p1, p2)

        if not isinstance(confidence, (int, float)):
            print(f"Invalid confidence type for pair ({numA}, {numB}): {confidence}")
            confidence = 0

        if decision == "Patient 1":
            preferred_pair = (numA, numB)
        elif decision == "Patient 2":
            preferred_pair = (numB, numA)
        else:
            preferred_pair = (numA, numB)

        # Only keep pairs with meaningful confidence
        results.append((preferred_pair, confidence))

    return results

def compute_pairs_and_ranking_for_all_doctors(patient_df):
    """
    For each doctor, do one pass over all unique patient pairs in patient_df,
    computing both pairwise decisions and overall win counts.
    Returns (overall_rankings, overall_pairs).
    """
    from doctor_heuristics import (
    doctor_decision_tree_1, doctor_decision_tree_2, doctor_decision_tree_3, doctor_decision_tree_4, doctor_decision_tree_5,
    doctor_decision_tree_6, doctor_decision_tree_7, doctor_decision_tree_8, doctor_decision_tree_9, doctor_decision_tree_10
)

    
    doctor_funcs = {
       "doctor1": doctor_decision_tree_1,
       "doctor2": doctor_decision_tree_2,
       "doctor3": doctor_decision_tree_3,
       "doctor4": doctor_decision_tree_4,
       "doctor5": doctor_decision_tree_5,
       "doctor6": doctor_decision_tree_6,
       "doctor7": doctor_decision_tree_7,
       "doctor8": doctor_decision_tree_8,
       "doctor9": doctor_decision_tree_9,
       "doctor10": doctor_decision_tree_10,
    }

    overall_rankings = {}
    overall_pairs = {}

    all_patients = patient_df['patient_num'].unique()
    n = len(all_patients)

    for doc_name, decision_func in doctor_funcs.items():
        print(f"[{doc_name}] Computing pairwise decisions and overall ranking ...")
        pairwise_decisions = []
        win_counts = {p: 0 for p in all_patients}

        # Loop over every unique pair (p1, p2) with p1 < p2
        for i in range(n):
            for j in range(i + 1, n):
                p1_dict = get_patient_dict(all_patients[i], patient_df)
                p2_dict = get_patient_dict(all_patients[j], patient_df)
                decision, confidence = decision_func(p1_dict, p2_dict)

                # Ensure the selected patient always appears first
                if decision == "Patient 1":
                    p_winner, p_loser = all_patients[i], all_patients[j]
                elif decision == "Patient 2":
                    p_winner, p_loser = all_patients[j], all_patients[i]
                else:
                    # Tie
                    p_winner, p_loser = all_patients[i], all_patients[j]
                    confidence = 0.5

                pairwise_decisions.append(((p_winner, p_loser), confidence))
                win_counts[p_winner] += 1

        # Sort patients by descending win count
        ranked_patients = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
        overall_rankings[doc_name] = ranked_patients
        overall_pairs[doc_name] = pairwise_decisions

    return overall_rankings, overall_pairs

def split_pairs_train_test(overall_pairs, df_train, df_test):
    """
    Splits the pairs from overall_pairs into train_pairs and test_pairs,
    depending on which patients are in train vs. test.
    """
    train_patients = set(df_train['patient_num'].unique())
    test_patients  = set(df_test['patient_num'].unique())

    train_pairs = {}
    test_pairs = {}

    for doc_name, pairs_list in overall_pairs.items():
        train_pairs[doc_name] = []
        test_pairs[doc_name] = []

        for ((p1_selected, p1_other), confidence) in pairs_list:
            # Both in train?
            if p1_selected in train_patients and p1_other in train_patients:
                train_pairs[doc_name].append(((p1_selected, p1_other), confidence))
            # Both in test?
            elif p1_selected in test_patients and p1_other in test_patients:
                test_pairs[doc_name].append(((p1_selected, p1_other), confidence))
            # Else: one is train, other is test => skip

        train_pairs[doc_name].sort(key=lambda x: x[1], reverse=True)
        test_pairs[doc_name].sort(key=lambda x: x[1], reverse=True)

    return train_pairs, test_pairs

def compute_train_test_rankings(overall_pairs, df_train, df_test):
    """
    Re-accumulate win counts for train/test sets separately,
    returning (train_rankings, test_rankings).
    """
    train_patients = set(df_train['patient_num'].unique())
    test_patients  = set(df_test['patient_num'].unique())

    train_rankings = {}
    test_rankings  = {}

    for doc_name, pairs_list in overall_pairs.items():
        train_wins = {p: 0 for p in train_patients}
        test_wins  = {p: 0 for p in test_patients}

        for ((p1_selected, p1_other), confidence) in pairs_list:
            # If both in train
            if p1_selected in train_patients and p1_other in train_patients:
                train_wins[p1_selected] += 1
            # If both in test
            elif p1_selected in test_patients and p1_other in test_patients:
                test_wins[p1_selected] += 1

        # Sort by descending # of wins
        train_rankings[doc_name] = sorted(train_wins.items(), key=lambda x: x[1], reverse=True)
        test_rankings[doc_name]  = sorted(test_wins.items(), key=lambda x: x[1], reverse=True)

    return train_rankings, test_rankings

def run_full_pipeline(patient_df, df_train, df_test):
    """
    Runs the entire pipeline and returns all outputs:
      overall_rankings, overall_pairs,
      all_train_pairs, all_test_pairs,
      train_rankings, test_rankings
    """
    print("[Pipeline] Computing overall rankings/pairs...")
    overall_rankings, overall_pairs = compute_pairs_and_ranking_for_all_doctors(patient_df)

    print("[Pipeline] Splitting pairs into train/test...")
    all_train_pairs, all_test_pairs = split_pairs_train_test(overall_pairs, df_train, df_test)

    print("[Pipeline] Computing train/test rankings separately...")
    train_rankings, test_rankings = compute_train_test_rankings(overall_pairs, df_train, df_test)

    return overall_rankings, overall_pairs, all_train_pairs, all_test_pairs, train_rankings, test_rankings

## MAYA D - added a check here that the train-test split loaded here matches the df_train, df_test split in parameters
def create_ranking_if_doesnt_exist(rankings_file, patient_df, df_train, df_test):

    # Transparently support a gzip-compressed cache: if the plain .pkl is
    # absent but a .pkl.gz sits next to it, load the compressed one. This keeps
    # the shipped repo small (260 MB -> 26 MB) without changing any behaviour.
    _rankings_path = rankings_file
    if not os.path.exists(_rankings_path) and os.path.exists(_rankings_path + ".gz"):
        import gzip
        print(f"'{_rankings_path}.gz' found. Loading compressed results...")
        with gzip.open(_rankings_path + ".gz", "rb") as f:
            saved_data = pickle.load(f)
        overall_rankings = saved_data["overall_rankings"]
        overall_pairs = saved_data["overall_pairs"]
        all_train_pairs = saved_data["all_train_pairs"]
        all_test_pairs = saved_data["all_test_pairs"]
        train_rankings = saved_data["train_rankings"]
        test_rankings = saved_data["test_rankings"]
        return (overall_rankings, overall_pairs, all_train_pairs,
                all_test_pairs, train_rankings, test_rankings)

    if os.path.exists(rankings_file):
        print(f"'{rankings_file}' found. Loading existing results...")
        with open(rankings_file, "rb") as f:
            saved_data = pickle.load(f)

        overall_rankings = saved_data["overall_rankings"]
        overall_pairs = saved_data["overall_pairs"]
        all_train_pairs = saved_data["all_train_pairs"]
        all_test_pairs = saved_data["all_test_pairs"]
        train_rankings = saved_data["train_rankings"]
        test_rankings = saved_data["test_rankings"]
# Check if patient IDs in saved train/test rankings match the provided ones function parameters
        doctor_check = 'doctor1' if train_rankings.get('doctor1') else 'doctor11'
        saved_train_patients = set([int(p[0]) for p in train_rankings[doctor_check]])
        saved_test_patients = set([int(p[0]) for p in test_rankings[doctor_check]])
        current_train_patients = set(df_train['patient_num'])
        current_test_patients = set(df_test['patient_num'])

        if saved_train_patients != current_train_patients or saved_test_patients != current_test_patients:
            raise ValueError("Mismatch between saved train/test splits and the ones provided.")
        return overall_rankings, overall_pairs, all_train_pairs, all_test_pairs, train_rankings, test_rankings
    else:
        print(f"'{rankings_file}' not found. Running the pipeline (this might be slow)...")
        # Run once
        (overall_rankings, overall_pairs,
        all_train_pairs, all_test_pairs,
        train_rankings, test_rankings) = run_full_pipeline(patient_df, df_train, df_test)

        # Save all objects in a dictionary
        saved_data = {
            "overall_rankings": overall_rankings,
            "overall_pairs": overall_pairs,
            "all_train_pairs": all_train_pairs,
            "all_test_pairs": all_test_pairs,
            "train_rankings": train_rankings,
            "test_rankings": test_rankings
        }

        with open(rankings_file, "wb") as f:
            pickle.dump(saved_data, f)

        print(f"Pipeline results saved to '{rankings_file}'.")
    return overall_rankings, overall_pairs, all_train_pairs, all_test_pairs, train_rankings, test_rankings

