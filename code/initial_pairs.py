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

from pairs_choosing_methods import (
    build_graph, identify_aut_ranked_pairs_all, identify_aut_ranked_pairs, 
    identify_recs_priority_pairs_all, identify_recs_priority_pairs, 
)
from utils import FEATURE_COLS, align_feature_columns
# automatic pairs
def get_automatic_pairs(patient_df, all_train_patients, all_test_patients, rec_cols):
    train_aut_pairs = identify_aut_ranked_pairs(
            patient_df[patient_df['patient_num'].isin(all_train_patients)],
            age_col="age",
            risk_col="risk",
            rec_cols=rec_cols,
            patient_id_col="patient_num"
        )
    train_aut_pairs_all = identify_aut_ranked_pairs_all(
            patient_df[patient_df['patient_num'].isin(all_train_patients)],
            age_col="age",
            risk_col="risk",
            rec_cols=rec_cols,
            patient_id_col="patient_num"
        )
    test_aut_pairs = identify_aut_ranked_pairs(
            patient_df[patient_df['patient_num'].isin(all_test_patients)],
            age_col="age",
            risk_col="risk",
            rec_cols=rec_cols,
            patient_id_col="patient_num"
        )
    return train_aut_pairs, train_aut_pairs_all, test_aut_pairs

# rec priority pairs
def get_rec_priority_pairs(df_train, df_test, rec_cols):
    train_rec_prior= identify_recs_priority_pairs(df_train, rec_cols=rec_cols, patient_id_col="patient_num")
    train_rec_prior_all = identify_recs_priority_pairs_all(df_train, rec_cols=rec_cols, patient_id_col="patient_num")
    test_rec_prior= identify_recs_priority_pairs(df_test, rec_cols=rec_cols, patient_id_col="patient_num")
    train_rec_prior = [((int(i), int(j)), conf) for ((i, j), conf) in train_rec_prior]
    test_rec_prior = [((int(i), int(j)), conf) for ((i, j), conf) in test_rec_prior]
    return train_rec_prior, train_rec_prior_all, test_rec_prior

# unclear what is this for
# auto_train_pairs = [((int(i), int(j)), conf) for ((i, j), conf) in train_aut_pairs]
# auto_test_pairs = [((int(i), int(j)), conf) for ((i, j), conf) in test_aut_pairs]

# pairs for exclusion
def get_pairs_to_exclude(train_rec_prior_all, train_aut_pairs_all):
    exclude_pairs= train_rec_prior_all+train_aut_pairs_all
    exclude_pairs = [
        (int(pair[0]), int(pair[1]))
        for item in (train_rec_prior_all + train_aut_pairs_all)
        for pair in [item[0] if isinstance(item[0], tuple) else item]
    ]
    return exclude_pairs

# graphs
def get_graphs(train_aut_pairs, train_rec_prior, test_aut_pairs, test_rec_prior):
    train_graph = build_graph(train_aut_pairs+train_rec_prior)
    test_graph  = build_graph(test_aut_pairs+test_rec_prior)
    return train_graph, test_graph

def get_initial_pairs(
        patient_df, all_train_patients, all_test_patients, rec_cols, df_train, df_test
    ):
    train_aut_pairs, train_aut_pairs_all, test_aut_pairs = get_automatic_pairs(
        patient_df, all_train_patients, all_test_patients, rec_cols
    )
    train_rec_prior, train_rec_prior_all, test_rec_prior = get_rec_priority_pairs(
        df_train, df_test, rec_cols
    )
    exclude_pairs = get_pairs_to_exclude(
        train_rec_prior_all, train_aut_pairs_all
    )
    train_graph, test_graph = get_graphs(
        train_aut_pairs, train_rec_prior, test_aut_pairs, test_rec_prior
    )
    return (
        train_aut_pairs, train_aut_pairs_all, test_aut_pairs,
        train_rec_prior, train_rec_prior_all, test_rec_prior,
        exclude_pairs, train_graph, test_graph 
    )

def create_initial_pairs_if_doesnt_exist(        
    initial_pairs_path, patient_df, all_train_patients, all_test_patients, rec_cols, df_train, df_test
):
    if os.path.exists(initial_pairs_path):
        print(f"'{initial_pairs_path}' found. Loading existing pairs...")
        with open(initial_pairs_path, "rb") as f:
            saved_data = pickle.load(f)
            train_aut_pairs = saved_data["train_aut_pairs"]
            train_aut_pairs_all = saved_data["train_aut_pairs_all"]
            test_aut_pairs = saved_data["test_aut_pairs"]
            train_rec_prior = saved_data["train_rec_prior"]
            train_rec_prior_all = saved_data["train_rec_prior_all"]
            test_rec_prior = saved_data["test_rec_prior"]
            exclude_pairs = saved_data["exclude_pairs"]
            train_graph = saved_data["train_graph"]
            test_graph = saved_data["test_graph"]

    else:
        print(f"'{initial_pairs_path}' not found. Creating initial pairs...")
        (train_aut_pairs, train_aut_pairs_all, test_aut_pairs,
        train_rec_prior, train_rec_prior_all, test_rec_prior,
        exclude_pairs, train_graph, test_graph 
        ) = get_initial_pairs(
        patient_df, all_train_patients, all_test_patients, rec_cols, df_train, df_test
        )    

        # Save all objects in a dictionary
        saved_data = {
            "train_aut_pairs": train_aut_pairs,
            "train_aut_pairs_all": train_aut_pairs_all,
            "test_aut_pairs": test_aut_pairs,
            "train_rec_prior": train_rec_prior,
            "train_rec_prior_all": train_rec_prior_all,
            "test_rec_prior": test_rec_prior,
            "exclude_pairs": exclude_pairs,
            "train_graph": train_graph,
            "test_graph": test_graph
        }

        with open(initial_pairs_path, "wb") as f:
            pickle.dump(saved_data, f)
            print(f"Initial pairs saved to '{initial_pairs_path}'.")

    return (
        train_aut_pairs, train_aut_pairs_all, test_aut_pairs,
        train_rec_prior, train_rec_prior_all, test_rec_prior,
        exclude_pairs, train_graph, test_graph 
    )

