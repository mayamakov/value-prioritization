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
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
# internal imports
from utils import compute_rec_weight_sum, get_weight
from utils import FEATURE_COLS, align_feature_columns

###########################
# systematic pairs
###########################

def select_systematic_pairs_list_only(
    patient_df,
    category_dict,
    category_importance,
    total_pairs=100,
    random_state=42,
    max_pairs_to_check=50000  # new param controlling partial sampling in each step
):
    """
    Multi-phase (A/B/C) pair selection:
      - Phase A: Within-Category
         * single-col vs single-col
         * entire category
      - Phase B: Between-categories
      - Phase C: Multi-categories
    Distribution of total_pairs: 50%/40%/10% or so.

    BUT the Hamming distance logic is:
      Priority = dist=2 => dist=3 => dist=4 => dist=1.
      If that doesn't yield enough pairs, we end up with fewer.

    Also uses partial sampling to speed up each sub-step.
    Because enumerating all pairs for each sub-step can be too large.

    Args:
      patient_df (pd.DataFrame):
          Must have columns:
            - 'patient_num' (unique)
            - 'rec1'..'rec21' (binary)
      category_dict (dict):
          e.g. {"labs": [...], "imaging": [...], ...}
      category_importance (dict):
          e.g. {"pharma":5,"consult":4,"imaging":3,"labs":2,"lifestyle":1,"admin":1}
      total_pairs (int):
          total number of pairs to collect across phases.
      random_state (int):
          seed for reproducibility.
      max_pairs_to_check (int):
          how many pairs to sample in each sub-step to avoid scanning all possible pairs.

    Returns:
      list of (p1, p2) up to total_pairs, with priority dist=2->3->4->1,
      short-circuiting and partial sampling to reduce runtime.
    """

    random.seed(random_state)
    total_pairs+=300
    # --------------------------------------------
    # 1) Build all pairs once (N choose 2).
    #    Then shuffle them. We'll do sub-samples as needed.
    # --------------------------------------------
    rec_cols = [c for c in patient_df.columns if c.startswith('rec')]
    patient_ids = patient_df['patient_num'].unique().tolist()
    all_pairs = list(combinations(patient_ids, 2))
    random.shuffle(all_pairs)

    # short-circuit if not enough pairs exist at all, but usually we won't rely on that
    # We'll do partial sampling anyway.

    # --------------------------------------------
    # 2) Hamming function
    # --------------------------------------------
    def compute_hamming(p1, p2, subset_cols=None):
        """
        Return Hamming distance for p1, p2 on subset_cols (if specified),
        otherwise on all rec_cols.
        """
        if subset_cols is None:
            cols = rec_cols
        else:
            cols = subset_cols
        row1 = patient_df.loc[patient_df['patient_num'] == p1, cols]
        row2 = patient_df.loc[patient_df['patient_num'] == p2, cols]
        if len(row1) != 1 or len(row2) != 1:
            return float('inf')
        va = row1.iloc[0].values
        vb = row2.iloc[0].values
        return sum(a != b for a, b in zip(va, vb))

    # --------------------------------------------
    # 3) distance priority filter: 2->3->4->1
    #    We sample up to max_pairs_to_check from candidate_pairs,
    #    compute distance, store in dist2,3,4,1, then pick as many as needed.
    # --------------------------------------------
    def filter_with_distance_priority(candidate_pairs, subset_cols, needed, sampling_size):
        """
        partial-sampling approach:
         1) sample 'sampling_size' from candidate_pairs
         2) compute hamming distance => sort them into dist2, dist3, dist4, dist1
         3) pick from dist2 first, then dist3, then dist4, then dist1,
            up to 'needed'.
         4) return that list of pairs
        If we want to do multiple rounds if not enough found, we can do so,
        but to keep it simpler, we do a single sample pass.
        """
        if sampling_size >= len(candidate_pairs):
            # sample entire set
            pairs_sampled = candidate_pairs
        else:
            # random.sample
            pairs_sampled = random.sample(candidate_pairs, sampling_size)

        dist2 = []
        dist3 = []
        dist4 = []
        dist1 = []

        for (p1, p2) in pairs_sampled:
            if len(dist2)+len(dist3)+len(dist4)+len(dist1) >= needed:
                # short-circuit if we already have enough stored
                # (we still need to label them by distance to pick priority in order,
                #  but let's do a minimal approach that might skip some. We'll keep code simpler.)
                break
            d = compute_hamming(p1, p2, subset_cols)
            if d == 2:
                dist2.append((p1,p2))
            elif d == 3:
                dist3.append((p1,p2))
            elif d == 4:
                dist4.append((p1,p2))
            elif d == 1:
                dist1.append((p1,p2))
            # ignore if d==0 or d>4

        random.shuffle(dist2)
        random.shuffle(dist3)
        random.shuffle(dist4)
        random.shuffle(dist1)

        results = []
        # pick from dist2
        needed_now = needed - len(results)
        from2 = dist2[:needed_now]
        results.extend(from2)
        # dist3
        needed_now = needed - len(results)
        if needed_now > 0:
            from3 = dist3[:needed_now]
            results.extend(from3)
        # dist4
        needed_now = needed - len(results)
        if needed_now > 0:
            from4 = dist4[:needed_now]
            results.extend(from4)
        # dist1
        needed_now = needed - len(results)
        if needed_now > 0:
            from1 = dist1[:needed_now]
            results.extend(from1)

        return results

    # --------------------------------------------
    # 4) distribute total_pairs among phases
    # --------------------------------------------
    phaseA_target = int(round(total_pairs * 0.5))  # e.g. 50%
    phaseB_target = int(round(total_pairs * 0.4))  # e.g. 40%
    phaseC_target = total_pairs - phaseA_target - phaseB_target

    phaseA_collected = []
    phaseB_collected = []
    phaseC_collected = []

    # --------------------------------------------
    # Phase A: within-category
    # --------------------------------------------
    sum_importance = sum(category_importance.values())
    # iterate over categories in random order or stable order
    for cat_name, cols in category_dict.items():
        cat_weight = category_importance.get(cat_name, 1)
        cat_target = int(round(phaseA_target * (cat_weight / float(sum_importance))))
        if cat_target <= 0:
            continue
        if len(phaseA_collected) >= phaseA_target:
            break

        needed_for_cat = cat_target - len(phaseA_collected)
        if needed_for_cat <= 0:
            continue

        # A.1 single-col vs single-col
        collected_for_cat = []
        from itertools import combinations as comb
        if len(cols) > 1:
            col_pairs = list(comb(cols, 2))
            random.shuffle(col_pairs)
            for (c1, c2) in col_pairs:
                if len(collected_for_cat) >= needed_for_cat:
                    break
                leftover = needed_for_cat - len(collected_for_cat)
                # call filter_with_distance_priority with partial sampling
                scenario = filter_with_distance_priority(
                    candidate_pairs=all_pairs,
                    subset_cols=[c1,c2],
                    needed=leftover,
                    sampling_size=max_pairs_to_check
                )
                collected_for_cat.extend(scenario)

        # A.2 entire category
        if len(collected_for_cat) < needed_for_cat:
            leftover = needed_for_cat - len(collected_for_cat)
            scenario = filter_with_distance_priority(
                candidate_pairs=all_pairs,
                subset_cols=cols,
                needed=leftover,
                sampling_size=max_pairs_to_check
            )
            collected_for_cat.extend(scenario)

        phaseA_collected.extend(collected_for_cat)
        if len(phaseA_collected) >= phaseA_target:
            break

    # --------------------------------------------
    # Phase B: between-categories
    # --------------------------------------------
    from itertools import combinations as comb
    cat_list = list(category_dict.keys())
    random.shuffle(cat_list)
    cat_pairs = list(comb(cat_list, 2))
    random.shuffle(cat_pairs)

    while len(phaseB_collected) < phaseB_target and cat_pairs:
        (catA, catB) = cat_pairs.pop()
        needed_b = phaseB_target - len(phaseB_collected)
        merged_cols = category_dict[catA] + category_dict[catB]
        scenario = filter_with_distance_priority(
            candidate_pairs=all_pairs,
            subset_cols=merged_cols,
            needed=needed_b,
            sampling_size=max_pairs_to_check
        )
        phaseB_collected.extend(scenario)

    # --------------------------------------------
    # Phase C: multi-category combos
    # --------------------------------------------
    needed_c = phaseC_target
    if needed_c > 0:
        cat_triplets = list(comb(list(category_dict.keys()), 3))
        random.shuffle(cat_triplets)
        while len(phaseC_collected) < needed_c and cat_triplets:
            (a, b, c) = cat_triplets.pop()
            leftover_c = needed_c - len(phaseC_collected)
            merged_cols = category_dict[a] + category_dict[b] + category_dict[c]
            scenario = filter_with_distance_priority(
                candidate_pairs=all_pairs,
                subset_cols=merged_cols,
                needed=leftover_c,
                sampling_size=max_pairs_to_check
            )
            phaseC_collected.extend(scenario)

    # --------------------------------------------
    # Combine everything
    # --------------------------------------------
    final_pairs = phaseA_collected + phaseB_collected + phaseC_collected
    random.shuffle(final_pairs)
    final_pairs = final_pairs[:total_pairs-300]

    print(f"Returning {len(final_pairs)} pairs (dist=2->3->4->1). Partial-sampling for each sub-step with sampling_size={max_pairs_to_check}.")
    return final_pairs

###########################
# k meams 
###########################

## original
def find_best_k_for_df(X, min_k=20, max_k=50, random_state=42):
    best_k = min_k
    best_score = -1
    for k in range(min_k, min(max_k, len(X)) + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        if k == 1:
            score = -1  # cannot compute silhouette for 1 cluster
        else:
            score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def select_pairs_kmeans_original(df, num_pairs=200, min_k=20, max_k=50, random_state=42):
    """
    Selects candidate patient pairs using KMeans clustering on the original DataFrame.

    The function:
      1. Drops the "patient_num" column to form the feature matrix.
      2. Scales the features.
      3. Finds the best number of clusters (using silhouette score).
      4. Assigns patients to clusters.
      5. Selects half of the pairs randomly from within clusters and half from across clusters.

    Args:
      df (pd.DataFrame): Original DataFrame containing a "patient_num" column and numeric features.
      num_pairs (int): Total number of pairs to select.
      min_k (int): Minimum number of clusters.
      max_k (int): Maximum number of clusters.
      random_state (int): Random state for reproducibility.

    Returns:
      list of tuples: List of (patient_num_A, patient_num_B) pairs.
    """
    df = df.copy()
    # Extract patient IDs and features
    patient_ids = df['patient_num'].values
    # Drop the patient ID column; assume the rest are features
    X = df.drop(columns=['patient_num']).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine the best number of clusters based on silhouette score
    best_k = find_best_k_for_df(X_scaled, min_k=min_k, max_k=max_k, random_state=random_state)
    #best_k=10
    print(f"Best K: {best_k}")
    kmeans = KMeans(n_clusters=best_k, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)

    # Create a mapping from cluster label to the list of patient IDs in that cluster
    clusters = {}
    for pid, label in zip(patient_ids, labels):
        clusters.setdefault(label, []).append(pid)

    # Determine how many pairs to select within clusters vs. across clusters
    within_target = num_pairs // 2
    across_target = num_pairs - within_target

    # Collect within-cluster pairs: for each cluster, compute all pairs and aggregate them
    within_pairs = []
    for clust_patients in clusters.values():
        if len(clust_patients) >= 2:
            pairs = list(combinations(clust_patients, 2))
            within_pairs.extend(pairs)

    random.shuffle(within_pairs)
    within_pairs = within_pairs[:within_target]

    # Collect across-cluster pairs: iterate over every pair of clusters
    across_pairs = []
    cluster_labels = list(clusters.keys())
    for i in range(len(cluster_labels)):
        for j in range(i + 1, len(cluster_labels)):
            for pid1 in clusters[cluster_labels[i]]:
                for pid2 in clusters[cluster_labels[j]]:
                    across_pairs.append((pid1, pid2))

    random.shuffle(across_pairs)
    across_pairs = across_pairs[:across_target]

    candidate_pairs = within_pairs + across_pairs
    random.shuffle(candidate_pairs)
    return candidate_pairs

# BEST K FOR CLUSTERING

def find_best_k(X, min_k=20, max_k=10, random_state=42):
    best_k = None
    best_score = -1
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X)
        labels = kmeans.labels_
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    best_k =best_k if best_k is not None and best_k <= X.shape[0] else min_k
    return best_k
    print(f"Best K: {best_k}")

#  REPLACEMENT CODE FOR KMEANS USING rec_weight_sum
def add_apca_plot(df, title="APCA Plot"):
    """
    Generates a 2D PCA scatter plot for the given DataFrame,
    assuming the DataFrame already has 'PC1' and 'PC2'.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='PC1',
        y='PC2',
        data=df,
        hue='cluster_weight',
        palette="viridis"
    )
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def cluster_patients_with_rec_weight(patient_df_scaled, rec_cols, min_k=2, max_k=10, random_state=42):
    """
    1) Compute 'rec_weight_sum' for each patient (summing all rec_cols).
    2) Cluster using [age, risk, rec_weight_sum].
    3) Store cluster labels in 'cluster_weight'.
    """
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
            'rec21': 1,
        }
        return weights_dict.get(col_name, 1)
    patient_df_scaled['rec_weight_sum'] = patient_df_scaled.apply(
        lambda row: sum(row[col] * get_weight(col) for col in rec_cols),
        axis=1
    )

    X = patient_df_scaled[['age', 'risk', 'rec_weight_sum']].values
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # ---------------------------------------------------------
    # שלב ג: הוסף עמודות "scaled" ל-DataFrame
    # ---------------------------------------------------------
    patient_df_scaled["age_scaled"] = X_scaled[:, 0]
    patient_df_scaled["risk_scaled"] = X_scaled[:, 1]
    patient_df_scaled["rec_sum_scaled"] = X_scaled[:, 2]
    # k = find_best_k(X_scaled, min_k=min_k, max_k=max_k, random_state=42)

    k = 50
    print(f"after scaler {patient_df_scaled}")
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    patient_df_scaled['cluster_weight'] = labels

    # Example: Suppose 'patient_df_scaled' is your DataFrame with the three scaled features.
    # It might also have other columns you want to keep (like a cluster label).
    # For demonstration, let's assume it has columns: 'age', 'risk', 'rec_weight_sum', plus maybe 'some_label'.

    # 1. Select the numeric columns for PCA
    numeric_cols = ['age', 'risk', 'rec_weight_sum']

    # 2. Create and fit PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(patient_df_scaled[numeric_cols])

    # 3. Insert the principal components back into the DataFrame
    patient_df_scaled['PC1'] = pca_result[:, 0]
    patient_df_scaled['PC2'] = pca_result[:, 1]

    # 4. (Optional) Check how much variance each principal component explains
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    add_apca_plot(patient_df_scaled, title="PCA of Age, Risk, Rec Weight Sum")

    return patient_df_scaled
import random
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def select_pairs_kmeans_weight(
    df, rec_cols, num_pairs=200,
    min_k=2, max_k=50, random_state=42,
    pca_dims=3
):
    """
    KMeans-based pair selection using PCA-reduced features [age, risk, rec_weight_sum].
    Ensures 90% of selected pairs have both patients with recommendations (has_rec),
    and 10% allow pairs where at least one has no recommendation.
    Balances within-cluster and across-cluster proportions.

    Args:
      df (pd.DataFrame): must contain 'patient_num', numeric rec_cols, 'age', 'risk'.
      rec_cols (list[str]): recommendation columns.
      num_pairs (int): total number of pairs to select.
      min_k, max_k (int): range of clusters for silhouette-based search.
      random_state (int): seed for reproducibility.
      pca_dims (int): number of principal components to reduce to before clustering.

    Returns:
      list of tuples: (patient_num_A, patient_num_B)
    """
    random.seed(random_state)
    df = df.copy()

    # Compute rec_weight_sum
    weights = {f: w for f, w in zip(rec_cols, [1.0, 2.0, 1.5, 1.5, 3, 4, 4, 4, 10, 10,
                                               10, 10, 10, 10, 9, 8, 8, 3, 1, 1, 1])}
    df['rec_weight_sum'] = df[rec_cols].mul(pd.Series(weights)).sum(axis=1)

    # Select features and scale
    features = df[['age', 'risk', 'rec_weight_sum']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # PCA dimension reduction
    n_comp = min(pca_dims, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)
    # Optional: print variance explained
    # print("PCA variance ratio:", pca.explained_variance_ratio_)

    # Determine best k via silhouette (if desired)
    # Here we fix k or use find_best_k; for now choose max_k
    k = max(min(max_k, len(df)), min_k)
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(X_reduced)
    df['cluster_weight'] = labels

    # Map patient to rec presence
    df['has_rec'] = df[rec_cols].sum(axis=1) > 0

    # Build within- and across-cluster pairs
    clusters = df.groupby('cluster_weight')['patient_num'].apply(list)
    all_within = [(a, b) for pts in clusters for a, b in combinations(pts, 2)]
    random.shuffle(all_within)

    cluster_keys = list(clusters.index)
    all_across = []
    for i in range(len(cluster_keys)):
        for j in range(i+1, len(cluster_keys)):
            for a in clusters[cluster_keys[i]]:
                for b in clusters[cluster_keys[j]]:
                    all_across.append((a,b))
    random.shuffle(all_across)

    # Proportional split
    total = len(all_within) + len(all_across)
    if total == 0:
        return []
    within_n = int(round(num_pairs * (len(all_within) / total)))
    across_n = num_pairs - within_n
    within_sel = all_within[:within_n]
    across_sel = all_across[:across_n]

    candidates = within_sel + across_sel

    # Enforce has_rec: 90% both True, 10% otherwise
    pos = [p for p in candidates if df.set_index('patient_num').loc[p[0],'has_rec'] and df.set_index('patient_num').loc[p[1],'has_rec']]
    non_pos = [p for p in candidates if p not in pos]
    pos_n = int(num_pairs * 0.9)
    non_pos_n = num_pairs - pos_n
    random.shuffle(pos)
    random.shuffle(non_pos)
    selected = pos[:pos_n] + non_pos[:non_pos_n]
    random.shuffle(selected)

    return selected


#this version of k means will make sure that 90% of pairs both patients have recommendations
# TDL - add PCA inside the clustering
# def select_pairs_kmeans_weight(patient_df_scaled, rec_cols, num_pairs=200,
#                                min_k=2, max_k=50, random_state=42):
#     """
#     KMeans-based pair selection using [age, risk, rec_weight_sum].
#     Ensures that 90% of selected pairs have both patients with at least one positive
#     in rec_cols, and only 10% of pairs can be pairs where at least one patient lacks a positive.
#     Additionally, balances the selection between within-cluster and across-cluster pairs
#     proportionally to the number of available pairs in each group.
#     """

#     random.seed(random_state)

#     # אם האשכולות לא חושבו עדיין, מבצעים clustering.
#     if 'cluster_weight' not in patient_df_scaled.columns:
#         cluster_patients_with_rec_weight(
#             patient_df_scaled, rec_cols=rec_cols,
#             min_k=min_k, max_k=max_k,
#             random_state=random_state
#         )

#     # מחשבים עמודה חדשה: True אם למטופל יש לפחות ערך חיובי בעמודות rec.
#     patient_df_scaled['has_rec'] = (patient_df_scaled[rec_cols].sum(axis=1) > 0)

#     # בניית קבוצות לפי האשכול (cluster_weight)
#     clusters = patient_df_scaled['cluster_weight'].unique()
#     cluster_groups = {
#         c: patient_df_scaled[patient_df_scaled['cluster_weight'] == c]['patient_num'].tolist()
#         for c in clusters
#     }
#     print(add_apca_plot(patient_df_scaled))

#     # יצירת זוגות בתוך הקבוצות (within-cluster)
#     all_within = []
#     for c, pids in cluster_groups.items():
#         for (p1, p2) in combinations(pids, 2):
#             all_within.append((p1, p2))
#     random.shuffle(all_within)

#     # יצירת זוגות בין הקבוצות (across-cluster)
#     all_across = []
#     cluster_list = list(cluster_groups.keys())
#     for i in range(len(cluster_list)):
#         for j in range(i+1, len(cluster_list)):
#             c1, c2 = cluster_list[i], cluster_list[j]
#             for p1 in cluster_groups[c1]:
#                 for p2 in cluster_groups[c2]:
#                     all_across.append((p1, p2))
#     random.shuffle(all_across)

#     # איזון פרופורציונלי בין within-cluster ל-across-cluster
#     total_within = len(all_within)
#     total_across = len(all_across)
#     total_available = total_within + total_across

#     if total_available == 0:
#         return []

#     within_target = int(round(num_pairs * (total_within / total_available)))
#     across_target = num_pairs - within_target

#     within_pairs = all_within[:within_target]
#     across_pairs = all_across[:across_target]

#     # שילוב הזוגות משתי הקבוצות
#     all_pairs = within_pairs + across_pairs

#     # יצירת מיפוי של patient_num -> has_rec
#     has_rec_dict = patient_df_scaled.set_index('patient_num')['has_rec'].to_dict()

#     # חלוקה לשתי קבוצות:
#     # - positive_pairs: זוגות בהם שני המטופלים עם המלצה חיובית
#     # - non_positive_pairs: זוגות בהם לפחות אחד מהם ללא המלצה חיובית
#     positive_pairs = [pair for pair in all_pairs if has_rec_dict.get(pair[0], False) and has_rec_dict.get(pair[1], False)]
#     non_positive_pairs = [pair for pair in all_pairs if not (has_rec_dict.get(pair[0], False) and has_rec_dict.get(pair[1], False))]

#     # קביעת יעדים: 90% זוגות חיוביים, 10% זוגות לא חיוביים.
#     positive_target = int(num_pairs * 0.9)
#     non_positive_target = num_pairs - positive_target

#     random.shuffle(positive_pairs)
#     random.shuffle(non_positive_pairs)
#     selected_positive = positive_pairs[:positive_target]
#     selected_non_positive = non_positive_pairs[:non_positive_target]

#     # שילוב סופי של הזוגות וערבובם
#     selected_pairs = selected_positive + selected_non_positive
#     random.shuffle(selected_pairs)

#     return selected_pairs

#############################################################
##   hamming distance  
#############################################################

def get_top_k_pairs_by_hamming_patients(
    patient_df,
    total_samples=50,
    id_col='patient_num',
    rec_prefix='rec',
    zero_dist_cap_fraction=0.05,  # up to 5% from distance=0
    random_state=42
):
    """
    1) Computes pairwise Hamming distances on columns starting with `rec_prefix` (0 <= dist <= 1).
    2) Separates zero-distance pairs from non-zero. Caps zero-distance at `zero_dist_cap_fraction`
       of the total sample size (e.g., 5%).
    3) Samples the rest of the pairs (distance>0) proportionally to their group size among all non-zero pairs.
    4) Returns a single list of (patient_i, patient_j) combining both zero-dist and non-zero-dist pairs.

    Args:
      patient_df (pd.DataFrame):
          Must contain `id_col` (e.g., 'patient_num') and columns named with `rec_prefix` (binary).
      total_samples (int):
          Desired total number of pairs to sample across all distances.
      id_col (str):
          Column with patient identifiers in `patient_df`.
      rec_prefix (str):
          Prefix for "recommendation" columns (binary).
      zero_dist_cap_fraction (float):
          Fraction of total_samples to allow from zero-distance pairs.
      random_state (int):
          Seed for reproducible random sampling.

    Returns:
      list of (patient_num_i, patient_num_j)
          A single list of pairs, where zero-dist pairs do not exceed the cap fraction.
    """
    random.seed(random_state)

    # 1) Identify recommendation columns
    rec_cols = [c for c in patient_df.columns if c.startswith(rec_prefix)]
    if not rec_cols:
        print(f"No columns start with prefix '{rec_prefix}'. Returning empty list.")
        return []

    df_rec = patient_df[rec_cols].copy()

    # 2) Compute pairwise Hamming distance for all rows
    dist_matrix = pairwise_distances(df_rec, metric='hamming')

    # 3) Build dictionary distance -> list of (row_i, row_j)
    #    including distance=0
    dist_dict = defaultdict(list)
    n = len(df_rec)
    for i in range(n):
        for j in range(i+1, n):
            d = dist_matrix[i, j]
            dist_dict[d].append((i, j))

    if not dist_dict:
        # No pairs found (should be impossible if n>1, but just in case)
        return []

    # Separate zero-dist from non-zero-dist
    zero_dist_pairs = dist_dict.pop(0.0, [])  # returns empty list if no exact 0.0 key
    nonzero_dist_dict = dist_dict  # what's left

    # -----------------------------------------------------
    # PART A: Handle zero-dist pairs with a cap
    # -----------------------------------------------------

    zero_dist_cap_num = int(round(total_samples * zero_dist_cap_fraction))  # e.g. 5% of total
    zero_dist_sample = []
    if zero_dist_pairs:
        if len(zero_dist_pairs) <= zero_dist_cap_num:
            # If there's fewer zero-dist pairs than the cap, take them all
            zero_dist_sample = zero_dist_pairs
        else:
            # Randomly sample up to the cap
            zero_dist_sample = random.sample(zero_dist_pairs, zero_dist_cap_num)
    actually_used_zero = len(zero_dist_sample)  # how many we ended up with

    # -----------------------------------------------------
    # PART B: Distribute the remainder to non-zero distances proportionally
    # -----------------------------------------------------

    remaining_samples = total_samples - actually_used_zero
    if remaining_samples <= 0:
        # If we've already used up the entire budget with zero-dist (very rare),
        # just return them
        final_pairs = zero_dist_sample
    else:
        # Build a plan for non-zero distances
        # 1) gather all non-zero pairs
        total_nonzero = sum(len(lst) for lst in nonzero_dist_dict.values())
        if total_nonzero == 0:
            # no non-zero pairs, just return the zero-dist
            final_pairs = zero_dist_sample
        else:
            # For each distance > 0, we compute how many we want to sample
            # proportionally to its group size among ALL non-zero pairs
            # (similar logic to previous function)

            # Sort distances
            sorted_distances = sorted(nonzero_dist_dict.keys())

            group_sampling_plan = []
            sum_samples_nonzero = 0

            for d in sorted_distances:
                group_size = len(nonzero_dist_dict[d])
                proportion = group_size / total_nonzero
                sample_size = int(round(proportion * remaining_samples))
                group_sampling_plan.append((d, group_size, sample_size))
                sum_samples_nonzero += sample_size

            # Adjust for rounding difference
            difference = remaining_samples - sum_samples_nonzero
            if difference != 0:
                # Sort by group_size desc so we can adjust biggest groups first
                group_sampling_plan.sort(key=lambda x: x[1], reverse=True)

                idx = 0
                while difference != 0 and idx < len(group_sampling_plan):
                    d, g_size, s_size = group_sampling_plan[idx]
                    if difference > 0:
                        if s_size < g_size:
                            group_sampling_plan[idx] = (d, g_size, s_size + 1)
                            difference -= 1
                    else:  # difference < 0
                        if s_size > 0:
                            group_sampling_plan[idx] = (d, g_size, s_size - 1)
                            difference += 1
                    idx += 1

                    if idx == len(group_sampling_plan) and difference != 0:
                        idx = 0

                # Sort back by distance ascending
                group_sampling_plan.sort(key=lambda x: x[0])

            # Now sample from each non-zero group
            nonzero_sample = []
            for (d, g_size, s_size) in group_sampling_plan:
                all_pairs_d = nonzero_dist_dict[d]
                if s_size >= len(all_pairs_d):
                    selected_pairs = all_pairs_d
                else:
                    selected_pairs = random.sample(all_pairs_d, s_size)

                nonzero_sample.extend(selected_pairs)

            # Combine zero-dist sample + non-zero-dist sample
            final_pairs = zero_dist_sample + nonzero_sample

    # -----------------------------------------------------
    # Convert row indices -> (patient_i, patient_j)
    # -----------------------------------------------------
    result_list = []
    for (row_i, row_j) in final_pairs:
        p_i = patient_df.iloc[row_i][id_col]
        p_j = patient_df.iloc[row_j][id_col]
        result_list.append((p_i, p_j))

    return result_list

###############################################################################
# 5. CHILD/CHAIN/OUTLIER/CONFLICT-BASED SELECTION
###############################################################################

def dfs_visited(graph_copy, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for (neighbor, _) in graph_copy[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
    return visited

def is_obsolete(pair, graph):
    node1, node2 = pair
    visited_from_1 = dfs_visited(graph, node1)
    visited_from_2 = dfs_visited(graph, node2)
    return (node2 in visited_from_1) or (node1 in visited_from_2)

def select_outlier_pairs(patient_df, feature_cols=None, outlier_prop=0.1,
                         random_state=42, max_outliers=50, max_inliers=200):
    if feature_cols is None:
        rec_cols = [c for c in patient_df.columns if c.startswith('rec')]
        feature_cols = ['age','risk'] + rec_cols

    X = patient_df[feature_cols].values
    iso = IsolationForest(contamination=outlier_prop, random_state=random_state)
    iso_labels = iso.fit_predict(X)
    patient_df['is_outlier'] = (iso_labels == -1)

    outliers_df = patient_df[patient_df['is_outlier'] == True]
    inliers_df  = patient_df[patient_df['is_outlier'] == False]

    if len(outliers_df) > max_outliers:
        outliers_df = outliers_df.sample(n=max_outliers, random_state=random_state)
    if len(inliers_df) > max_inliers:
        inliers_df = inliers_df.sample(n=max_inliers, random_state=random_state)

    outlier_pairs = []
    for out_row in outliers_df.itertuples():
        for in_row in inliers_df.itertuples():
            outlier_pairs.append((out_row.patient_num, in_row.patient_num))

    patient_df.drop(columns=['is_outlier'], inplace=True)
    return outlier_pairs

###############################################
##conflict pairs 
###############################################

def select_conflict_pairs_3d_percentiles(patient_df,
                                        age_col='age',
                                        risk_col='risk',
                                        alpha=1.0,
                                        top_n=300,
                                        max_patients=1000,
                                        random_state=42):
    """
    1. מחשב rec_weight_sum ויוצר age_rank, risk_rank, rec_sum_rank.
    2. מגדיר קונפליקט לפי: conflict = |d_age - d_risk| + alpha * d_rec
    3. מוצא את ספי 10% ו-90% על פי ערך הקונפליקט.
    4. מחלק את הזוגות ל-3 קטגוריות (נמוך, בינוני, גבוה):
         - conflict <  p10_threshold    => נמוך
         - p10_threshold <= conflict <= p90_threshold => בינוני
         - conflict >  p90_threshold    => גבוה
    5. מדגמים באופן אקראי top_n/3 זוגות מכל קטגוריה (אם אפשר),
       ומחזירים רשימה עם סה"כ top_n זוגות.
    """
    # הגדרת seed לרנדום
    random.seed(random_state)

    #df = patient_df.copy()
    if len(df) > max_patients:
        df = df.sample(n=max_patients, random_state=random_state).reset_index(drop=True)

    # אם טרם חושב rec_weight_sum, נחשב
    if 'rec_weight_sum' not in df.columns:
        df['rec_weight_sum'] = df.apply(compute_rec_weight_sum, axis=1)

    # דירוגים
    df['age_rank'] = df[age_col].rank(method='dense')
    df['risk_rank'] = df[risk_col].rank(method='dense')
    df['rec_sum_rank'] = df['rec_weight_sum'].rank(method='dense')

    # בונים את רשימת כל הזוגות (או אפשר רק חלק)
    pairs_list = []
    for (_, row1), (_, row2) in combinations(df.iterrows(), 2):
        d_age = abs(row1['age_rank'] - row2['age_rank'])
        d_risk = abs(row1['risk_rank'] - row2['risk_rank'])
        d_rec = abs(row1['rec_sum_rank'] - row2['rec_sum_rank'])

        # הגדרת פונקציית הקונפליקט: |d_age - d_risk| + alpha*d_rec
        conflict_val = abs(d_age - d_risk) + alpha * d_rec

        pairs_list.append({
            'pair': (row1['patient_num'], row2['patient_num']),
            'conflict': conflict_val
        })

    # אם אין מספיק זוגות, מחזירים את כולם
    if len(pairs_list) <= top_n:
        return [p['pair'] for p in pairs_list]

    # מיון לפי ערך קונפליקט (עולה או יורד כרצונך; כאן עולה)
    pairs_list.sort(key=lambda x: x['conflict'])

    # מציאת ערכי סף לפי אחוזון 10% ו-90%
    # (כי sorted לפי conflict עולה => אינדקס 0.1*len => p10)
    n = len(pairs_list)
    p10_index = int(0.2 * n)
    p90_index = int(0.8 * n)

    p10_threshold = pairs_list[p10_index]['conflict']
    p90_threshold = pairs_list[p90_index]['conflict']

    # חלוקה ל-3 קטגוריות
    cat_low = [p for p in pairs_list if p['conflict'] < p10_threshold]
    cat_med = [p for p in pairs_list if p10_threshold <= p['conflict'] <= p90_threshold]
    cat_high= [p for p in pairs_list if p['conflict'] > p90_threshold]

    # כמה זוגות נרצה מכל קטגוריה? -> top_n / 3
    seg_size = top_n // 3

    final_pairs = []

    # דוגמים באופן אקראי מכל קטגוריה
    if len(cat_low) > seg_size:
        final_pairs += random.sample(cat_low, seg_size)
    else:
        final_pairs += cat_low  # אם יש פחות, קח את כולם

    if len(cat_med) > seg_size:
        final_pairs += random.sample(cat_med, seg_size)
    else:
        final_pairs += cat_med

    if len(cat_high) > seg_size:
        final_pairs += random.sample(cat_high, seg_size)
    else:
        final_pairs += cat_high

    # אם מסיבה כלשהי יש חוסר/עודף, אפשר להשלים או לקצץ:
    if len(final_pairs) < top_n:
        # לדוגמה, להשלים באופן רנדומלי מהקטגוריה הגדולה ביותר (או אחרת)
        shortfall = top_n - len(final_pairs)
        leftover = cat_high if len(cat_high) > seg_size else cat_med
        if len(leftover) > shortfall:
            final_pairs += random.sample(leftover, shortfall)
        else:
            final_pairs += leftover

    if len(final_pairs) > top_n:
        # צמצום אם יש עודף
        final_pairs = random.sample(final_pairs, top_n)

    # מערבבים את הסדר הסופי אם רוצים
    random.shuffle(final_pairs)

    # מחזירים רק את רשימת הזוגות (p1, p2)
    return [item['pair'] for item in final_pairs]

def select_conflict_pairs_3d(patient_df,
                             age_col='age',
                             risk_col='risk',
                             alpha_rec=2.0,        # משקל להפרש המלצות
                             top_n=100,
                             max_patients=1000):
    """
    1. חישוב rec_weight_sum (אם טרם חושב).
    2. בניית age_rank, risk_rank, rec_sum_rank.
    3. חישוב conflict_value עבור כל זוג:
       conflict_value = max(d_age, d_risk, alpha_rec * d_rec)
                        - min(d_age, d_risk, alpha_rec * d_rec)
       (כך הפרשי ההמלצות נכנסים פי alpha_rec)
    4. החזרת top_n זוגות עם conflict הגבוה ביותר.
    """
    df = patient_df.copy()

    # אם יש מעל max_patients, נדגום
    if len(df) > max_patients:
        df = df.sample(n=max_patients, random_state=42).reset_index(drop=True)

    # שלב א: לחשב rec_weight_sum אם לא קיים
    if 'rec_weight_sum' not in df.columns:
        df['rec_weight_sum'] = df.apply(compute_rec_weight_sum, axis=1)

    # שלב ב: ליצור rank לשלושת המשתנים
    df['age_rank'] = df[age_col].rank(method='dense')
    df['risk_rank'] = df[risk_col].rank(method='dense')
    df['rec_sum_rank'] = df['rec_weight_sum'].rank(method='dense')

    # שלב ג: עבור כל צמד (i, j), לחשב הפרש במדרג
    from itertools import combinations
    pairs_list = []
    for (i, row1), (j, row2) in combinations(df.iterrows(), 2):
        d_age = abs(row1['age_rank'] - row2['age_rank'])
        d_risk = abs(row1['risk_rank'] - row2['risk_rank'])
        d_rec = abs(row1['rec_sum_rank'] - row2['rec_sum_rank'])

        # הפרשי המלצות מוכפלים ב-alpha_rec
        d_rec_scaled = alpha_rec * d_rec

        # נוסחת conflict (מקס-מינימום) עם הגדלת השפעת d_rec
        conflict_value = max(d_age, d_risk, d_rec_scaled) - min(d_age, d_risk, d_rec_scaled)

        pairs_list.append({
            'pair': (row1['patient_num'], row2['patient_num']),
            'conflict': conflict_value
        })

    # שלב ד: למיין לפי conflict מהגדול לקטן ולקחת את top_n
    pairs_list.sort(key=lambda x: x['conflict'], reverse=True)
    if len(pairs_list) > top_n:
        pairs_list = pairs_list[:top_n]

    return [item['pair'] for item in pairs_list]

def select_conflict_pairs(patient_df, age_col='age', risk_col='risk',
                          top_n=100, max_patients=1000):
    df = patient_df.copy()
    if len(df) > max_patients:
        df = df.sample(n=max_patients, random_state=42).reset_index(drop=True)

    df['age_rank']  = df[age_col].rank(method='dense')
    df['risk_rank'] = df[risk_col].rank(method='dense')

    pairs_list = []
    for (i, row1), (j, row2) in combinations(df.iterrows(), 2):
        age_diff  = abs(row1['age_rank'] - row2['age_rank'])
        risk_diff = abs(row1['risk_rank'] - row2['risk_rank'])
        conflict_value = abs(age_diff - risk_diff)
        pairs_list.append({
            'pair': (row1['patient_num'], row2['patient_num']),
            'conflict': conflict_value
        })

    pairs_list.sort(key=lambda x: x['conflict'], reverse=True)
    if len(pairs_list) > top_n:
        pairs_list = pairs_list[:top_n]

    return [item['pair'] for item in pairs_list]

def compute_conflict_and_rec_diff(row1, row2, age_col='age', risk_col='risk'):
    """
    Computes the conflict value and recommendation difference between two patients.
    """
    d_age = abs(row1['age_rank'] - row2['age_rank'])
    d_risk = abs(row1['risk_rank'] - row2['risk_rank'])
    conflict_value = abs(d_age - d_risk)
    rec_diff = abs(row1['rec_weight_sum'] - row2['rec_weight_sum'])
    return conflict_value, rec_diff

def select_conflict_pairs_balanced(patient_df, age_col='age', risk_col='risk',
                                  top_n=100, max_patients=1000,
                                  conflict_low_threshold=5, conflict_high_threshold=10,
                                  rec_diff_low_threshold=3, rec_diff_high_threshold=8,
                                  random_state=42):
    """
    Modified conflict-based pair selection with balanced selection within priority 1.
    - If |d_age - d_risk| <= conflict_low_threshold:
        Select pairs with 0 < rec_diff <= rec_diff_low_threshold
    - If |d_age - d_risk| >= conflict_high_threshold:
        Select pairs with rec_diff ==0 or rec_diff >= rec_diff_high_threshold
    - Else:
        Less priority
    Ensures equal representation from both subgroups within priority 1.
    Returns top_n pairs prioritized as per above.
    """
    # הגדרת seed לרנדום
    random.seed(random_state)

    df = patient_df.copy()
    if len(df) > max_patients:
        df = df.sample(n=max_patients, random_state=random_state).reset_index(drop=True)

    # חישוב rec_weight_sum אם לא קיים
    if 'rec_weight_sum' not in df.columns:
        df['rec_weight_sum'] = df.apply(compute_rec_weight_sum, axis=1)

    # דירוגים
    df['age_rank'] = df[age_col].rank(method='dense')
    df['risk_rank'] = df[risk_col].rank(method='dense')
    df['rec_sum_rank'] = df['rec_weight_sum'].rank(method='dense')

    # בניית רשימת זוגות עם conflict_value ו-rec_diff
    pairs_list = []
    for (i, row1), (j, row2) in combinations(df.iterrows(), 2):
        conflict_value, rec_diff = compute_conflict_and_rec_diff(row1, row2, age_col, risk_col)

        # הגדרת קטגוריית עדיפות
        if conflict_value <= conflict_low_threshold:
            if 0 < rec_diff <= rec_diff_low_threshold:
                priority = 1
                subgroup = 'low_conflict_low_rec_diff'
            else:
                priority = 3
                subgroup = None
        elif conflict_value >= conflict_high_threshold:
            if rec_diff == 0 or rec_diff >= rec_diff_high_threshold:
                priority = 1
                subgroup = 'high_conflict_rec_diff'
            else:
                priority = 3
                subgroup = None
        else:
            priority = 2  # עדיפות בינונית
            subgroup = None

        pairs_list.append({
            'pair': (row1['patient_num'], row2['patient_num']),
            'conflict': conflict_value,
            'rec_diff': rec_diff,
            'priority': priority,
            'subgroup': subgroup
        })

    # חלוקה לקבוצות
    priority1_low = [p for p in pairs_list if p['priority'] == 1 and p['subgroup'] == 'low_conflict_low_rec_diff']
    priority1_high = [p for p in pairs_list if p['priority'] == 1 and p['subgroup'] == 'high_conflict_rec_diff']
    priority2 = [p for p in pairs_list if p['priority'] == 2]
    priority3 = [p for p in pairs_list if p['priority'] == 3]

    # מספר זוגות לכל קבוצה בתוך priority 1
    priority1_each = top_n // 2  # שני subgroups

    selected_pairs = []
    seen_pairs = set()

    # בחירת זוגות מקבוצה א׳
    random.shuffle(priority1_low)
    for p in priority1_low:
        if len(selected_pairs) >= top_n:
            break
        pair = p['pair']
        if pair not in seen_pairs and (pair[1], pair[0]) not in seen_pairs:
            selected_pairs.append(pair)
            seen_pairs.add(pair)
            if len(selected_pairs) >= priority1_each:
                break

    # בחירת זוגות מקבוצה ב׳
    random.shuffle(priority1_high)
    for p in priority1_high:
        if len(selected_pairs) >= top_n:
            break
        pair = p['pair']
        if pair not in seen_pairs and (pair[1], pair[0]) not in seen_pairs:
            selected_pairs.append(pair)
            seen_pairs.add(pair)
            if len(selected_pairs) >= 2 * priority1_each:
                break

    # אם לא הספקנו לבחור את כל זוגות priority 1, נשלים את הזוגות הנוספים מהקבוצות הקיימות
    if len(selected_pairs) < 2 * priority1_each:
        remaining = 2 * priority1_each - len(selected_pairs)
        combined_priority1 = priority1_low + priority1_high
        random.shuffle(combined_priority1)
        for p in combined_priority1:
            if len(selected_pairs) >= 2 * priority1_each:
                break
            pair = p['pair']
            if pair not in seen_pairs and (pair[1], pair[0]) not in seen_pairs:
                selected_pairs.append(pair)
                seen_pairs.add(pair)
                remaining -= 1
                if remaining <= 0:
                    break

    # בחירת זוגות מעדיפות בינוניות
    needed_from_priority2 = top_n - len(selected_pairs)
    if needed_from_priority2 > 0:
        random.shuffle(priority2)
        for p in priority2:
            if len(selected_pairs) >= top_n:
                break
            pair = p['pair']
            if pair not in seen_pairs and (pair[1], pair[0]) not in seen_pairs:
                selected_pairs.append(pair)
                seen_pairs.add(pair)

    # בחירת זוגות מעדיפות נמוכות
    needed_from_priority3 = top_n - len(selected_pairs)
    if needed_from_priority3 > 0:
        random.shuffle(priority3)
        for p in priority3:
            if len(selected_pairs) >= top_n:
                break
            pair = p['pair']
            if pair not in seen_pairs and (pair[1], pair[0]) not in seen_pairs:
                selected_pairs.append(pair)
                seen_pairs.add(pair)

    return selected_pairs

###############################################
# child / chain 
###############################################

def select_child_pairs(graph):
    child_pairs = []
    for parent, children in graph.items():
        confidence_groups = defaultdict(list)
        for child, confidence in children:
            confidence_groups[confidence].append(child)
        for confidence, grouped_children in confidence_groups.items():
            if len(grouped_children) > 1:
                for i in range(len(grouped_children)):
                    for j in range(i+1, len(grouped_children)):
                        child_pairs.append((grouped_children[i], grouped_children[j]))
    return list(set(child_pairs))

def find_all_possible_chains(graph_copy, print_flag=False, print_text="All chains:"):
    all_paths = []
    def dfs_paths(gc, start, path):
        stack = [(start, path)]
        while stack:
            node, current_path = stack.pop()
            current_path.append(node)
            if not gc[node]:
                all_paths.append(current_path.copy())
            for neighbor, _ in gc[node]:
                if neighbor not in current_path:
                    stack.append((neighbor, current_path.copy()))
            current_path.pop()

    for start_node in list(graph_copy.keys()):
        dfs_paths(graph_copy, start_node, [])

    if print_flag:
        print(print_text, all_paths)
    return all_paths

def is_subsequence(small, large):
    it = iter(large)
    return all(item in it for item in small)

def remove_contained_chains(chains, print_flag=False, print_text="Unique chains:"):
    chains.sort(key=len, reverse=True)
    unique_chains = []
    for chain in chains:
        if not any(is_subsequence(chain, other) for other in unique_chains):
            unique_chains.append(chain)
    if print_flag:
        print(print_text, unique_chains)
    return unique_chains

def select_chain_pairs(chains):
    heads_tails = []
    for chain in chains:
        if chain:
            heads_tails.append((chain[0], chain[-1], set(chain)))

    selected_pairs = []
    num_chains = len(heads_tails)
    for i in range(num_chains):
        for j in range(i+1, num_chains):
            head1, tail1, set1 = heads_tails[i]
            head2, tail2 = heads_tails[j][0], heads_tails[j][1]
            set2 = heads_tails[j][2]
            if set1.isdisjoint(set2):
                selected_pairs.append((head1, tail2))
                selected_pairs.append((head2, tail1))
    return list(set(selected_pairs))

def choose_random_pairs(pairs, num_pairs):
    return random.sample(pairs, min(num_pairs, len(pairs)))

###############################################################################
# 6. BUILD GRAPH & RANKED PAIRS
###############################################################################

def build_graph(ranked_pairs, print_flag=False, print_text="Graph:"):
    g = defaultdict(list)
    for ((parent, child), confidence) in ranked_pairs:
        g[parent].append((child, confidence))
    return g

def transitive_reduction(graph):
    reduced_graph = defaultdict(list)
    graph_copy = copy.deepcopy(graph)
    for node in graph:
        reachable = {}
        for neighbor, conf in graph[node]:
            if neighbor not in reachable:
                reachable[neighbor] = dfs_visited(graph_copy, neighbor)
        neighbors = [(n, c) for (n, c) in graph[node]]
        for (neighbor, confidence) in neighbors:
            skip = any(
                neighbor in reachable[other] for (other, _) in neighbors if other != neighbor
            )
            if not skip:
                if neighbor not in [n for (n, _) in reduced_graph[node]]:
                    reduced_graph[node].append((neighbor, confidence))
    return reduced_graph

###############################################################################
# 7. PATIENT PAIR SELECTION UTILS
###############################################################################

def select_patient_pairs(data, age_threshold=10, risk_threshold=10):
    """
    Picks pairs based on:
     1) identical recommendations => large diffs in age/risk
     2) different recommendations => small diffs in age/risk

    Returns a DataFrame with columns ['pair','criteria'].
    """
    # Identify recommendation columns (exclude known columns)
    recommendation_cols = [
        c for c in data.columns
        if c not in ['patient_num','age','risk','cluster','cluster_weight']
    ]

    pairs = []
    for (i, row1), (j, row2) in combinations(data.iterrows(), 2):
        # Are they identical in rec columns?
        identical = all(row1[col] == row2[col] for col in recommendation_cols)
        # Are they different?
        different = any(row1[col] != row2[col] for col in recommendation_cols)

        # If identical => large diff in age & risk
        if identical:
            if (abs(row1['age'] - row2['age']) > age_threshold and
                abs(row1['risk'] - row2['risk']) > risk_threshold):
                pairs.append({
                    'pair': (row1['patient_num'], row2['patient_num']),
                    'criteria': 'Age-Risk Tradeoff'
                })

        # If different => small diff in age & risk
        if different:
            if (abs(row1['age'] - row2['age']) < age_threshold and
                abs(row1['risk'] - row2['risk']) < risk_threshold):
                pairs.append({
                    'pair': (row1['patient_num'], row2['patient_num']),
                    'criteria': 'Recommendation Comparison'
                })

    return pd.DataFrame(pairs)

###############################################################################
# 8. AUTO-RANKED PAIRS (OPTIONAL)
###############################################################################

def is_superset(patient_i, patient_j, columns):
    return all(patient_i[col] >= patient_j[col] for col in columns)

def is_exactset(patient_i, patient_j, columns):
    return all(patient_i[col] == patient_j[col] for col in columns)

def identify_aut_ranked_pairs_all(patient_df, age_col, risk_col, rec_cols, patient_id_col):
    """
    If i is 'better' => ( (i,j), 6 ).
    If identical => ( (i,j), 1 ).
    """

    pair_conf_dict = {}

    for i in range(len(patient_df)):
        for j in range(len(patient_df)):
            if i == j:
                continue
            p_i = patient_df.iloc[i]
            p_j = patient_df.iloc[j]
            id_i = int(p_i[patient_id_col])
            id_j = int(p_j[patient_id_col])

            pair = (id_i, id_j)

            def update(pair, conf):
                if pair not in pair_conf_dict or conf > pair_conf_dict[pair]:
                    pair_conf_dict[pair] = conf

            if (p_i[age_col] <= p_j[age_col]
                and p_i[risk_col] >= p_j[risk_col]
                and is_superset(p_i, p_j, rec_cols)
                and not (
                    p_i[age_col] == p_j[age_col]
                    and p_i[risk_col] == p_j[risk_col]
                    and is_exactset(p_i, p_j, rec_cols)
                )):
                update(pair, 6)

            if (p_i[age_col] == p_j[age_col]
                and p_i[risk_col] == p_j[risk_col]
                and is_exactset(p_i, p_j, rec_cols)
                and id_i < id_j):
                update(pair, 1)

            if (p_i[age_col] <= p_j[age_col]
                and p_i[risk_col] > p_j[risk_col]
                and is_exactset(p_i, p_j, rec_cols)):
                update(pair, 4)

            if (p_i[age_col] < p_j[age_col]
                and p_i[risk_col] >= p_j[risk_col]
                and is_exactset(p_i, p_j, rec_cols)):
                update(pair, 4)

    all_pairs = list(pair_conf_dict.items())
    return all_pairs

def identify_aut_ranked_pairs(patient_df, age_col, risk_col, rec_cols, patient_id_col):
    """
    If i is 'better' => ( (i,j), 6 ).
    Returns a list of unique ranked pairs with the highest confidence per (i,j).
    """
    pair_conf_dict = {}

    for i in range(len(patient_df)):
        for j in range(len(patient_df)):
            if i == j:
                continue
            p_i = patient_df.iloc[i]
            p_j = patient_df.iloc[j]
            id_i = int(p_i[patient_id_col])
            id_j = int(p_j[patient_id_col])

            pair = (id_i, id_j)

            def update(pair, conf):
                if pair not in pair_conf_dict or conf > pair_conf_dict[pair]:
                    pair_conf_dict[pair] = conf

            if (p_i[age_col] <= p_j[age_col]
                and p_i[risk_col] >= p_j[risk_col]
                and is_superset(p_i, p_j, rec_cols)
                and not (
                    p_i[age_col] == p_j[age_col]
                    and p_i[risk_col] == p_j[risk_col]
                    and is_exactset(p_i, p_j, rec_cols)
                )):
                update(pair, 6)

            if (p_i[age_col] == p_j[age_col]
                and p_i[risk_col] == p_j[risk_col]
                and is_exactset(p_i, p_j, rec_cols)
                and id_i < id_j):
                update(pair, 1)

            if (p_i[age_col] <= p_j[age_col]
                and p_i[risk_col] > p_j[risk_col]
                and is_exactset(p_i, p_j, rec_cols)):
                update(pair, 4)

            if (p_i[age_col] < p_j[age_col]
                and p_i[risk_col] >= p_j[risk_col]
                and is_exactset(p_i, p_j, rec_cols)):
                update(pair, 4)

    all_pairs = list(pair_conf_dict.items())
    if len(all_pairs) > 2000:
        all_pairs = random.sample(all_pairs, 2000)

    return all_pairs

def identify_aut_ranked_pairs_unique(patient_df, age_col, risk_col, rec_cols, patient_id_col):
    """
    Returns a list of ( (patientA, patientB), label ) where each patient appears at most once.
    Uses 'identify_aut_ranked_pairs' but ensures if a is 'better than' multiple patients,
    it is chosen in only one pair.
    """

    # 1) Get all the pairs from your existing function
    all_pairs = identify_aut_ranked_pairs(
        patient_df, age_col, risk_col, rec_cols, patient_id_col
    )

    used_patients = set()  # To keep track of patients already used
    final_pairs = []

    # 2) Iterate over the pairs in the order they were generated
    #    (or in a sorted order if you need a specific priority).
    for pair, label in all_pairs:
        p1, p2 = pair
        # 3) Only pick the pair if both patients are not yet used
        if p1 not in used_patients and p2 not in used_patients:
            final_pairs.append((pair, label))
            # Mark them as used so they won't appear again
            used_patients.add(p1)
            used_patients.add(p2)
    if len(pairs) > 500:
        pairs = random.sample(pairs, 500)
    return final_pairs

def remove_duplicate_pairs(original_pairs, auto_ranked_pairs):
    """
    מסיר זוגות מתוך original_pairs שמופיעים ב-auto_ranked_pairs, בלי תלות בסדר המספרים בזוג.

    Args:
        original_pairs: רשימה של זוגות, כל זוג הוא (i, j).
        auto_ranked_pairs: רשימה של זוגות אוטומטיים, כל זוג הוא ((i, j), conf).

    Returns:
        cleaned_pairs: רשימה חדשה של זוגות עם זוגות כפולים מוסרים.
    """
    # יצירת סט של זוגות אוטומטיים, כשהזוגות ממוינים כדי להתעלם מהסדר
    auto_set = set()
    for pair_conf in auto_ranked_pairs:
        pair = pair_conf[0]
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            print(f"מתעלמים מזוג לא תקין ב-auto_ranked_pairs: {pair}")
            continue
        try:
            sorted_pair = tuple(sorted(pair))
            auto_set.add(sorted_pair)
        except TypeError as e:
            print(f"שגיאה במיון הזוג {pair} ב-auto_ranked_pairs: {e}")
            continue

    # סינון original_pairs
    cleaned_pairs = []
    for pair in original_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            print(f"מתעלמים מזוג לא תקין ב-original_pairs: {pair}")
            continue
        try:
            sorted_pair = tuple(sorted(pair))
        except TypeError as e:
            print(f"שגיאה במיון הזוג {pair} ב-original_pairs: {e}")
            continue
        if sorted_pair not in auto_set:
            cleaned_pairs.append(pair)

    return cleaned_pairs

#############################
# rec priority
#############################
def identify_recs_priority_pairs_all(patient_df, rec_cols, patient_id_col):
    """
    בונה זוגות שבהם מטופלים עם המלצות מועדפים על כאלה בלי, ומונעת כפילויות.

    Returns:
        list of ((id_with_rec, id_without_rec), 5)
    """

    def has_recommendations(patient):
        # נחשב שמספר > 0 בעמודות ההמלצות מעיד על קיום המלצה
        return any(patient[col] > 0 for col in rec_cols)

    # ✅ שינוי 1: יצירת סט כדי לעקוב אחרי זוגות שכבר נוספו
    pairs_set = set()
    final_pairs = []

    n = len(patient_df)
    for i in range(n):
        patient_i = patient_df.iloc[i]
        id_i = int(patient_i[patient_id_col])
        if has_recommendations(patient_i):
            for j in range(n):
                if i == j:
                    continue
                patient_j = patient_df.iloc[j]
                id_j = int(patient_j[patient_id_col])
                if not has_recommendations(patient_j):
                    pair = (id_i, id_j)

                    # ✅ שינוי 2: בדיקה אם הזוג כבר קיים — אם לא, נוסיף
                    if pair not in pairs_set:
                        pairs_set.add(pair)
                        final_pairs.append((pair, 6))

    return final_pairs

def identify_recs_priority_pairs(patient_df, rec_cols, patient_id_col):
    """
    בונה זוגות שבהם מטופלים עם המלצות מועדפים על כאלה בלי, ומונעת כפילויות.

    Returns:
        list of ((id_with_rec, id_without_rec), 5)
    """

    def has_recommendations(patient):
        # נחשב שמספר > 0 בעמודות ההמלצות מעיד על קיום המלצה
        return any(patient[col] > 0 for col in rec_cols)

    # ✅ שינוי 1: יצירת סט כדי לעקוב אחרי זוגות שכבר נוספו
    pairs_set = set()
    final_pairs = []

    n = len(patient_df)
    for i in range(n):
        patient_i = patient_df.iloc[i]
        id_i = int(patient_i[patient_id_col])
        if has_recommendations(patient_i):
            for j in range(n):
                if i == j:
                    continue
                patient_j = patient_df.iloc[j]
                id_j = int(patient_j[patient_id_col])
                if not has_recommendations(patient_j):
                    pair = (id_i, id_j)

                    # ✅ שינוי 2: בדיקה אם הזוג כבר קיים — אם לא, נוסיף
                    if pair not in pairs_set:
                        pairs_set.add(pair)
                        final_pairs.append((pair, 6))

    # ✅ שינוי 3: שמירה על הגבלה ל-300 אם יש יותר מדי
    if len(final_pairs) >500:
        final_pairs = random.sample(final_pairs, 500)

    return final_pairs

def identify_random_pairs(data, num_pairs=None, id_column="patient_num", rec_columns=None):
    """
    מחזירה זוגות אקראיים מהדאטה לפי האיזון הבא:
      - 90% מהזוגות יהיו בין מטופלים שיש להם לפחות המלצה (כלומר, באחד מעמודות rec1 עד rec21 יש ערך 1)
      - 10% מהזוגות יהיו זוגות רנדומליים לחלוטין (מהמאגר כולו)

    Args:
      data (pd.DataFrame): דאטה הכוללת עמודת מזהה המטופל (ברירת מחדל "patient_num") ועמודות המלצות.
      num_pairs (int, optional): מספר זוגות סך הכל. אם לא מוגדר, הפונקציה תחזיר את כל הזוגות האפשריים.
      id_column (str, optional): שם העמודה עם המזהה (ברירת מחדל "patient_num").
      rec_columns (list, optional): רשימת שמות העמודות שמכילות את ההמלצות.
                                    אם לא מוגדרת, נניח ["rec1", "rec2", …, "rec21"].

    Returns:
      list of tuples: רשימת זוגות בצורה (מטופל1, מטופל2)
    """
    # אם לא סופקו עמודות המלצות, נניח את rec1 עד rec21
    if rec_columns is None:
        rec_columns = [f"rec{i}" for i in range(1, 22)]

    # זיהוי מטופלים עם לפחות המלצה אחת
    rec_mask = data[rec_columns].gt(0).any(axis=1)
    recommended_patients = data.loc[rec_mask, id_column].unique()

    # יצירת כל הזוגות האפשריים מבין המטופלים עם המלצה
    recommended_pairs = list(combinations(recommended_patients, 2))

    # אם num_pairs לא מוגדר – נחזיר את כל הזוגות האפשריים (או ניתן לשנות לפי צורך)
    if num_pairs is None:
        # במקרה זה נחזיר את כל זוגות ההמלצות יחד עם זוגות רנדומליים מכלל המטופלים
        all_patients = data[id_column].unique()
        all_random_pairs = list(combinations(all_patients, 2))
        candidate_pairs = list(set(recommended_pairs) | set(all_random_pairs))
        return candidate_pairs

    # נבחר 90% זוגות ממערכת ההמלצות
    num_rec_pairs = int(round(0.9 * num_pairs))
    # נבחר 10% זוגות רנדומליים לחלוטין
    num_random_pairs = num_pairs - num_rec_pairs

    if len(recommended_pairs) < num_rec_pairs:
        sampled_rec_pairs = recommended_pairs
    else:
        sampled_rec_pairs = random.sample(recommended_pairs, num_rec_pairs)

    # זוגות רנדומליים מכלל המטופלים
    all_patients = data[id_column].unique()
    all_random_pairs = list(combinations(all_patients, 2))
    if len(all_random_pairs) < num_random_pairs:
        sampled_random_pairs = all_random_pairs
    else:
        sampled_random_pairs = random.sample(all_random_pairs, num_random_pairs)

    candidate_pairs = sampled_rec_pairs + sampled_random_pairs
    random.shuffle(candidate_pairs)
    return candidate_pairs

################################
# influential_pairs 
##############################

# sorting function
def select_influential_pairs(common_dict, num_pairs):
    """
    מחזיר רשימה של עד num_pairs זוגות מתוך המילון common_dict,
    שבו המפתח הוא (i,j) והערך הוא מספר הרופאים.
    """
    # מיון הזוגות לפי מספר רופאים (יורד)
    sorted_pairs = sorted(common_dict.items(), key=lambda x: x[1], reverse=True)
    # רק המפתחות (i,j)
    return [pair for pair, cnt in sorted_pairs[:num_pairs]]


################################
# at least 1 reccomendation  
##############################
def identify_random_pairs_with_recommendations(patient_df, num_pairs, random_state=42):
    """
    Randomly selects *patient_num* pairs (לא index) כך שלשניהם יש לפחות המלצה אחת.
    """
    import random, itertools
    random.seed(random_state)

    rec_cols = [c for c in patient_df.columns if c.startswith('rec')]

    # -- מטופלים שיש להם לפחות המלצה
    patients_with_rec = patient_df.loc[
        patient_df[rec_cols].sum(axis=1) > 0, 'patient_num'
    ].tolist()

    if len(patients_with_rec) < 2:
        return []

    all_pairs = list(itertools.combinations(patients_with_rec, 2))
    random.shuffle(all_pairs)
    return all_pairs[:num_pairs]

# ===== UNCAPPED wrappers for the math part =====

def get_aut_pairs_for_math(patient_df, age_col, risk_col, rec_cols, patient_id_col):
    """
    מחזיר את כל הזוגות האוטומטיים (ללא תקרה), ע"י שימוש ב-identify_aut_ranked_pairs_all.
    """
    return identify_aut_ranked_pairs_all(
        patient_df=patient_df,
        age_col=age_col,
        risk_col=risk_col,
        rec_cols=rec_cols,
        patient_id_col=patient_id_col
    )

def get_rec_priority_pairs_for_math(patient_df, rec_cols, patient_id_col):
    """
    מחזיר את כל זוגות ה-rec-priority (ללא תקרה), ע"י שימוש ב-identify_recs_priority_pairs_all.
    """
    return identify_recs_priority_pairs_all(
        patient_df=patient_df,
        rec_cols=rec_cols,
        patient_id_col=patient_id_col
    )