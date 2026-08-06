import random
import numpy as np

from utils import take_until_quota
from pairs_choosing_methods import (
    select_pairs_kmeans_weight, 
    select_outlier_pairs, 
    select_conflict_pairs_balanced,
    get_top_k_pairs_by_hamming_patients,
    select_child_pairs,
    select_chain_pairs,
    remove_contained_chains,
    find_all_possible_chains,
    select_systematic_pairs_list_only,
    select_influential_pairs,
    identify_random_pairs_with_recommendations,    
    identify_random_pairs
    )

###############################################################################
#   robust quota handling + adaptive oversampling
###############################################################################

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import random


def identify_candidate_pairs_for_stable_set(
    graph,
    patient_df_scaled,
    patient_df,
    common_pairs_dict,
    num_pairs           = 400,
    alpha_kmeans_d      = 0.60,
    alpha_outlier       = 0.15,
    alpha_conflict      = 0.10,
    alpha_pat           = 0.10,
    alpha_child         = 0.03,
    alpha_chain         = 0.02,
    alpha_random        = 0.00,     # אחוז רנדום “אמיתי”
    alpha_sys           = 0.00,
    alpha_inf           = 0.00,
    alpha_llm           = 0.00,
    exclude_pairs       = None,
    print_flag          = False,
    print_text          = "",
    min_k               = 2,
    max_k               = 10,
    random_state        = 42
):
    # === שילוב שיטת LLM כבחירת זוגות מוכנה מראש ===
    # if alpha_llm == 1:
    #     return selected_pairs

    category_dict = {
        'labs':      ['rec1','rec2','rec3','rec4'],
        'imaging':   ['rec5','rec6','rec7','rec8'],
        'pharma':    ['rec9','rec10','rec11','rec12','rec13','rec14','rec15'],
        'consult':   ['rec16','rec17','rec18'],
        'lifestyle': ['rec19','rec20'],
        'admin':     ['rec21']
    }
    category_importance = {
        'pharma':    5,
        'consult':   4,
        'imaging':   3,
        'labs':      2,
        'lifestyle': 1,
        'admin':     1
    }

    # ---------- utilities ----------
    def _pair2key(pair):
        """Normalize a pair to an unordered key (no duplicates (a,b)/(b,a))."""
        a, b = int(pair[0]), int(pair[1])
        return (a, b) if a <= b else (b, a)

    def _coerce_pair(x):
        """Accept (a,b) or ((a,b), anything...), return (a,b) or None if invalid."""
        if isinstance(x, (list, tuple)):
            if len(x) >= 2 and isinstance(x[0], (int, np.integer)) and isinstance(x[1], (int, np.integer)):
                return int(x[0]), int(x[1])
            if len(x) >= 1 and isinstance(x[0], (list, tuple)) and len(x[0]) == 2:
                a, b = x[0]
                if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                    return int(a), int(b)
        return None

    # Build a fast allow-list of patients who have at least one recommendation
    rec_cols_df = [c for c in patient_df.columns if c.startswith('rec')]
    if rec_cols_df:
        has_rec_mask = (patient_df[rec_cols_df] > 0).any(axis=1)
        allowed_patients = set(patient_df.loc[has_rec_mask, 'patient_num'].astype(int))
    else:
        # If there are no rec columns at all, we cannot filter on “has rec”.
        # In that case, we skip this constraint (keeps backward compatibility).
        allowed_patients = None  # means: do not filter by recs

    # ---------- forbidden (כבר דורגו / נשללו) ----------
    exclude_keys = set()
    if exclude_pairs:
        for item in exclude_pairs:
            # item can be (a,b) or ((a,b), meta...)
            pr = _coerce_pair(item)
            if pr is not None:
                exclude_keys.add(_pair2key(pr))

    candidate_pairs: List[Tuple[int, int]] = []

    # Track pairs we have already emitted (dedup over all stages)
    seen_keys = set(exclude_keys)

    def _take_until_quota_enforcing(raw_pairs, needed: int) -> List[Tuple[int, int]]:
        """
        Local intake that:
        - normalizes pair shape,
        - skips self-pairs,
        - enforces 'both patients have rec' (if rec columns exist),
        - enforces no duplicates (unordered),
        - respects exclude_keys and seen_keys.
        """
        out: List[Tuple[int, int]] = []
        if needed <= 0:
            return out
        for itm in raw_pairs:
            if len(out) >= needed:
                break
            pr = _coerce_pair(itm)
            if pr is None:
                continue
            a, b = int(pr[0]), int(pr[1])
            if a == b:
                continue
            key = _pair2key((a, b))

            # Skip forbidden / already taken
            if key in seen_keys:
                continue

            # Enforce both patients have at least one recommendation (when we can)
            if allowed_patients is not None:
                if (a not in allowed_patients) or (b not in allowed_patients):
                    continue

            # Accept
            seen_keys.add(key)
            out.append((a, b))
        return out

    # ======================================================================
    #  פונקציית עזר: מריצה “שלב” עם oversampling אדפטיבי
    # ======================================================================
    def run_stage(name, need, produce_func, max_tries=3, init_factor=3):
        """מוסיפה זוגות ל-candidate_pairs (עד *need*) מתוך produce_func."""
        if need <= 0:
            return
        factor   = init_factor
        obtained = 0
        tries    = 0
        while obtained < need and tries < max_tries:
            raw = produce_func(max(1, need * factor))
            got = _take_until_quota_enforcing(raw, need - obtained)
            candidate_pairs.extend(got)
            obtained += len(got)
            factor  *= 2   # מגדילים overshoot
            tries   += 1
            if print_flag:
                print(f"[{name}]  pass {tries}:  +{len(got)}  (total {obtained}/{need})")
        if print_flag and obtained < need:
            print(f"[{name}]  WARNING – חסרים {need-obtained} זוגות")

    # ======================================================================
    # 1) K-MEANS
    # ======================================================================
    run_stage(
        "KMEANS",
        int(num_pairs * alpha_kmeans_d),
        lambda n: select_pairs_kmeans_weight(
            patient_df_scaled,
            rec_cols=[c for c in patient_df.columns if c.startswith('rec')],
            num_pairs=n,
            min_k=50,
            max_k=max_k,
            random_state=random_state)
    )

    # 2) OUTLIERS
    run_stage(
        "OUTLIER",
        int(num_pairs * alpha_outlier),
        lambda n: select_outlier_pairs(
            patient_df,
            random_state=random_state,
            max_outliers=50,
            max_inliers=200)[:n]
    )

    # 3) CONFLICT
    run_stage(
        "CONFLICT",
        int(num_pairs * alpha_conflict),
        lambda n: select_conflict_pairs_balanced(
            patient_df, top_n=max(2000, n))[:n]
    )

    # 4) HAMMING / PAT
    run_stage(
        "HAMMING",
        int(num_pairs * alpha_pat),
        lambda n: get_top_k_pairs_by_hamming_patients(patient_df,
                                                      total_samples=n)
    )

    # 5) CHILD
    run_stage(
        "CHILD",
        int(num_pairs * alpha_child),
        lambda n: random.sample(select_child_pairs(graph),
                                k=min(n*3, len(select_child_pairs(graph))))
    )

    # 6) CHAIN
    run_stage(
        "CHAIN",
        int(num_pairs * alpha_chain),
        lambda n: random.sample(
            select_chain_pairs(
                remove_contained_chains(
                    find_all_possible_chains(graph, False), False)),
            k=min(n*3, len(select_chain_pairs(
                remove_contained_chains(
                    find_all_possible_chains(graph, False), False))))
        )
    )

    # 7) SYSTEMATIC
    run_stage(
        "SYSTEM",
        int(num_pairs * alpha_sys),
        lambda n: select_systematic_pairs_list_only(
            patient_df,
            category_dict,
            category_importance,
            total_pairs=n,
            random_state=random_state)
    )

    # 8) INFLUENTIAL
    if alpha_inf > 0 and common_pairs_dict:
        run_stage(
            "INFLUENTIAL",
            int(num_pairs * alpha_inf),
            lambda n: select_influential_pairs(common_pairs_dict, n*3)
        )

    # 9) RANDOM – אם מישהו בחר אחוז רנדום אמיתי
    if alpha_random > 0:
        run_stage(
            "RANDOM",
            int(num_pairs * alpha_random),
            lambda n: identify_random_pairs_with_recommendations(patient_df, n*4)
        )

    # 10) RANDOM – fill any remaining gap
    still_need = num_pairs - len(candidate_pairs)
    if still_need > 0:
        random_pairs = identify_random_pairs(patient_df, still_need*4)
        random.shuffle(random_pairs)
        fill = _take_until_quota_enforcing(random_pairs, still_need)
        candidate_pairs.extend(fill)
        if print_flag:
            print(f"[RANDOM-FILL]  added {len(fill)}/{still_need}")

    # Final trim (already deduped all along)
    if len(candidate_pairs) > num_pairs:
        candidate_pairs = candidate_pairs[:num_pairs]

    if print_flag:
        print(f"{print_text} ⇒ returning {len(candidate_pairs)}/{num_pairs} pairs")

    return candidate_pairs
    

# version with no control over duplicates and only recs pairs
# def identify_candidate_pairs_for_stable_set(
#     graph,
#     patient_df_scaled,
#     patient_df,
#     common_pairs_dict,
#     num_pairs           = 400,
#     alpha_kmeans_d      = 0.60,
#     alpha_outlier       = 0.15,
#     alpha_conflict      = 0.10,
#     alpha_pat           = 0.10,
#     alpha_child         = 0.03,
#     alpha_chain         = 0.02,
#     alpha_random        = 0.00,     # אחוז רנדום “אמיתי”
#     alpha_sys           = 0.00,
#     alpha_inf           = 0.00,
#     alpha_llm           = 0.00,
#     exclude_pairs       = None,
#     print_flag          = False,
#     print_text          = "",
#     min_k               = 2,
#     max_k               = 10,
#     random_state        = 42
# ):
#     # === שילוב שיטת LLM כבחירת זוגות מוכנה מראש ===
#     # if alpha_llm == 1:
#     #     return selected_pairs
      
    
    
#     category_dict = {
#         'labs':      ['rec1','rec2','rec3','rec4'],
#         'imaging':   ['rec5','rec6','rec7','rec8'],
#         'pharma':    ['rec9','rec10','rec11','rec12','rec13','rec14','rec15'],
#         'consult':   ['rec16','rec17','rec18'],
#         'lifestyle': ['rec19','rec20'],
#         'admin':     ['rec21']
#     }
#     category_importance = {
#         'pharma':    5,
#         'consult':   4,
#         'imaging':   3,
#         'labs':      2,
#         'lifestyle': 1,
#         'admin':     1
#     }    
#     # ---------- normalise key ----------
#     def _pair2key(pair):
#         return tuple(sorted(pair))

#     # ---------- forbidden (כבר דורגו / נשללו) ----------
#     exclude_keys = set()
#     if exclude_pairs:
#         for item in exclude_pairs:
#             key = _pair2key(item if len(item) == 2 else item[0])
#             exclude_keys.add(key)

#     candidate_pairs = []

#     # ======================================================================
#     #  פונקציית עזר: מריצה “שלב” עם oversampling אדפטיבי         # <<< NEW >>>
#     # ======================================================================
#     def run_stage(name, need, produce_func, max_tries=3, init_factor=3):
#         """מוסיפה זוגות ל-candidate_pairs (עד *need*) מתוך produce_func."""
#         if need <= 0:
#             return
#         factor   = init_factor
#         obtained = 0
#         tries    = 0
#         while obtained < need and tries < max_tries:
#             raw = produce_func(need * factor)
#             got = take_until_quota(raw, need - obtained, exclude_keys)
#             candidate_pairs.extend(got)
#             obtained += len(got)
#             factor  *= 2   # מגדילים overshoot
#             tries   += 1
#             if print_flag:
#                 print(f"[{name}]  pass {tries}:  +{len(got)}  (total {obtained}/{need})")
#         if print_flag and obtained < need:
#             print(f"[{name}]  WARNING – חסרים {need-obtained} זוגות")

#     # ======================================================================
#     # 1) K-MEANS
#     # ======================================================================
#     run_stage(
#         "KMEANS",
#         int(num_pairs * alpha_kmeans_d),
#         lambda n: select_pairs_kmeans_weight(
#             patient_df_scaled,
#             rec_cols=[c for c in patient_df.columns if c.startswith('rec')],
#             num_pairs=n,
#             min_k=50,
#             max_k=max_k,
#             random_state=random_state)
#     )

#     # 2) OUTLIERS
#     run_stage(
#         "OUTLIER",
#         int(num_pairs * alpha_outlier),
#         lambda n: select_outlier_pairs(
#             patient_df,
#             random_state=random_state,
#             max_outliers=50,
#             max_inliers=200)[:n]
#     )

#     # 3) CONFLICT
#     run_stage(
#         "CONFLICT",
#         int(num_pairs * alpha_conflict),
#         lambda n: select_conflict_pairs_balanced(
#             patient_df, top_n=max(2000, n))[:n]
#     )

#     # 4) HAMMING / PAT
#     run_stage(
#         "HAMMING",
#         int(num_pairs * alpha_pat),
#         lambda n: get_top_k_pairs_by_hamming_patients(patient_df,
#                                                       total_samples=n)
#     )

#     # 5) CHILD
#     run_stage(
#         "CHILD",
#         int(num_pairs * alpha_child),
#         lambda n: random.sample(select_child_pairs(graph), k=min(n*3,
#                                                                  len(select_child_pairs(graph))))
#     )

#     # 6) CHAIN
#     run_stage(
#         "CHAIN",
#         int(num_pairs * alpha_chain),
#         lambda n: random.sample(
#             select_chain_pairs(
#                 remove_contained_chains(
#                     find_all_possible_chains(graph, False), False)),
#             k=min(n*3, len(select_chain_pairs(
#                 remove_contained_chains(
#                     find_all_possible_chains(graph, False), False))))
#         )
#     )

#     # 7) SYSTEMATIC
#     run_stage(
#         "SYSTEM",
#         int(num_pairs * alpha_sys),
#         lambda n: select_systematic_pairs_list_only(
#             patient_df,
#             category_dict,
#             category_importance,
#             total_pairs=n,
#             random_state=random_state)
#     )

#     # 8) INFLUENTIAL
#     if alpha_inf > 0 and common_pairs_dict:
#         run_stage(
#             "INFLUENTIAL",
#             int(num_pairs * alpha_inf),
#             lambda n: select_influential_pairs(common_pairs_dict, n*3)
#         )

#     # 9) RANDOM – אם מישהו בחר אחוז רנדום אמיתי
#     if alpha_random > 0:
#         run_stage(
#             "RANDOM",
#             int(num_pairs * alpha_random),
#             lambda n: identify_random_pairs_with_recommendations(patient_df, n*4)
#         )
#     # 10) RANDOM – fill any remaining gap                    # <<< CHG >>>
#     still_need = num_pairs - len(candidate_pairs)
#     if still_need > 0:
#         random_pairs = identify_random_pairs(patient_df, still_need*4)
#         random.shuffle(random_pairs)
#         fill = take_until_quota(random_pairs, still_need, exclude_keys)
#         candidate_pairs.extend(fill)
#         if print_flag:
#             print(f"[RANDOM-FILL]  added {len(fill)}/{still_need}")

#     # -----------------------------------------------------------------
#     # סגירה סופית
#     # -----------------------------------------------------------------
#     if len(candidate_pairs) > num_pairs:
#         candidate_pairs = candidate_pairs[:num_pairs]

#     if print_flag:
#         print(f"{print_text} ⇒ returning {len(candidate_pairs)}/{num_pairs} pairs")

#     return candidate_pairs


def check_ranked_pairs_issues(ranked_pairs, patient_df):
    """
    Checks for possible issues in ranked pairs:
    - Conflicting pairs
    - Duplicate pairs
    - Uninformative pairs (identical feature vectors)
    - Low feature variability
    """

    # ====================
    # 1. Prepare DataFrame
    # ====================

    # Ensure no trailing spaces in column names
    patient_df.columns = patient_df.columns.str.strip()

    # Set 'patient_num' as the index instead of dropping it
    df_indexed = patient_df.set_index('patient_num')

    # Convert each row to a dict {column_name: value} for quick lookup
    patient_features = df_indexed.to_dict(orient='index')

    # =======================
    # 2. Initialize variables
    # =======================

    conflicting_pairs = []
    duplicate_pairs = []
    uninformative_pairs = []

    # Keep track of pairs to detect duplicates
    seen_pairs = set()

    # ============================
    # 3. Check each ranked pair
    # ============================

    for (patient_i, patient_j), confidence in ranked_pairs:

        # Duplicate check means exact same order (patient_i, patient_j)
        if (patient_i, patient_j) in seen_pairs:
            duplicate_pairs.append((patient_i, patient_j))
        else:
            seen_pairs.add((patient_i, patient_j))

        # Conflict check means we have the reverse pair (patient_j, patient_i) somewhere
        if (patient_j, patient_i) in seen_pairs:
            conflicting_pairs.append((patient_i, patient_j))

        # Uninformative pair check - do both patients exist, and do they have identical features?
        if (patient_i in patient_features) and (patient_j in patient_features):
            feat_i = df_indexed.loc[patient_i].values
            feat_j = df_indexed.loc[patient_j].values
            if np.array_equal(feat_i, feat_j):
                uninformative_pairs.append((patient_i, patient_j))
        else:
            # Optional: warn if the patient isn't found
            print(f"Warning: Patient {patient_i} or {patient_j} not found in DataFrame index.")

    # ===================================
    # 4. Check overall feature variability
    # ===================================
    # Now that 'patient_num' is the index, the columns are the actual features
    all_features = df_indexed.values  # shape: [num_patients, num_features]
    feature_variability = np.std(all_features, axis=0)

    # Threshold for low variability (example: < 0.01)
    low_variability_features = np.where(feature_variability < 0.01)[0]

    # ===================
    # 5. Print a summary
    # ===================
    print("=== Ranked Pairs Issues Report ===")
    print(f"Number of ranked pairs: {len(ranked_pairs)}")

    print(f"Number of conflicting pairs: {len(conflicting_pairs)}")
    if conflicting_pairs:
        print("Example conflicting pair:", conflicting_pairs[:5])

    print(f"Number of duplicate pairs: {len(duplicate_pairs)}")
    if duplicate_pairs:
        print("Example duplicate pair:", duplicate_pairs[:5])

    print(f"Number of uninformative pairs: {len(uninformative_pairs)}")
    if uninformative_pairs:
        print("Example uninformative pair:", uninformative_pairs[:5])

    print(f"Number of low variability features: {len(low_variability_features)}")
    if len(low_variability_features) > 0:
        print("Low variability feature indices:", low_variability_features)

    # ===================
    # 6. Return the stats
    # ===================
    return {
        "conflicting_pairs": conflicting_pairs,
        "duplicate_pairs": duplicate_pairs,
        "uninformative_pairs": uninformative_pairs,
        "low_variability_features": low_variability_features
    }
