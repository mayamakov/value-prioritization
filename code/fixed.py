# fixed.py
from concurrent import futures
import random
import math
import numpy as np
import pandas as pd
import torch
import xgboost as xgb  # חשוב עבור Booster/DMatrix

# pair selection & models
from pairs_selection import identify_candidate_pairs_for_stable_set
from logistic_regression import run_single_logistic_experiment
from xgb import run_single_xgb_experiment
from ranknet import run_single_ranknet_experiment

# math (graph) methods
from math_completion import compute_math_scores
from pairs_choosing_methods import get_aut_pairs_for_math, get_rec_priority_pairs_for_math

# metrics & utils
from metrics import pairwise_accuracy, calc_auc, evaluate_ranking_metrics_for_doctor
from utils import filter_pairs_by_doctor_truth, get_alpha_name_from_row


# --------------------------------------------------------------------
# Helpers: normalize & filter OVERALL inputs
# --------------------------------------------------------------------

def _normalize_overall_ranking(ranking):
    """
    הופך כל אחד מהפורמטים הבאים ל-[(pid, score)]:
      - [pid, pid, ...]
      - [(pid, score), ...]
      - {pid: score, ...}
    """
    if ranking is None:
        return []

    if isinstance(ranking, dict):
        out = []
        for pid, sc in ranking.items():
            try:
                pid_i = int(pid)
            except Exception:
                continue
            try:
                sc_f = float(sc)
            except Exception:
                sc_f = 0.0
            out.append((pid_i, sc_f))
        return out

    if isinstance(ranking, (list, tuple)):
        if not ranking:
            return []
        first = ranking[0]
        if isinstance(first, (list, tuple)):
            out = []
            for tup in ranking:
                if not tup:
                    continue
                try:
                    pid_i = int(tup[0])
                except Exception:
                    continue
                sc_f = 0.0
                if len(tup) > 1:
                    try:
                        sc_f = float(tup[1])
                    except Exception:
                        sc_f = 0.0
                out.append((pid_i, sc_f))
            return out
        else:
            out = []
            for pid in ranking:
                try:
                    pid_i = int(pid)
                except Exception:
                    continue
                out.append((pid_i, 0.0))
            return out

    return []


def _filter_pairs_and_ranking_for_allowed(pairs_list, ranking_list, allowed_pids: set):
    """
    Returns:
      • filtered_pairs: [((i,j), conf)]  (conf=1.0 if not provided)
      • filtered_rank:  [(pid, score)]   (keeps score; filtered to allowed_pids)
    """
    filtered_pairs = []
    for e in (pairs_list or []):
        # ((i, j), conf)
        if isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[0], (list, tuple)) and len(e[0]) == 2:
            (i, j), conf = e
            try:
                i, j = int(i), int(j)
                conf = float(conf)
            except Exception:
                continue
            if i in allowed_pids and j in allowed_pids:
                filtered_pairs.append(((i, j), conf))
        # (i, j) → conf = 1.0
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            i, j = e
            try:
                i, j = int(i), int(j)
            except Exception:
                continue
            if i in allowed_pids and j in allowed_pids:
                filtered_pairs.append(((i, j), 1.0))

    # normalize gold ranking & keep scores
    norm_rank = _normalize_overall_ranking(ranking_list)
    filtered_rank = []
    for pid, sc in norm_rank:
        try:
            pid_i = int(pid)
            sc_f = float(sc)
        except Exception:
            continue
        if pid_i in allowed_pids:
            filtered_rank.append((pid_i, sc_f))

    return filtered_pairs, filtered_rank


# --------------------------------------------------------------------
# Helpers for building ALL/OVERALL matrices
# --------------------------------------------------------------------

def _build_all_matrix_aligned_to_train(df_train_scaled, df_test_scaled):
    """איחוד train+test ל־ALL מיושר לפיצ'רים של train."""
    df_all = pd.concat([df_train_scaled, df_test_scaled], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['patient_num']).copy()
    feature_cols = [c for c in df_train_scaled.columns if c != 'patient_num']
    for col in feature_cols:
        if col not in df_all.columns:
            df_all[col] = 0.0
    extras = [c for c in df_all.columns if c not in feature_cols + ['patient_num']]
    if extras:
        df_all = df_all.drop(columns=extras)
    X_all = df_all[feature_cols].values
    pids = df_all['patient_num'].astype(int).tolist()
    return df_all, X_all, pids, feature_cols


def _build_all_matrix_from_full(df_overall_scaled, df_train_scaled):
    """בונה מטריצת OVERALL מתוך df_overall_scaled, מיושרת לפיצ'רי ה-train."""
    df_all = df_overall_scaled.drop_duplicates(subset=['patient_num']).copy()
    feature_cols = [c for c in df_train_scaled.columns if c != 'patient_num']
    for col in feature_cols:
        if col not in df_all.columns:
            df_all[col] = 0.0
    extras = [c for c in df_all.columns if c not in feature_cols + ['patient_num']]
    if extras:
        df_all = df_all.drop(columns=extras)
    X_all = df_all[feature_cols].values
    pids = df_all['patient_num'].astype(int).tolist()
    return df_all, X_all, pids, feature_cols


def _concat_all(df_train_scaled, df_test_scaled):
    df_all = pd.concat([df_train_scaled, df_test_scaled], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['patient_num']).copy()
    feature_cols = [c for c in df_all.columns if c != 'patient_num']
    X_all = df_all[feature_cols].values
    pids = df_all['patient_num'].astype(int).tolist()
    return df_all, X_all, pids, feature_cols


# --------------------------------------------------------------------
# Graph helpers for math_* methods
# --------------------------------------------------------------------

def _pairs_to_edges_for_math(pairs):
    """pairs יכול להיות [(i,j)] או [((i,j), conf)] → [(i,j)]"""
    if not pairs:
        return []
    out = []
    for item in pairs:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
            (i, j), _ = item
            out.append((int(i), int(j)))
        elif isinstance(item, tuple) and len(item) == 2:
            i, j = item
            out.append((int(i), int(j)))
    return out


def _build_full_train_graph_for_math(docs_range, all_train_pairs, df_train,
                                     age_col='age', risk_col='risk',
                                     patient_id_col='patient_num'):
    """מאחד את כל הזוגות ב-TRAIN + אוטומטי + rec-priority (UNCAPPED)."""
    rec_cols = [c for c in df_train.columns if str(c).lower().startswith('rec')]
    edges = set()

    # 1) כל זוגות הרופאים
    if isinstance(all_train_pairs, dict):
        for i in docs_range:
            dk = f"doctor{i}"
            if dk in all_train_pairs:
                for e in _pairs_to_edges_for_math(all_train_pairs[dk]):
                    edges.add(e)

    # 2) זוגות אוטומטיים – ללא תקרה
    aut_uncapped = get_aut_pairs_for_math(
        patient_df=df_train,
        age_col=age_col,
        risk_col=risk_col,
        rec_cols=rec_cols,
        patient_id_col=patient_id_col
    )
    for e in _pairs_to_edges_for_math(aut_uncapped):
        edges.add(e)

    # 3) rec-priority – ללא תקרה
    recp_uncapped = get_rec_priority_pairs_for_math(
        patient_df=df_train,
        rec_cols=rec_cols,
        patient_id_col=patient_id_col
    )
    for e in _pairs_to_edges_for_math(recp_uncapped):
        edges.add(e)

    return {"edges": list(edges)}


# --------------------------------------------------------------------
# Scoring helper: robust prediction across model types
# --------------------------------------------------------------------

def _predict_scores_any(model, X):
    """
    מוציא וקטור ציונים רציף לכל מטופל מכל מודל:
      - XGBoost Booster: booster.predict(DMatrix)
      - XGBClassifier/XGBRanker: booster.predict(DMatrix) או predict_proba
      - sklearn LogisticRegression: predict_proba / decision_function
      - RankNet (PyTorch): forward(X) → vector
    """
    t = str(type(model))

    # XGBoost Booster טהור
    if "xgboost.core.Booster" in t:
        return model.predict(xgb.DMatrix(X))

    # עטיפות XGB (sklearn API)
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            return booster.predict(xgb.DMatrix(X))
        except Exception:
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)[:, 1]
            return model.predict(X)

    # sklearn קלאסי
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass

    # RankNet (PyTorch)
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(X, dtype=np.float32))
            out = model(x)
            arr = out.detach().cpu().numpy()
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            return arr

    # fallback
    return np.asarray(model.predict(X), dtype=float)


# --------------------------------------------------------------------
# Local completion for math_* (topo with priorities)
# --------------------------------------------------------------------
from collections import defaultdict
import heapq

def _normalize_scores(scores: dict) -> dict:
    if not scores:
        return {}
    vals = list(scores.values())
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.0 for k in scores}
    return {k: (scores[k] - vmin) / (vmax - vmin) for k in scores}


def _build_adj_simple(edges, patients):
    out_edges = defaultdict(set)
    in_edges  = defaultdict(set)
    for u, v in edges:
        if u == v:
            continue
        out_edges[u].add(v)
        in_edges[v].add(u)
    for p in patients:
        out_edges.setdefault(p, set())
        in_edges.setdefault(p, set())
    return out_edges, in_edges


def _kahn_with_priority(out_edges, in_edges, priority, secondary, tertiary):
    indeg = {u: len(in_edges[u]) for u in in_edges}
    remaining = set(out_edges.keys()) | set(in_edges.keys())
    heap = []
    for u in remaining:
        if indeg.get(u, 0) == 0:
            heapq.heappush(heap, (-priority.get(u, 0.0),
                                  -secondary.get(u, 0.0),
                                  -tertiary.get(u, 0.0), u))
    order = []
    while remaining:
        if heap:
            _, _, _, u = heapq.heappop(heap)
        else:
            # מחזור: בוחרים את ה"עדיף" וממשיכים
            u = max(remaining, key=lambda x: (priority.get(x, 0.0),
                                              secondary.get(x, 0.0),
                                              tertiary.get(x, 0.0)))
        if u not in remaining:
            continue
        order.append(u)
        remaining.remove(u)
        for v in list(out_edges.get(u, [])):
            if v in remaining:
                indeg[v] = max(0, indeg.get(v, 0) - 1)
                if indeg[v] == 0:
                    heapq.heappush(heap, (-priority.get(v, 0.0),
                                          -secondary.get(v, 0.0),
                                          -tertiary.get(v, 0.0), v))
    return order


def complete_total_order_local(train_edges, patients_df, math_scores_by_method,
                               priority_method="math_a_domcount",
                               secondary_method="math_b_beaten_inv"):
    patients = patients_df["patient_num"].astype(int).tolist()
    out_edges, in_edges = _build_adj_simple(train_edges, patients)
    priority  = _normalize_scores(math_scores_by_method.get(priority_method, {}))
    secondary = _normalize_scores(math_scores_by_method.get(secondary_method, {})) if secondary_method else {p: 0.0 for p in patients}
    tertiary  = {p: -float(p) for p in patients}  # deterministic tie-break
    topo_order = _kahn_with_priority(out_edges, in_edges, priority, secondary, tertiary)
    N = len(topo_order)
    pos_score = {pid: float(N - idx) for idx, pid in enumerate(topo_order)}
    eps = 1e-3
    finished_scores = {
        pid: pos_score[pid] + eps * priority.get(pid, 0.0) + (eps**2) * secondary.get(pid, 0.0)
        for pid in topo_order
    }
    return sorted(finished_scores.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------
# One doctor run (כולל חישוב OVERALL אך ורק כאן)
# --------------------------------------------------------------------

def single_doctor_experiment(
    doctor_key,
    pairs_chosen_by_method,
    train_aut_pairs,
    train_rec_prior,
    df_train_scaled,
    df_test_scaled,
    all_train_pairs,
    all_test_pairs,
    train_rankings,
    test_rankings,
    alpha_name,
    model_type,
    xgb_params,
    num_boost_round,
    conf=True,
    recs_only=True,
    llm=False,
    pairs_are_filtered: bool = False,
    add_auto_pairs=True,
    weighted_metics=False,
    math_scores_by_method=None,
    # OVERALL
    overall_pairs=None,
    overall_rankings=None,
    df_overall_scaled=None,   # אופציונלי: אם יש DF מלא ל-OVERALL
):
    # map pairs to this doctor's truth if needed
    if pairs_are_filtered:
        chosen_train_pairs = list(pairs_chosen_by_method)  # [((i,j), conf)]
    else:
        chosen_train_pairs = filter_pairs_by_doctor_truth(
            pairs_chosen_by_method, all_train_pairs, doctor_key
        )

    if add_auto_pairs:
        chosen_train_pairs = chosen_train_pairs + train_aut_pairs + train_rec_prior

    # -------- math_* : COMPLETE from partial graph + OVERALL metrics --------
    if isinstance(model_type, str) and model_type.startswith("math_"):
        if overall_pairs is None or overall_rankings is None:
            raise ValueError("overall_pairs / overall_rankings are required for math_* runs.")

        train_graph_edges = _pairs_to_edges_for_math(list(chosen_train_pairs))

        # DF מלא אם ניתן, אחרת איחוד train+test
        if df_overall_scaled is not None:
            df_all = df_overall_scaled.drop_duplicates(subset=['patient_num']).copy()
        else:
            df_all, _, _, _ = _concat_all(df_train_scaled, df_test_scaled)

        math_scores_by_method_local = compute_math_scores({"edges": train_graph_edges}, df_all)

        completed_ranking = complete_total_order_local(
            train_edges=train_graph_edges,
            patients_df=df_all,
            math_scores_by_method=math_scores_by_method_local,
            priority_method=model_type,
            secondary_method="math_b_beaten_inv"
        )

        score_dict_all = dict(completed_ranking)
        ranked_all = sorted(score_dict_all.items(), key=lambda kv: kv[1], reverse=True)

        # סינון יעדים ל-PID-ים שקיבלו ציון
        allowed = set(int(pid) for pid, _ in ranked_all)
        f_pairs, f_rank = _filter_pairs_and_ranking_for_allowed(
            (overall_pairs or {}).get(doctor_key, []),
            (overall_rankings or {}).get(doctor_key, []),
            allowed
        )

        ov_acc = pairwise_accuracy(score_dict_all, f_pairs) if f_pairs else None
        ov_auc = calc_auc(score_dict_all, f_pairs) if f_pairs else None
        ov_rank_metrics = evaluate_ranking_metrics_for_doctor(ranked_all, f_rank) if f_rank else {}

        results = {}
        if ov_acc is not None:
            results['overall_accuracy'] = ov_acc
        if ov_auc is not None:
            results['overall_auc'] = ov_auc
        for k, v in ov_rank_metrics.items():
            results[f'overall_{k}'] = v

        return doctor_key, results, None

    # -------- learned models: train/test as before --------
    if model_type == 'logistic_regression':
        model, results = run_single_logistic_experiment(
            doctor_key,
            chosen_train_pairs,
            all_train_pairs,
            all_test_pairs,
            df_train_scaled,
            df_test_scaled,
            train_rankings,
            test_rankings,
            conf=conf,
            recs_only=recs_only,
            llm=llm
        )

    elif model_type in ('lambdamart', 'xgb'):
        model, results = run_single_xgb_experiment(
            doctor_key=doctor_key,
            chosen_train_pairs=chosen_train_pairs,
            all_train_pairs=all_train_pairs,
            all_test_pairs=all_test_pairs,
            df_train_scaled=df_train_scaled,
            df_test_scaled=df_test_scaled,
            # אם ב-xgb.py יש פרמטרים האלה – העבירי אותם. אם לא קיימים שם, אפשר למחוק את שתי השורות הבאות:
            df_overall_scaled=df_overall_scaled,
            train_rankings=train_rankings,
            test_rankings=test_rankings,
            overall_rankings=overall_rankings,
            xgb_params=xgb_params,
            num_boost_round=num_boost_round,
            conf=conf,
            recs_only=recs_only,
            llm=llm
        )

    elif model_type == 'ranknet':
        model, results = run_single_ranknet_experiment(
            doctor_key,
            chosen_train_pairs,
            all_train_pairs,
            all_test_pairs,
            df_train_scaled,
            df_test_scaled,
            train_rankings,
            test_rankings,
            seed=42,
            hidden_dim=128,
            lr=1e-3,
            num_epochs=500,
            print_flag=False,
            recs_only=recs_only,
            llm=llm,
            track_influence=False,
            weighted_metics=weighted_metics
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- חישוב OVERALL למודלים מלומדים: אך ורק כאן ---
    if (model_type in ('logistic_regression', 'lambdamart', 'xgb', 'ranknet')
        and overall_pairs is not None and overall_rankings is not None):

        # אם פונקציה פנימית החזירה בטעות overall_* – נמחק כדי שלא נדרוס בטעות
        for k in list(results.keys()):
            if k.startswith("overall_"):
                results.pop(k)

        # בניית מטריצת OVERALL (מיושרת לפיצ'רים של ה-train)
        if df_overall_scaled is not None:
            _, X_overall, pids, _ = _build_all_matrix_from_full(df_overall_scaled, df_train_scaled)
        else:
            _, X_overall, pids, _ = _build_all_matrix_aligned_to_train(df_train_scaled, df_test_scaled)

        # ניבוי ציונים, תמיכה בכל סוגי המודלים
        y_pred_overall = _predict_scores_any(model, X_overall)
        y_pred_overall = np.asarray(y_pred_overall, dtype=float)

        # נרמול מונוטוני (AUC/דירוג לא יפגעו)
        if np.allclose(y_pred_overall.min(), y_pred_overall.max()):
            y_pred_overall = np.zeros_like(y_pred_overall)
        else:
            y_pred_overall = (y_pred_overall - y_pred_overall.min()) / (y_pred_overall.max() - y_pred_overall.min())

        score_dict_all = {int(pid): float(s) for pid, s in zip(pids, y_pred_overall)}
        ranked_all = sorted(score_dict_all.items(), key=lambda kv: kv[1], reverse=True)

        allowed = set(score_dict_all.keys())
        f_pairs, f_rank = _filter_pairs_and_ranking_for_allowed(
            (overall_pairs or {}).get(doctor_key, []),
            (overall_rankings or {}).get(doctor_key, []),
            allowed
        )

        ov_acc = pairwise_accuracy(score_dict_all, f_pairs) if f_pairs else None
        ov_auc = calc_auc(score_dict_all, f_pairs) if f_pairs else None
        ov_rank_metrics = evaluate_ranking_metrics_for_doctor(ranked_all, f_rank) if f_rank else {}

        if ov_acc is not None:
            results['overall_accuracy'] = ov_acc
        if ov_auc is not None:
            results['overall_auc'] = ov_auc
        for k, v in ov_rank_metrics.items():
            results[f'overall_{k}'] = v

    return doctor_key, results, model


# --------------------------------------------------------------------
# Run alpha (10 doctors) — קריאה נקייה ואחידה
# --------------------------------------------------------------------

def single_alpha_experiment(
        idx,
        alpha_set,
        train_graph,
        test_graph,
        exclude_pairs,
        train_aut_pairs,
        train_rec_prior,
        df_train,
        df_train_scaled,
        df_test,
        df_test_scaled,
        patient_df,
        all_train_pairs,
        all_test_pairs,
        train_rankings,
        test_rankings,
        model_type,
        docs_range=None,
        xgb_params=None,
        num_boost_round=None,
        llm=False,
        # OVERALL
        overall_pairs=None,
        overall_rankings=None,
        df_overall_scaled=None,   # מומלץ להעביר patient_df_scaled אם יש
):
    alpha_name = get_alpha_name_from_row(alpha_set)
    print(f"\nAlphaSet {idx}: {alpha_set}")
    random.seed(42)
    if docs_range is None:
        docs_range = range(1, 11)

    # build candidate pairs for this alpha
    raw_train_pairs = identify_candidate_pairs_for_stable_set(
        graph               = train_graph,
        patient_df_scaled   = df_train_scaled,
        patient_df          = df_train,
        common_pairs_dict   = {},
        num_pairs           = alpha_set[0],
        alpha_kmeans_d      = alpha_set[1],
        alpha_outlier       = alpha_set[2],
        alpha_conflict      = alpha_set[3],
        alpha_pat           = alpha_set[4],
        alpha_child         = alpha_set[5],
        alpha_chain         = alpha_set[6],
        alpha_random        = alpha_set[7],
        alpha_sys           = alpha_set[8],
        alpha_inf           = alpha_set[9],
        alpha_llm           = alpha_set[10],
        exclude_pairs       = exclude_pairs,
        random_state        = 42
    )

    is_math = isinstance(model_type, str) and model_type.startswith("math_")
    if is_math:
        metric_keys = ['overall_accuracy','overall_auc','overall_ndcg','overall_tau','overall_map','overall_rbo']
    else:
        metric_keys = [
            'train_accuracy','test_accuracy','train_auc','test_auc',
            'train_ndcg','test_ndcg','train_tau','test_tau',
            'train_map','test_map','train_rbo','test_rbo',
            'overall_accuracy','overall_auc','overall_ndcg','overall_tau','overall_map','overall_rbo'
        ]

    per_doctor_local = {
        f'doctor{i}': {'alpha_set': [], **{k: [] for k in metric_keys}}
        for i in docs_range
    }
    totals = {k: 0.0 for k in metric_keys}
    allowed_metric_keys = set(metric_keys)

    def run_for_doctor(i):
        doctor_key = f'doctor{i}'
        doctor_key, results, _ = single_doctor_experiment(
            doctor_key,
            raw_train_pairs,
            train_aut_pairs,
            train_rec_prior,
            df_train_scaled,
            df_test_scaled,
            all_train_pairs,
            all_test_pairs,
            train_rankings,
            test_rankings,
            alpha_name,
            model_type,
            xgb_params,
            num_boost_round,
            conf=True,
            recs_only=True,
            llm=llm,
            math_scores_by_method=None,  # לא נדרש במסלולי learned
            overall_pairs=overall_pairs,
            overall_rankings=overall_rankings,
            df_overall_scaled=df_overall_scaled,   # אם יש DF מלא – מצוין
        )
        return doctor_key, results

    # נריץ במקביל על רופאים – Thread ולא Process (אין בעיות pickling/interrupt)
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        futs = [executor.submit(run_for_doctor, i) for i in docs_range]
        for fut in futures.as_completed(futs):
            doctor_key, metrics = fut.result()
            per_doctor_local[doctor_key]['alpha_set'].append(alpha_name)
            for k, v in metrics.items():
                if k in allowed_metric_keys and v is not None:
                    try:
                        vf = float(v)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(vf) or math.isinf(vf):
                        continue
                    per_doctor_local[doctor_key][k].append(vf)
                    totals[k] += vf

    alpha_result = {'alpha_set': alpha_name, 'model_type': model_type}
    num_docs = len(list(docs_range)) or 1
    for k in totals:
        alpha_result[k] = totals[k] / num_docs

    return idx, alpha_result, per_doctor_local


# from concurrent import futures
# import random
# import math
# import numpy as np
# import pandas as pd
# import torch

# # pair selection & models
# from pairs_selection import identify_candidate_pairs_for_stable_set
# from logistic_regression import run_single_logistic_experiment
# from xgb import run_single_xgb_experiment
# from ranknet import run_single_ranknet_experiment

# # math (graph) methods
# from math_completion import compute_math_scores
# from pairs_choosing_methods import get_aut_pairs_for_math, get_rec_priority_pairs_for_math

# # metrics & utils
# from metrics import pairwise_accuracy, calc_auc, evaluate_ranking_metrics_for_doctor
# from utils import filter_pairs_by_doctor_truth, get_alpha_name_from_row

# # --------------------------------------------------------------------
# # Helpers: normalize & filter OVERALL inputs
# # --------------------------------------------------------------------

# def _normalize_overall_ranking(ranking):
#     """
#     הופך כל אחד מהפורמטים הבאים ל-[(pid, score)]:
#       - [pid, pid, ...]
#       - [(pid, score), ...]
#       - {pid: score, ...}
#     """
#     if ranking is None:
#         return []

#     if isinstance(ranking, dict):
#         out = []
#         for pid, sc in ranking.items():
#             try:
#                 pid_i = int(pid)
#             except Exception:
#                 continue
#             try:
#                 sc_f = float(sc)
#             except Exception:
#                 sc_f = 0.0
#             out.append((pid_i, sc_f))
#         return out

#     if isinstance(ranking, (list, tuple)):
#         if not ranking:
#             return []
#         first = ranking[0]
#         if isinstance(first, (list, tuple)):
#             out = []
#             for tup in ranking:
#                 if not tup:
#                     continue
#                 try:
#                     pid_i = int(tup[0])
#                 except Exception:
#                     continue
#                 sc_f = 0.0
#                 if len(tup) > 1:
#                     try:
#                         sc_f = float(tup[1])
#                     except Exception:
#                         sc_f = 0.0
#                 out.append((pid_i, sc_f))
#             return out
#         else:
#             out = []
#             for pid in ranking:
#                 try:
#                     pid_i = int(pid)
#                 except Exception:
#                     continue
#                 out.append((pid_i, 0.0))
#             return out

#     return []

# def _filter_pairs_and_ranking_for_allowed(pairs_list, ranking_list, allowed_pids: set):
#     """
#     Returns:
#       • filtered_pairs: [((i,j), conf)]  (conf=1.0 if not provided)
#       • filtered_rank:  [(pid, score)]   (keeps score; filtered to allowed_pids)
#     """
#     filtered_pairs = []
#     for e in (pairs_list or []):
#         # ((i, j), conf)
#         if isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[0], (list, tuple)) and len(e[0]) == 2:
#             (i, j), conf = e
#             try:
#                 i, j = int(i), int(j)
#                 conf = float(conf)
#             except Exception:
#                 continue
#             if i in allowed_pids and j in allowed_pids:
#                 filtered_pairs.append(((i, j), conf))
#         # (i, j) → conf = 1.0
#         elif isinstance(e, (list, tuple)) and len(e) == 2:
#             i, j = e
#             try:
#                 i, j = int(i), int(j)
#             except Exception:
#                 continue
#             if i in allowed_pids and j in allowed_pids:
#                 filtered_pairs.append(((i, j), 1.0))

#     # normalize the gold ranking and KEEP (pid, score) tuples
#     norm_rank = _normalize_overall_ranking(ranking_list)
#     filtered_rank = []
#     for pid, sc in norm_rank:
#         try:
#             pid_i = int(pid)
#             sc_f = float(sc)
#         except Exception:
#             continue
#         if pid_i in allowed_pids:
#             filtered_rank.append((pid_i, sc_f))

#     return filtered_pairs, filtered_rank



# # --------------------------------------------------------------------
# # Helpers for building ALL/OVERALL matrices
# # --------------------------------------------------------------------

# def _build_all_matrix_aligned_to_train(df_train_scaled, df_test_scaled):
#     """איחוד train+test ל־ALL מיושר לפיצ'רים של train."""
#     df_all = pd.concat([df_train_scaled, df_test_scaled], ignore_index=True)
#     df_all = df_all.drop_duplicates(subset=['patient_num']).copy()
#     feature_cols = [c for c in df_train_scaled.columns if c != 'patient_num']
#     for col in feature_cols:
#         if col not in df_all.columns:
#             df_all[col] = 0.0
#     extras = [c for c in df_all.columns if c not in feature_cols + ['patient_num']]
#     if extras:
#         df_all = df_all.drop(columns=extras)
#     X_all = df_all[feature_cols].values
#     pids = df_all['patient_num'].astype(int).tolist()
#     return df_all, X_all, pids, feature_cols


# def _build_all_matrix_from_full(df_overall_scaled, df_train_scaled):
#     """בונה מטריצת OVERALL מתוך df_overall_scaled, מיושרת לפיצ'רי ה-train."""
#     df_all = df_overall_scaled.drop_duplicates(subset=['patient_num']).copy()
#     feature_cols = [c for c in df_train_scaled.columns if c != 'patient_num']
#     for col in feature_cols:
#         if col not in df_all.columns:
#             df_all[col] = 0.0
#     extras = [c for c in df_all.columns if c not in feature_cols + ['patient_num']]
#     if extras:
#         df_all = df_all.drop(columns=extras)
#     X_all = df_all[feature_cols].values
#     pids = df_all['patient_num'].astype(int).tolist()
#     return df_all, X_all, pids, feature_cols

# # --------------------------------------------------------------------
# # Graph helpers for math_* methods
# # --------------------------------------------------------------------

# def _pairs_to_edges_for_math(pairs):
#     """pairs יכול להיות [(i,j)] או [((i,j), conf)] → [(i,j)]"""
#     if not pairs:
#         return []
#     out = []
#     for item in pairs:
#         if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
#             (i, j), _ = item
#             out.append((int(i), int(j)))
#         elif isinstance(item, tuple) and len(item) == 2:
#             i, j = item
#             out.append((int(i), int(j)))
#     return out


# def _build_full_train_graph_for_math(docs_range, all_train_pairs, df_train,
#                                      age_col='age', risk_col='risk',
#                                      patient_id_col='patient_num'):
#     """מאחד את כל הזוגות ב-TRAIN + אוטומטי + rec-priority (UNCAPPED)."""
#     rec_cols = [c for c in df_train.columns if str(c).lower().startswith('rec')]
#     edges = set()

#     # 1) כל זוגות הרופאים
#     if isinstance(all_train_pairs, dict):
#         for i in docs_range:
#             dk = f"doctor{i}"
#             if dk in all_train_pairs:
#                 for e in _pairs_to_edges_for_math(all_train_pairs[dk]):
#                     edges.add(e)

#     # 2) זוגות אוטומטיים – ללא תקרה
#     aut_uncapped = get_aut_pairs_for_math(
#         patient_df=df_train,
#         age_col=age_col,
#         risk_col=risk_col,
#         rec_cols=rec_cols,
#         patient_id_col=patient_id_col
#     )
#     for e in _pairs_to_edges_for_math(aut_uncapped):
#         edges.add(e)

#     # 3) rec-priority – ללא תקרה
#     recp_uncapped = get_rec_priority_pairs_for_math(
#         patient_df=df_train,
#         rec_cols=rec_cols,
#         patient_id_col=patient_id_col
#     )
#     for e in _pairs_to_edges_for_math(recp_uncapped):
#         edges.add(e)

#     return {"edges": list(edges)}

# # --------------------------------------------------------------------
# # Scoring helpers for learned models
# # --------------------------------------------------------------------

# def _concat_all(df_train_scaled, df_test_scaled):
#     df_all = pd.concat([df_train_scaled, df_test_scaled], ignore_index=True)
#     df_all = df_all.drop_duplicates(subset=['patient_num']).copy()
#     feature_cols = [c for c in df_all.columns if c != 'patient_num']
#     X_all = df_all[feature_cols].values
#     pids = df_all['patient_num'].astype(int).tolist()
#     return df_all, X_all, pids, feature_cols


# def _score_learned_model_all(model, X_all):
#     # sklearn
#     if hasattr(model, "predict_proba"):
#         try:
#             return np.asarray(model.predict_proba(X_all)[:, 1], dtype=float)
#         except Exception:
#             pass
#     if hasattr(model, "decision_function"):
#         try:
#             return np.asarray(model.decision_function(X_all), dtype=float)
#         except Exception:
#             pass
#     # XGB
#     try:
#         preds = model.predict(X_all, output_margin=True)
#         return np.asarray(preds, dtype=float)
#     except TypeError:
#         try:
#             preds = np.asarray(model.predict(X_all), dtype=float)
#             if np.unique(np.round(preds, 6)).size <= 2 and hasattr(model, "predict_proba"):
#                 return np.asarray(model.predict_proba(X_all)[:, 1], dtype=float)
#             return preds
#         except Exception:
#             pass
#     except Exception:
#         pass
#     # RankNet (PyTorch)
#     if isinstance(model, torch.nn.Module):
#         model.eval()
#         with torch.no_grad():
#             x = torch.from_numpy(np.asarray(X_all, dtype=np.float32))
#             out = model(x)
#             arr = out.detach().cpu().numpy()
#             if arr.ndim == 2 and arr.shape[1] == 1:
#                 arr = arr[:, 0]
#             return np.asarray(arr, dtype=float)
#     # fallback
#     return np.zeros((X_all.shape[0],), dtype=float)

# # --------------------------------------------------------------------
# # Local completion for math_* (topo with priorities)
# # --------------------------------------------------------------------
# from collections import defaultdict
# import heapq

# def _normalize_scores(scores: dict) -> dict:
#     if not scores:
#         return {}
#     vals = list(scores.values())
#     vmin, vmax = min(vals), max(vals)
#     if vmax == vmin:
#         return {k: 0.0 for k in scores}
#     return {k: (scores[k] - vmin) / (vmax - vmin) for k in scores}


# def _build_adj_simple(edges, patients):
#     out_edges = defaultdict(set)
#     in_edges  = defaultdict(set)
#     for u, v in edges:
#         if u == v:
#             continue
#         out_edges[u].add(v)
#         in_edges[v].add(u)
#     for p in patients:
#         out_edges.setdefault(p, set())
#         in_edges.setdefault(p, set())
#     return out_edges, in_edges


# def _kahn_with_priority(out_edges, in_edges, priority, secondary, tertiary):
#     indeg = {u: len(in_edges[u]) for u in in_edges}
#     remaining = set(out_edges.keys()) | set(in_edges.keys())
#     heap = []
#     for u in remaining:
#         if indeg.get(u, 0) == 0:
#             heapq.heappush(heap, (-priority.get(u, 0.0),
#                                   -secondary.get(u, 0.0),
#                                   -tertiary.get(u, 0.0), u))
#     order = []
#     while remaining:
#         if heap:
#             _, _, _, u = heapq.heappop(heap)
#         else:
#             # מחזור: בוחרים את ה"עדיף" וממשיכים
#             u = max(remaining, key=lambda x: (priority.get(x, 0.0),
#                                               secondary.get(x, 0.0),
#                                               tertiary.get(x, 0.0)))
#         if u not in remaining:
#             continue
#         order.append(u)
#         remaining.remove(u)
#         for v in list(out_edges.get(u, [])):
#             if v in remaining:
#                 indeg[v] = max(0, indeg.get(v, 0) - 1)
#                 if indeg[v] == 0:
#                     heapq.heappush(heap, (-priority.get(v, 0.0),
#                                           -secondary.get(v, 0.0),
#                                           -tertiary.get(v, 0.0), v))
#     return order


# def complete_total_order_local(train_edges, patients_df, math_scores_by_method,
#                                priority_method="math_a_domcount",
#                                secondary_method="math_b_beaten_inv"):
#     patients = patients_df["patient_num"].astype(int).tolist()
#     out_edges, in_edges = _build_adj_simple(train_edges, patients)
#     priority  = _normalize_scores(math_scores_by_method.get(priority_method, {}))
#     secondary = _normalize_scores(math_scores_by_method.get(secondary_method, {})) if secondary_method else {p: 0.0 for p in patients}
#     tertiary  = {p: -float(p) for p in patients}  # deterministic tie-break
#     topo_order = _kahn_with_priority(out_edges, in_edges, priority, secondary, tertiary)
#     N = len(topo_order)
#     pos_score = {pid: float(N - idx) for idx, pid in enumerate(topo_order)}
#     eps = 1e-3
#     finished_scores = {
#         pid: pos_score[pid] + eps * priority.get(pid, 0.0) + (eps**2) * secondary.get(pid, 0.0)
#         for pid in topo_order
#     }
#     return sorted(finished_scores.items(), key=lambda kv: kv[1], reverse=True)

# # --------------------------------------------------------------------
# # One doctor run
# # --------------------------------------------------------------------

# def single_doctor_experiment(
#     doctor_key,
#     pairs_chosen_by_method,
#     train_aut_pairs,
#     train_rec_prior,
#     df_train_scaled,
#     df_test_scaled,
#     all_train_pairs,
#     all_test_pairs,
#     train_rankings,
#     test_rankings,
#     alpha_name,
#     model_type,
#     xgb_params,
#     num_boost_round,
#     conf=True,
#     recs_only=True,
#     llm=False,
#     pairs_are_filtered: bool = False,
#     add_auto_pairs=True,
#     weighted_metics=False,
#     math_scores_by_method=None,
#     # OVERALL
#     overall_pairs=None,
#     overall_rankings=None,
#     df_overall_scaled=None,   # אם קיים: כל ה-patients scaled
# ):
#     # map pairs to this doctor's truth if needed
#     if pairs_are_filtered:
#         chosen_train_pairs = list(pairs_chosen_by_method)  # [((i,j), conf)]
#     else:
#         chosen_train_pairs = filter_pairs_by_doctor_truth(
#             pairs_chosen_by_method, all_train_pairs, doctor_key
#         )

#     if add_auto_pairs:
#         chosen_train_pairs = chosen_train_pairs + train_aut_pairs + train_rec_prior

#     # -------- math_* : COMPLETE from partial graph + OVERALL metrics --------
#     if isinstance(model_type, str) and model_type.startswith("math_"):
#         if overall_pairs is None or overall_rankings is None:
#             raise ValueError("overall_pairs / overall_rankings are required for math_* runs.")

#         train_graph_edges = _pairs_to_edges_for_math(list(chosen_train_pairs))

#         # DF מלא אם ניתן, אחרת איחוד train+test
#         if df_overall_scaled is not None:
#             df_all = df_overall_scaled.drop_duplicates(subset=['patient_num']).copy()
#         else:
#             df_all, _, _, _ = _concat_all(df_train_scaled, df_test_scaled)

#         math_scores_by_method_local = compute_math_scores({"edges": train_graph_edges}, df_all)

#         completed_ranking = complete_total_order_local(
#             train_edges=train_graph_edges,
#             patients_df=df_all,
#             math_scores_by_method=math_scores_by_method_local,
#             priority_method=model_type,
#             secondary_method="math_b_beaten_inv"
#         )

#         score_dict_all = dict(completed_ranking)
#         ranked_all = sorted(score_dict_all.items(), key=lambda kv: kv[1], reverse=True)

#         # סינון יעדים ל-PID-ים שקיבלו ציון
#         allowed = set(int(pid) for pid, _ in ranked_all)
#         f_pairs, f_rank = _filter_pairs_and_ranking_for_allowed(
#             (overall_pairs or {}).get(doctor_key, []),
#             (overall_rankings or {}).get(doctor_key, []),
#             allowed
#         )
#         # ground-truth ranking as list of pids
#         f_rank_pids = [pid for pid, _ in f_rank]

#         ov_acc = pairwise_accuracy(score_dict_all, f_pairs)
#         ov_auc = calc_auc(score_dict_all, f_pairs)
#         # ov_rank_metrics = evaluate_ranking_metrics_for_doctor(ranked_all, f_rank_pids)
#         ov_rank_metrics = evaluate_ranking_metrics_for_doctor(ranked_all, f_rank)

#         results = {
#             'overall_accuracy': ov_acc,
#             'overall_auc': ov_auc,
#             **{f'overall_{k}': v for k, v in ov_rank_metrics.items()}
#         }
#         return doctor_key, results, None

#     # -------- learned models: train/test as before --------
#     if model_type == 'logistic_regression':
#         model, results = run_single_logistic_experiment(
#             doctor_key,
#             chosen_train_pairs,
#             all_train_pairs,
#             all_test_pairs,
#             df_train_scaled,
#             df_test_scaled,
#             train_rankings,
#             test_rankings,
#             conf=conf,
#             recs_only=recs_only,
#             llm=llm
#         )

#     elif model_type in ('lambdamart', 'xgb'):
#         model, results = run_single_xgb_experiment(
#             doctor_key,
#             chosen_train_pairs,
#             all_train_pairs,
#             all_test_pairs,
#             df_train_scaled,
#             df_test_scaled,
#             df_overall_scaled,
#             train_rankings,
#             test_rankings,
#             overall_rankings,
#             xgb_params,
#             num_boost_round,
#             conf=conf,
#             recs_only=recs_only,
#             llm=llm
#         )

#     elif model_type == 'ranknet':
#         model, results = run_single_ranknet_experiment(
#             doctor_key,
#             chosen_train_pairs,
#             all_train_pairs,
#             all_test_pairs,
#             df_train_scaled,
#             df_test_scaled,
#             train_rankings,
#             test_rankings,
#             seed=42,
#             hidden_dim=128,
#             lr=1e-3,
#             num_epochs=500,
#             print_flag=False,
#             recs_only=recs_only,
#             llm=llm,
#             track_influence=False,
#             weighted_metics=weighted_metics
#         )
#     else:
#         raise ValueError(f"Unknown model_type: {model_type}")
#         # --- אחרי שהגדרנו model, results במסלול learned models, ולפני return ---
#     if (model_type in ('logistic_regression', 'lambdamart', 'xgb', 'ranknet')
#         and overall_pairs is not None and overall_rankings is not None):

#         # בונה מטריצת OVERALL מיושרת לפיצ'רים של ה-train
#         if df_overall_scaled is not None:
#             _, X_overall, pids, _ = _build_all_matrix_from_full(df_overall_scaled, df_train_scaled)
#         else:
#             _, X_overall, pids, _ = _build_all_matrix_aligned_to_train(df_train_scaled, df_test_scaled)

#         # ניבוי ציונים לכל המטופלים (אותו flow כמו בטסט)
#         try:
#             y_pred_overall = model.predict(X_overall)
#         except Exception:
#             y_pred_overall = _score_learned_model_all(model, X_overall)

#         y_pred_overall = np.asarray(y_pred_overall, dtype=float)
#         # נרמול יציב לטווח [0,1] (חשוב רק לסקלת AUC/accuracy אחידה; הדירוג נשמר)
#         if np.allclose(y_pred_overall.min(), y_pred_overall.max()):
#             y_pred_overall = np.zeros_like(y_pred_overall)
#         else:
#             y_pred_overall = (y_pred_overall - y_pred_overall.min()) / (y_pred_overall.max() - y_pred_overall.min())

#         score_dict_all = {int(pid): float(s) for pid, s in zip(pids, y_pred_overall)}
#         ranked_all = sorted(score_dict_all.items(), key=lambda kv: kv[1], reverse=True)

#         allowed = set(score_dict_all.keys())
#         f_pairs, f_rank = _filter_pairs_and_ranking_for_allowed(
#             (overall_pairs or {}).get(doctor_key, []),
#             (overall_rankings or {}).get(doctor_key, []),
#             allowed
#         )

#         # אם אין זוגות אחרי סינון – אל תכניסי 0.0 "מזויף"
#         ov_acc = pairwise_accuracy(score_dict_all, f_pairs) if f_pairs else None
#         ov_auc = calc_auc(score_dict_all, f_pairs)           if f_pairs else None

#         ov_rank_metrics = evaluate_ranking_metrics_for_doctor(ranked_all, f_rank) if f_rank else {}

#         # עדכון results – רק אם יש ערך (ימנע ממוצע 0.0 ברמת alpha)
#         if ov_acc is not None:
#             results['overall_accuracy'] = ov_acc
#         if ov_auc is not None:
#             results['overall_auc'] = ov_auc
#         for k, v in ov_rank_metrics.items():
#             results[f'overall_{k}'] = v

#     return doctor_key, results, model

# # --------------------------------------------------------------------
# # Run alpha (10 doctors)
# # --------------------------------------------------------------------

# def single_alpha_experiment(
#         idx,
#         alpha_set,
#         train_graph,
#         test_graph,
#         exclude_pairs,
#         train_aut_pairs,
#         train_rec_prior,
#         df_train,
#         df_train_scaled,
#         df_test,
#         df_test_scaled,
#         patient_df,
#         all_train_pairs,
#         all_test_pairs,
#         train_rankings,
#         test_rankings,
#         model_type,
#         docs_range=None,
#         xgb_params=None,
#         num_boost_round=None,
#         llm=False,
#         # OVERALL
#         overall_pairs=None,
#         overall_rankings=None,
#         df_overall_scaled=None,   # מומלץ להעביר patient_df_scaled
# ):
#     alpha_name = get_alpha_name_from_row(alpha_set)
#     print(f"\nAlphaSet {idx}: {alpha_set}")
#     random.seed(42)
#     if docs_range is None:
#         docs_range = range(1, 11)

#     # build candidate pairs for this alpha
#     raw_train_pairs = identify_candidate_pairs_for_stable_set(
#         graph               = train_graph,
#         patient_df_scaled   = df_train_scaled,
#         patient_df          = df_train,
#         common_pairs_dict   = {},
#         num_pairs           = alpha_set[0],
#         alpha_kmeans_d      = alpha_set[1],
#         alpha_outlier       = alpha_set[2],
#         alpha_conflict      = alpha_set[3],
#         alpha_pat           = alpha_set[4],
#         alpha_child         = alpha_set[5],
#         alpha_chain         = alpha_set[6],
#         alpha_random        = alpha_set[7],
#         alpha_sys           = alpha_set[8],
#         alpha_inf           = alpha_set[9],
#         alpha_llm           = alpha_set[10],
#         exclude_pairs       = exclude_pairs,
#         random_state        = 42
#     )

#     is_math = isinstance(model_type, str) and model_type.startswith("math_")
#     if is_math:
#         metric_keys = ['overall_accuracy','overall_auc','overall_ndcg','overall_tau','overall_map','overall_rbo']
#     else:
#         metric_keys = [
#             'train_accuracy','test_accuracy','train_auc','test_auc',
#             'train_ndcg','test_ndcg','train_tau','test_tau',
#             'train_map','test_map','train_rbo','test_rbo',
#             'overall_accuracy','overall_auc','overall_ndcg','overall_tau','overall_map','overall_rbo'
#         ]

#     per_doctor_local = {
#         f'doctor{i}': {'alpha_set': [], **{k: [] for k in metric_keys}}
#         for i in docs_range
#     }
#     totals = {k: 0.0 for k in metric_keys}
#     allowed_metric_keys = set(metric_keys)

#     def run_for_doctor(i):
#         doctor_key = f'doctor{i}'
#         doctor_key, results, _ = single_doctor_experiment(
#             doctor_key,
#             raw_train_pairs,
#             train_aut_pairs,
#             train_rec_prior,
#             df_train_scaled,
#             df_test_scaled,
#             all_train_pairs,
#             all_test_pairs,
#             train_rankings,
#             test_rankings,
#             alpha_name,
#             model_type,
#             xgb_params,
#             num_boost_round,
#             conf=True,
#             recs_only=True,
#             llm=llm,
#             math_scores_by_method=None,  # לא נדרש כאן
#             overall_pairs=overall_pairs,
#             overall_rankings=overall_rankings,
#             df_overall_scaled=df_overall_scaled,   # מעבירים DF מלא אם יש
#         )
#         return doctor_key, results

#     with futures.ThreadPoolExecutor(max_workers=10) as executor:
#         futs = [executor.submit(run_for_doctor, i) for i in docs_range]
#         for fut in futures.as_completed(futs):
#             doctor_key, metrics = fut.result()
#             per_doctor_local[doctor_key]['alpha_set'].append(alpha_name)
#             for k, v in metrics.items():
#                 if k in allowed_metric_keys and v is not None:
#                     try:
#                         vf = float(v)
#                     except (TypeError, ValueError):
#                         continue
#                     if math.isnan(vf) or math.isinf(vf):
#                         continue
#                     per_doctor_local[doctor_key][k].append(vf)
#                     totals[k] += vf

#     alpha_result = {'alpha_set': alpha_name, 'model_type': model_type}
#     num_docs = len(list(docs_range)) or 1
#     for k in totals:
#         alpha_result[k] = totals[k] / num_docs

#     return idx, alpha_result, per_doctor_local



# # ---------------- run alpha (10 doctors) ----------------
# def single_alpha_experiment(
#         idx,
#         alpha_set,
#         train_graph,
#         test_graph,
#         exclude_pairs,
#         train_aut_pairs,
#         train_rec_prior,
#         df_train,
#         df_train_scaled,
#         df_test,
#         df_test_scaled,
#         patient_df,
#         all_train_pairs,
#         all_test_pairs,
#         train_rankings,
#         test_rankings,
#         model_type,
#         docs_range=None,
#         xgb_params=None,
#         num_boost_round=None,
#         llm=False,
#         # NEW: pass overall
#         overall_pairs=None,
#         overall_rankings=None,
#         df_overall_scaled=None,   # <<<<<<<< DF מלא (patient_df_scaled)
# ):
#     alpha_name = get_alpha_name_from_row(alpha_set)
#     print(f"\nAlphaSet {idx}: {alpha_set}")
#     random.seed(42)
#     if docs_range is None:
#         docs_range = range(1, 11)

#     raw_train_pairs = identify_candidate_pairs_for_stable_set(
#         graph               = train_graph,
#         patient_df_scaled   = df_train_scaled,
#         patient_df          = df_train,
#         common_pairs_dict   = {},
#         num_pairs           = alpha_set[0],
#         alpha_kmeans_d      = alpha_set[1],
#         alpha_outlier       = alpha_set[2],
#         alpha_conflict      = alpha_set[3],
#         alpha_pat           = alpha_set[4],
#         alpha_child         = alpha_set[5],
#         alpha_chain         = alpha_set[6],
#         alpha_random        = alpha_set[7],
#         alpha_sys           = alpha_set[8],
#         alpha_inf           = alpha_set[9],
#         alpha_llm           = alpha_set[10],
#         exclude_pairs       = exclude_pairs,
#         random_state        = 42
#     )

#     is_math = isinstance(model_type, str) and model_type.startswith("math_")
#     if is_math:
#         metric_keys = ['overall_accuracy','overall_auc','overall_ndcg','overall_tau','overall_map','overall_rbo']
#     else:
#         metric_keys = [
#             'train_accuracy','test_accuracy','train_auc','test_auc',
#             'train_ndcg','test_ndcg','train_tau','test_tau',
#             'train_map','test_map','train_rbo','test_rbo',
#             'overall_accuracy','overall_auc','overall_ndcg','overall_tau','overall_map','overall_rbo'
#         ]

#     per_doctor_local = {
#         f'doctor{i}': {'alpha_set': [], **{k: [] for k in metric_keys}}
#         for i in docs_range
#     }
#     totals = {k: 0.0 for k in metric_keys}
#     allowed_metric_keys = set(metric_keys)

#     def run_for_doctor(i):
#         doctor_key = f'doctor{i}'
#         doctor_key, results, _ = single_doctor_experiment(
#             doctor_key,
#             raw_train_pairs,
#             train_aut_pairs,
#             train_rec_prior,
#             df_train_scaled,
#             df_test_scaled,
#             all_train_pairs,
#             all_test_pairs,
#             train_rankings,
#             test_rankings,
#             alpha_name,
#             model_type,
#             xgb_params,
#             num_boost_round,
#             conf=True,
#             recs_only=True,
#             llm=llm,
#             math_scores_by_method=None,
#             overall_pairs=overall_pairs,
#             overall_rankings=overall_rankings,
#             df_overall_scaled=df_overall_scaled,   # <<<<<<<< מעבירים DF מלא
#         )
#         return doctor_key, results

#     with futures.ThreadPoolExecutor(max_workers=10) as executor:
#         futs = [executor.submit(run_for_doctor, i) for i in docs_range]
#         for fut in futures.as_completed(futs):
#             doctor_key, metrics = fut.result()
#             per_doctor_local[doctor_key]['alpha_set'].append(alpha_name)
#             for k, v in metrics.items():
#                 if k in allowed_metric_keys and v is not None:
#                     try:
#                         vf = float(v)
#                     except (TypeError, ValueError):
#                         continue
#                     if math.isnan(vf) or math.isinf(vf):
#                         continue
#                     per_doctor_local[doctor_key][k].append(vf)
#                     totals[k] += vf

#     alpha_result = {'alpha_set': alpha_name, 'model_type': model_type}
#     num_docs = len(list(docs_range)) or 1
#     for k in totals:
#         alpha_result[k] = totals[k] / num_docs

#     return idx, alpha_result, per_doctor_local


################################
#########################ישן

# from concurrent import futures
# import random
# import math
# import numpy as np
# import pandas as pd
# import torch

# # pair selection & models
# from pairs_selection import identify_candidate_pairs_for_stable_set
# from logistic_regression import run_single_logistic_experiment
# from xgb import run_single_xgb_experiment
# from ranknet import run_single_ranknet_experiment

# # math (graph) methods
# from math_completion import compute_math_scores
# from pairs_choosing_methods import get_aut_pairs_for_math, get_rec_priority_pairs_for_math

# # metrics & utils
# from metrics import pairwise_accuracy, calc_auc, evaluate_ranking_metrics_for_doctor
# from utils import filter_pairs_by_doctor_truth, get_alpha_name_from_row


# # ---------------- helpers for math FULL graph ----------------
# def _pairs_to_edges_for_math(pairs):
#     """pairs יכול להיות [(i,j)] או [((i,j), conf)] → [(i,j)]"""
#     if not pairs:
#         return []
#     out = []
#     for item in pairs:
#         if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
#             (i, j), _ = item
#             out.append((int(i), int(j)))
#         elif isinstance(item, tuple) and len(item) == 2:
#             i, j = item
#             out.append((int(i), int(j)))
#     return out

# def _build_full_train_graph_for_math(docs_range, all_train_pairs, df_train,
#                                      age_col='age', risk_col='risk',
#                                      patient_id_col='patient_num'):
#     """
#     מאחד:
#       • כל זוגות הרופאים ב-TRAIN
#       • כל הזוגות האוטומטיים (UNCAPPED) – מחושבים מה-DF
#       • כל זוגות ה-rec-priority (UNCAPPED) – מחושבים מה-DF
#     ומחזיר dict {'edges': [(u,v), ...]} שמתאים ל-compute_math_scores
#     """
#     rec_cols = [c for c in df_train.columns if str(c).lower().startswith('rec')]
#     edges = set()

#     # 1) כל זוגות הרופאים
#     if isinstance(all_train_pairs, dict):
#         for i in docs_range:
#             dk = f"doctor{i}"
#             if dk in all_train_pairs:
#                 for e in _pairs_to_edges_for_math(all_train_pairs[dk]):
#                     edges.add(e)

#     # 2) זוגות אוטומטיים – ללא תקרה
#     aut_uncapped = get_aut_pairs_for_math(
#         patient_df=df_train,
#         age_col=age_col,
#         risk_col=risk_col,
#         rec_cols=rec_cols,
#         patient_id_col=patient_id_col
#     )
#     for e in _pairs_to_edges_for_math(aut_uncapped):
#         edges.add(e)

#     # 3) rec-priority – ללא תקרה
#     recp_uncapped = get_rec_priority_pairs_for_math(
#         patient_df=df_train,
#         rec_cols=rec_cols,
#         patient_id_col=patient_id_col
#     )
#     for e in _pairs_to_edges_for_math(recp_uncapped):
#         edges.add(e)

#     return {"edges": list(edges)}


# # ---------------- helpers to score ALL patients for learned models ----------------
# def _concat_all(df_train_scaled, df_test_scaled):
#     """איחוד (עם הסרת כפילויות) ל־ALL, החזרת (df_all, X_all, pids, feature_cols)."""
#     df_all = pd.concat([df_train_scaled, df_test_scaled], ignore_index=True)
#     df_all = df_all.drop_duplicates(subset=['patient_num']).copy()
#     feature_cols = [c for c in df_all.columns if c != 'patient_num']
#     X_all = df_all[feature_cols].values
#     pids = df_all['patient_num'].astype(int).tolist()
#     return df_all, X_all, pids, feature_cols

# def _score_learned_model_all(model, X_all):
#     """
#     מנסה להוציא ציון רציף לכל מטופל מכל מודל:
#       - sklearn LogisticRegression: predict_proba / decision_function
#       - XGB/LGBM: predict
#       - PyTorch RankNet: forward(X) → vector
#     מחזיר np.array בגודל [N].
#     """
#     # sklearn-like
#     if hasattr(model, "predict_proba"):
#         try:
#             probs = model.predict_proba(X_all)[:, 1]
#             return np.asarray(probs, dtype=float)
#         except Exception:
#             pass
#     if hasattr(model, "decision_function"):
#         try:
#             dec = model.decision_function(X_all)
#             return np.asarray(dec, dtype=float)
#         except Exception:
#             pass
#     if hasattr(model, "predict"):
#         try:
#             pred = model.predict(X_all)
#             return np.asarray(pred, dtype=float)
#         except Exception:
#             pass
#     # torch (RankNet)
#     if isinstance(model, torch.nn.Module):
#         model.eval()
#         with torch.no_grad():
#             x = torch.from_numpy(np.asarray(X_all, dtype=np.float32))
#             out = model(x)
#             # out יכול להיות Nx1 או N; נמיר ל־numpy 1D
#             arr = out.detach().cpu().numpy()
#             if arr.ndim == 2 and arr.shape[1] == 1:
#                 arr = arr[:, 0]
#             return np.asarray(arr, dtype=float)

#     # fallback: אפסים (לא אמור לקרות)
#     return np.zeros((X_all.shape[0],), dtype=float)


# # ---------------- one doctor run ----------------
# def single_doctor_experiment(
#     doctor_key,
#     pairs_chosen_by_method,  # per method of this alpha
#     train_aut_pairs,
#     train_rec_prior,
#     df_train_scaled,
#     df_test_scaled,
#     all_train_pairs,
#     all_test_pairs,
#     train_rankings,
#     test_rankings,
#     alpha_name,
#     model_type,
#     xgb_params,
#     num_boost_round,
#     conf=True,
#     recs_only=True,
#     llm=False,
#     pairs_are_filtered: bool = False,
#     add_auto_pairs=True,
#     weighted_metics=False,
#     math_scores_by_method=None,
#     # NEW for overall
#     overall_pairs=None,
#     overall_rankings=None
# ):
#     # map pairs to this doctor's truth if needed
#     if pairs_are_filtered:
#         chosen_train_pairs = list(pairs_chosen_by_method)  # [((i,j), conf)]
#     else:
#         chosen_train_pairs = filter_pairs_by_doctor_truth(
#             pairs_chosen_by_method, all_train_pairs, doctor_key
#         )

#     # extend with auto/rec-priority if requested
#     if add_auto_pairs:
#         chosen_train_pairs = chosen_train_pairs + train_aut_pairs + train_rec_prior

#     # -------- math_* : only OVERALL metrics --------
#     if isinstance(model_type, str) and model_type.startswith("math_"):
#         if math_scores_by_method is None:
#             raise ValueError("math_scores_by_method is None for math_* run.")
#         if overall_pairs is None or overall_rankings is None:
#             raise ValueError("overall_pairs / overall_rankings are required for math_* runs.")

#         score_dict = math_scores_by_method[model_type]  # {pid: score}

#         # overall metrics only
#         overall_acc = pairwise_accuracy(score_dict, overall_pairs[doctor_key])
#         overall_auc = calc_auc(score_dict, overall_pairs[doctor_key])
#         scores_list = list(score_dict.items())
#         overall_rank_metrics = evaluate_ranking_metrics_for_doctor(
#             scores_list, overall_rankings[doctor_key]
#         )
#         results = {
#             'overall_accuracy': overall_acc,
#             'overall_auc': overall_auc,
#             **{f'overall_{k}': v for k, v in overall_rank_metrics.items()}
#         }
#         return doctor_key, results, None

#     # -------- learned models: train/test as before + add OVERALL --------
#     if model_type == 'logistic_regression':
#         model, results = run_single_logistic_experiment(
#             doctor_key,
#             chosen_train_pairs,
#             all_train_pairs,
#             all_test_pairs,
#             df_train_scaled,
#             df_test_scaled,
#             train_rankings,
#             test_rankings,
#             conf=conf,
#             recs_only=recs_only,
#             llm=llm
#         )

#     elif model_type in ('lambdamart', 'xgb'):
#         model, results = run_single_xgb_experiment(
#             doctor_key,
#             chosen_train_pairs,
#             all_train_pairs,
#             all_test_pairs,
#             df_train_scaled,
#             df_test_scaled,
#             train_rankings,
#             test_rankings,
#             xgb_params,
#             num_boost_round,
#             conf=conf,
#             recs_only=recs_only,
#             llm=llm
#         )

#     elif model_type == 'ranknet':
#         model, results = run_single_ranknet_experiment(
#             doctor_key,
#             chosen_train_pairs,
#             all_train_pairs,
#             all_test_pairs,
#             df_train_scaled,
#             df_test_scaled,
#             train_rankings,
#             test_rankings,
#             seed=42,
#             hidden_dim=128,
#             lr=1e-3,
#             num_epochs=500,
#             print_flag=False,
#             recs_only=recs_only,
#             llm=llm,
#             track_influence=False,
#             weighted_metics=weighted_metics
#         )
#     else:
#         raise ValueError(f"Unknown model_type: {model_type}")

#     # append OVERALL metrics for learned models (if overall_* provided)
#     if overall_pairs is not None and overall_rankings is not None:
#         # build ALL matrix & score
#         _, X_all, pids, _ = _concat_all(df_train_scaled, df_test_scaled)
#         scores_all = _score_learned_model_all(model, X_all)
#         score_dict_all = {int(pid): float(s) for pid, s in zip(pids, scores_all)}

#         # compute overall
#         ov_acc = pairwise_accuracy(score_dict_all, overall_pairs[doctor_key])
#         ov_auc = calc_auc(score_dict_all, overall_pairs[doctor_key])
#         ov_rank_metrics = evaluate_ranking_metrics_for_doctor(
#             list(score_dict_all.items()), overall_rankings[doctor_key]
#         )
#         results.update({
#             'overall_accuracy': ov_acc,
#             'overall_auc': ov_auc,
#             **{f'overall_{k}': v for k, v in ov_rank_metrics.items()}
#         })

#     return doctor_key, results, model


# # ---------------- run alpha (10 doctors) ----------------
# def single_alpha_experiment(
#         idx,
#         alpha_set,
#         train_graph,
#         test_graph,
#         exclude_pairs,
#         train_aut_pairs,
#         train_rec_prior,
#         df_train,
#         df_train_scaled,
#         df_test,
#         df_test_scaled,
#         patient_df,
#         all_train_pairs,
#         all_test_pairs,
#         train_rankings,
#         test_rankings,
#         model_type,
#         docs_range=None,
#         xgb_params=None,
#         num_boost_round=None,
#         llm=False,
#         # NEW: pass overall
#         overall_pairs=None,
#         overall_rankings=None
# ):
#     alpha_name = get_alpha_name_from_row(alpha_set)
#     print(f"\nAlphaSet {idx}: {alpha_set}")
#     random.seed(42)
#     if docs_range is None:
#         docs_range = range(1, 11)

#     # build candidate pairs for this alpha
#     raw_train_pairs = identify_candidate_pairs_for_stable_set(
#         graph               = train_graph,
#         patient_df_scaled   = df_train_scaled,
#         patient_df          = df_train,
#         common_pairs_dict   = {},
#         num_pairs           = alpha_set[0],
#         alpha_kmeans_d      = alpha_set[1],
#         alpha_outlier       = alpha_set[2],
#         alpha_conflict      = alpha_set[3],
#         alpha_pat           = alpha_set[4],
#         alpha_child         = alpha_set[5],
#         alpha_chain         = alpha_set[6],
#         alpha_random        = alpha_set[7],
#         alpha_sys           = alpha_set[8],
#         alpha_inf           = alpha_set[9],
#         alpha_llm           = alpha_set[10],
#         exclude_pairs       = exclude_pairs,
#         random_state        = 42
#     )

#     # math scores (only if math_*)
#     is_math = isinstance(model_type, str) and model_type.startswith("math_")
#     math_scores_by_method = None
#     if is_math:
#         # בונים גרף מלא מה-train + אוטומטי/rec ללא תקרה
#         train_graph_full_for_math = _build_full_train_graph_for_math(
#             docs_range=docs_range,
#             all_train_pairs=all_train_pairs,
#             df_train=df_train,
#             age_col='age',
#             risk_col='risk'
#         )
#         math_scores_by_method = compute_math_scores(train_graph_full_for_math, df_train)

#     # metric keys
#     if is_math:
#         metric_keys = [
#             'overall_accuracy', 'overall_auc',
#             'overall_ndcg', 'overall_tau', 'overall_map', 'overall_rbo'
#         ]
#     else:
#         metric_keys = [
#             'train_accuracy','test_accuracy',
#             'train_auc','test_auc',
#             'train_ndcg','test_ndcg',
#             'train_tau','test_tau',
#             'train_map','test_map',
#             'train_rbo','test_rbo',
#             # add ALL for learned models
#             'overall_accuracy','overall_auc',
#             'overall_ndcg','overall_tau','overall_map','overall_rbo'
#         ]

#     per_doctor_local = {
#         f'doctor{i}': {'alpha_set': [], **{k: [] for k in metric_keys}}
#         for i in docs_range
#     }
#     totals = {k: 0.0 for k in metric_keys}
#     allowed_metric_keys = set(metric_keys)

#     def run_for_doctor(i):
#         doctor_key = f'doctor{i}'
#         doctor_key, results, _ = single_doctor_experiment(
#             doctor_key,
#             raw_train_pairs,
#             train_aut_pairs,
#             train_rec_prior,
#             df_train_scaled,
#             df_test_scaled,
#             all_train_pairs,
#             all_test_pairs,
#             train_rankings,
#             test_rankings,
#             alpha_name,
#             model_type,
#             xgb_params,
#             num_boost_round,
#             conf=True,
#             recs_only=True,
#             llm=llm,
#             math_scores_by_method=math_scores_by_method,
#             overall_pairs=overall_pairs,
#             overall_rankings=overall_rankings
#         )
#         return doctor_key, results

#     # parallel over doctors
#     with futures.ThreadPoolExecutor(max_workers=10) as executor:
#         futs = [executor.submit(run_for_doctor, i) for i in docs_range]
#         for fut in futures.as_completed(futs):
#             doctor_key, metrics = fut.result()
#             per_doctor_local[doctor_key]['alpha_set'].append(alpha_name)

#             for k, v in metrics.items():
#                 if k in allowed_metric_keys and v is not None:
#                     try:
#                         vf = float(v)
#                     except (TypeError, ValueError):
#                         continue
#                     if math.isnan(vf) or math.isinf(vf):
#                         continue
#                     per_doctor_local[doctor_key][k].append(vf)
#                     totals[k] += vf

#     # averages across doctors
#     alpha_result = {'alpha_set': alpha_name, 'model_type': model_type}
#     num_docs = len(list(docs_range)) or 1
#     for k in totals:
#         alpha_result[k] = totals[k] / num_docs

#     return idx, alpha_result, per_doctor_local


# # from concurrent import futures
# # import random 

# # # from utils import filter_pairs_by_doctor_truth
# # from pairs_selection import identify_candidate_pairs_for_stable_set
# # from logistic_regression import run_single_logistic_experiment
# # from xgb import run_single_xgb_experiment
# # from ranknet import run_single_ranknet_experiment
# # from pairs_selection import identify_candidate_pairs_for_stable_set
# # from math_completion import compute_math_scores, run_single_math_experiment
# # from pairs_choosing_methods import get_aut_pairs_for_math, get_rec_priority_pairs_for_math
# # from utils import filter_pairs_by_doctor_truth
# # from utils import get_alpha_name_from_row


# # ####helper for the math part 
# # def _pairs_to_edges_for_math(pairs):
# #     """pairs יכול להיות [(i,j)] או [((i,j), conf)] → מחזיר [(i,j)]"""
# #     if not pairs:
# #         return []
# #     out = []
# #     for item in pairs:
# #         if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
# #             (i, j), _ = item
# #             out.append((int(i), int(j)))
# #         elif isinstance(item, tuple) and len(item) == 2:
# #             i, j = item
# #             out.append((int(i), int(j)))
# #     return out

# # def _build_full_train_graph_for_math(docs_range, all_train_pairs, df_train,
# #                                      age_col='age', risk_col='risk',
# #                                      patient_id_col='patient_num'):
# #     """
# #     מאחד:
# #       • כל זוגות הרופאים ב-TRAIN
# #       • כל הזוגות האוטומטיים (UNCAPPED) – מחושבים עכשיו מה-DF
# #       • כל זוגות ה-rec-priority (UNCAPPED) – מחושבים עכשיו מה-DF
# #     ומחזיר dict {'edges': [(u,v), ...]} שמתאים ל-compute_math_scores
# #     """
# #     rec_cols = [c for c in df_train.columns if str(c).lower().startswith('rec')]

# #     edges = set()

# #     # 1) כל זוגות הרופאים
# #     if isinstance(all_train_pairs, dict):
# #         for i in docs_range:
# #             dk = f"doctor{i}"
# #             if dk in all_train_pairs:
# #                 for e in _pairs_to_edges_for_math(all_train_pairs[dk]):
# #                     edges.add(e)

# #     # 2) זוגות אוטומטיים – ללא תקרה (נבנים מחדש מה-DF)
# #     aut_uncapped = get_aut_pairs_for_math(
# #         patient_df=df_train,
# #         age_col=age_col,
# #         risk_col=risk_col,
# #         rec_cols=rec_cols,
# #         patient_id_col=patient_id_col
# #     )
# #     for e in _pairs_to_edges_for_math(aut_uncapped):
# #         edges.add(e)

# #     # 3) rec-priority – ללא תקרה (נבנים מחדש מה-DF)
# #     recp_uncapped = get_rec_priority_pairs_for_math(
# #         patient_df=df_train,
# #         rec_cols=rec_cols,
# #         patient_id_col=patient_id_col
# #     )
# #     for e in _pairs_to_edges_for_math(recp_uncapped):
# #         edges.add(e)

# #     return {"edges": list(edges)}


# # #one doctor, one alpha=(one method, one num pairs), one model type
# # def single_doctor_experiment(  
# #     doctor_key,
# #     pairs_chosen_by_method, # per method of this alpha
# #     train_aut_pairs, 
# #     train_rec_prior,
# #     df_train_scaled,
# #     df_test_scaled,
# #     all_train_pairs,
# #     all_test_pairs,
# #     train_rankings,
# #     test_rankings,
# #     alpha_name,
# #     model_type,
# #     xgb_params,
# #     num_boost_round,
# #     conf=True,     
# #     recs_only=True,
# #     llm=False,
# #     pairs_are_filtered: bool = False,
# #     add_auto_pairs=True,
# #     weighted_metics=False,
# #     math_scores_by_method=None

# # ):
# #     # If the pairs were already mapped to the doctor's truth (with confidences), use them as-is.
# #     if pairs_are_filtered:
# #         # Expect list of ((i,j), conf)
# #         chosen_train_pairs = list(pairs_chosen_by_method)
# #     else:
# #         # Backward-compatible path: pairs_chosen_by_method is list[(i,j)]
# #         chosen_train_pairs = filter_pairs_by_doctor_truth(pairs_chosen_by_method, all_train_pairs, doctor_key)
   
# #     # ── build training set (extend with automatic & rec-priority pairs) ──────
# #     if add_auto_pairs: 
# #         chosen_train_pairs = (
# #             chosen_train_pairs + train_aut_pairs + train_rec_prior
# #         )

# #     if model_type == 'logistic_regression':
# #         model, results = run_single_logistic_experiment(
# #             doctor_key,
# #             chosen_train_pairs,
# #             all_train_pairs,
# #             all_test_pairs,
# #             df_train_scaled,
# #             df_test_scaled,
# #             train_rankings,
# #             test_rankings,
# #             conf=conf,     
# #             recs_only=recs_only,
# #             llm=llm
# #         )
    
# #     elif model_type.startswith("math_"):
# #         # אין מודל לאמן – רק ציונים מחושבים מראש
# #         _, results = run_single_math_experiment(
# #             doctor_key=doctor_key,
# #             method_name=model_type,
# #             math_scores_by_method=math_scores_by_method,
# #             df_train_scaled=df_train_scaled,
# #             df_test_scaled=df_test_scaled,
# #             all_train_pairs=all_train_pairs,
# #             all_test_pairs=all_test_pairs,
# #             train_rankings=train_rankings,
# #             test_rankings=test_rankings,
# #             recs_only=recs_only,
# #             llm=llm
# #         )
# #         return doctor_key, results, None

# #     elif model_type == 'lambdamart' or model_type=='xgb':
# #         model, results = run_single_xgb_experiment(
# #             doctor_key,
# #             chosen_train_pairs,
# #             all_train_pairs,
# #             all_test_pairs,
# #             df_train_scaled,
# #             df_test_scaled,
# #             train_rankings,
# #             test_rankings,
# #             xgb_params,  # parameters for XGBoost
# #             num_boost_round, # number of boosting rounds
# #             conf=conf,     
# #             recs_only=recs_only,
# #             llm=llm
# #         )

# #     elif model_type == 'ranknet':
# #         model, results = run_single_ranknet_experiment(
# #             doctor_key,
# #             chosen_train_pairs,
# #             all_train_pairs,
# #             all_test_pairs,
# #             df_train_scaled,
# #             df_test_scaled,
# #             train_rankings,
# #             test_rankings,
# #             seed=42,
# #             hidden_dim=128,
# #             lr=1e-3,
# #             num_epochs=500,
# #             print_flag=False,
# #             recs_only=recs_only,
# #             llm=llm, 
# #             track_influence=False,   # to track influential pairs
# #             weighted_metics=weighted_metics
# #         )    
    
# #     return doctor_key, results, model

# # # 10 doctors, one alpha=(one method, one num pairs), one model type


# # def single_alpha_experiment(
# #         idx,
# #         alpha_set,
# #         train_graph,
# #         test_graph,
# #         exclude_pairs,
# #         train_aut_pairs,
# #         train_rec_prior,
# #         df_train,
# #         df_train_scaled,
# #         df_test,
# #         df_test_scaled,
# #         patient_df,
# #         all_train_pairs,
# #         all_test_pairs,
# #         train_rankings,
# #         test_rankings,
# #         model_type,
# #         docs_range=None,
# #         xgb_params=None,
# #         num_boost_round=None,
# #         llm=False,
# # ):
# #     import math
# #     from utils import get_alpha_name_from_row  # לוודא קיים

# #     alpha_name = get_alpha_name_from_row(alpha_set)
# #     print(f"\nAlphaSet {idx}: {alpha_set}")
# #     random.seed(42)
# #     if docs_range is None:
# #         docs_range = range(1, 11)

# #     # ── build candidate pair lists for this alpha-set ────────────────────────
# #     raw_train_pairs = identify_candidate_pairs_for_stable_set(
# #         graph               = train_graph,
# #         patient_df_scaled   = df_train_scaled,
# #         patient_df          = df_train,
# #         common_pairs_dict   = {},
# #         num_pairs           = alpha_set[0],
# #         alpha_kmeans_d      = alpha_set[1],
# #         alpha_outlier       = alpha_set[2],
# #         alpha_conflict      = alpha_set[3],
# #         alpha_pat           = alpha_set[4],
# #         alpha_child         = alpha_set[5],
# #         alpha_chain         = alpha_set[6],
# #         alpha_random        = alpha_set[7],
# #         alpha_sys           = alpha_set[8],
# #         alpha_inf           = alpha_set[9],
# #         alpha_llm           = alpha_set[10],
# #         exclude_pairs       = exclude_pairs,
# #         random_state        = 42
# #     )

# #     # ---------- לחשב ציוני מתמטיקה פעם אחת אם model_type הוא math_* ----------
# #     is_math = isinstance(model_type, str) and model_type.startswith("math_")
# #     math_scores_by_method = None
# #     if is_math:
# #         train_graph_full_for_math = _build_full_train_graph_for_math(
# #             docs_range=docs_range,
# #             all_train_pairs=all_train_pairs,
# #             df_train=df_train,
# #             age_col='age',
# #             risk_col='risk'
# #         )
# #         math_scores_by_method = compute_math_scores(train_graph_full_for_math, df_train)

# #     # ---------- הגדרת מפתחות מטריקות לפי סוג המודל ----------
# #     if is_math:
# #         metric_keys = [
# #             'overall_accuracy', 'overall_auc',
# #             'overall_ndcg', 'overall_tau', 'overall_map', 'overall_rbo'
# #         ]
# #     else:
# #         metric_keys = [
# #             'train_accuracy','test_accuracy',
# #             'train_auc','test_auc',
# #             'train_ndcg','test_ndcg',
# #             'train_tau','test_tau',
# #             'train_map','test_map',
# #             'train_rbo','test_rbo'
# #         ]

# #     # ---------- per_doctor_local & totals ----------
# #     per_doctor_local = {
# #         f'doctor{i}': {'alpha_set': [], **{k: [] for k in metric_keys}}
# #         for i in docs_range
# #     }
# #     totals = {k: 0.0 for k in metric_keys}
# #     allowed_metric_keys = set(metric_keys)

# #     # ---------- הרצה לרופא יחיד ----------
# #     def run_for_doctor(i):
# #         doctor_key = f'doctor{i}'
# #         doctor_key, results, _ = single_doctor_experiment(
# #             doctor_key,
# #             raw_train_pairs,
# #             train_aut_pairs,
# #             train_rec_prior,
# #             df_train_scaled,
# #             df_test_scaled,
# #             all_train_pairs,
# #             all_test_pairs,
# #             train_rankings,
# #             test_rankings,
# #             alpha_name,
# #             model_type,
# #             xgb_params,
# #             num_boost_round,
# #             conf=True,
# #             recs_only=True,
# #             llm=llm,
# #             math_scores_by_method=math_scores_by_method
# #         )
# #         return doctor_key, results

# #     # ---------- מקביליות על כל הרופאים ואיסוף מטריקות ----------
# #     with futures.ThreadPoolExecutor(max_workers=10) as executor:
# #         futs = [executor.submit(run_for_doctor, i) for i in docs_range]
# #         for fut in futures.as_completed(futs):
# #             doctor_key, metrics = fut.result()
# #             per_doctor_local[doctor_key]['alpha_set'].append(alpha_name)

# #             # אסוף רק את המפתחות הרלוונטיים, וסנן None/NaN/Inf
# #             for k, v in metrics.items():
# #                 if k in allowed_metric_keys:
# #                     if v is None:
# #                         continue
# #                     try:
# #                         vf = float(v)
# #                     except (TypeError, ValueError):
# #                         continue
# #                     if math.isnan(vf) or math.isinf(vf):
# #                         continue

# #                     per_doctor_local[doctor_key][k].append(vf)
# #                     totals[k] += vf

# #     # ── compute alpha-level averages ─────────────────────
# #     alpha_result = {'alpha_set': alpha_name, 'model_type': model_type}
# #     num_docs = len(list(docs_range)) if len(list(docs_range)) > 0 else 1
# #     for k in totals:
# #         alpha_result[k] = totals[k] / num_docs

# #     return idx, alpha_result, per_doctor_local




# # # def single_alpha_experiment(
# # #         idx,
# # #         alpha_set,
# # #         train_graph,
# # #         test_graph,
# # #         exclude_pairs,
# # #         train_aut_pairs,
# # #         train_rec_prior,
# # #         df_train,
# # #         df_train_scaled,
# # #         df_test,
# # #         df_test_scaled,
# # #         patient_df,
# # #         all_train_pairs,
# # #         all_test_pairs,
# # #         train_rankings,
# # #         test_rankings,
# # #         model_type,
# # #         docs_range=None,
# # #         xgb_params=None,
# # #         num_boost_round=None,
# # #         llm=False,

# # # ):
# # #     alpha_name = get_alpha_name_from_row(alpha_set)
# # #     print(f"\nAlphaSet {idx}: {alpha_set}")
# # #     random.seed(42)
# # #     if docs_range is None:
# # #         docs_range=range(1, 11)

# # #     # ── build candidate pair lists for this alpha-set ────────────────────────
# # #     raw_train_pairs = identify_candidate_pairs_for_stable_set(
# # #         graph               = train_graph,
# # #         patient_df_scaled   = df_train_scaled,
# # #         patient_df          = df_train,
# # #         common_pairs_dict   = {},
# # #         num_pairs           = alpha_set[0],
# # #         alpha_kmeans_d      = alpha_set[1],
# # #         alpha_outlier       = alpha_set[2],
# # #         alpha_conflict      = alpha_set[3],
# # #         alpha_pat           = alpha_set[4],
# # #         alpha_child         = alpha_set[5],
# # #         alpha_chain         = alpha_set[6],
# # #         alpha_random        = alpha_set[7],
# # #         alpha_sys           = alpha_set[8],
# # #         alpha_inf           = alpha_set[9],
# # #         alpha_llm           = alpha_set[10],
# # #         exclude_pairs       = exclude_pairs,
# # #         random_state        = 42
# # #     )
# # #     # ---------- NEW: לחשב ציוני מתמטיקה פעם אחת אם model_type הוא math_* ----------
# # #     math_scores_by_method = None  # <<< NEW
# # #     if isinstance(model_type, str) and model_type.startswith("math_"):  # <<< NEW
# # #         train_graph_full_for_math = _build_full_train_graph_for_math(    # <<< NEW
# # #             docs_range=docs_range,
# # #             all_train_pairs=all_train_pairs,
# # #             df_train=df_train,
# # #             age_col='age',
# # #             risk_col='risk'
# # #         )
# # #         # חישוב 6 השיטות על כל הזוגות (ללא קיטום)                        # <<< NEW
# # #         math_scores_by_method = compute_math_scores(                      # <<< NEW
# # #             train_graph_full_for_math, df_train
# # #         )

# # #     per_doctor_local = {f'doctor{i}': {
# # #         'alpha_set'                    : [],
# # #         'train_accuracy'               : [],
# # #         'test_accuracy'                : [],
# # #         'train_auc'                    : [],
# # #         'test_auc'                     : [], 
# # #         'train_ndcg'                   : [],
# # #         'test_ndcg'                    : [],
# # #         'train_tau'                    : [],
# # #         'test_tau'                     : [],
# # #         'train_map'                 : [],
# # #         'test_map'                  : [],
# # #         'train_rbo'                    : [],
# # #         'test_rbo'                     : []
# # #     } for i in docs_range}

# # #     totals = {
# # #         'train_accuracy': 0.0, 'test_accuracy': 0.0,
# # #         'train_auc':0.0, 'test_auc':0.0,
# # #         'train_ndcg': 0.0, 'test_ndcg': 0.0,
# # #         'train_tau': 0.0, 'test_tau': 0.0,
# # #         'train_map': 0.0, 'test_map': 0.0,
# # #         'train_rbo': 0.0, 'test_rbo': 0.0
# # #     }

# # #     def run_for_doctor(i):
# # #         doctor_key = f'doctor{i}'
# # #         doctor_key, results, _ = single_doctor_experiment(
# # #             doctor_key,
# # #             raw_train_pairs,
# # #             train_aut_pairs,
# # #             train_rec_prior,
# # #             df_train_scaled,
# # #             df_test_scaled,
# # #             all_train_pairs,
# # #             all_test_pairs,
# # #             train_rankings,
# # #             test_rankings,
# # #             alpha_name,
# # #             model_type,
# # #             xgb_params,
# # #             num_boost_round,
# # #             conf=True,
# # #             recs_only=True,
# # #             llm=llm,
# # #             math_scores_by_method=math_scores_by_method
# # #         )
# # #         return doctor_key, results

# # #     # define once, right before the loop over futures
# # #     allowed_metric_keys = set(totals.keys())  # exactly the metrics you average
    
# # #     with futures.ThreadPoolExecutor(max_workers=10) as executor:
# # #         futs = [executor.submit(run_for_doctor, i) for i in docs_range]
# # #         for fut in futures.as_completed(futs):
# # #             doctor_key, metrics = fut.result()
# # #             # keep a record of the alpha-set used (always available)
# # #             per_doctor_local[doctor_key]['alpha_set'].append(alpha_name)
# # #             # merge only known numeric metrics into per_doctor_local and totals
# # #             for k, v in metrics.items():
# # #                 if k in allowed_metric_keys:
# # #                     per_doctor_local[doctor_key][k].append(v)
# # #                     totals[k] += float(v)  # ensure numeric


# # #     # ── compute alpha-level averages ─────────────────────
# # #     alpha_result = {'alpha_set': alpha_name}
# # #     for k in totals:
# # #         alpha_result[k] = totals[k] / 10
    
# # #     alpha_result['model_type'] = model_type
# # #     return idx, alpha_result, per_doctor_local

