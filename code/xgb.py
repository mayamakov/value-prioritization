# imports
import xgboost as xgb
import numpy as np

from utils import filter_patients_with_recs
from metrics import pairwise_accuracy, calc_auc, evaluate_ranking_metrics_for_doctor


# def build_xgb_pairwise_dataset(pair_list, patient_df, conf=False):  
#     features = []
#     labels = []
#     weights = []

#     feature_dict = {
#         int(row['patient_num']): row.drop('patient_num').values
#         for _, row in patient_df.iterrows()
#     }

#     for ((i, j), weight) in pair_list:
#         f_i = feature_dict.get(int(i))
#         f_j = feature_dict.get(int(j))
#         if f_i is None or f_j is None:
#             continue
#         features.extend([f_i, f_j])
#         labels.extend([1, 0])  # Winner first
#         if conf:
#             weight = int(weight)
#             # weights.extend([weight, weight])
#             weights.append(weight)

#     X = np.array(features)
#     y = np.array(labels)
#     group_sizes = [2] * (len(y) // 2)
#     if conf:
#         weights = np.array(weights)
#         return  X, y, group_sizes, weights
#     return X, y, group_sizes

def build_xgb_pairwise_dataset(pair_list, patient_df, conf=False):
    features, labels, group_sizes, group_weights = [], [], [], []
    feat = {int(r['patient_num']): r.drop('patient_num').values for _, r in patient_df.iterrows()}

    for ((i, j), w) in pair_list:
        fi, fj = feat.get(int(i)), feat.get(int(j))
        if fi is None or fj is None:
            continue
        features.extend([fi, fj])
        labels.extend([1, 0])
        group_sizes.append(2)           # כל זוג = קבוצה
        if conf:
            group_weights.append(float(w))

    X = np.asarray(features)
    y = np.asarray(labels)
    if conf:
        gw = np.asarray(group_weights, dtype=np.float32)  # <<< חשוב: float32
        return X, y, group_sizes, gw
    return X, y, group_sizes

# def train_xgb_ranker_group2(X, y, group, params, num_boost_round, w=None):
#     if w is not None:
#         dtrain = xgb.DMatrix(X, label=y, weight=w)
#     else:
#         dtrain = xgb.DMatrix(X, label=y)
#     dtrain.set_group(group)
#     model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
#     return model

def train_xgb_ranker_group2(X, y, group, params, num_boost_round, w=None):
    dtrain = xgb.DMatrix(X, label=y)
    # קודם קובעים קבוצות
    dtrain.set_group(group)
    # אם יש משקל פר-קבוצה: האורך חייב להיות == len(group)
    if w is not None:
        dtrain.set_weight(np.asarray(w, dtype=np.float32))
    # ודאי שה־objective הוא דירוג
    params = dict(params)
    params.setdefault("objective", "rank:ndcg")  # או 'rank:pairwise'
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    return model

# def score_patients_xgb(model, patient_df, recs_only=False):
#     if recs_only:
#         patient_df = filter_patients_with_recs(patient_df)
#     patient_ids = patient_df['patient_num'].values
#     X = patient_df.drop(columns=['patient_num']).values
#     dtest = xgb.DMatrix(X)
#     scores = model.predict(dtest)
#     return list(zip(patient_ids, scores))
# def score_patients_xgb(model, patient_df, recs_only=False):
#     if recs_only:
#         patient_df = filter_patients_with_recs(patient_df)
#     patient_ids = patient_df['patient_num'].values
#     X = patient_df.drop(columns=['patient_num']).values
#     dtest = xgb.DMatrix(X)
#     # אם זה Ranker – יש להגדיר group גם ב-predict
#     try:
#         if hasattr(model, "attributes") and "objective" in model.attributes():
#             objective = model.attributes()["objective"]
#         else:
#             objective = None
#         if objective and "rank" in objective:
#             group = np.array([len(X)])   # קבוצה אחת גדולה
#             dtest.set_group(group)
#     except Exception:
#         pass
#     scores = model.predict(dtest)
#     return list(zip(patient_ids, scores))
def score_patients_xgb(model, patient_df, recs_only=False):
    import xgboost as xgb
    if recs_only:
        patient_df = filter_patients_with_recs(patient_df)
    patient_ids = patient_df['patient_num'].values
    X = patient_df.drop(columns=['patient_num']).values
    dtest = xgb.DMatrix(X)
    scores = model.predict(dtest)
    return list(zip(patient_ids, scores))

def run_single_xgb_experiment(
    doctor_key,
    chosen_train_pairs,
    all_train_pairs,
    all_test_pairs,
    df_train_scaled,
    df_test_scaled,
    df_overall_scaled, 
    train_rankings,
    test_rankings,
    overall_rankings,
    xgb_params,  # parameters for XGBoost
    num_boost_round, # number of boosting rounds
    conf=True,
    recs_only=False,
    llm=False
):
    
    if conf:
        X, y, group, weights = build_xgb_pairwise_dataset(chosen_train_pairs, df_train_scaled, conf=True)
        model = train_xgb_ranker_group2(X, y, group, params=xgb_params, num_boost_round=num_boost_round, w=weights)
    else:
        X, y, group = build_xgb_pairwise_dataset(chosen_train_pairs, df_train_scaled)
        model = train_xgb_ranker_group2(X, y, group, params=xgb_params, num_boost_round=num_boost_round)


    train_scores = score_patients_xgb(model, df_train_scaled, recs_only=recs_only)
    test_scores  = score_patients_xgb(model, df_test_scaled,  recs_only=recs_only)

    train_score_dict = {pid: score for pid, score in train_scores}
    test_score_dict  = {pid: score for pid, score in test_scores}

    train_acc = pairwise_accuracy(train_score_dict, all_train_pairs[doctor_key])
    test_acc  = pairwise_accuracy(test_score_dict,  all_test_pairs[doctor_key])
    train_auc = calc_auc(train_score_dict, all_train_pairs[doctor_key])
    test_auc  = calc_auc(test_score_dict,  all_test_pairs[doctor_key])

    if llm:
        return model, {
            'doctor': doctor_key,
            'train_accuracy': train_acc,
            'test_accuracy':  test_acc,
            'train_auc':      train_auc,
            'test_auc':       test_auc
        }

    train_rank_metrics = evaluate_ranking_metrics_for_doctor(
        train_scores, train_rankings[doctor_key]
    )
    test_rank_metrics  = evaluate_ranking_metrics_for_doctor(
        test_scores,  test_rankings[doctor_key]
    )

    # === בלוק Overall (חדש, מבודד) ===
    overall_metrics = {}
    if (df_overall_scaled is not None) and (overall_rankings is not None):
        overall_scores = score_patients_xgb(model, df_overall_scaled, recs_only=recs_only)
        # אם ה-GT שלך ל-Overall לפי doctor_key (כמו train/test):
        gt_overall = overall_rankings[doctor_key]
        # הערה חשובה: אם יש בעיית dtype ב-PID ב-GT, אפשר להמיר רק כאן בלי לגעת ב-test/train:
        # overall_scores = [(int(pid), s) for pid, s in overall_scores]

        overall_metrics = evaluate_ranking_metrics_for_doctor(
            overall_scores, gt_overall
        )

    return model, {
        'doctor': doctor_key,
        'train_accuracy': train_acc,
        'test_accuracy':  test_acc,
        'train_auc':      train_auc,
        'test_auc':       test_auc,
        **{f'train_{k}': v for k, v in train_rank_metrics.items()},
        **{f'test_{k}':  v for k, v in test_rank_metrics.items()},
        **{f'overall_{k}': v for k, v in overall_metrics.items()}
    }




# XGB tuning 

# OPTUNA TRIALS: 
# # best custom metric in multi objective 
#   'test_accuracy': 0.9108601202404808,
#   'test_auc': 0.967830499071088,
#   'test_map5': 0.0,
#   'test_map10': 0.0,
#   'test_map20': 0.5464,
#   'test_kendall': 0.836,
#   'test_rbo': 0.4857,
#   'test_ndcg': 0.982,

#   'params_eta': 0.0853738129093829,
#   'params_gamma': 1.1669153292386532,
#   'params_lambda': 1.685441147898798,
#   'params_max_depth': 8,
#   'params_min_child_weight': 3,
#   'params_num_boost_round': 295,
#   'params_objective': 'rank:ndcg',

# # best accuracy in single objective 
# Best params: {'eta': 0.10171250777389301, 'max_depth': 14, 'lambda': 4.580294906446968, 'min_child_weight': 2, 'gamma': 0.7138016750924401, 'num_boost_round': 508}
# Best mean accuracy: 0.9033621208215175

# # best custom metric in single objective 
# Best params: {'eta': 0.47022729768745564, 'max_depth': 2, 'lambda': 2.264473433160305, 'min_child_weight': 18, 'gamma': 1.778333323505484, 'num_boost_round': 140}
# Best mean custom metric: 0.1752353824885025