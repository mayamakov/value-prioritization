import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import filter_patients_with_recs
from metrics import (
    pairwise_accuracy,
    calc_auc,
    evaluate_ranking_metrics_for_doctor
)


def build_pairwise_dataset_logistic_regression(pair_list, patient_df, conf=False):
    """
    Converts (i > j) pairs into feature difference vectors and labels.
    Adds inverse pairs for balance.
    """
    X = []
    y = []
    weights = []
    # Create a fast lookup for patient features
    feature_dict = {
    int(row['patient_num']): row.drop('patient_num').values
    for _, row in patient_df.iterrows()
    }
    for ((i, j), weight) in pair_list:
        i, j = int(i), int(j)
        weight = int(weight)
        f_i = feature_dict.get(i)
        f_j = feature_dict.get(j)
        if f_i is None or f_j is None:
            continue

        diff = f_i - f_j
        X.append(diff)
        y.append(1)  # i preferred

        X.append(-diff)
        y.append(0)  # j preferred
        if conf:
            weights.append(weight) 
            weights.append(weight) 
    if conf:
        return np.array(X), np.array(y), np.array(weights)
    return np.array(X), np.array(y)

def train_logistic_ranker(X, y, w=None):
    model = LogisticRegression(max_iter=1000, n_jobs=1)
    if w is not None:
        model.fit(X, y, sample_weight=w)
    else:
        model.fit(X, y)
    return model

def score_patients_logistic(model, patient_df, recs_only=False):
    if recs_only:
        patient_df = filter_patients_with_recs(patient_df)
    patient_ids = patient_df['patient_num'].values
    X = patient_df.drop(columns=['patient_num']).values
    scores = model.decision_function(X)
    return list(zip(patient_ids, scores))

def run_single_logistic_experiment(
    doctor_key,
    chosen_train_pairs,
    all_train_pairs,
    all_test_pairs,
    df_train_scaled,
    df_test_scaled,
    train_rankings,
    test_rankings,
    conf=False,     
    recs_only=False,
    llm=False
):
    if conf:
        X, y, w = build_pairwise_dataset_logistic_regression(chosen_train_pairs, df_train_scaled, conf=True)
        model = train_logistic_ranker(X, y, w)
    else:
        X, y = build_pairwise_dataset_logistic_regression(chosen_train_pairs, df_train_scaled)
        model = train_logistic_ranker(X, y)
    
    train_scores = score_patients_logistic(model, df_train_scaled, recs_only=recs_only)
    test_scores = score_patients_logistic(model, df_test_scaled, recs_only=recs_only)
    # for faster lookup
    train_score_dict = {pid: score for pid, score in train_scores}
    test_score_dict = {pid: score for pid, score in test_scores}
  
    train_acc = pairwise_accuracy(train_score_dict, all_train_pairs[doctor_key])
    test_acc = pairwise_accuracy(test_score_dict, all_test_pairs[doctor_key])
    train_auc = calc_auc(train_score_dict, all_train_pairs[doctor_key])
    test_auc = calc_auc(test_score_dict, all_test_pairs[doctor_key])

    if llm:
        return model, {
        'doctor': doctor_key,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
    }
    
    train_rank_metrics = evaluate_ranking_metrics_for_doctor(train_scores, train_rankings[doctor_key])
    test_rank_metrics = evaluate_ranking_metrics_for_doctor(test_scores, test_rankings[doctor_key])

    return model, {
        'doctor': doctor_key,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        **{f'train_{k}': v for k, v in train_rank_metrics.items()},
        **{f'test_{k}': v for k, v in test_rank_metrics.items()}
    }