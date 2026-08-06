# ====================================================================
# PATCH FOR data_loading.py (old code, 21 recs version)
# ====================================================================
# REPLACE the entire content of data_loading.py with this file.
# ====================================================================

from create_ranking_data import create_ranking_if_doesnt_exist
from initial_pairs import create_initial_pairs_if_doesnt_exist
from setup import rankings_file, patient_df_path, initial_pairs_path
from utils import (
    split_and_scale_data, load_patient_df, get_rec_cols,
    add_path_features,
)


def _add_paths_everywhere(patient_df, df_train, df_train_scaled,
                          df_test, df_test_scaled, patient_df_scaled):
    """Adds 6 path features to all DataFrames at once."""
    patient_df        = add_path_features(patient_df)
    df_train          = add_path_features(df_train)
    df_train_scaled   = add_path_features(df_train_scaled)
    df_test           = add_path_features(df_test)
    df_test_scaled    = add_path_features(df_test_scaled)
    patient_df_scaled = add_path_features(patient_df_scaled)
    return patient_df, df_train, df_train_scaled, df_test, df_test_scaled, patient_df_scaled


def init_data():
    # 1) Load & split
    patient_df = load_patient_df(patient_df_path)
    rec_cols   = get_rec_cols(patient_df)
    (all_train_patients, all_test_patients,
     df_train, df_train_scaled,
     df_test,  df_test_scaled,
     patient_df_scaled) = split_and_scale_data(patient_df)

    # 2) Add the 6 path features BEFORE creating rankings/pairs,
    #    so the graph/scoring functions also see them.
    (patient_df, df_train, df_train_scaled,
     df_test, df_test_scaled, patient_df_scaled) = _add_paths_everywhere(
        patient_df, df_train, df_train_scaled,
        df_test, df_test_scaled, patient_df_scaled,
    )

    # 3) Overall doctor rankings
    (overall_ranking, overall_pairs,
     all_train_pairs, all_test_pairs,
     train_rankings, test_rankings) = create_ranking_if_doesnt_exist(
        rankings_file,
        patient_df,
        df_train,
        df_test,
    )

    # 4) Initial automatic / rec-priority pairs
    (train_aut_pairs, train_aut_pairs_all,
     test_aut_pairs,
     train_rec_prior, train_rec_prior_all,
     test_rec_prior, exclude_pairs,
     train_graph, test_graph) = create_initial_pairs_if_doesnt_exist(
        initial_pairs_path,
        patient_df,
        all_train_patients,
        all_test_patients,
        rec_cols,
        df_train,
        df_test,
    )

    return (
        patient_df,
        df_train, df_train_scaled,
        df_test,  df_test_scaled,
        overall_ranking, overall_pairs,
        all_train_pairs, all_test_pairs,
        train_rankings, test_rankings,
        train_aut_pairs, train_rec_prior,
        exclude_pairs, train_graph, test_graph,
        patient_df_scaled,
    )


def init_data_with_automatic_test():
    # 1) Load & split
    patient_df = load_patient_df(patient_df_path)
    rec_cols   = get_rec_cols(patient_df)
    (all_train_patients, all_test_patients,
     df_train, df_train_scaled,
     df_test,  df_test_scaled,
     patient_df_scaled) = split_and_scale_data(patient_df)

    # 2) Add path features
    (patient_df, df_train, df_train_scaled,
     df_test, df_test_scaled, patient_df_scaled) = _add_paths_everywhere(
        patient_df, df_train, df_train_scaled,
        df_test, df_test_scaled, patient_df_scaled,
    )

    # 3) Overall doctor rankings
    (overall_ranking, overall_pairs,
     all_train_pairs, all_test_pairs,
     train_rankings, test_rankings) = create_ranking_if_doesnt_exist(
        rankings_file,
        patient_df,
        df_train,
        df_test,
    )

    # 4) Initial automatic / rec-prior pairs
    (train_aut_pairs, train_aut_pairs_all,
     test_aut_pairs,
     train_rec_prior, train_rec_prior_all,
     test_rec_prior, exclude_pairs,
     train_graph, test_graph) = create_initial_pairs_if_doesnt_exist(
        initial_pairs_path,
        patient_df,
        all_train_patients,
        all_test_patients,
        rec_cols,
        df_train,
        df_test,
    )

    return (
        patient_df,
        df_train, df_train_scaled,
        df_test,  df_test_scaled,
        overall_ranking, overall_pairs,
        all_train_pairs, all_test_pairs,
        train_rankings, test_rankings,
        train_aut_pairs, train_rec_prior,
        test_aut_pairs, test_rec_prior,
        exclude_pairs, train_graph, test_graph,
    )
