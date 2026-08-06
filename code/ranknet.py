# ====================================================================
# PATCH FOR ranknet.py (old code, 21 recs version)
# ====================================================================
# REPLACE the entire content of ranknet.py with this file.
# 
# Key changes from original ranknet.py:
# 1. FEATURE_COLS imported from utils (not redefined here) -> 29 features
#    (23 base + 6 path).
# 2. NEW ListwiseBatchDataset (replaces PairwiseDatasetWithConfidence)
#    -> packs unique patients per batch, much faster.
# 3. SOFT confidence-aware labels: rating 1..6 -> target in [0.55, 1.00]
# 4. NO confidence weighting in the loss (soft labels already encode it).
# 5. Small L2 penalty on raw scores to stabilize utility magnitude.
# 6. RankNet architecture unchanged (same as the new code).
# ====================================================================

from collections import defaultdict
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

# Import FEATURE_COLS from utils (29 cols: 23 base + 6 path)
from utils import (
    filter_patients_with_recs,
    align_feature_columns,
    FEATURE_COLS,
)
from metrics import (
    pairwise_accuracy,
    calc_auc,
    evaluate_ranking_metrics_for_doctor,
    pairwise_accuracy_weighted,
    calc_auc_weighted,
)
from setup import batch_size


# =====================================================================
# Listwise Dataset (replaces PairwiseDatasetWithConfidence)
# =====================================================================
class ListwiseBatchDataset(Dataset):
    """
    Groups ranked_pairs into chunks of `pairs_per_batch`, builds per-chunk
    unique-patient feature matrix, and produces pair-index tensors that
    reference rows in that matrix. Much faster than per-pair lookups.
    """
    def __init__(self, ranked_pairs, patient_df, pairs_per_batch=32):
        self.ranked_pairs = ranked_pairs
        self.patient_df = align_feature_columns(patient_df.copy())
        self.pairs_per_batch = pairs_per_batch

        # Feature lookup by patient_num
        self._feat_lookup = {}
        for _, row in self.patient_df.iterrows():
            pid = int(row['patient_num'])
            self._feat_lookup[pid] = row[FEATURE_COLS].values.astype(np.float32)

        self._batches = self._build_batches()

    @staticmethod
    def confidence_to_soft_label(confidence):
        """
        Maps confidence (1..6) to soft label in [0.55, 1.00].
        Higher confidence => stronger preference target.
        """
        conf = int(round(float(confidence)))
        conf = max(1, min(6, conf))
        return 0.55 + (conf - 1) * (1.00 - 0.55) / 5.0

    def _build_batches(self):
        batches = []
        for i in range(0, len(self.ranked_pairs), self.pairs_per_batch):
            batches.append(self.ranked_pairs[i:i + self.pairs_per_batch])
        return batches

    def __len__(self):
        return len(self._batches)

    def __getitem__(self, batch_idx):
        chunk = self._batches[batch_idx]

        unique_patients = []
        patient_to_local_idx = {}
        for (a, b), _ in chunk:
            a, b = int(a), int(b)
            if a not in patient_to_local_idx:
                patient_to_local_idx[a] = len(unique_patients)
                unique_patients.append(a)
            if b not in patient_to_local_idx:
                patient_to_local_idx[b] = len(unique_patients)
                unique_patients.append(b)

        feats = np.stack([self._feat_lookup[pid] for pid in unique_patients])

        pair_indices = []
        confidences = []
        labels = []
        for (a, b), conf in chunk:
            a, b = int(a), int(b)
            pair_indices.append([patient_to_local_idx[a], patient_to_local_idx[b]])
            confidences.append(float(conf))
            labels.append(self.confidence_to_soft_label(conf))

        return {
            'patient_features': torch.tensor(feats, dtype=torch.float32),
            'pair_indices':     torch.tensor(pair_indices, dtype=torch.long),
            'confidences':      torch.tensor(confidences, dtype=torch.float32),
            'labels':           torch.tensor(labels, dtype=torch.float32),
        }


# =====================================================================
# Model (unchanged architecture from original; matches new code)
# =====================================================================
class RankNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.10,
                 use_layernorm=True, residual=True):
        super().__init__()
        self.input_dim = input_dim
        self.act = nn.GELU()
        self.dp = nn.Dropout(dropout)

        h1 = hidden_dim
        h2 = hidden_dim
        h3 = max(32, hidden_dim // 2)
        h4 = max(16, hidden_dim // 4)

        self.fc_in  = nn.Linear(input_dim, h1)
        self.fc_h1  = nn.Linear(h1, h2)
        self.fc_h2  = nn.Linear(h2, h3)
        self.fc_h3  = nn.Linear(h3, h4)
        self.fc_out = nn.Linear(h4, 1)

        self.use_layernorm = use_layernorm
        self.residual = residual

        self.n1 = nn.LayerNorm(h1) if use_layernorm else nn.Identity()
        self.n2 = nn.LayerNorm(h2) if use_layernorm else nn.Identity()
        self.n3 = nn.LayerNorm(h3) if use_layernorm else nn.Identity()
        self.n4 = nn.LayerNorm(h4) if use_layernorm else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pad_trunc(self, x):
        cur = x.size(1)
        if cur < self.input_dim:
            pad = torch.zeros(x.size(0), self.input_dim - cur,
                              device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif cur > self.input_dim:
            x = x[:, :self.input_dim]
        return x

    def forward(self, x):
        x = self._pad_trunc(x)
        x = self.fc_in(x);  x = self.act(x); x = self.dp(x); x = self.n1(x)
        y = self.fc_h1(x);  y = self.act(y); y = self.dp(y); y = self.n2(y)
        if self.residual and y.shape == x.shape:
            x = x + y
        else:
            x = y
        x = self.fc_h2(x);  x = self.act(x); x = self.dp(x); x = self.n3(x)
        x = self.fc_h3(x);  x = self.act(x); x = self.dp(x); x = self.n4(x)
        return self.fc_out(x)


def build_ranknet_dataset(pair_list, patient_df, pairs_per_batch=32):
    return ListwiseBatchDataset(pair_list, patient_df,
                                pairs_per_batch=pairs_per_batch)


# =====================================================================
# Training (soft-label BCE, no confidence weighting, +L2 on scores)
# =====================================================================
def train_model(
    model,
    dataset,
    criterion,
    optimizer,
    num_epochs,
    patience,
    min_delta,
    print_flag=False,
):
    n_batches = len(dataset)
    batch_indices = list(range(n_batches))
    train_idx, val_idx = train_test_split(batch_indices, test_size=0.2, random_state=42)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val, no_imp, stop = float('inf'), 0, False

    for epoch in range(num_epochs):
        if stop:
            if print_flag:
                print(f"Early stopping at epoch {epoch+1}")
            break

        model.train()
        train_loss_sum, n_train = 0.0, 0
        np.random.shuffle(train_idx)

        for b_idx in train_idx:
            batch = dataset[b_idx]
            patient_feats = batch['patient_features']
            pair_indices  = batch['pair_indices']
            labels        = batch['labels']

            scores = model(patient_feats).squeeze(-1)
            score_w = scores[pair_indices[:, 0]]
            score_l = scores[pair_indices[:, 1]]
            diff = score_w - score_l

            loss_pair = criterion(diff, labels).mean()
            score_reg = (scores ** 2).mean() * 0.001
            loss = loss_pair + score_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            n_train += labels.size(0)

        avg_train = train_loss_sum / max(1, n_train)

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for b_idx in val_idx:
                batch = dataset[b_idx]
                patient_feats = batch['patient_features']
                pair_indices  = batch['pair_indices']
                labels        = batch['labels']
                scores = model(patient_feats).squeeze(-1)
                score_w = scores[pair_indices[:, 0]]
                score_l = scores[pair_indices[:, 1]]
                diff = score_w - score_l
                loss = criterion(diff, labels).mean()
                val_loss_sum += loss.item() * labels.size(0)
                n_val += labels.size(0)

        avg_val = val_loss_sum / max(1, n_val)
        scheduler.step(avg_val)

        if print_flag:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train {avg_train:.4f} | Val {avg_val:.4f} | LR {lr:.6f}")

        if best_val - avg_val > min_delta:
            best_val, no_imp = avg_val, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                if print_flag:
                    print("Early stopping triggered.")
                stop = True

    return []


def score_patients_ranknet(model, patient_df, recs_only=False):
    if recs_only:
        patient_df = filter_patients_with_recs(patient_df)
    patient_df = align_feature_columns(patient_df.copy())
    model.eval()
    patient_ids = patient_df['patient_num'].values
    X = torch.tensor(patient_df[FEATURE_COLS].values, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X).squeeze().numpy()
    return list(zip(patient_ids, scores))


def train_ranknet(
    model,
    dataset,
    batch_size,
    lr,
    num_epochs,
    print_flag=False,
    track_influence=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    train_model(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        patience=5,
        min_delta=0.00001,
        print_flag=print_flag,
    )

    return model, []


def run_single_ranknet_experiment(
    doctor_key,
    chosen_train_pairs,
    all_train_pairs,
    all_test_pairs,
    df_train_scaled,
    df_test_scaled,
    train_rankings,
    test_rankings,
    seed,
    hidden_dim=128,
    lr=1e-3,
    num_epochs=500,
    print_flag=False,
    recs_only=False,
    llm=False,
    track_influence=False,
    weighted_metics=False,
):
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    dataset = build_ranknet_dataset(chosen_train_pairs, df_train_scaled,
                                    pairs_per_batch=32)
    input_dim = len(FEATURE_COLS)
    model = RankNet(input_dim=input_dim, hidden_dim=hidden_dim)

    model, _ = train_ranknet(
        model, dataset,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        print_flag=print_flag,
        track_influence=track_influence,
    )

    train_scores = score_patients_ranknet(model, df_train_scaled, recs_only=recs_only)
    test_scores  = score_patients_ranknet(model, df_test_scaled,  recs_only=recs_only)

    train_score_dict = {pid: score for pid, score in train_scores}
    test_score_dict  = {pid: score for pid, score in test_scores}

    train_acc = pairwise_accuracy(train_score_dict, all_train_pairs[doctor_key])
    test_acc  = pairwise_accuracy(test_score_dict,  all_test_pairs[doctor_key])
    train_auc = calc_auc(train_score_dict, all_train_pairs[doctor_key])
    test_auc  = calc_auc(test_score_dict,  all_test_pairs[doctor_key])

    if not weighted_metics:
        if llm:
            return model, {
                'doctor':         doctor_key,
                'train_accuracy': train_acc,
                'test_accuracy':  test_acc,
                'train_auc':      train_auc,
                'test_auc':       test_auc,
            }

        train_rank_metrics = evaluate_ranking_metrics_for_doctor(
            train_scores, train_rankings[doctor_key]
        )
        test_rank_metrics = evaluate_ranking_metrics_for_doctor(
            test_scores, test_rankings[doctor_key]
        )

        return model, {
            'doctor':         doctor_key,
            'train_accuracy': train_acc,
            'test_accuracy':  test_acc,
            'train_auc':      train_auc,
            'test_auc':       test_auc,
            **{f'train_{k}': v for k, v in train_rank_metrics.items()},
            **{f'test_{k}':  v for k, v in test_rank_metrics.items()},
        }

    weighted_train_acc = pairwise_accuracy_weighted(
        train_score_dict, all_train_pairs[doctor_key],
        scheme="pow", alpha=2.0, min_w=0.0,
    )
    weighted_test_acc = pairwise_accuracy_weighted(
        test_score_dict, all_test_pairs[doctor_key],
        scheme="pow", alpha=2.0, min_w=0.0,
    )
    weighted_train_auc = calc_auc_weighted(
        train_score_dict, all_train_pairs[doctor_key],
        scheme="pow", alpha=2.0, min_w=0.0,
    )
    weighted_test_auc = calc_auc_weighted(
        test_score_dict, all_test_pairs[doctor_key],
        scheme="pow", alpha=2.0, min_w=0.0,
    )

    return model, {
        'doctor':                  doctor_key,
        'train_accuracy':          round(train_acc, 5),
        'test_accuracy':           round(test_acc, 5),
        'train_auc':               round(train_auc, 5),
        'test_auc':                round(test_auc, 5),
        'weighted_train_accuracy': round(weighted_train_acc, 5),
        'weighted_test_accuracy':  round(weighted_test_acc, 5),
        'weighted_train_auc':      round(weighted_train_auc, 5),
        'weighted_test_auc':       round(weighted_test_auc, 5),
    }
