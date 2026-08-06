# imports
import math
from sklearn.metrics import roc_auc_score
from scipy.stats import kendalltau
import numpy as np  
from statistics import mean

# NDCG
def compare_model_and_doctor_ndcg(model_rankings, doctor_rankings, k=None):
    """
    מחשבת את NDCG האמיתי לפי הדוקטור, בלי תלות בסדר של המודל
    """

    # שלב 1: בנה מילון רלוונטיות קבוע
    sorted_list = sorted(doctor_rankings, key=lambda x: x[1], reverse=True)
    max_rank = len(sorted_list)
    relevance_dict = {patient: max_rank - idx for idx, (patient, _) in enumerate(sorted_list)}

    if not relevance_dict:
        return 0.0 #None

    # שלב 2: סדר המטופלים לפי תחזיות המודל
    model_order = [patient for patient, score in sorted(model_rankings, key=lambda x: x[1], reverse=True)]

    # שלב 3: שלוף רלוונטיות לפי מילון קבוע
    relevances = [relevance_dict.get(patient, 0) for patient in model_order]

    # חישוב DCG
    dcg_value = compute_dcg(relevances, k)

    # חישוב IDCG
    ideal_relevances = sorted(relevance_dict.values(), reverse=True)
    idcg_value = compute_dcg(ideal_relevances, k)

    if idcg_value == 0:
        return 0.0

    return dcg_value / idcg_value

def compute_dcg(relevances, k=None):
    """
    relevances: list של ערכי רלוונטיות (int>=0 או float)
    k: cutoff למקום ה‑k (אם None – בלי חיתוך)
    """
    if k is not None:
        relevances = relevances[:k]
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)
    return dcg

# map
def average_precision_at_k(predicted_ranking, gold_ranking, k):
    """
    מחשבת את ה-Average Precision (AP) עד קאט-אוף k.

    Args:
        predicted_ranking (list): רשימה ממוינת (מהגבוה לנמוך) של מזהי מטופלים כפי שמפיק המודל.
        gold_ranking (list): רשימה ממוינת (מהגבוה לנמוך) של מזהי מטופלים לפי הדירוג האידיאלי (Gold Standard).
        k (int): קאט-אוף – עד איזה מיקום לחשב את ה-AP.

    Returns:
        float: ערך ה-Average Precision@k.
    """
    # נגדיר את קבוצת הרלוונטיים כ- top k מה-gold ranking
    relevant_set = set(gold_ranking[:k])
    if not relevant_set:
        return 0.0

    sum_precision = 0.0
    relevant_found = 0.0

    # נעבור על כל הפריטים ב-predicted ranking (עד קאט-אוף k)
    for i, patient in enumerate(predicted_ranking[:k]):
        if patient in relevant_set:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            sum_precision += precision_at_i

    # AP הוא ממוצע של Precision@i עבור כל הרלוונטיים (מספר הרלוונטיים הוא k במקרה זה)
    return sum_precision / len(relevant_set)

def compute_map_at_cutoffs(predicted_ranking, gold_ranking, cutoff_values):
    """
    מחשבת את ה-MAP עבור רשימת קאט-אוף שונים.

    Args:
        predicted_ranking (list): רשימה ממוינת (מהגבוה לנמוך) של מזהי מטופלים כפי שמפיק המודל.
        gold_ranking (list): רשימה ממוינת (מהגבוה לנמוך) של מזהי מטופלים לפי Gold Standard.
        cutoff_values (list): רשימה של ערכי קאט-אוף (לדוגמה, [20, 40, 60]).

    Returns:
        dict: מילון שבו המפתחות הם ערכי הקאט-אוף והערך הוא AP@k עבור כל אחד מהם.
    """
    map_dict = {}
    for k in cutoff_values:
        ap = average_precision_at_k(predicted_ranking, gold_ranking, k)
        map_dict[k] = ap
    return map_dict

# kendells tau
def compute_kendall_tau(ranking1, ranking2):
    """
    מחשבת את מקדם קנדל טאו (Kendall's Tau) בין שני סדרי דירוג.

    Args:
        ranking1 (list): רשימה ממוינת (מהגבוה לנמוך) של פריטים, לדוגמה, מזהי מטופלים.
        ranking2 (list): רשימה ממוינת (מהגבוה לנמוך) של אותם פריטים, לפי דירוג אחר (Gold Standard).

    Returns:
        tuple: (tau, p_value) – מקדם קנדל וטבלת p.
    """
    # בניית מילונים: פריט -> מיקום (0-indexed)
    rank_dict1 = {item: i for i, item in enumerate(ranking1)}
    rank_dict2 = {item: i for i, item in enumerate(ranking2)}

    # מציאת הפריטים המשותפים
    common_items = list(set(ranking1) & set(ranking2))
    if not common_items:
        return 0.0,0.0 #None, None

    # בניית רשימות המיקומים עבור הפריטים המשותפים
    ranks1 = [rank_dict1[item] for item in common_items]
    ranks2 = [rank_dict2[item] for item in common_items]

    tau, p_value = kendalltau(ranks1, ranks2)
    return tau, p_value

# rbo
def compute_rbo(list1, list2, p=0.98):
    """
    מחשבת את ה-Rank-Biased Overlap (RBO) בין שתי רשימות דירוג.

    נוסחת ה-RBO (גרסה פשוטה עבור רשימות סופיות):
      RBO = (1-p) * sum_{d=1}^{k} [ (Overlap_d / d) * p^(d-1) ]
    כאשר:
      - Overlap_d הוא מספר הפריטים המשותפים ב-top-d של כל רשימה.
      - k = max(len(list1), len(list2))
      - p הוא פרמטר persistence (0 < p < 1), כאשר ערך גבוה יותר מפחית את השפעת החלק העליון.

    Args:
        list1 (list): רשימה ממוינת (מהגבוה לנמוך) של פריטים.
        list2 (list): רשימה ממוינת (מהגבוה לנמוך) של אותם הפריטים.
        p (float, optional): פרמטר persistence (ברירת מחדל 0.9).

    Returns:
        float: ערך ה-RBO (בין 0 ל-1).
    """
    # נקבע k = max(len(list1), len(list2))
    k = max(len(list1), len(list2))
    rbo_sum = 0.0
    for d in range(1, k+1):
        # נלקח את ה-top-d בכל רשימה (אם הרשימה קצרה, משתמשים בכל הרשימה)
        set1 = set(list1[:d]) if d <= len(list1) else set(list1)
        set2 = set(list2[:d]) if d <= len(list2) else set(list2)
        overlap = len(set1.intersection(set2))
        rbo_sum += (overlap / d) * (p ** (d - 1))
    rbo = (1 - p) * rbo_sum
    return rbo

def pairwise_accuracy(score_dict, pair_list):
    correct = 0
    total = 0

    for ((i, j), _) in pair_list:
        if i not in score_dict or j not in score_dict:
            continue
        if score_dict[i] > score_dict[j]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0

def calc_auc(score_dict, pair_list):
    y_true = []
    y_scores = []

    for ((i, j), _) in pair_list:
        if i not in score_dict or j not in score_dict:
            continue

        s_i = score_dict[i]
        s_j = score_dict[j]

        y_scores.append(s_i - s_j)
        y_true.append(1)

        y_scores.append(s_j - s_i)
        y_true.append(0)

    if len(set(y_true)) < 2:
        return None

    return roc_auc_score(y_true, y_scores)

def evaluate_ranking_metrics_for_doctor(
    scores,
    gold_ranking,
    cutoff_values=[10, 20, 50, 75, 100, 200],
):
    """
    Computes ranking metrics for a single doctor's predicted scores.
    Returns average MAP across the given cutoffs.
    """
    # Extract full ordered list of patient IDs from ranking
    gold_ranking_full = [pid for pid, _ in gold_ranking]

    # Sort predicted scores by descending order
    sorted_pred_pids = [pid for pid, _ in sorted(scores, key=lambda x: x[1], reverse=True)]

    # Metrics
    ndcg = compare_model_and_doctor_ndcg(scores, gold_ranking, k=100) 
    map_at_k_dict = compute_map_at_cutoffs(sorted_pred_pids, gold_ranking_full, cutoff_values)
    map_avg = mean(map_at_k_dict.values()) if map_at_k_dict else 0.0
    tau, _ = compute_kendall_tau(sorted_pred_pids, gold_ranking_full)
    rbo = compute_rbo(sorted_pred_pids, gold_ranking_full)

    return {
        'ndcg': round(ndcg, 3),
        'map': round(map_avg, 3),  # average MAP across cutoffs
        'tau': round(float(tau), 3),
        'rbo': round(rbo, 3),
    }

#######המטריקות עם הקונפינדס - אקיורסי וAUC
# ## שימי לב שאפשר לקנות את המשקולות של הקונפידנס כמה 5 ישפיע יותר מ4 וכו... 
# ========================
# Pairwise Accuracy & AUC עם קונפידנס לזוגות (1..5)
# ========================

from typing import Dict, Iterable, Tuple, Optional, Any
from sklearn.metrics import roc_auc_score
import math

def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def conf_weight(
    conf: Optional[float],
    *,
    scheme: str = "pow",   # "pow" (מומלץ), "linear", "exp", "steps"
    alpha: float = 2.0,    # ככל שגבוה יותר—5 "סופר" הרבה יותר מ־1–2
    min_w: float = 0.0,    # אם רוצים ש־c=1 כמעט לא ישפיע: השאירו 0.0
    max_conf: float = 5.0  # טווח קונפידנס עליון
) -> float:
    """
    המרה של קונפידנס זוגי conf∈{1..5} למשקל w∈[0,1] כדי שזוגות עם conf=5
    ישפיעו משמעותית יותר מזוגות עם conf=1–2 (שכמעט רנדומליים).

    פרמטרים:
      scheme:
        - "pow"    (מומלץ): w = x**alpha, כאשר x = (conf-1)/(max_conf-1)
        - "linear" : w = x (שטוח יותר; פחות מדגיש 5)
        - "exp"    : w ~ exp(beta*(conf-1)) מנורמל ל-[0..1], עם beta=alpha
        - "steps"  : מדרגות קשיחות בטבלה (ראו למטה)
      alpha:
        שולט על "חדות" המיפוי. ככל ש-α גדול יותר—c=5 גדל משמעותית,
        ו-c=1–2 מדוכאים. ערכי התחלה טובים: 1.5, 2.0, 2.5, 3.0.
      min_w:
        משקל מינימלי (למשל 0.0 כדי שכמעט לא נספור c=1).
      conf=None:
        שומר תאימות לאחור: weight=1.0 (כלומר בלי שקילה).

    דוגמאות משקל עבור c=1..5 (לאחר נרמול x=(c-1)/4):

    • scheme="pow" (w = x**alpha):
        alpha=1.0  → [0.00, 0.25, 0.50, 0.75, 1.00]
        alpha=1.5  → [0.00, 0.125, 0.354, 0.650, 1.00]
        alpha=2.0  → [0.00, 0.063, 0.250, 0.563, 1.00]   ← ברירת מחדל טובה
        alpha=2.5  → [0.00, 0.031, 0.177, 0.487, 1.00]
        alpha=3.0  → [0.00, 0.016, 0.125, 0.422, 1.00]

      המשמעות: עם alpha גבוה, 1–2 כמעט לא משפיעים ו-5 שולט.

    • scheme="exp" (מנורמל ל-[0..1], עם beta=alpha):
        beta=0.8  → [0.00, 0.052, 0.168, 0.426, 1.00]
        beta=1.0  → [0.00, 0.032, 0.119, 0.356, 1.00]
        beta=1.5  → [0.00, 0.009, 0.047, 0.221, 1.00]

      exp מדכא חזק יותר את 2–3 יחסית ל-"pow" באותם פרמטרים.

    • scheme="steps": (שנו לפי טעם/נתונים)
        {1: 0.00, 2: 0.10, 3: 0.40, 4: 0.70, 5: 1.00}

    המלצה פרקטית:
      - התחילי עם scheme="pow", alpha=2.0 (ו-min_w=0.0 כדי ש-1 כמעט לא ישפיע).
      - אם את רוצה להתעלם עוד יותר מ-1–2 → alpha=2.5–3.0.
      - אם את רוצה רכות → alpha=1.5.
      - ניתן לבחור scheme="exp" אם רוצים הדגשה אפילו חזקה יותר לטופ.

    החזרה:
      float ב-[0..1]: משקל לשימוש בכל מטריקות זוגיות/הפסדים/דגימות.
    """
    if conf is None:
        return 1.0

    c = float(_clip(conf, 1.0, max_conf))
    x = (c - 1.0) / (max_conf - 1.0)  # 1 → 0, 5 → 1

    if scheme == "linear":
        w = x
    elif scheme == "pow":
        w = x ** float(alpha)
    elif scheme == "exp":
        beta = float(alpha)
        # מנרמלים כך ש- conf=1 → 0 ו- conf=max_conf → 1
        w = (math.exp(beta * (c - 1.0)) - 1.0) / (math.exp(beta * (max_conf - 1.0)) - 1.0)
    elif scheme == "steps":
        table = {1: 0.0, 2: 0.1, 3: 0.4, 4: 0.7, 5: 1.0}
        w = table.get(int(round(c)), x)
    else:
        w = x

    return float(_clip(w, min_w, 1.0))

def pairwise_accuracy_weighted(
    score_dict: Dict[Any, float],
    pair_list: Iterable[Tuple[Tuple[Any, Any], Optional[float]]],
    *,
    scheme: str = "pow",
    alpha: float = 2.0,
    min_w: float = 0.0
) -> float:
    """
    Pairwise Accuracy משוקלל לפי קונפידנס:
      pair_list: [((i, j), conf), ...]  כאשר הדוקטור העדיף i על j.
      score_dict: ציוני המודל לכל מטופל.

    הלוגיקה:
      - מחושב משקל לכל זוג לפי conf_weight(conf, scheme, alpha, min_w)
      - מזכים את המודל אם s_i > s_j ומחברים עם המשקל
      - דיוק = (סכום משקלי הצלחות) / (סכום משקלים)

    אם conf=None → משקל 1.0 (תאימות לאחור).
    """
    correct, total = 0.0, 0.0
    for ((i, j), conf) in pair_list:
        si = score_dict.get(i)
        sj = score_dict.get(j)
        if si is None or sj is None:
            continue
        w = conf_weight(conf, scheme=scheme, alpha=alpha, min_w=min_w)
        if si > sj:
            correct += w
        total += w
    return 0.0 if total == 0.0 else (correct / total)

def calc_auc_weighted(
    score_dict: Dict[Any, float],
    pair_list: Iterable[Tuple[Tuple[Any, Any], Optional[float]]],
    *,
    scheme: str = "pow",
    alpha: float = 2.0,
    min_w: float = 0.0
) -> Optional[float]:
    """
    AUC על הפרשי ציונים Δ=s_i-s_j, עם sample_weight = conf_weight(conf,..).
    כל זוג תורם שתי דגימות: (Δ, y=1) ו-(-Δ, y=0) עם אותו משקל.

    אם אין שתי מחלקות שונות ב-y_true (למשל מעט מדי דגימות) → מחזירים None.
    """
    y_true, y_scores, wts = [], [], []
    for ((i, j), conf) in pair_list:
        si = score_dict.get(i)
        sj = score_dict.get(j)
        if si is None or sj is None:
            continue
        w = conf_weight(conf, scheme=scheme, alpha=alpha, min_w=min_w)
        d = si - sj
        y_scores.extend([d, -d])
        y_true.extend([1, 0])   # i>j הוא ה"פוזיטיב"
        wts.extend([w, w])

    if len(set(y_true)) < 2 or len(y_scores) == 0:
        return None

    return float(roc_auc_score(y_true, y_scores, sample_weight=wts))
