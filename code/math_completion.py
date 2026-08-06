# ===== math_completion.py =====
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import pandas as pd
from metrics import (
    pairwise_accuracy, calc_auc, evaluate_ranking_metrics_for_doctor
)

Pair = Tuple[int, int]  # (winner, loser)

# ---------- helpers to read/build edges ----------
def _extract_edges_from_graph(train_graph) -> List[Pair]:
    if train_graph is None:
        return []
    if hasattr(train_graph, "edges"):
        try:
            return list(train_graph.edges())
        except Exception:
            pass
    if isinstance(train_graph, dict):
        if "edges" in train_graph and isinstance(train_graph["edges"], (list, tuple)):
            return list(train_graph["edges"])
        if "out_edges" in train_graph and isinstance(train_graph["out_edges"], dict):
            edges: List[Pair] = []
            for u, outs in train_graph["out_edges"].items():
                for v in outs:
                    edges.append((u, v))
            return edges
    return []

def build_graph_from_pairs(pairs_list: Optional[List[Pair]]) -> dict:
    """
    בונה אובייקט גרף בסגנון {"edges": [...] } מרשימת זוגות (מנצח, מפסיד).
    מיועד לשימוש בכל איטרציה כדי להזין ל-compute_math_scores גרף *חלקי*.
    """
    return {"edges": list(pairs_list) if pairs_list else []}

def _build_adj(edges: List[Pair], patients: List[int]):
    out_edges = defaultdict(set)
    in_edges  = defaultdict(set)
    for u, v in edges:
        if u == v:  # ignore self loops
            continue
        out_edges[u].add(v)
        in_edges[v].add(u)
    for p in patients:
        out_edges.setdefault(p, set())
        in_edges.setdefault(p, set())
    return out_edges, in_edges

# ---------- 6 methods ----------
def _dominance_count(out_edges, patients):      # (a)
    return {p: len(out_edges[p]) for p in patients}

def _beaten_count(in_edges, patients):          # (b) (raw indegree; נמוך=טוב)
    return {p: len(in_edges[p]) for p in patients}

def _ratio_ab(a_scores, b_scores):              # (c)
    return {p: a_scores[p] / (b_scores[p] if b_scores[p] != 0 else 1) for p in a_scores}

def _scc_kosaraju(out_edges, patients):
    visited, order = set(), []
    def dfs1(u):
        visited.add(u)
        for v in out_edges[u]:
            if v not in visited:
                dfs1(v)
        order.append(u)
    for p in patients:
        if p not in visited:
            dfs1(p)

    rev = defaultdict(set)
    for u in out_edges:
        for v in out_edges[u]:
            rev[v].add(u)

    comp_id = {}
    def dfs2(u, cid):
        comp_id[u] = cid
        for v in rev[u]:
            if v not in comp_id:
                dfs2(v, cid)

    cid = 0
    for u in reversed(order):
        if u not in comp_id:
            dfs2(u, cid)
            cid += 1
    return comp_id, cid

def _compress_to_dag(out_edges, comp_id, n_comp):
    comp_out = defaultdict(set)
    for u in out_edges:
        cu = comp_id[u]
        for v in out_edges[u]:
            cv = comp_id[v]
            if cu != cv:
                comp_out[cu].add(cv)
    for c in range(n_comp):
        comp_out.setdefault(c, set())
    return comp_out

def _topo_order(dag_out):
    indeg = defaultdict(int)
    nodes = list(dag_out.keys())
    for u in nodes:
        for v in dag_out[u]:
            indeg[v] += 1
        indeg.setdefault(u, indeg.get(u, 0))
    q = deque([u for u in nodes if indeg[u] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in dag_out[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order

def _longest_paths_in_DAG(dag_out, topo):
    # ending at
    L_end = {u: 1 for u in dag_out}
    for u in topo:
        for v in dag_out[u]:
            L_end[v] = max(L_end[v], L_end[u] + 1)
    # starting from
    L_start = {u: 1 for u in dag_out}
    for u in reversed(topo):
        for v in dag_out[u]:
            L_start[u] = max(L_start[u], 1 + L_start[v])
    return L_end, L_start

def _lift_back_to_patients(L_end_c, L_start_c, comp_id):
    end_scores, start_scores = {}, {}
    for p, c in comp_id.items():
        end_scores[p]   = L_end_c[c]
        start_scores[p] = L_start_c[c]
    return end_scores, start_scores

# ---------- core scoring ----------
def compute_math_scores(train_graph, patients_df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    """
    מחשב ציוני MATH נתונים גרף (רצוי חלקי) ורשימת מטופלים.
    מחזיר dict: method_name -> {patient_num: score_float}, גבוה=טוב.
    """
    patients = patients_df["patient_num"].tolist()
    edges = _extract_edges_from_graph(train_graph)
    out_edges, in_edges = _build_adj(edges, patients)

    a_scores = _dominance_count(out_edges, patients)           # (a)
    b_raw    = _beaten_count(in_edges, patients)               # (b) נמוך=טוב → נהפוך לגבוה=טוב
    b_max    = max(b_raw.values()) if b_raw else 0
    b_scores = {p: (b_max - b_raw[p]) for p in patients}       # “כמה לא גברו עליי”

    c_scores = _ratio_ab(a_scores, b_raw)                      # (c)

    comp_id, n_comp = _scc_kosaraju(out_edges, patients)       # (d)(e)(f)
    dag_out = _compress_to_dag(out_edges, comp_id, n_comp)
    topo = _topo_order(dag_out)
    L_end_c, L_start_c = _longest_paths_in_DAG(dag_out, topo)
    d_scores, e_scores = _lift_back_to_patients(L_end_c, L_start_c, comp_id)
    f_scores = {p: d_scores[p] / (e_scores[p] if e_scores[p] != 0 else 1) for p in patients}

    return {
        "math_a_domcount"    : a_scores,
        "math_b_beaten_inv"  : b_scores,
        "math_c_ratio"       : c_scores,
        "math_d_chain_end"   : d_scores,
        "math_e_chain_start" : e_scores,
        "math_f_chain_ratio" : f_scores,
    }

def compute_math_scores_from_pairs(pairs_list: Optional[List[Pair]], patients_df: pd.DataFrame
                                   ) -> Dict[str, Dict[int, float]]:
    """
    wrapper נוח: קחי רשימת זוגות (של האיטרציה הנוכחית), בני ממנה גרף חלקי,
    והחזירי ציוני MATH על בסיסו.
    """
    partial_graph = build_graph_from_pairs(pairs_list)
    return compute_math_scores(partial_graph, patients_df)

# ---------- run one math "model" ----------
def run_single_math_experiment(
    doctor_key: str,
    method_name: str,                 # למשל: "math_a_domcount"
    math_scores_by_method: dict,      # פלט compute_math_scores(…)
    df_train_scaled, df_test_scaled,  # נשאר לחתימה בלבד
    all_train_pairs, all_test_pairs,  # נשאר לחתימה בלבד
    train_rankings, test_rankings,    # נשאר לחתימה בלבד
    recs_only: bool = True,
    llm: bool = False,
    eval_pairs=None,                  # ← היה overall_pairs
    eval_rankings=None,               # ← היה overall_rankings
    # תאימות לאחור: אם מישהו עוד מעביר overall_*, נקלוט את זה:
    overall_pairs=None,
    overall_rankings=None,
):
    """
    מריץ הערכה לשיטה מתמטית אחת על בסיס ציונים שחושבו מגרף *חלקי* (math_scores_by_method),
    אבל מודד מול eval_pairs/eval_rankings (בדרך כלל ה-GT המלא) כדי להשוות הוגן.
    """
    # תאימות לאחור: אם eval_* לא הועברו אבל overall_* כן — השתמש בהן.
    if eval_pairs is None and overall_pairs is not None:
        eval_pairs = overall_pairs
    if eval_rankings is None and overall_rankings is not None:
        eval_rankings = overall_rankings

    if eval_pairs is None or eval_rankings is None:
        raise ValueError("eval_pairs/eval_rankings must be provided for math models.")
    if doctor_key not in eval_pairs or doctor_key not in eval_rankings:
        raise KeyError(f"{doctor_key} missing in eval_pairs/eval_rankings")

    # ציוני השיטה לכל המטופלים (נגזרו מהגרף החלקי של האיטרציה)
    score_dict = math_scores_by_method[method_name]  # {pid: score}

    # מדדים pairwise על *כל* הזוגות של היעד שהעברת (בדרך כלל overall/ALL)
    acc = pairwise_accuracy(score_dict, eval_pairs[doctor_key])
    auc = calc_auc(score_dict, eval_pairs[doctor_key])

    # דירוג מלא שנגזר מהגרף החלקי; נמדד מול יעד הדירוג (בדרך כלל overall)
    scores_list = list(score_dict.items())
    rank_metrics = evaluate_ranking_metrics_for_doctor(
        scores_list, eval_rankings[doctor_key]
    )

    results = {
        'doctor': doctor_key,
        'overall_accuracy': acc,
        'overall_auc': auc,
        **{f'overall_{k}': v for k, v in rank_metrics.items()},
    }
    return None, results

# ---------- optional convenience: one-shot run from pairs ----------
def run_single_math_on_partial_graph(
    doctor_key: str,
    method_name: str,
    pairs_for_alpha: Optional[List[Pair]],   # הזוגות של האיטרציה לרופא הזה
    patients_df: pd.DataFrame,
    df_train_scaled, df_test_scaled,
    all_train_pairs, all_test_pairs,
    train_rankings, test_rankings,
    eval_pairs, eval_rankings,
    recs_only: bool = True,
    llm: bool = False,
):
    """
    פונקציית נוחות: מקבלת את רשימת הזוגות החלקית של האיטרציה,
    מחשבת ציוני MATH מהגרף החלקי, ומחזירה תוצאות הערכה מול היעד.
    """
    math_scores_by_method = compute_math_scores_from_pairs(pairs_for_alpha, patients_df)
    return run_single_math_experiment(
        doctor_key=doctor_key,
        method_name=method_name,
        math_scores_by_method=math_scores_by_method,
        df_train_scaled=df_train_scaled, df_test_scaled=df_test_scaled,
        all_train_pairs=all_train_pairs, all_test_pairs=all_test_pairs,
        train_rankings=train_rankings, test_rankings=test_rankings,
        recs_only=recs_only, llm=llm,
        eval_pairs=eval_pairs, eval_rankings=eval_rankings,
    )



# # ===== math_completion.py =====
# from collections import defaultdict, deque
# import pandas as pd
# from metrics import (
#     pairwise_accuracy, calc_auc, evaluate_ranking_metrics_for_doctor
# )

# # ---------- helpers to read edges ----------
# def _extract_edges_from_graph(train_graph):
#     if train_graph is None:
#         return []
#     if hasattr(train_graph, "edges"):
#         try:
#             return list(train_graph.edges())
#         except Exception:
#             pass
#     if isinstance(train_graph, dict):
#         if "edges" in train_graph and isinstance(train_graph["edges"], (list, tuple)):
#             return list(train_graph["edges"])
#         if "out_edges" in train_graph and isinstance(train_graph["out_edges"], dict):
#             edges = []
#             for u, outs in train_graph["out_edges"].items():
#                 for v in outs:
#                     edges.append((u, v))
#             return edges
#     return []

# def _build_adj(edges, patients):
#     out_edges = defaultdict(set)
#     in_edges  = defaultdict(set)
#     for u, v in edges:
#         if u == v:  # ignore self loops
#             continue
#         out_edges[u].add(v)
#         in_edges[v].add(u)
#     for p in patients:
#         out_edges.setdefault(p, set())
#         in_edges.setdefault(p, set())
#     return out_edges, in_edges

# # ---------- 6 methods ----------
# def _dominance_count(out_edges, patients):      # (a)
#     return {p: len(out_edges[p]) for p in patients}

# def _beaten_count(in_edges, patients):          # (b) (raw indegree; נמוך=טוב)
#     return {p: len(in_edges[p]) for p in patients}

# def _ratio_ab(a_scores, b_scores):              # (c)
#     return {p: a_scores[p] / (b_scores[p] if b_scores[p] != 0 else 1) for p in a_scores}

# def _scc_kosaraju(out_edges, patients):
#     visited, order = set(), []
#     def dfs1(u):
#         visited.add(u)
#         for v in out_edges[u]:
#             if v not in visited:
#                 dfs1(v)
#         order.append(u)
#     for p in patients:
#         if p not in visited:
#             dfs1(p)

#     rev = defaultdict(set)
#     for u in out_edges:
#         for v in out_edges[u]:
#             rev[v].add(u)

#     comp_id = {}
#     def dfs2(u, cid):
#         comp_id[u] = cid
#         for v in rev[u]:
#             if v not in comp_id:
#                 dfs2(v, cid)

#     cid = 0
#     for u in reversed(order):
#         if u not in comp_id:
#             dfs2(u, cid)
#             cid += 1
#     return comp_id, cid

# def _compress_to_dag(out_edges, comp_id, n_comp):
#     comp_out = defaultdict(set)
#     for u in out_edges:
#         cu = comp_id[u]
#         for v in out_edges[u]:
#             cv = comp_id[v]
#             if cu != cv:
#                 comp_out[cu].add(cv)
#     for c in range(n_comp):
#         comp_out.setdefault(c, set())
#     return comp_out

# def _topo_order(dag_out):
#     indeg = defaultdict(int)
#     nodes = list(dag_out.keys())
#     for u in nodes:
#         for v in dag_out[u]:
#             indeg[v] += 1
#         indeg.setdefault(u, indeg.get(u, 0))
#     from collections import deque
#     q = deque([u for u in nodes if indeg[u] == 0])
#     order = []
#     while q:
#         u = q.popleft()
#         order.append(u)
#         for v in dag_out[u]:
#             indeg[v] -= 1
#             if indeg[v] == 0:
#                 q.append(v)
#     return order

# def _longest_paths_in_DAG(dag_out, topo):
#     # ending at
#     L_end = {u: 1 for u in dag_out}
#     for u in topo:
#         for v in dag_out[u]:
#             L_end[v] = max(L_end[v], L_end[u] + 1)
#     # starting from
#     L_start = {u: 1 for u in dag_out}
#     for u in reversed(topo):
#         for v in dag_out[u]:
#             L_start[u] = max(L_start[u], 1 + L_start[v])
#     return L_end, L_start

# def _lift_back_to_patients(L_end_c, L_start_c, comp_id):
#     end_scores, start_scores = {}, {}
#     for p, c in comp_id.items():
#         end_scores[p]   = L_end_c[c]
#         start_scores[p] = L_start_c[c]
#     return end_scores, start_scores

# def compute_math_scores(train_graph, patients_df):
#     """
#     מחזיר dict: method_name -> {patient_num: score_float}, גבוה=טוב.
#     """
#     patients = patients_df["patient_num"].tolist()
#     edges = _extract_edges_from_graph(train_graph)
#     out_edges, in_edges = _build_adj(edges, patients)

#     a_scores = _dominance_count(out_edges, patients)           # (a)
#     b_raw    = _beaten_count(in_edges, patients)               # (b) נמוך=טוב → נהפוך לגבוה=טוב
#     b_max    = max(b_raw.values()) if b_raw else 0
#     b_scores = {p: (b_max - b_raw[p]) for p in patients}       # “כמה לא גברו עליי”

#     c_scores = _ratio_ab(a_scores, b_raw)                      # (c)

#     comp_id, n_comp = _scc_kosaraju(out_edges, patients)       # (d)(e)(f)
#     dag_out = _compress_to_dag(out_edges, comp_id, n_comp)
#     topo = _topo_order(dag_out)
#     L_end_c, L_start_c = _longest_paths_in_DAG(dag_out, topo)
#     d_scores, e_scores = _lift_back_to_patients(L_end_c, L_start_c, comp_id)
#     f_scores = {p: d_scores[p] / (e_scores[p] if e_scores[p] != 0 else 1) for p in patients}

#     return {
#         "math_a_domcount"    : a_scores,
#         "math_b_beaten_inv"  : b_scores,
#         "math_c_ratio"       : c_scores,
#         "math_d_chain_end"   : d_scores,
#         "math_e_chain_start" : e_scores,
#         "math_f_chain_ratio" : f_scores,
#     }

# # ---------- run one math "model" ----------
# # ---------- run one math "model" ----------
# def run_single_math_experiment(
#     doctor_key: str,
#     method_name: str,                 # למשל: "math_a_domcount"
#     math_scores_by_method: dict,      # פלט compute_math_scores
#     df_train_scaled, df_test_scaled,  # נשאר לחתימה בלבד
#     all_train_pairs, all_test_pairs,  # נשאר לחתימה בלבד
#     train_rankings, test_rankings,    # נשאר לחתימה בלבד
#     recs_only: bool = True,
#     llm: bool = False,
#     overall_pairs=None,
#     overall_rankings=None,
# ):
#     # ציוני השיטה לכל המטופלים
#     score_dict = math_scores_by_method[method_name]  # {pid: score}

#     # ---- נשתמש ב-ground-truth האמיתי של ALL ----
#     if overall_pairs is None or overall_rankings is None:
#         raise ValueError("Overall pairs/rankings must be provided for math models.")
#     if doctor_key not in overall_pairs or doctor_key not in overall_rankings:
#         raise KeyError(f"{doctor_key} missing in overall_pairs/overall_rankings")

#     # ---- Accuracy / AUC על כל הזוגות ----
#     overall_acc = pairwise_accuracy(score_dict, overall_pairs[doctor_key])
#     overall_auc = calc_auc(score_dict, overall_pairs[doctor_key])

#     # ---- דירוגים כוללים ----
#     overall_scores_list = list(score_dict.items())
#     overall_rank_metrics = evaluate_ranking_metrics_for_doctor(
#         overall_scores_list, overall_rankings[doctor_key]
#     )

#     results = {
#         'doctor': doctor_key,
#         'overall_accuracy': overall_acc,
#         'overall_auc': overall_auc,
#         **{f'overall_{k}': v for k, v in overall_rank_metrics.items()},
#     }
#     return None, results
