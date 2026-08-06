import random
from typing import Dict, Tuple, Set, Any, List


def doctor_decision_tree_1(p1: Dict, p2: Dict) -> Tuple[str, int]:
    TREATMENT_RECS: Set[str] = {"rec9", "rec10", "rec11", "rec12", "rec13", "rec14", "rec15"}
    CONSULT_IMAGING: Set[str] = {"rec5", "rec6", "rec7", "rec8", "rec18"}
    HIGH_PRIORITY_RECS: Set[str] = {"rec10", "rec11", "rec12", "rec13", "rec14", "rec15"}
    LOW_VALUE_RECS: Set[str] = {"rec1", "rec2"}

    CLOSE_RISK_DELTA = 4.0
    YOUNG_AGE = 40
    VERY_OLD_AGE = 90
    ELDERLY_AGE = 85
    MID_AGE = 75
    RISK_HIGH = 8.0
    RISK_MODERATE_YOUNG = 6.0

    def _risk(p): return float(p.get("risk", p.get("Risk", 0.0)))
    def _age(p): return float(p.get("age", p.get("Age", 0.0)))
    def _extract_recs(p):
        if "recommendations" in p:
            return {str(r).lower() for r in p["recommendations"]}
        return {k.lower() for k, v in p.items() if k.lower().startswith("rec") and v}

    def _rec_sum(p): return int(p.get("rec_sum", len(_extract_recs(p))))

    r1, r2 = _risk(p1), _risk(p2)
    a1, a2 = _age(p1), _age(p2)
    recs1, recs2 = _extract_recs(p1), _extract_recs(p2)
    s1, s2 = _rec_sum(p1), _rec_sum(p2)

    p1_t = any(r in TREATMENT_RECS for r in recs1)
    p2_t = any(r in TREATMENT_RECS for r in recs2)
    p1_ci = any(r in CONSULT_IMAGING for r in recs1)
    p2_ci = any(r in CONSULT_IMAGING for r in recs2)
    p1_hp = any(r in HIGH_PRIORITY_RECS for r in recs1)
    p2_hp = any(r in HIGH_PRIORITY_RECS for r in recs2)
    p1_lv = any(r in LOW_VALUE_RECS for r in recs1)
    p2_lv = any(r in LOW_VALUE_RECS for r in recs2)

    # Case 1: One treated
    if p1_t and not p2_t:
        if a1 > VERY_OLD_AGE:
            if r1 < 3 and p2_ci: return ("Patient 2", 5)
            if p1_lv: return ("Patient 2", 4)
        if a2 < YOUNG_AGE and p2_ci and r2 > 7: return ("Patient 2", 5)
        if p1_hp and not p2_hp: return ("Patient 1", 5)
        return ("Patient 1", 4)
    if p2_t and not p1_t:
        if a2 > VERY_OLD_AGE:
            if r2 < 3 and p1_ci: return ("Patient 1", 5)
            if p2_lv: return ("Patient 1", 4)
        if a1 < YOUNG_AGE and p1_ci and r1 > 7: return ("Patient 1", 5)
        if p2_hp and not p1_hp: return ("Patient 2", 5)
        return ("Patient 2", 4)

    # Case 2: Both treated
    if p1_t and p2_t:
        if p1_hp and not p2_hp: return ("Patient 1", 5)
        if p2_hp and not p1_hp: return ("Patient 2", 5)
        if p1_lv and not p2_lv: return ("Patient 2", 4)
        if p2_lv and not p1_lv: return ("Patient 1", 4)
        if abs(r1 - r2) > CLOSE_RISK_DELTA:
            return ("Patient 1", 4) if r1 > r2 else ("Patient 2", 4)
        if s1 > s2: return ("Patient 1", 4)
        if s2 > s1: return ("Patient 2", 4)
        if abs(a1 - a2) > 20: return ("Patient 1", 3) if a1 < a2 else ("Patient 2", 3)
        if r1 > RISK_HIGH and r2 <= RISK_HIGH: return ("Patient 1", 3)
        if r2 > RISK_HIGH and r1 <= RISK_HIGH: return ("Patient 2", 3)
        if r1 != r2: return ("Patient 1", 2) if r1 > r2 else ("Patient 2", 2)
        if a1 != a2: return ("Patient 1", 2) if a1 < a2 else ("Patient 2", 2)
        return (random.choice(["Patient 1", "Patient 2"]), 1)

    # Case 3: Neither treated
    p1_young_trigger = (a1 < 50 and p1_ci and r1 > RISK_MODERATE_YOUNG)
    p2_young_trigger = (a2 < 50 and p2_ci and r2 > RISK_MODERATE_YOUNG)
    if p1_young_trigger and not p2_young_trigger: return ("Patient 1", 4)
    if p2_young_trigger and not p1_young_trigger: return ("Patient 2", 4)
    if a1 > ELDERLY_AGE and a2 < 60 and s1 < 3 and r2 > 7: return ("Patient 2", 3)
    if a2 > ELDERLY_AGE and a1 < 60 and s2 < 3 and r1 > 7: return ("Patient 1", 3)
    if s1 > s2 and a1 < MID_AGE: return ("Patient 1", 3)
    if s2 > s1 and a2 < MID_AGE: return ("Patient 2", 3)
    if abs(a1 - a2) >= 30:
        if s1 > s2: return ("Patient 1", 2)
        if s2 > s1: return ("Patient 2", 2)
    if a1 > ELDERLY_AGE and a2 > ELDERLY_AGE and s1 <= 2 and s2 <= 2:
        return ("Patient 1", 1) if a1 < a2 else ("Patient 2", 1)
    if s1 != s2: return ("Patient 1", 1) if s1 > s2 else ("Patient 2", 1)
    if r1 != r2: return ("Patient 1", 1) if r1 > r2 else ("Patient 2", 1)
    if a1 != a2: return ("Patient 1", 1) if a1 < a2 else ("Patient 2", 1)
    return (random.choice(["Patient 1", "Patient 2"]), 1)



def doctor_decision_tree_2(p1: Dict, p2: Dict) -> Tuple[str, int]:
    # ---- Recommendation groups (כפי שהיו במיפוי המקורי) ----
    PREVENTIVE_RECS: Set[str] = {"rec9", "rec19", "rec20"}                 # התחלת טיפול מניעתי / הפסקת הרגל מזיק
    LIFESTYLE_RECS:  Set[str] = {"rec19", "rec20"}                         # שינוי/שיפור אורח-חיים
    MONITORING_RECS: Set[str] = {"rec1", "rec4", "rec5", "rec18"}          # בדיקות בסיס, ניטור, הדמיה שגרתית
    INVASIVE_RECS:   Set[str] = {"rec10", "rec11", "rec12", "rec13", "rec14", "rec15"}  # טיפולים/פרוצדורות פולשניים

    # ---- Helpers --------------------------------------------------------
    def _recs(p) -> Set[str]:
        """Handles both list-format and wide recX: True format."""
        if "recommendations" in p:
            return {str(r).lower() for r in p["recommendations"]}
        return {k.lower() for k, v in p.items() if k.lower().startswith("rec") and v}

    # Boolean features
    def has_any(p): return bool(_recs(p))
    def has_prev(p): return bool(_recs(p) & PREVENTIVE_RECS)
    def has_life(p): return bool(_recs(p) & LIFESTYLE_RECS)
    def has_mon(p):  return bool(_recs(p) & MONITORING_RECS)
    def has_inv(p):  return bool(_recs(p) & INVASIVE_RECS)

    def prev_bundle(p): return has_prev(p) or has_life(p) or has_mon(p)
    def only_prev_bundle(p):  # “LI טהור” – prevention/lifestyle/monitoring בלבד
        recs = _recs(p)
        return recs and not (recs & INVASIVE_RECS) and prev_bundle(p)
    def only_inv(p):          # רק פולשני
        recs = _recs(p)
        return recs and not (recs & (PREVENTIVE_RECS | LIFESTYLE_RECS | MONITORING_RECS)) and (recs & INVASIVE_RECS)

    # אות בטיחותי (החלפת טיפול)
    def has_safety_replace(p): return "rec13" in _recs(p)

    # Counters
    def cnt_prev(p): return len(_recs(p) & PREVENTIVE_RECS)
    def cnt_life(p): return len(_recs(p) & LIFESTYLE_RECS)
    def cnt_mon(p):  return len(_recs(p) & MONITORING_RECS)
    def cnt_inv(p):  return len(_recs(p) & INVASIVE_RECS)

    # Risk adjusted for age
    def adj(p): return float(p.get("risk", 0.0)) - 0.12 * float(p.get("age", 0.0))

    # ---- Extract features ------------------------------------------------
    r1, r2 = float(p1.get("risk", 0.0)), float(p2.get("risk", 0.0))
    a1, a2 = float(p1.get("age", 0.0)),  float(p2.get("age", 0.0))
    s1, s2 = adj(p1), adj(p2)

    p1_any, p2_any = has_any(p1), has_any(p2)
    p1_prev, p2_prev = prev_bundle(p1), prev_bundle(p2)
    p1_life, p2_life = has_life(p1), has_life(p2)
    p1_mon,  p2_mon  = has_mon(p1),  has_mon(p2)
    p1_inv,  p2_inv  = has_inv(p1),  has_inv(p2)

    p1_only_prev, p2_only_prev = only_prev_bundle(p1), only_prev_bundle(p2)
    p1_srep, p2_srep = has_safety_replace(p1), has_safety_replace(p2)

    # Scale-aware thresholds
    scale = 40 if max(r1, r2) > 20 else 10
    def risk_gap(val40, val10): return val40 if scale == 40 else val10
    def very_high(r):           return r >= (20 if scale == 40 else 8)
    def big_gap():              return abs(r1 - r2) >= risk_gap(10, 4)
    def close_gap():
        base = risk_gap(5, 2)
        if a1 >= 75 or a2 >= 75: base += risk_gap(1, 1)
        elif a1 <= 40 and a2 <= 40: base -= 1
        return abs(r1 - r2) <= base

    # “LI-score” (לשוברי-שוויון)
    def li_score(life, mon, prev, inv):
        return 2*life + 2*mon + 1*prev - 2*inv
    li1, li2 = li_score(p1_life, p1_mon, p1_prev, p1_inv), li_score(p2_life, p2_mon, p2_prev, p2_inv)

    # ---- Deterministic final tie-breaker ---------------------------------
    def final_tie():
        if p1_only_prev != p2_only_prev:
            return ("Patient 1", 1) if p1_only_prev else ("Patient 2", 1)
        if li1 != li2:
            return ("Patient 1", 1) if li1 > li2 else ("Patient 2", 1)
        if s1 != s2:
            return ("Patient 1", 1) if s1 > s2 else ("Patient 2", 1)
        if r1 != r2:
            return ("Patient 1", 1) if r1 > r2 else ("Patient 2", 1)
        rs1, rs2 = int(p1.get("rec_sum", 0)), int(p2.get("rec_sum", 0))
        if rs1 != rs2:
            return ("Patient 1", 1) if rs1 > rs2 else ("Patient 2", 1)
        if a1 != a2:
            return ("Patient 1", 1) if a1 < a2 else ("Patient 2", 1)
        return (random.choice(["Patient 1", "Patient 2"]), 1)

    # ================= Decision cascade =================

    # 0) Safety-replacement (rec13) one-sided
    if p1_srep != p2_srep:
        return (("Patient 1", 5) if p1_srep else ("Patient 2", 5)) if close_gap() \
               else (("Patient 1", 4) if p1_srep else ("Patient 2", 4))

    # 1) “יש המלצות” מול “אין”
    if p1_any != p2_any:
        return ("Patient 1", 5) if p1_any else ("Patient 2", 5)
    if not p1_any:  # שניהם ריקים
        if r1 != r2:
            return ("Patient 1", 4) if r1 > r2 else ("Patient 2", 4)
        if a1 != a2:
            return ("Patient 1", 3) if a1 < a2 else ("Patient 2", 3)
        return final_tie()

    # 2) שניהם עם bundle מניעתי
    if p1_prev and p2_prev:
        if (p1_only_prev != p2_only_prev) and close_gap():
            return ("Patient 1", 4) if p1_only_prev else ("Patient 2", 4)
        if (p1_life and p1_mon) != (p2_life and p2_mon) and close_gap():
            return ("Patient 1", 4) if (p1_life and p1_mon) else ("Patient 2", 4)
        if (a1 >= 75 or a2 >= 75) and (p1_inv != p2_inv) and not big_gap():
            return ("Patient 1", 3) if not p1_inv else ("Patient 2", 3)
        if close_gap() and li1 != li2:
            return ("Patient 1", 3) if li1 > li2 else ("Patient 2", 3)
        if very_high(r1) != very_high(r2):
            hi = "Patient 1" if very_high(r1) else "Patient 2"
            if (p1_inv if hi == "Patient 1" else p2_inv):
                return (hi, 3)
        if s1 > s2 + risk_gap(2, 1):
            return ("Patient 1", 3)
        if s2 > s1 + risk_gap(2, 1):
            return ("Patient 2", 3)
        rs1, rs2 = int(p1.get("rec_sum", 0)), int(p2.get("rec_sum", 0))
        if rs1 != rs2:
            return ("Patient 1", 2) if rs1 > rs2 else ("Patient 2", 2)
        if a1 != a2 and close_gap():
            return ("Patient 1", 2) if a1 < a2 else ("Patient 2", 2)
        return final_tie()

    # 3) אף אחד לא עם bundle מניעתי
    if not p1_prev and not p2_prev:
        if p1_inv and p2_inv and big_gap():
            return ("Patient 1", 3) if r1 > r2 else ("Patient 2", 3)
        if p1_inv != p2_inv and close_gap():
            return ("Patient 1", 3) if not p1_inv else ("Patient 2", 3)
        if very_high(r1) != very_high(r2):
            hi = "Patient 1" if very_high(r1) else "Patient 2"
            if (p1_inv if hi == "Patient 1" else p2_inv):
                return (hi, 3)
        if s1 > s2 + risk_gap(1.5, 1.0):
            return ("Patient 1", 2)
        if s2 > s1 + risk_gap(1.5, 1.0):
            return ("Patient 2", 2)
        if close_gap() and a1 != a2:
            return ("Patient 1", 2) if a1 < a2 else ("Patient 2", 2)
        if p1_any != p2_any:
            return ("Patient 1", 2) if p1_any else ("Patient 2", 2)
        return final_tie()

    # 4) בדיוק צד אחד עם bundle מניעתי
    if p1_prev != p2_prev:
        if big_gap():
            return ("Patient 1", 4) if r1 > r2 else ("Patient 2", 4)
        if (p1_only_prev or p2_only_prev) and close_gap():
            return ("Patient 1", 4) if p1_only_prev else ("Patient 2", 4)
        if (p1_prev and p2_inv and close_gap()) or (p2_prev and p1_inv and close_gap()):
            return ("Patient 1", 3) if p1_prev else ("Patient 2", 3)
        if close_gap():
            if (a1 <= 40 and p1_prev) or (a2 <= 40 and p2_prev):
                return ("Patient 1", 3) if p1_prev else ("Patient 2", 3)
            if (a1 >= 75 or a2 >= 75) and p1_inv != p2_inv:
                return ("Patient 1", 3) if not p1_inv else ("Patient 2", 3)
        if s1 != s2:
            return ("Patient 1", 3) if s1 > s2 else ("Patient 2", 3)
        if (p1_prev and p1_life and p1_mon) != (p2_prev and p2_life and p2_mon):
            return ("Patient 1", 2) if (p1_prev and p1_life and p1_mon) else ("Patient 2", 2)
        return final_tie()

    # ---- Fallback (should rarely be reached) ----
    return final_tie()




def doctor_decision_tree_3(p1: Dict, p2: Dict) -> Tuple[str, int]:
    # ---- Recommendation groups (IDs based on your schema) ----
    HIGH_YIELD_LABS: Set[str]      = {"rec2", "rec3"}          # advanced dx labs; pathophysiology work‑up
    HIGH_YIELD_IMG_PROC: Set[str]  = {"rec6", "rec7"}          # advanced imaging; diagnostic procedure
    BASELINE_CLARITY: Set[str]     = {"rec1", "rec5"}          # basic lab; routine imaging (when it advances clarity)
    CONSULT: Set[str]              = {"rec8"}                  # specialist consultation for complex cases

    # Lower‑clarity (for this frame): lifestyle/admin/monitoring‑only steps (kept for gentle de‑weighting)
    LOWER_CLARITY: Set[str]        = {"rec4", "rec19", "rec20"}  # routine monitoring lab; lifestyle counseling

    # ---- Helpers ----
    def _recs(p) -> Set[str]: return set(p.get("recommendations", []))
    def has_any_clarity(p) -> bool:
        recs = _recs(p)
        return bool(recs & (HIGH_YIELD_LABS | HIGH_YIELD_IMG_PROC | BASELINE_CLARITY | CONSULT))

    def hy_lab_cnt(p) -> int: return len(_recs(p) & HIGH_YIELD_LABS)
    def hy_img_cnt(p) -> int: return len(_recs(p) & HIGH_YIELD_IMG_PROC)
    def base_cnt(p)   -> int: return len(_recs(p) & BASELINE_CLARITY)
    def has_consult(p)-> bool: return bool(_recs(p) & CONSULT)
    def has_rec6(p)   -> bool: return "rec6" in _recs(p)
    def lower_only(p) -> bool:
        recs = _recs(p)
        return len(recs) > 0 and recs.issubset(LOWER_CLARITY)

    # Slight age moderation (still secondary to clarity for close calls)
    def adj_risk(p) -> float:
        return float(p.get("risk", 0)) - 0.08 * float(p.get("age", 0))

    # Clarity score (used for tie‑breakers, not as a primary scorer)
    def clarity_score(p) -> int:
        # Strong weight for high‑yield items; breadth bonus; smaller weight for baseline; consult bonus
        score  = 3 * hy_lab_cnt(p) + 3 * hy_img_cnt(p)
        if hy_lab_cnt(p) > 0 and hy_img_cnt(p) > 0:
            score += 2  # breadth bonus: lab + imaging/procedure
        score += min(base_cnt(p), 1)  # at most +1 from baseline clarity
        if has_consult(p):
            score += 1
        # De‑emphasize lower‑clarity‑only bundles
        if lower_only(p):
            score -= 1
        return score

    # ---- Extract features ----
    p1_has = has_any_clarity(p1)
    p2_has = has_any_clarity(p2)
    r1, r2 = float(p1.get("risk", 0)), float(p2.get("risk", 0))
    a1, a2 = float(p1.get("age", 0)),  float(p2.get("age", 0))
    ar1, ar2 = adj_risk(p1), adj_risk(p2)
    cs1, cs2 = clarity_score(p1), clarity_score(p2)

    # ---- 1) Neither has clarity‑oriented steps ----
    if not p1_has and not p2_has:
        # Fall back to plain risk first
        if r1 > r2 + 8:
            return ("Patient 1", 4)
        if r2 > r1 + 8:
            return ("Patient 2", 4)
        # Then age‑adjusted risk
        if ar1 > ar2:
            return ("Patient 1", 3)
        if ar2 > ar1:
            return ("Patient 2", 3)
        # Younger very slight edge
        if a1 < a2:
            return ("Patient 1", 2)
        if a2 < a1:
            return ("Patient 2", 2)
        return (random.choice(["Patient 1", "Patient 2"]), 1)

    # ---- 2) Exactly one has clarity‑oriented steps ----
    if p1_has and not p2_has:
        return ("Patient 1", 5)
    if p2_has and not p1_has:
        return ("Patient 2", 5)

    # ---- 3) Both have clarity‑oriented steps ----
    # 3a) Clear risk override if very large gap
    if r1 >= r2 + 10:
        return ("Patient 1", 4)
    if r2 >= r1 + 10:
        return ("Patient 2", 4)

    # 3b) Breadth advantage (lab + imaging/procedure)
    p1_breadth = (hy_lab_cnt(p1) > 0 and hy_img_cnt(p1) > 0)
    p2_breadth = (hy_lab_cnt(p2) > 0 and hy_img_cnt(p2) > 0)
    if p1_breadth and not p2_breadth:
        return ("Patient 1", 4)
    if p2_breadth and not p1_breadth:
        return ("Patient 2", 4)

    # 3c) Specific high‑value signal: rec6 (advanced imaging) one‑sided
    if has_rec6(p1) and not has_rec6(p2):
        return ("Patient 1", 4)
    if has_rec6(p2) and not has_rec6(p1):
        return ("Patient 2", 4)

    # 3d) Specialist consult can tip balance if paired with at least one high‑yield item
    p1_hi = (hy_lab_cnt(p1) + hy_img_cnt(p1)) > 0
    p2_hi = (hy_lab_cnt(p2) + hy_img_cnt(p2)) > 0
    if has_consult(p1) and p1_hi and not (has_consult(p2) and p2_hi):
        return ("Patient 1", 3)
    if has_consult(p2) and p2_hi and not (has_consult(p1) and p1_hi):
        return ("Patient 2", 3)

    # 3e) Compare clarity scores (meaningful clarity advantage)
    if cs1 >= cs2 + 2:
        return ("Patient 1", 3)
    if cs2 >= cs1 + 2:
        return ("Patient 2", 3)

    # 3f) When clarity is similar and risks are close, lean to age‑adjusted risk
    if abs(r1 - r2) <= 7:
        if ar1 > ar2:
            return ("Patient 1", 2)
        if ar2 > ar1:
            return ("Patient 2", 2)

    # 3g) Baseline clarity tie‑break if still close
    if base_cnt(p1) > base_cnt(p2):
        return ("Patient 1", 2)
    if base_cnt(p2) > base_cnt(p1):
        return ("Patient 2", 2)

    # 3h) rec_sum tie‑break
    if p1.get("rec_sum", 0) > p2.get("rec_sum", 0):
        return ("Patient 1", 2)
    if p2.get("rec_sum", 0) > p1.get("rec_sum", 0):
        return ("Patient 2", 2)

    # 3i) Absolute fallback: younger first
    if a1 < a2:
        return ("Patient 1", 1)
    if a2 < a1:
        return ("Patient 2", 1)
    return (random.choice(["Patient 1", "Patient 2"]), 1)


def doctor_decision_tree_4(p1, p2):
    """
    Input:  p1, p2 – dicts with keys: 'age', 'risk', 'recommendations' (iterable of 'recX'), 'rec_sum'
    Output: tuple ("Patient 1"|"Patient 2", confidence 1..5)
    Symmetric: never relies on input order; always compares features of both patients.
    """

    # --- Categories (compatible with your existing rec codes) ---
    measurement_recs     = {'rec1', 'rec2', 'rec3', 'rec4'}        # מדידה/בדיקות בסיסיות
    imaging_recs         = {'rec5', 'rec6', 'rec7', 'rec8', 'rec18'}
    treatment_start_upg  = {'rec9', 'rec10', 'rec11', 'rec12', 'rec14'}
    treatment_replace    = {'rec13'}                                # replacement due to contraindication
    lifestyle_recs       = {'rec15', 'rec16', 'rec17'}
    consultation_recs    = {'rec19', 'rec20'}

    def count_in(s, recs): return sum(1 for r in recs if r in s)

    def feats(p):
        recs = set(p.get('recommendations', []))
        age  = float(p.get('age', 0.0))
        risk = float(p.get('risk', 0.0))
        m  = count_in(measurement_recs,    recs)
        i  = count_in(imaging_recs,        recs)
        ts = count_in(treatment_start_upg, recs)
        tr = count_in(treatment_replace,   recs)
        ls = count_in(lifestyle_recs,      recs)
        c  = count_in(consultation_recs,   recs)
        li = m + ls + c                     # low‑intensity bundle
        inv = i + ts                        # escalation bundle (imaging/treatment)
        any_rec = (m+i+ts+tr+ls+c) > 0

        # adjusted risk prefers higher risk but penalizes advanced age (safety‑first tilt)
        adj_risk = risk - 0.08 * age

        return {
            "age": age, "risk": risk, "adj": adj_risk,
            "m": m, "i": i, "ts": ts, "tr": tr, "ls": ls, "c": c,
            "li": li, "inv": inv, "any": any_rec,
            "only_li": (li > 0 and inv == 0 and tr == 0),
            "only_imaging": (i > 0 and ts == 0 and tr == 0 and li == 0),
            "only_treat": (ts > 0 and i == 0 and tr == 0 and li == 0),
            "has_imaging": i > 0,
            "has_treat": ts > 0,
            "has_replace": tr > 0,
            "recs_set": recs
        }

    f1, f2 = feats(p1), feats(p2)

    def prefer(pn, conf):
        return (pn, conf)

    # helpers
    def risk_diff_close(r1, r2, age1, age2):
        # dynamic closeness: looser if both older (lean conservative); tighter if both young
        base = 2.0
        if age1 >= 75 or age2 >= 75:
            base = 3.0
        if (age1 <= 40 and age2 <= 40):
            base = 1.5
        return abs(r1 - r2) <= base

    def younger_is(pn1, pn2):
        if f1["age"] < f2["age"]:
            return pn1
        if f2["age"] < f1["age"]:
            return pn2
        # Ages equal → break tie randomly
        return random.choice([pn1, pn2])
    # ─────────────────────────────────────────────────────────────────
    # A) Safety correction first: treatment replacement (rec13)  → conf 5/4
    #    Prefer the one needing a safety‑driven replacement, especially when risks are close.
    # ─────────────────────────────────────────────────────────────────
    if f1["has_replace"] != f2["has_replace"]:
        pn = "Patient 1" if f1["has_replace"] else "Patient 2"
        # if risks close → very decisive; else still prefer but downgrade a bit
        if risk_diff_close(f1["risk"], f2["risk"], f1["age"], f2["age"]):
            return prefer(pn, 5)
        return prefer(pn, 4)

    if f1["has_replace"] and f2["has_replace"]:
        # both require safety replacement → choose higher adjusted risk, tie‑break younger
        if f1["adj"] != f2["adj"]:
            return prefer("Patient 1" if f1["adj"] > f2["adj"] else "Patient 2", 4)
        return prefer(younger_is("Patient 1","Patient 2"), 3)

    # ─────────────────────────────────────────────────────────────────
    # B) No recommendations vs. has recommendations  → conf 5
    #    If both empty, pick higher risk (safety watch if risk is high).
    # ─────────────────────────────────────────────────────────────────
    if f1["any"] != f2["any"]:
        return prefer("Patient 1" if f1["any"] else "Patient 2", 5)
    if not f1["any"] and not f2["any"]:
        if f1["risk"] != f2["risk"]:
            return prefer("Patient 1" if f1["risk"] > f2["risk"] else "Patient 2", 4)
        # same risk & no recs → prefer younger to enable conservative monitoring
        return prefer(younger_is("Patient 1","Patient 2"), 3)

    # ─────────────────────────────────────────────────────────────────
    # C) When risks are close → favor low‑intensity pathway
    #    Prefer (measurement + lifestyle + consult) over (imaging/treatment),
    #    unless one patient’s risk is clearly higher.
    # ─────────────────────────────────────────────────────────────────
    if risk_diff_close(f1["risk"], f2["risk"], f1["age"], f2["age"]):
        # preference score: LI wins; penalize invasive escalation in older adults
        def li_pref_score(f):
            penalty_age = 0.5 if f["age"] >= 75 else 0.0
            return 2.0 * f["li"] + 1.0 * f["c"] + 0.5 * f["m"] + 0.5 * f["ls"] - (f["inv"] + penalty_age * f["inv"])

        s1, s2 = li_pref_score(f1), li_pref_score(f2)
        if s1 != s2:
            return prefer("Patient 1" if s1 > s2 else "Patient 2", 4)

        # tie → prefer who lacks escalation entirely (pure LI)
        if f1["only_li"] != f2["only_li"]:
            return prefer("Patient 1" if f1["only_li"] else "Patient 2", 4)

        # still tie → prefer higher adjusted risk (penalizes very old)
        if f1["adj"] != f2["adj"]:
            return prefer("Patient 1" if f1["adj"] > f2["adj"] else "Patient 2", 3)

    # ─────────────────────────────────────────────────────────────────
    # D) Imaging‑only vs. Low‑intensity‑only  → prefer LI when risks close  (conf 4)
    # ─────────────────────────────────────────────────────────────────
    if (f1["only_imaging"] and f2["only_li"]) or (f2["only_imaging"] and f1["only_li"]):
        if risk_diff_close(f1["risk"], f2["risk"], f1["age"], f2["age"]):
            return prefer("Patient 2" if f2["only_li"] else "Patient 1", 4)

    # ─────────────────────────────────────────────────────────────────
    # E) High risk permits escalation (model should learn nonlinearity)
    #    If one is clearly high‑risk and has treatment start/upgrade, allow prioritizing them.
    #    (Works both for 0‑10 and 0‑40 scales via dual threshold.)
    # ─────────────────────────────────────────────────────────────────
    def very_high_risk(r):  # covers both scales
        return (r >= 8.0) or (r >= 20.0)

    if very_high_risk(f1["risk"]) != very_high_risk(f2["risk"]):
        high = f1 if very_high_risk(f1["risk"]) else f2
        high_pn = "Patient 1" if high is f1 else "Patient 2"
        if high["has_treat"] or high["has_imaging"]:
            # decisive if the high‑risk patient already has escalation recs
            return prefer(high_pn, 4)

    # ─────────────────────────────────────────────────────────────────
    # F) Older adults: penalize escalation unless risk gap is large
    # ─────────────────────────────────────────────────────────────────
    def big_risk_gap(r1, r2): return abs(r1 - r2) >= 4.0

    if (f1["age"] >= 75 or f2["age"] >= 75) and (f1["inv"] != f2["inv"]):
        # prefer the one with *less* escalation when the risk gap is not big
        if not big_risk_gap(f1["risk"], f2["risk"]):
            return prefer("Patient 1" if f1["inv"] < f2["inv"] else "Patient 2", 3)

    # ─────────────────────────────────────────────────────────────────
    # G) Both with treatment: prefer fewer upgrades + higher adj risk
    # ─────────────────────────────────────────────────────────────────
    if f1["has_treat"] and f2["has_treat"]:
        # fewer concurrent escalations (ts+i) is slightly preferred (minimalist staging)
        esc1, esc2 = f1["ts"] + f1["i"], f2["ts"] + f2["i"]
        if esc1 != esc2:
            return prefer("Patient 1" if esc1 < esc2 else "Patient 2", 3)
        if f1["adj"] != f2["adj"]:
            return prefer("Patient 1" if f1["adj"] > f2["adj"] else "Patient 2", 3)
        # tie‑break by lower age (safer tolerance to basic steps)
        return prefer(younger_is("Patient 1","Patient 2"), 2)

    # ─────────────────────────────────────────────────────────────────
    # H) Weighted recommendation load (stages before procedures)
    # ─────────────────────────────────────────────────────────────────
    def weighted_load(f):
        # LI counts positively; escalation counts negatively
        return (2.0 * f["li"] + 0.5 * f["m"]) - (1.5 * f["i"] + 1.5 * f["ts"])

    wl1, wl2 = weighted_load(f1), weighted_load(f2)
    if wl1 != wl2:
        # if risks are near, LI‑heavier wins with higher confidence; otherwise lower
        conf = 4 if risk_diff_close(f1["risk"], f2["risk"], f1["age"], f2["age"]) else 3
        return prefer("Patient 1" if wl1 > wl2 else "Patient 2", conf)

    # ─────────────────────────────────────────────────────────────────
    # I) Adjusted risk (nonlinear preference already embedded via age penalty)
    # ─────────────────────────────────────────────────────────────────
    if f1["adj"] != f2["adj"]:
        return prefer("Patient 1" if f1["adj"] > f2["adj"] else "Patient 2", 2)

    # ─────────────────────────────────────────────────────────────────
    # J) rec_sum as last structured signal; then age as final tie‑break
    # ─────────────────────────────────────────────────────────────────
    rs1, rs2 = float(p1.get('rec_sum', 0.0)), float(p2.get('rec_sum', 0.0))
    if rs1 != rs2:
        return prefer("Patient 1" if rs1 > rs2 else "Patient 2", 2)

    # Final fallback – prefer younger (room for conservative build‑up)
    return prefer("Patient 1" if f1["age"] < f2["age"] else "Patient 2", 1)


def doctor_decision_tree_5(p1: Dict, p2: Dict) -> Tuple[str, int]:
    """
    Dr#5 – Treatment‑First Optimizer (per DL mapping):
    נותן משקל גבוה להתחלת/שדרוג/החלפת טיפול כעת, ייעוץ משנה‑ניהול, דיאגנוסטיקה מכריעה (מהירה),
    ובדיקות בסיס/מדידות כ‑prerequisites. לייף‑סטייל משני אלא אם מאפשר את התוכנית.
    סימטרי ודטרמיניסטי. פלט: ("Patient X", conf 1..5)
    """

    # ---- Exact DL-based sets ----
    INIT_TREAT:    Set[str] = {"rec10", "rec11"}       # initiate first-line / initiate advanced
    UPGRADE_TREAT: Set[str] = {"rec12"}                # upgrade
    REPLACE_TREAT: Set[str] = {"rec13"}                # replacement (safety)
    ADV_LABS:      Set[str] = {"rec2", "rec3"}         # advanced labs
    ADV_IMG:       Set[str] = {"rec6"}                 # advanced imaging
    ROUTINE_IMG:   Set[str] = {"rec5"}                 # routine imaging
    BASE_LABS:     Set[str] = {"rec1", "rec4"}         # baseline/routine labs & monitoring
    MEASURE:       Set[str] = {"rec8"}                 # measurement
    CONSULT:       Set[str] = {"rec16", "rec17", "rec18"}  # specialist consultations
    LIFESTYLE:     Set[str] = {"rec19"}                # lifestyle improvement
    ADMIN:         Set[str] = {"rec21"}                # curation/admin

    # ---- Robust helpers (work with list or wide-form recX:True) ----
    def recs(p) -> Set[str]:
        if "recommendations" in p:
            return {str(r).lower() for r in p["recommendations"]}
        return {k.lower() for k, v in p.items() if k.lower().startswith("rec") and v}

    def has_any_of(p, S: Set[str]) -> bool: return bool(recs(p) & S)
    def cnt_any_of(p, S: Set[str]) -> int:  return len(recs(p) & S)

    # Treatment ladder & readiness/diagnostics
    def has_init(p):    return has_any_of(p, INIT_TREAT)
    def has_upg(p):     return has_any_of(p, UPGRADE_TREAT)
    def has_rep(p):     return has_any_of(p, REPLACE_TREAT)
    def has_tx(p):      return has_init(p) or has_upg(p) or has_rep(p)

    def has_fast_dx(p): return has_any_of(p, ADV_LABS | ADV_IMG)         # fast turnaround
    def has_rout_img(p):return has_any_of(p, ROUTINE_IMG)
    def safety_ready(p):return has_any_of(p, BASE_LABS) or has_any_of(p, MEASURE)

    def has_consult(p): return has_any_of(p, CONSULT)
    def has_admin(p):   return has_any_of(p, ADMIN)
    def has_life(p):    return has_any_of(p, LIFESTYLE)

    # Features
    r1, r2 = float(p1.get("risk", 0.0)), float(p2.get("risk", 0.0))
    a1, a2 = float(p1.get("age", 0.0)),  float(p2.get("age", 0.0))
    rec1, rec2 = recs(p1), recs(p2)

    # Flags
    p1_rep, p2_rep = has_rep(p1), has_rep(p2)
    p1_upg, p2_upg = has_upg(p1), has_upg(p2)
    p1_ini, p2_ini = has_init(p1), has_init(p2)
    p1_tx,  p2_tx  = has_tx(p1),   has_tx(p2)

    p1_fast, p2_fast = has_fast_dx(p1), has_fast_dx(p2)
    p1_rimg, p2_rimg = has_rout_img(p1), has_rout_img(p2)
    p1_safe, p2_safe = safety_ready(p1), safety_ready(p2)

    p1_cons, p2_cons = has_consult(p1), has_consult(p2)
    p1_admin, p2_admin = has_admin(p1), has_admin(p2)
    p1_life, p2_life = has_life(p1), has_life(p2)

    # Counts (robust — independent of "types()")
    tct1 = cnt_any_of(p1, INIT_TREAT | UPGRADE_TREAT | REPLACE_TREAT)
    tct2 = cnt_any_of(p2, INIT_TREAT | UPGRADE_TREAT | REPLACE_TREAT)
    ict1 = cnt_any_of(p1, ADV_IMG | ROUTINE_IMG)
    ict2 = cnt_any_of(p2, ADV_IMG | ROUTINE_IMG)
    bct1 = cnt_any_of(p1, BASE_LABS | ADV_LABS)
    bct2 = cnt_any_of(p2, BASE_LABS | ADV_LABS)

    # Scale/age-aware risk gaps
    scale = 40 if max(r1, r2) > 20 else 10
    def gap(x): return abs(r1 - r2) >= x
    def base_close():
        b = 7 if scale == 40 else 3
        if a1 >= 80 or a2 >= 80: b += 1
        elif a1 <= 40 and a2 <= 40: b -= 1
        return b
    huge_gap   = gap(10 if scale == 40 else 4)
    big_gap    = gap(7  if scale == 40 else 3)
    close_risk = abs(r1 - r2) <= base_close()
    very_close = abs(r1 - r2) <= (3 if scale == 40 else 1)

    # Momentum for management (any major forward step)
    mom1 = p1_tx or p1_fast or p1_rimg or p1_cons
    mom2 = p2_tx or p2_fast or p2_rimg or p2_cons

    # Loads for later tie-breaking
    def escalation_load(tct, ict) -> float: return 1.6 * tct + 1.2 * ict
    def readiness_load(bct, safe, meas: bool) -> float:
        return 1.1 * bct + (1.0 if safe else 0.0) + (0.8 if meas else 0.0)

    el1, el2 = escalation_load(tct1, ict1), escalation_load(tct2, ict2)
    rl1, rl2 = readiness_load(bct1, p1_safe, has_any_of(p1, MEASURE)), readiness_load(bct2, p2_safe, has_any_of(p2, MEASURE))

    def rung(rep, upg, ini) -> int: return 3 if rep else 2 if upg else 1 if ini else 0
    rung1, rung2 = rung(p1_rep, p1_upg, p1_ini), rung(p2_rep, p2_upg, p2_ini)

    # Deterministic final tiebreaker
    def final_tiebreak() -> Tuple[str, int]:
        if close_risk and el1 != el2:
            return ("Patient 1", 1) if el1 < el2 else ("Patient 2", 1)
        if rl1 != rl2:
            return ("Patient 1", 1) if rl1 > rl2 else ("Patient 2", 1)
        if r1 != r2:
            return ("Patient 1", 1) if r1 > r2 else ("Patient 2", 1)
        if a1 != a2:
            return ("Patient 1", 1) if a1 < a2 else ("Patient 2", 1)
        rs1, rs2 = int(p1.get("rec_sum", 0)), int(p2.get("rec_sum", 0))
        if rs1 != rs2:
            return ("Patient 1", 1) if rs1 > rs2 else ("Patient 2", 1)
        return ("Patient 1", 1)

    # ===================== Branchy cascade (treatment‑first) =====================

    # A) Replacement (safety) one‑sided — strongest; but needs prerequisites if missing
    if p1_rep != p2_rep:
        # If replacement side lacks safety yet the other is safe and can act fast → prefer the ready side (risks close)
        if p1_rep and not p1_safe and p2_safe and (p2_fast or p2_upg or p2_ini or p2_cons) and close_risk:
            return ("Patient 2", 5)
        if p2_rep and not p2_safe and p1_safe and (p1_fast or p1_upg or p1_ini or p1_cons) and close_risk:
            return ("Patient 1", 5)
        return ("Patient 1", 5) if (p1_rep and close_risk) else ("Patient 2", 5) if (p2_rep and close_risk) \
               else ("Patient 1", 4) if p1_rep else ("Patient 2", 4)

    # B) Upgrade one‑sided (no replacement on either) — stronger with safety/fast‑dx
    if (p1_upg != p2_upg) and not (p1_rep or p2_rep):
        side = "Patient 1" if p1_upg else "Patient 2"
        side_safe = p1_safe if side == "Patient 1" else p2_safe
        side_fast = p1_fast if side == "Patient 1" else p2_fast
        conf = 5 if (close_risk and (side_safe or side_fast)) else 4
        return (side, conf)

    # C) Initiation one‑sided (no higher rung) — prefer if safety present
    if (p1_ini != p2_ini) and not (p1_rep or p2_rep or p1_upg or p2_upg):
        side = "Patient 1" if p1_ini else "Patient 2"
        side_safe = p1_safe if side == "Patient 1" else p2_safe
        return (side, 5 if (close_risk and side_safe) else 4)

    # D) Both on treatment path — safety first; elderly penalty for heavy escalation when close
    if p1_tx and p2_tx:
        if p1_safe != p2_safe:
            return ("Patient 1", 4) if p1_safe else ("Patient 2", 4)
        if (a1 >= 80 or a2 >= 80) and close_risk and el1 != el2:
            return ("Patient 1", 3) if el1 < el2 else ("Patient 2", 3)
        if rung1 != rung2:
            return ("Patient 1", 3) if rung1 > rung2 else ("Patient 2", 3)

    # E) Decisive diagnostics — prefer fast (adv labs/img) over slow; routine imaging weaker
    if p1_fast != p2_fast:
        return ("Patient 1", 4) if p1_fast else ("Patient 2", 4)
    if p1_rimg != p2_rimg:
        # routine imaging only nudges when other has no fast/consult momentum
        other = "Patient 2" if p1_rimg else "Patient 1"
        other_fast = p2_fast if other == "Patient 2" else p1_fast
        other_cons = p2_cons if other == "Patient 2" else p1_cons
        if not (other_fast or other_cons):
            return ("Patient 1", 3) if p1_rimg else ("Patient 2", 3)

    # F) Consultation one‑sided — prefer if it unlocks decisions (tx/fast) or other lacks momentum/safety
    if p1_cons != p2_cons:
        if p1_cons and (p1_tx or p1_fast or (not mom2 or not p2_safe)):
            return ("Patient 1", 4 if close_risk else 3)
        if p2_cons and (p2_tx or p2_fast or (not mom1 or not p1_safe)):
            return ("Patient 2", 4 if close_risk else 3)
        return ("Patient 1", 3) if p1_cons else ("Patient 2", 3)

    # G) Huge-risk override when both have momentum
    if mom1 and mom2 and huge_gap:
        return ("Patient 1", 4) if r1 > r2 else ("Patient 2", 4)

    # H) Treatment ladder depth → treatment count → diagnostic breadth → imaging count
    if rung1 != rung2:
        return ("Patient 1", 3) if rung1 > rung2 else ("Patient 2", 3)
    if tct1 != tct2:
        return ("Patient 1", 3) if tct1 > tct2 else ("Patient 2", 3)
    dx_comp1 = int(bool(recs(p1) & ADV_IMG)) + int(bool(recs(p1) & ROUTINE_IMG)) + int(bool(recs(p1) & ADV_LABS))
    dx_comp2 = int(bool(recs(p2) & ADV_IMG)) + int(bool(recs(p2) & ROUTINE_IMG)) + int(bool(recs(p2) & ADV_LABS))
    if dx_comp1 != dx_comp2:
        return ("Patient 1", 3) if dx_comp1 > dx_comp2 else ("Patient 2", 3)
    if ict1 != ict2:
        return ("Patient 1", 2) if ict1 > ict2 else ("Patient 2", 2)

    # I) Nonlinear patterns (speed/feasibility): young+high risk+fast/upgrade; elderly+slow without readiness
    young_hi_1 = (a1 <= 45 and r1 >= (20 if scale == 40 else 8) and (p1_upg or p1_fast))
    young_hi_2 = (a2 <= 45 and r2 >= (20 if scale == 40 else 8) and (p2_upg or p2_fast))
    if young_hi_1 != young_hi_2:
        return ("Patient 1", 3) if young_hi_1 else ("Patient 2", 3)

    elder_slow_1 = (a1 >= 80 and not p1_fast and not p1_safe and p1_rimg)  # slowish pathway
    elder_slow_2 = (a2 >= 80 and not p2_fast and not p2_safe and p2_rimg)
    if close_risk and (elder_slow_1 != elder_slow_2):
        return ("Patient 1", 3) if elder_slow_1 else ("Patient 2", 3)

    # J) Readiness advantage in close risk
    if close_risk and (p1_safe != p2_safe):
        return ("Patient 1", 2) if p1_safe else ("Patient 2", 2)

    # K) Admin only if immediately required and the other lacks momentum/safety
    if p1_admin != p2_admin:
        if p1_admin and not (mom2 or p2_safe):
            return ("Patient 1", 2)
        if p2_admin and not (mom1 or p1_safe):
            return ("Patient 2", 2)

    # L) Remaining ties → big gap: risk; very close: age; else deterministic tiebreak
    if big_gap:
        return ("Patient 1", 2) if r1 > r2 else ("Patient 2", 2)
    if very_close and a1 != a2:
        return ("Patient 1", 2) if a1 < a2 else ("Patient 2", 2)
    return final_tiebreak()







###############################################################################
# 4. DOCTORS 6..10  –  “formula” style (same API as the trees)
###############################################################################

# -----------------------------------------------------------
# helpers  (shared by all five functions)
# -----------------------------------------------------------
REC_PREFIX = "rec"

def _recs(patient):
    """
    Return a set of 'recN' strings – works with either:
        • patient['recommendations']  → list of "recX"
        • patient['recommendations_num'] → list/tuple of 0‑1 flags (index 0 == rec1)
    """
    if "recommendations" in patient and patient["recommendations"]:
        return {str(r).lower() for r in patient["recommendations"]}
    if "recommendations_num" in patient:
        flags = patient["recommendations_num"]
        return {f"{REC_PREFIX}{i+1}" for i, f in enumerate(flags) if f}
    # wide form recX:True
    return {k.lower() for k, v in patient.items()
            if k.lower().startswith(REC_PREFIX) and v}

# --- Recommendation groups (same sets used בכל העצים) -----------------------
TREAT_START   = {"rec9",  "rec10", "rec11"}          # initiate / start
TREAT_UPGRADE = {"rec12", "rec14"}                   # intensify / upgrade
TREAT_REPLACE = {"rec13", "rec15"}                   # replacement (safety)
INVASIVE_TX   = TREAT_START | TREAT_UPGRADE | TREAT_REPLACE | {"rec12", "rec14", "rec15"}

ADV_IMG       = {"rec6"}                             # advanced imaging
DIAG_PROC     = {"rec7"}                             # diagnostic procedure
ADV_LABS      = {"rec2", "rec3"}                     # advanced / targeted labs
ROUT_IMG      = {"rec5"}                             # routine imaging
BASIC_TESTS   = {"rec1", "rec4"}                     # basic / baseline panels
MEASURES      = {"rec8"}                             # key measurements
CONSULTS      = {"rec16", "rec17", "rec18"}          # specialist consults
LIFESTYLE     = {"rec19", "rec20"}                   # improve / stop harmful habits

PREVENT_SET   = {"rec9", "rec19", "rec20"}           # preventive treatment start + lifestyle
MONITOR_SET   = BASIC_TESTS | ROUT_IMG | {"rec18"}   # early detection / monitoring

# Safety‑ready pre‑requisite
def safety_ready(recs): return bool(recs & (BASIC_TESTS | MEASURES))

# -----------------------------------------------------------
# Doctor 6  –  Near‑Term Risk Reducer (Actionability‑First)
# -----------------------------------------------------------

def doctor_decision_tree_6(p1, p2):
    # fast decisive win when exactly one patient has **no** recommendations
    if p1["rec_sum"] == 0 and p2["rec_sum"] > 0:
        return ("Patient 2", 5)
    if p2["rec_sum"] == 0 and p1["rec_sum"] > 0:
        return ("Patient 1", 5)

    # ---------- inner scorer -------------------------------------------------
    def score_actionability(pt):
        r     = _recs(pt)
        risk  = float(pt["risk"])
        age   = float(pt["age"])
        score = 0.0

        # (1) tiered raw‑risk anchor  –  nonlinear
        if   risk >= 30: score += 15
        elif risk >= 20: score += 10
        elif risk >= 10: score +=  6
        else:            score +=  2* (risk / 10)

        # (2) treatment ladder (replacement > upgrade > initiation)
        if r & TREAT_REPLACE: score += 12
        if r & TREAT_UPGRADE: score +=  9
        if r & TREAT_START:   score +=  6

        # (3) synergy: treatment + (decisive dx OR consult) – near‑term plan builder
        if (r & INVASIVE_TX) and (r & (ADV_LABS | ADV_IMG | CONSULTS)):
            score += 5

        # (4) device / procedure starts
        if r & DIAG_PROC:
            score += 4

        # (5) safety readiness  (baseline labs / measurement)
        if safety_ready(r):
            score += 3
        elif r & INVASIVE_TX:            # escalation without prerequisites → penalty
            score -= 6

        # (6) age interaction (time‑to‑benefit / reversibility)
        if age > 80:
            score -= 8
        elif age > 70:
            score -= 4
        elif age < 40 and risk >= 15:    # very young + sizeable risk
            score += 4

        # (7) immediacy kicker: very high risk *and* escalation now
        if risk >= 25 and (r & INVASIVE_TX):
            score += 0.5 * (risk - 25)

        return score

    s1, s2 = score_actionability(p1), score_actionability(p2)

    # ---------- convert score gap → confidence -------------------------------
    gap = abs(s1 - s2)
    if gap >= 12:
        conf = 5
    elif gap >= 7:
        conf = 4
    elif gap >= 3:
        conf = 3
    else:
        conf = 2

    if s1 > s2:
        return ("Patient 1", conf)
    elif s2 > s1:
        return ("Patient 2", conf)

    # perfect tie → pick higher raw‑risk; still deterministic
    if p1["risk"] != p2["risk"]:
        return ("Patient 1", 2) if p1["risk"] > p2["risk"] else ("Patient 2", 2)
    return (random.choice(["Patient 1", "Patient 2"]), 1)

# -----------------------------------------------------------
# Doctor 7  –  Prevention & Lifestyle Optimizer (Long‑Horizon)
# -----------------------------------------------------------
def doctor_decision_tree_7(p1, p2):
    if p1["rec_sum"] == 0 and p2["rec_sum"] > 0:  return ("Patient 2", 5)
    if p2["rec_sum"] == 0 and p1["rec_sum"] > 0:  return ("Patient 1", 5)

    def score_prevention(p):
        r = _recs(p)
        age, risk = p["age"], p["risk"]
        s = 0
        # preventive initiation / lifestyle / monitoring
        s += 6 * bool(r & PREVENT_SET)
        s += 4 * bool(r & LIFESTYLE)
        s += 3 * bool(r & MONITOR_SET)
        # reward bundle synergies
        if r & LIFESTYLE and safety_ready(r): s += 3
        # penalize invasive unless risk high
        if r & INVASIVE_TX:
            s -= 4
            if risk >= 20: s += 4   # cancel penalty if risk high
        # age weighting (younger more years to benefit)
        s += max(0, 40 - age) * 0.2
        # subtle risk bonus
        s += 0.2 * risk
        return s

    s1, s2 = score_prevention(p1), score_prevention(p2)
    if abs(s1 - s2) >= 10: return ("Patient 1", 5) if s1 > s2 else ("Patient 2", 5)
    if abs(s1 - s2) >= 5:  return ("Patient 1", 4) if s1 > s2 else ("Patient 2", 4)
    if s1 != s2:
        return ("Patient 1", 3) if s1 > s2 else ("Patient 2", 3)
    return (random.choice(["Patient 1", "Patient 2"]), 1)

# -----------------------------------------------------------
# Doctor 8  –  Diagnostic‑Clarity Seeker
# -----------------------------------------------------------
def doctor_decision_tree_8(p1, p2):
    if p1["rec_sum"] == 0 and p2["rec_sum"] > 0:  return ("Patient 2", 5)
    if p2["rec_sum"] == 0 and p1["rec_sum"] > 0:  return ("Patient 1", 5)

    def score_clarity(p):
        r = _recs(p)
        risk = p["risk"]
        s = 0
        # high‑information diagnostics
        s += 8 * bool(r & (ADV_IMG | DIAG_PROC | ADV_LABS))
        # specialist consult that resolves complexity
        s += 4 * bool(r & CONSULTS)
        # routine imaging/labs if they unlock treatment start (proxy: +monitor+inv)
        if r & ROUT_IMG and (r & INVASIVE_TX):
            s += 3
        # penalize low‑info tests without follow‑up
        if (r & ROUT_IMG) and not (r & (TREAT_START | TREAT_UPGRADE)):
            s -= 1
        # risk bonus but smaller
        s += 0.25 * risk
        return s

    s1, s2 = score_clarity(p1), score_clarity(p2)
    if abs(s1 - s2) >= 6: return ("Patient 1", 5) if s1 > s2 else ("Patient 2", 5)
    if abs(s1 - s2) >= 3: return ("Patient 1", 4) if s1 > s2 else ("Patient 2", 4)
    if s1 != s2:
        return ("Patient 1", 3) if s1 > s2 else ("Patient 2", 3)
    return (random.choice(["Patient 1", "Patient 2"]), 1)

# -----------------------------------------------------------
# Doctor 9  –  Safety‑First Minimalist
# -----------------------------------------------------------
def doctor_decision_tree_9(p1, p2):
    if p1["rec_sum"] == 0 and p2["rec_sum"] > 0:  return ("Patient 2", 5)
    if p2["rec_sum"] == 0 and p1["rec_sum"] > 0:  return ("Patient 1", 5)

    def score_minimalist(p):
        r = _recs(p)
        age, risk = p["age"], p["risk"]
        s = 0
        # reversible / low‑intensity steps
        s += 6 * bool(r & (BASIC_TESTS | MEASURES))
        s += 5 * bool(r & LIFESTYLE)
        s += 4 * bool(r & MONITOR_SET)
        # treatment replacement as safety action
        s += 7 * bool(r & TREAT_REPLACE)
        # penalize escalation
        if r & INVASIVE_TX:
            s -= 5
        if r & (ADV_IMG | DIAG_PROC):
            s -= 3
        # risk still matters
        s += 0.3 * risk
        # older → lean even more minimalist
        s += (88 - age) * 0.1
        return s

    s1, s2 = score_minimalist(p1), score_minimalist(p2)
    if abs(s1 - s2) >= 7: return ("Patient 1", 5) if s1 > s2 else ("Patient 2", 5)
    if abs(s1 - s2) >= 3: return ("Patient 1", 4) if s1 > s2 else ("Patient 2", 4)
    if s1 != s2:
        return ("Patient 1", 3) if s1 > s2 else ("Patient 2", 3)
    return (random.choice(["Patient 1", "Patient 2"]), 1)

# -----------------------------------------------------------
# Doctor 10  –  Momentum Builder (Plan‑Enabling Steward)
# -----------------------------------------------------------
def doctor_decision_tree_10(p1, p2):
    def score_momentum(p):
        r = _recs(p)
        risk, age = p["risk"], p["age"]
        s = 0
        # treatment ladder
        s += 9 * bool(r & TREAT_REPLACE)
        s += 7 * bool(r & TREAT_UPGRADE)
        s += 6 * bool(r & TREAT_START)
        # consults that change management
        s += 5 * bool(r & CONSULTS)
        # decisive diagnostics with near‑term readout
        s += 4 * bool(r & (ADV_LABS | ADV_IMG))
        # prerequisite readiness
        if safety_ready(r):
            s += 3
        # admin that immediately enables
        s += 2 * bool("rec21" in r)
        # risk weight
        s += 0.35 * risk
        # mild penalty if prerequisites missing but escalation present
        if (r & INVASIVE_TX) and not safety_ready(r):
            s -= 4
        # timeliness: older pts lose small amount for long‑horizon plans
        s -= 0.05 * age
        return s

    s1, s2 = score_momentum(p1), score_momentum(p2)
    if abs(s1 - s2) >= 10: return ("Patient 1", 5) if s1 > s2 else ("Patient 2", 5)
    if abs(s1 - s2) >= 5:  return ("Patient 1", 4) if s1 > s2 else ("Patient 2", 4)
    if s1 != s2:
        return ("Patient 1", 3) if s1 > s2 else ("Patient 2", 3)
    return (random.choice(["Patient 1", "Patient 2"]), 1)



    