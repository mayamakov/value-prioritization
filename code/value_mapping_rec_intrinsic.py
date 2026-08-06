# -*- coding: utf-8 -*-
"""value_mapping_rec_intrinsic.py - SHIM (single source of truth = v10).
Was a separate v9 mapping; now re-exports value_mapping_v10 so the pipeline
has ONE mapping. Edit the mapping table in paper_value_config_v10.py.
"""
from value_mapping_v10 import *  # noqa
from value_mapping_v10 import (
    ETHICAL_DIMENSIONS, REC_INTRINSIC_MAPPING,
    patient_to_values, aggregate_shap_to_ethical_values,
    normalize_profiles, make_pricebased_justice, priority_to_need,
    # v11 support-weighting (item 1):
    compute_support_weights, set_support_weights, load_support_weights,
)