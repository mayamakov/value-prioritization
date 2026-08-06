# model_type ="ranknet"
model_type ="ranknet"
CPU_CORES = 250 
batch_size = 64
# paths
rankings_file = "doctor_rankings.pkl"
patient_df_path = "synthetic_data (7).xlsx" 
common_pairs_dict_file = 'common_pairs_dict.pkl'
initial_pairs_path = 'initial_pairs.pkl'

# THRESHOLD CONSTANTS for a 0..60 risk scale
THRESH_05 = 3.0   # 0.05 * 60
THRESH_1  = 6.0   # 0.1  * 60
THRESH_25 = 15.0  # 0.25 * 60
THRESH_4  = 24.0  # 0.4  * 60
THRESH_5  = 30.0  # 0.5  * 60
THRESH_7  = 42.0  # 0.7  * 60
THRESH_8  = 48.0  # 0.8  * 60
THRESH_9  = 54.0  # 0.9  * 60

# BASELINE SCORES (used by Doctors 6–10)
baseline_scores = [
    2,   # index 0  -> rec1
    4,   # index 1  -> rec2
    5,   # index 2  -> rec3
    3,   # index 3  -> rec4
    3,   # index 4  -> rec5
    3,   # index 5  -> rec6
    4,   # index 6  -> rec7
    3,   # index 7  -> rec8
    2,   # index 8  -> rec9
    7,   # index 9  -> rec10
    10,  # index 10 -> rec11
    12,  # index 11 -> rec12
    12,  # index 12 -> rec13
    20,  # index 13 -> rec14
    12,  # index 14 -> rec15
    14,  # index 15 -> rec16
    7,   # index 16 -> rec17
    5,   # index 17 -> rec18
    3,   # index 18 -> rec19
    2,   # index 19 -> rec20
    2,   # index 20 -> rec21
]
