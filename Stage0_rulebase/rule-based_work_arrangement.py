import pandas as pd
import re

# Test set
test_input_file_path = "./job_data_files/raw-data/work_arrangements_test_set.csv"
output_file_path = "./job_data_files/df_output_work_rule_based_testset.csv"


df_work = pd.read_csv(test_input_file_path)


signal_phrases = {
    'remote': [
        r'\bremote\b',
        r'\bwork from home\b',
        r'\bwfh\b',
        r'\btelecommut(e|ing)\b',
        r'\bvirtual\b'
    ],
    'onsite': [
        r'\bin person\b',
        r'\bon[ -]?site\b',
        r'\bat our offices?\b',
        r'\bon campus\b'
    ],
    'hybrid': [
        r'\bhybrid\b',
        r'\bflexible\b',
        r'\bmixed[- ]mode\b',
    ]
}

def detect_label(text):
    t = text.lower()
    if any(re.search(p, t) for p in signal_phrases['remote']) and \
       any(re.search(p, t) for p in signal_phrases['onsite']):
        return 'Hybrid'
    if any(re.search(p, t) for p in signal_phrases['hybrid']):
        return 'Hybrid'
    if any(re.search(p, t) for p in signal_phrases['remote']):
        return 'Remote'
    return 'Onsite'




df_work['predicted'] = df_work['job_ad'].apply(detect_label)
accuracy_dev = (df_work['predicted'] == df_work['y_true']).mean()
print(f"Accuracy: {accuracy_dev:.2%}")

# Keep only certain columns
df_work = df_work.rename(columns={'predicted': 'y_pred'})

df_work = df_work[['y_true', 'y_pred']]

df_work.to_csv(output_file_path, index=False)


# df_work_test['predicted'] = df_work_test['job_ad'].apply(classify_job_ad)
# accuracy_test = (df_work_test['predicted'] == df_work_test['y_true']).mean()
# print(f" Accuracy: {accuracy_test:.2%}")




