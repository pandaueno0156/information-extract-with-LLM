import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import re
import os

def extract_salary_format(text):
    match = re.search(r"\d{3,6}-\d{3,6}-[A-Z]{3,4}-[A-Z]+", text)
    return match.group(0) if match else "0-0-None-None"


def parse_salary_string(y_pred):
    result = {
        'min_salary': None,
        'max_salary': None,
        'currency': None,
        'frequency': None
    }

    if isinstance(y_pred, str) and '-' in y_pred:
        parts = [p.strip() for p in y_pred.split('-')]

        if len(parts) > 0:
            result['min_salary'] = parts[0]
        if len(parts) > 1:
            result['max_salary'] = parts[1]
        if len(parts) > 2:
            result['currency'] = parts[2]
        if len(parts) > 3:
            result['frequency'] = parts[3]

    return pd.Series(result)


def eval_salary(df):


    df = df.iloc[:567].copy()

    df['parsed_pred'] = df['y_pred'].apply(extract_salary_format)
    df[['min_salary_pred', 'max_salary_pred', 'currency_pred', 'frequency_pred']] = df['parsed_pred'].apply(parse_salary_string)

    df[['min_salary_true', 'max_salary_true', 'currency_true', 'frequency_true']] = df['complete'].apply(parse_salary_string)

    total = len(df)
    acc_min = (df['min_salary_pred'] == df['min_salary_true']).sum() / total
    acc_max = (df['max_salary_pred'] == df['max_salary_true']).sum() / total
    acc_currency = (df['currency_pred'] == df['currency_true']).sum() / total
    acc_freq = (df['frequency_pred'] == df['frequency_true']).sum() / total
    avg_acc = (acc_min + acc_max + acc_currency + acc_freq) / 4

    print('\n----- Salary Format Evaluation -----')
    print(f"Min Salary Accuracy: {acc_min:.2%}")
    print(f"Max Salary Accuracy: {acc_max:.2%}")
    print(f"Currency Accuracy: {acc_currency:.2%}")
    print(f"Frequency Accuracy: {acc_freq:.2%}")
    print(f"Overall Average Accuracy: {avg_acc:.2%}")
    print('---------------------------------------\n')

def get_task_type(prompt):
    if '[TASK: Salary]' in prompt:
        return 'Salary'
    elif '[TASK: Seniority]' in prompt:
        return 'Seniority'
    elif '[TASK: Work Arrangement]' in prompt:
        return 'Work Arrangement'
    else:
        return 'Unknown'

def eval_data(b):

    y_true = b['complete']
    y_pred = b['y_pred']

    b['val'] = y_true == y_pred

    b['task_type'] = b['prompt'].apply(get_task_type)

    for task in b['task_type'].unique():
        if task != 'Unknown':
            task_data = b[b['task_type'] == task]
            correct = sum(task_data['val'])
            total = len(task_data)
            print(f'{task}: {correct} / {total} - {correct/total * 100:.2f}%')
    print()
    print('-' * 8, 'sklearn metrics', '-' * 8)
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")


def main_eval(path_):
    title = path_.split('\\')[-1]
    print('\n', '='*20, title, '='*20)
    df = pd.read_json(path_)
    if 'y_true' in df.columns:
        df = df.rename(columns={'y_true': 'complete'})
    if 'y_predicted' in df.columns:
        df = df.rename(columns={'y_predicted': 'y_pred'})
    eval_data(df)
    eval_salary(df)


if __name__ == '__main__':

    # Please adjust the folder path to run the testing json files
    folder_path = r'results_collection/general/'
    # folder_path = r'results_collection/langchain/'
    # folder_path = r'results_collection/autogen/'

    files = [folder_path+f for f in os.listdir(folder_path) if f.endswith('.json')]
    for i in files:
        print(f'\nFile name: {i}')
        main_eval(i)

