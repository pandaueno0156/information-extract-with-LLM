import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import argparse


def preprocess_compensation_finder(row):
    job_detail = row['job_ad_details']
    nation_desc = row['nation_short_desc']

    soup = BeautifulSoup(job_detail, "html.parser")

    compensation_text = soup.find(string=lambda t: t.strip() == "Compensation")
    compensation = compensation_text.find_next("p").text.strip() if compensation_text else None

    compensation_number = re.findall(r'\d[\d,]*', str(compensation))
    comp_numbers = [int(num.replace(',', '')) for num in compensation_number]

    compensation_range_text = soup.find(string=lambda t: t.strip() == "Compensation Range")
    comp_range = compensation_range_text.find_next("p").text.strip() if compensation_range_text else None

    numbers = re.findall(r'\d[\d,]*', str(comp_range))
    numbers = [int(num.replace(',', '')) for num in numbers]
    sorted_numbers = sorted(numbers)

    if nation_desc == 'PH':
        if comp_numbers:
            min_sal_1 = max_sal_1 = round(comp_numbers[0])
        elif sorted_numbers:
            min_sal_1 = max_sal_1 = round(sorted_numbers[0]) if len(sorted_numbers) == 1 else (round(sorted_numbers[0]), round(sorted_numbers[1]))
        else:
            min_sal_1 = max_sal_1 = None
    else:
        if sorted_numbers:
            min_sal_1 = max_sal_1 = round(sorted_numbers[0]) if len(sorted_numbers) == 1 else (round(sorted_numbers[0]), round(sorted_numbers[1]))
        elif compensation:
            min_sal_1 = max_sal_1 = round(int(compensation))
        else:
            min_sal_1 = max_sal_1 = None

    return pd.Series([min_sal_1, max_sal_1])


def make_lowercase(text):
    return str(text).lower()


payment_keyword = {
    'WEEKLY': ['per week'],
    'ANNUAL': ['per year', 'per annum', 'p.a.', 'yearly', 'annually'],
    'HOURLY': ['per hour', 'p.h.', 'ph', 'p/hr', 'hourly', 'hr'],
    'DAILY': ['per day'],
    'MONTHLY': ['per month', 'p.m.', 'monthly']
}


def preprocess_payment_frequency(text):
    for frequency, keywords in payment_keyword.items():
        for keyword in keywords:
            if keyword in text:
                return frequency
    return None


def preprocess_find_salary_range(text):
    if not text or pd.isna(text):
        return pd.Series([None, None])

    pattern = r'\$?(\d{1,7}(?:,\d{3})*(?:\.\d+)?%?)'
    matches = re.findall(pattern, text)

    clean_numbers = []
    for match in matches:
        if '%' in match:
            continue
        match = match.replace('$', '').replace(',', '')
        try:
            number = round(float(match))
            if number >= 10:
                clean_numbers.append(number)
        except ValueError:
            pass

    if clean_numbers:
        return pd.Series([min(clean_numbers), max(clean_numbers)])
    else:
        return pd.Series([None, None])


def preprocessing_payment_freq3(row):
    nation_desc, min_sal1, min_sal2, max_sal1, max_sal2, freq1, freq2 = (
        row['nation_short_desc'], row['min_sal_1'], row['min_sal_2'], row['max_sal_1'], row['max_sal_2'], row['freq_1'], row['freq_2']
    )

    if nation_desc == 'AUS':
        if ((pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2))) and not pd.notna(freq2):
            return freq2 or freq1 or 'ANNUAL'
        else:
            return 'ANNUAL'
    elif nation_desc == 'ID':
        return 'MONTHLY'
    elif nation_desc in ['MY', 'NZ', 'PH']:
        if (pd.notna(min_sal1) or pd.notna(min_sal2)) and not freq2:
            return 'MONTHLY'
        return freq2 or freq1
    else:
        return freq2 or freq1


currency_dict = {
    'AUS': 'AUD', 'SG': 'SGD', 'HK': 'HKD', 'ID': 'IDR',
    'TH': 'THB', 'NZ': 'NZD', 'MY': 'MYR', 'PH': 'PHP'
}


def preprocessing_min_max_currency(row):
    min_sal1, min_sal2, max_sal1, max_sal2, nation_desc = (
        row['min_sal_1'], row['min_sal_2'], row['max_sal_1'], row['max_sal_2'], row['nation_short_desc']
    )
    currency = currency_dict.get(nation_desc, 'None')

    if pd.notna(min_sal2) and pd.notna(max_sal2):
        return pd.Series([min_sal2, max_sal2, currency])
    elif pd.notna(min_sal1) and pd.notna(max_sal1):
        return pd.Series([min_sal1, max_sal1, currency])
    else:
        return pd.Series([0, 0, "None"])


def y_prediction(row):
    min_sal3, max_sal3, freq3, currency = row['min_sal_3'], row['max_sal_3'], row['freq_3'], row['currency']
    if min_sal3 != 0 and max_sal3 != 0:
        return f"{int(min_sal3)}-{int(max_sal3)}-{currency}-{freq3}"
    return "0-0-None-None"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output_path', type=str, required=True, help='Output CSV file path')
    args = parser.parse_args()

    df_sal = pd.read_csv(args.input_path)

    df_sal[['min_sal_1', 'max_sal_1']] = df_sal.apply(preprocess_compensation_finder, axis=1)
    df_sal[['min_sal_1', 'max_sal_1']] = df_sal[['min_sal_1', 'max_sal_1']].astype('Int64')

    df_sal['cleaned_job_ad_details'] = df_sal['job_ad_details'].apply(make_lowercase)
    df_sal['cleaned_salary_additional_text'] = df_sal['salary_additional_text'].apply(make_lowercase)

    df_sal['freq_1'] = df_sal['cleaned_job_ad_details'].apply(preprocess_payment_frequency)
    df_sal[['min_sal_2', 'max_sal_2']] = df_sal['cleaned_salary_additional_text'].apply(preprocess_find_salary_range)
    df_sal['freq_2'] = df_sal['cleaned_salary_additional_text'].apply(preprocess_payment_frequency)

    df_sal['freq_3'] = df_sal.apply(preprocessing_payment_freq3, axis=1)
    df_sal[['min_sal_3', 'max_sal_3', 'currency']] = df_sal.apply(preprocessing_min_max_currency, axis=1)

    df_sal['y_pred'] = df_sal.apply(y_prediction, axis=1)

    df_sal = df_sal[['y_true', 'y_pred']]
    df_sal.to_csv(args.output_path, index=False)

    accuracy = (df_sal['y_true'] == df_sal['y_pred']).mean()
    print(f'Accuracy: {args.input_path} - {accuracy:.2%}')
