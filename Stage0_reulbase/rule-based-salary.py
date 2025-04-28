import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

# Train Set
input_file_path = "./job_data_files/raw-data/salary_labelled_test_set.csv"
output_file_path = "./job_data_files/df_output_salary_rule_based_testset.csv"

# Test Set
# input_file_path = "./job_data_files/salary_labelled_test_set.csv"
# output_file_path = "./job_data_files/adjusted_salary_labelled_test_set.csv"


df_sal = pd.read_csv(input_file_path)

def preprocess_compensation_finder(row):
    job_detail = row['job_ad_details']
    nation_desc = row['nation_short_desc']

    soup = BeautifulSoup(job_detail, "html.parser")

    compensation_text = soup.find(string = lambda t: t.strip() == "Compensation")
    compensation = compensation_text.find_next("p").text.strip() if compensation_text else None

    compensation_number = re.findall(r'\d[\d,]*', str(compensation))
    comp_numbers = [int(num.replace(',', '')) for num in compensation_number]


    compensation_range_text = soup.find(string = lambda t: t.strip() == "Compensation Range")
    comp_range = compensation_range_text.find_next("p").text.strip() if compensation_range_text else None
    # print(f'comp_range: {comp_range}')

    # convert this comp_range to a list
    numbers = re.findall(r'\d[\d,]*', str(comp_range))

    # convert to integers(remove comma)
    numbers = [int(num.replace(',', '')) for num in numbers]
    sorted_numbers = sorted(numbers)

    # Couple senarios
    if nation_desc == 'PH':
        # Use comp_range is not accurate for PH
        # Use compensation as min, max instead
        if comp_numbers:
            min_sal_1 = round(comp_numbers[0])
            max_sal_1 = round(comp_numbers[0])
        elif not comp_numbers and sorted_numbers:
            # no compensation value
            if len(sorted_numbers) >= 2:
                min_sal_1 = round(sorted_numbers[0])
                max_sal_1 = round(sorted_numbers[1])
            if len(sorted_numbers) == 1:
                min_sal_1 = round(sorted_numbers[0])
                max_sal_1 = round(sorted_numbers[0])
        elif not comp_numbers and not sorted_numbers:
            min_sal_1 = None
            max_sal_1 = None
    elif nation_desc != "PH":
        # All other countries
        if compensation and sorted_numbers:
            # If comp_range exists and compensation value exists, we take comp_range
            if len(sorted_numbers) >= 2:
                min_sal_1 = round(sorted_numbers[0])
                max_sal_1 = round(sorted_numbers[1])
            if len(sorted_numbers) == 1:
                min_sal_1 = round(sorted_numbers[0])
                max_sal_1 = round(sorted_numbers[0])
        elif not compensation and sorted_numbers:
            # If comp_range exists and compensation value does not exist, we take comp_range
            if len(sorted_numbers) >= 2:
                min_sal_1 = round(sorted_numbers[0])
                max_sal_1 = round(sorted_numbers[1])
            if len(sorted_numbers) == 1:
                min_sal_1 = round(sorted_numbers[0])
                max_sal_1 = round(sorted_numbers[0])
        elif not sorted_numbers and compensation:
            # If comp_range does not exist and compensation value does exist, we take compensation for both min_sal_1 and max_sal_1
            min_sal_1 = round(compensation)
            max_sal_1 = round(compensation)
        elif not sorted_numbers and not compensation:
            # If both comp_range  and compensation value does not exist, min_sal_1 and max_sal_1 == None
            min_sal_1 = None
            max_sal_1 = None

    return pd.Series([min_sal_1, max_sal_1])

df_sal[['min_sal_1', 'max_sal_1']] = df_sal.apply(preprocess_compensation_finder, axis=1)
# To get rid of floating points
df_sal[['min_sal_1', 'max_sal_1']] = df_sal[['min_sal_1', 'max_sal_1']].astype('Int64')

# Cleaned salary addtional column for search
def make_lowercase(column_data):
    lowercase_data = str(column_data).lower()

    return lowercase_data


df_sal["cleaned_job_ad_details"] = df_sal['job_ad_details'].apply(make_lowercase)

df_sal["cleaned_salary_additional_text"] = df_sal["salary_additional_text"].apply(make_lowercase)

payment_keyword = {
    'WEEKLY': ['per week'],
    'ANNUAL' : ['per year', 'per annum', 'p.a.', 'yearly', 'annually'],
    'HOURLY': ['per hour', 'p.h.', 'ph', 'p/hr', 'hourly', 'hr'],
    'DAILY': ['per day'],
    'MONTHLY': ['per month', 'p.m.', 'monthly']
}

def preprocess_payment_frquency_finder(column_data):
    # Search translated_job_ad_detail to identify payment frequency
    for frequency, keywords in payment_keyword.items():
        for keyword in keywords:
            if keyword in column_data:
                return frequency
    return None

df_sal['freq_1'] = df_sal['cleaned_job_ad_details'].apply(preprocess_payment_frquency_finder)

def preprocess_find_salary_range(column_data):
    
    if not column_data or pd.isna(column_data):
        return pd.Series([None, None])  # Always return two values: (None, None)
        
    # Pattern for dollar amounts, percentages, and numbers
    pattern = r'\$?(\d{1,7}(?:,\d{3})*(?:\.\d+)?%?)'

    # Find all valid matches
    matches = re.findall(pattern, column_data)
    
    clean_numbers = []

    # Means there are no matches
    if not matches:
        return pd.Series([None, None])  # Ensure we always return two values
    
    # Process each match
    for match in matches:
        if '%' in match:
            continue  # Skip numbers with percentage
        match = match.replace('$', '').replace(',', '')
        try:
            number = round(float(match))
            if number >= 10:
                clean_numbers.append(number)
        except ValueError:
            pass
    
    if len(clean_numbers) > 1:
        min_sal_2 = min(clean_numbers)
        max_sal_2 = max(clean_numbers)
    elif len(clean_numbers) == 1:
        min_sal_2 = clean_numbers[0]
        max_sal_2 = clean_numbers[0]
    else:
        return pd.Series([None, None]) # Ensure we always return two values if no valid numbers
    
    return pd.Series([min_sal_2, max_sal_2])  # Always return two values

# Apply the function and assign the result to 'min_sal_2' and 'max_sal_2'
df_sal[['min_sal_2', 'max_sal_2']] = df_sal['cleaned_salary_additional_text'].apply(preprocess_find_salary_range)
df_sal['freq_2'] = df_sal['cleaned_salary_additional_text'].apply(preprocess_payment_frquency_finder)


# Final payment frequency prediction using freq3 predition based on statistics of data
def preprocessing_payment_freq3(row):
    nation_desc = row['nation_short_desc']
    min_sal1 = row['min_sal_1']
    min_sal2 = row['min_sal_2']
    max_sal1 = row['max_sal_1']
    max_sal2 = row['max_sal_2']
    freq1 = row['freq_1']
    freq2 = row['freq_2']

    if nation_desc == 'AUS':
        # We assume all salary in AUS to be annual if it has values and no freq2
        if (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and not pd.notna(freq2):
            return freq2
        elif (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and not pd.notna(freq1):
            return freq1
        else:
            return 'ANNUAL'
    elif nation_desc == 'ID':
        # We assume all salary in ID to be monthly if it has values
        if (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)):
            return 'MONTHLY'
    elif nation_desc == 'MY':
        # We assume all salary in MY to be monthly if no fre2 but with values
        if (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and not freq2:
            return 'MONTHLY'
        else:
            if freq2:
                return freq2
            elif freq1:
                return freq1
            else:
                return None
    elif nation_desc == 'NZ':
        # We assume all salary in MY to be monthly if no fre2 but with values
        if (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and ((not freq2) or (not freq1)):
            return 'MONTHLY'
        else:
            if freq2:
                return freq2
            elif freq1:
                return freq1
            else:
                return None
    elif nation_desc == 'PH':
        # We assume all salary in PH to be monthly if no fre2 but with values
        # PH only has Monthly or Daily for frequency payment
        if (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and ((not freq2) or (not freq1)):
            # We assume if the min_val is greater than 10000
            if pd.notna(min_sal1) and pd.notna(min_sal2):
                if min(min_sal1, min_sal2) > 10000:
                    # it is monthly
                    return 'MONTHLY'
                else:
                    # otherwise it is daily
                    return 'DAILY'
            elif pd.notna(min_sal1) and not pd.notna(min_sal2):
                if min_sal1 > 10000:
                    # it is monthly
                    return 'MONTHLY'
                else:
                    # otherwise it is daily
                    return 'DAILY'
            elif not pd.notna(min_sal1) and pd.notna(min_sal2):
                if min_sal2 > 10000:
                    # it is monthly
                    return 'MONTHLY'
                else:
                    # otherwise it is daily
                    return 'DAILY'
        else:
            if freq2:
                return freq2
            elif freq1:
                return freq1
            else:
                return None   
    elif nation_desc == 'HK':
        # We assume all salary in HK as freq2 if values are available
        if (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and freq2:
            return freq2
        elif (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and freq1:
            return freq1
        else:
            None
    elif nation_desc == 'SG':
        # We assume all salary in SG as freq2 if values are available
        if (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and freq2:
            return freq2
        elif (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and freq1:
            return freq1
        else:
            None            
    elif nation_desc == 'TH':
        # We assume all salary in TH as freq2 if values are available
        if (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and freq2:
            return freq2
        elif (pd.notna(min_sal1) and pd.notna(max_sal1)) or (pd.notna(min_sal2) and pd.notna(max_sal2)) and freq1:
            return freq1
        else:
            None    

df_sal['freq_3'] = df_sal.apply(preprocessing_payment_freq3, axis=1)
currency_dict = {'AUS': 'AUD', 
            'SG': 'SGD', 
            'HK': 'HKD', 
            'ID': 'IDR', 
            'TH': 'THB', 
            'NZ': 'NZD', 
            'MY': 'MYR', 
            'PH': 'PHP'}

# Final min_sal_3 and max_sal_3 and currency are used for prediction
def preprocessing_min_max_currency(row):
    min_sal1 = row['min_sal_1']
    min_sal2 = row['min_sal_2']
    max_sal1 = row['max_sal_1']
    max_sal2 = row['max_sal_2']
    nation_desc = row['nation_short_desc']

    currency = currency_dict[nation_desc]

    if pd.notna(min_sal2) and pd.notna(max_sal2):
        # min_sal2 and max_sal2 has higher preference over min_sal1 and max_sal1
        return pd.Series([min_sal2, max_sal2, currency])
    elif pd.notna(min_sal1) and pd.notna(max_sal1):
        return pd.Series([min_sal1, max_sal1, currency])
    else:
        return pd.Series([0, 0, "None"])

df_sal[["min_sal_3", "max_sal_3", "currency"]] = df_sal.apply(preprocessing_min_max_currency, axis=1)

# Create y-predict column
def y_prediction(row):
    min_sal3 = row['min_sal_3']
    max_sal3 = row['max_sal_3']
    freq3 = row['freq_3']
    currency = row['currency']
    if pd.notna(min_sal3) and pd.notna(max_sal3):
        if min_sal3 != 0 and max_sal3 != 0:
            sol = str(int(min_sal3)) + "-" + str(int(max_sal3)) + "-" + str(currency) + "-" + str(freq3)
        else:
            sol = "0-0-None-None"
        return sol
    else:
        sol = "0-0-None-None"
        return sol


df_sal["y_pred"] = df_sal.apply(y_prediction, axis=1)

# Keep only certain columns
df_sal = df_sal[["y_true","y_pred"]]
df_sal.to_csv(output_file_path, index=False)

accuracy = (df_sal['y_true'] == df_sal['y_pred']).mean()
print(f'Accuracy: {input_file_path} - {accuracy:.2%}')
