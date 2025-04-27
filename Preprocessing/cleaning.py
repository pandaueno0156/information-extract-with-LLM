import pandas as pd
from bs4 import BeautifulSoup
import re


def html_parse(details_str):
    if pd.isna(details_str):
        return ""

    soup = BeautifulSoup(details_str, 'html.parser')

    for br in soup.find_all('br'):
        br.replace_with('\n')
    for p in soup.find_all('p'):
        p.append('\n')

    text = soup.get_text(separator='\n', strip=True)
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    return text


def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F" 
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF" 
        u"\U0001F1E0-\U0001F1FF" 
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text_fields(job_title, job_ad_details):
    clean_ad = html_parse(job_ad_details)
    clean_title = html_parse(job_title)
    clean_title = remove_emojis(clean_title)
    return clean_title, clean_ad


def clean_salary(row):
    job_title, job_ad_details = clean_text_fields(
        row["job_title"].strip() if pd.notna(row["job_title"]) else "",
        row["job_ad_details"].strip() if pd.notna(row["job_ad_details"]) else ""
    )

    nation_desc = str(row["nation_short_desc"]).strip() if pd.notna(row["nation_short_desc"]) else ""
    salary_text = str(row["salary_additional_text"]).strip() if pd.notna(row["salary_additional_text"]) else ""

    prompt = (
        job_title + " " +
        job_ad_details + " " +
        nation_desc + " " +
        salary_text
    )
    return html_parse(prompt)


def clean_seniority(row):
    job_title, job_ad_details = clean_text_fields(
        row["job_title"].strip(), row["job_ad_details"].strip()
    )
    prompt = (
        job_title + " " +
        row['job_summary'].strip() + " " +
        job_ad_details + " " +
        row["classification_name"].strip() + " " +
        row["subclassification_name"].strip()
    )
    return html_parse(prompt)



def clean_arrangement(row):
    return html_parse(row["job_ad"].strip() if pd.notna(row["job_ad"]) else "")


if __name__ == '__main__':

    # training sets
    salary_df = pd.read_csv(r'G:/Coding/PycharmProject/AdSeek_final/job_data_files/salary_labelled_development_set.csv')
    salary_df['prompt'] = salary_df.apply(clean_salary, axis=1)
    salary_df['complete'] = salary_df["y_true"].str.strip()
    salary_df[['prompt', 'complete']].to_csv(r'After_cleaning/salary_training.csv', index=False)

    seniority_df = pd.read_csv(r'G:/Coding/PycharmProject/AdSeek_final/job_data_files/seniority_labelled_development_set.csv')
    seniority_df['prompt'] = seniority_df.apply(clean_seniority, axis=1)
    seniority_df['complete'] = seniority_df["y_true"].str.strip()
    seniority_df[['prompt', 'complete']].to_csv(r'After_cleaning/seniority_training.csv', index=False)

    arr_df = pd.read_csv(r'G:/Coding/PycharmProject/AdSeek_final/job_data_files/work_arrangements_development_set.csv')
    arr_df['prompt'] = arr_df.apply(clean_arrangement, axis=1)
    arr_df['complete'] = arr_df["y_true"].str.strip()
    arr_df[['prompt', 'complete']].to_csv(r'After_cleaning/work_training.csv', index=False)

    # test sets
    salary_df = pd.read_csv(r'G:/Coding/PycharmProject/AdSeek_final/job_data_files/salary_labelled_test_set.csv')
    salary_df['prompt'] = salary_df.apply(clean_salary, axis=1)
    salary_df['complete'] = salary_df["y_true"].str.strip()
    salary_df[['prompt', 'complete']].to_csv(r'After_cleaning/salary_test.csv', index=False)

    seniority_df = pd.read_csv(r'G:/Coding/PycharmProject/AdSeek_final/job_data_files/seniority_labelled_test_set.csv')
    seniority_df['prompt'] = seniority_df.apply(clean_seniority, axis=1)
    seniority_df['complete'] = seniority_df["y_true"].str.strip()
    seniority_df[['prompt', 'complete']].to_csv(r'After_cleaning/seniority_test.csv', index=False)

    arr_df = pd.read_csv(r'G:/Coding/PycharmProject/AdSeek_final/job_data_files/work_arrangements_test_set.csv')
    arr_df['prompt'] = arr_df.apply(clean_arrangement, axis=1)
    arr_df['complete'] = arr_df["y_true"].str.strip()
    arr_df[['prompt', 'complete']].to_csv(r'After_cleaning/work_test.csv', index=False)
