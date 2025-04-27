import json
import pandas as pd


def combine_and_export_to_json(df_list, output_path):
    combined_data = []
    for df in df_list:
        for _, row in df.iterrows():
            combined_data.append({
                "prompt": row["prompt"],
                "complete": row["complete"]
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"Splitted data saved to {output_path}")


def parse_text(text, task):
    text = text.strip()

    if task == 'salary':
        return (
            "You are an information extraction assistant.\n"
            "Your task is to extract salary information from a job advertisement.\n"
            "STRICTLY follow the format: [min]-[max]-[currency]-[unit]\n"
            "Example: 32-45-AUD-HOURLY\n"
            "Do NOT add explanations, comments, or extra text.\n"
            "Do NOT include anything else.\n"
            "If no salary is found, return exactly: 0-0-None-None\n"
            "Return the result ONLY.\n\n"
            f"Job Advertisement:\n{text}\n"
        )

    elif task == 'seniority':
        return (
            "You are a seniority classification assistant.\n"
            "Extract the most suitable seniority level from the job ad.\n"
            "Choose ONE from the following: trainee, entry level, intermediate, experienced, senior, lead\n"
            "Output must be a SINGLE WORD from the list.\n"
            "Do NOT explain your reasoning.\n"
            "Do NOT return full sentences, just the one word.\n\n"
            f"Job Advertisement:\n{text}\n"
        )

    elif task == 'arrangement':
        return (
            "You are a job arrangement classifier.\n"
            "Classify the work arrangement of the job ad.\n"
            "Output must be ONLY ONE word from: OnSite, Remote, Hybrid\n"
            "Do NOT explain.\n"
            "Do NOT include labels or additional words.\n"
            "The output must be EXACTLY one of the three options, nothing else.\n\n"
            f"Job Advertisement:\n{text}\n"
        )

    else:
        raise ValueError("Unknown task.")


if __name__ == '__main__':
    salary_df = pd.read_csv(r'After_cleaning/salary_test.csv')
    salary_df['prompt'] = salary_df.prompt.apply(lambda x: parse_text(x, 'salary'))

    seniority_df = pd.read_csv(r'After_cleaning/seniority_test.csv')
    seniority_df['prompt'] = seniority_df.prompt.apply(lambda x: parse_text(x, 'seniority'))

    work_df = pd.read_csv(r'After_cleaning/work_test.csv')
    work_df['prompt'] = work_df.prompt.apply(lambda x: parse_text(x, 'arrangement'))

    combined_df = pd.concat([salary_df, seniority_df, work_df], ignore_index=True)

    combined_df.to_json(r'p_engineering_testsets/p_engineering_testset_qwen_2.json', orient='records', indent=2)






