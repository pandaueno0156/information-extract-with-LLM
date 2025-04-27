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

    prompt_dict = {'salary': '[TASK: Salary] Extract salary information from the job ad. \n\n',
                   'seniority': '[TASK: Seniority] Extract seniority level from the job ad. \n\n',
                   'arrangement': '[TASK: Work Arrangement] Extract work arrangement from the job ad. \n\n'}
    prompt = prompt_dict[task]

    return prompt + 'job ad: \n{q}\n\n'.format(q=text)




if __name__ == '__main__':
    salary_df = pd.read_csv(r'After_cleaning/salary_training.csv')
    salary_df['prompt'] = salary_df.prompt.apply(lambda x: parse_text(x, 'salary'))

    seniority_df = pd.read_csv(r'After_cleaning/seniority_training.csv')
    seniority_df['prompt'] = seniority_df.prompt.apply(lambda x: parse_text(x, 'seniority'))

    work_df = pd.read_csv(r'After_augment/work_training_aug.csv')
    work_df['prompt'] = work_df.prompt.apply(lambda x: parse_text(x, 'arrangement'))

    combined_df = pd.concat([salary_df, seniority_df, work_df], ignore_index=True)

    combined_df.to_json(r'ready2train&test/agument_work_5316.json', orient='records', indent=2)






