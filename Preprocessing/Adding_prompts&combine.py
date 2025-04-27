
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

    text = html_parse(text)

    return prompt + 'job ad: \n{q}\n\n'.format(q=text)

salary_df = pd.read_csv(r'G:/Coding/PycharmProject/AdSeek_final/job_data_files/salary_labelled_development_set.csv')

salary_df['prompt'] = salary_df.prompt_.apply(lambda x: parse_text(x, 'salary'))