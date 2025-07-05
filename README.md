# Information Extraction with LLMs (SEEK Industry project)

## Contribution:

Kunisuke Ishigaki 

Xiao Chen 

WanJie Mao 

Zhuo Chen 


---

## Overview:

#### This project focuses on the processing, fine-tuning, and inference of job advertisement texts.
#### It is structured across several stages:

- Stage 0: Data Cleaning & Rule-Based Modeling

- Stage 1: Prompt Engineering, Non fine-tuning methods

- Stage 2: Prompt-Tuning, LoRA-Based Lightweight Fine-Tuning, Agent Development

- Stage 3: Advanced Methods Exploration

- Stage 4: General inference, Langchain inference, Verification

The project systematically progresses from initial data preparation to advanced model deployment.


## Project Structure

```
AdSeek_ready2submit/
│
├── Preprocessing/            # Data cleaning and feature engineering
│   ├── After_augment/
│   ├── After_cleaning/
│   ├── p_engineering_testsets/
│   ├── ready2train_test/
│   ├── Adding_prompts&combine.py
│   ├── cleaning.py
│   ├── data_augment.ipynb
│   ├── p_engineering_concat.py
│   └── prompt_injection&combine.py
│
├── Stage0_rulebase/            # Stage 0: Rule-based model
│   ├── rule-based_senority.py
│   ├── rule-based_work_arrangement.py
│   └── rule-based-salary.py            # Supports CLI-based batch inference
│
├── Stage1_prompt_only/         # Stage 1: Prompt-only inference
│   ├── Claude_pe.ipynb
│   ├── GPT_pe.ipynb
│   ├── Qwen05B_p_eng_inference.ipynb
│   └── Qwen25_inference.py             # Supports CLI-based batch inference
│
├── Stage2_LoRA_pt_agents/      # Stage 2: Lightweight fine-tuning with LoRA
│   ├── agent/                          # Synthetic data generation using Claude and GPT
│   │   ├── claude_multiAgent.py        # Autogen Claude Multi-Task Agent Pipeline
│   │   └── gpt_mulitAgent.py           # Autogen GPT Multi-Task Agent Pipeline 
│   │ 
│   ├── LoRA_qwen05.ipynb
│   ├── prompt_tuning_qwen05.ipynb
│   ├── Qwen05_LoRA.py                  # Supports CLI-based LoRA fine-tuning
│   └── qwen-0.5b-llm_kaggle.ipynb
│
├── Stage3_advanced/            # Stage 3: Advanced techniques (DAPT, POS/NER-based tuning)
│   ├── DAPT.ipynb
│   ├── qwen-05b-train-dapt-PrompTuning.ipynb   # Pretrained DAPT and Prompt tuning
│   ├── qwen-05b-train-dapt-LoRA.ipynb          # Pretrained DAPT and LoRA
│   ├── LoRA_classification_head.ipynb
│   ├── POS_NER_prompt_tuning_1st.ipynb
│   └── POS_NER_prompt_tuning_2nd.ipynb
│  
├── Stage4_inference_and_verification/  # Stage 4: Inference, Langchain, Verification
│   ├── inference_general.py
│   ├── inference_langchain.py
│
├── eval.py                             # Evaluation script for inference results
├── README.md                           # Project overview and usage guide
└── requirements.txt                    # Python package dependencies
```

---

## Quick Start

### 1. Environment Setup

Install required packages:

```bash
pip install -r requirements.txt
```

Additionally, for `spaCy` models:

```bash
python -m spacy download en_core_web_sm
```

### 2. Rule-based model (Stage 0) 

Use `rule-based-salary.py` to extraction salary information:

```bash
python Stage0_rulebase/rule-based-salary.py \
  --input_path "./job_data_files/raw-data/salary_labelled_test_set.csv" \
  --output_path "./job_data_files/df_output_salary_rule_based_testset.csv"

```

---

### 3. Non fine-tuning methods (Stage 1)

Use `Qwen25_inference.py` for batch inference:

```bash
python Stage1_prompt_only/Qwen25_inference.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --input_json "Preprocessing/p_engineering_testsets/p_engineering_testset.json" \
  --output_json "results_collection/qwen25_pe_results.json" \
  --batch_size 8 \
  --max_new_tokens 20
```

| Argument | Description | Default |
|:---------|:------------|:--------|
| `--model_path` | Path to Hugging Face model | Required |
| `--input_json` | Input JSON file path | Required |
| `--output_json` | Output JSON save path | Required |
| `--batch_size` | Inference batch size | 8 |
| `--max_new_tokens` | Maximum tokens to generate | 20 |
| `--prompt_column` | Column for input prompts | prompt |

---

### 4. Prompt-tuning, LoRA Fine-Tuning and Synthetic Agent Data (Stage 2) 

Use `Qwen05_LoRA.py` for LoRA-based fine-tuning:

```bash
python Stage2_LoRA_pt_agents/Qwen05_LoRA.py \
  --json_path "Preprocessing/ready2train_test/train.json" \
  --model_save_path "results_collection/qwen05_lora_finetuned/"
```

| Argument | Description | |
|:---------|:-------------|:--|
| `--json_path` | Path to training dataset (JSON format) | Required |
| `--model_save_path` | Directory to save fine-tuned model | Required |

Fine-tuning is performed on **Qwen2.5-0.5B-Instruct** using LoRA (Low-Rank Adaptation) for efficient parameter-efficient training.

---

### 5. Evaluation

After inference or fine-tuning, evaluate results using `eval.py`:

```bash
python eval.py
```

Metrics include: Self-defined Salary Accuracy, Accuracy, Recall, Macro F1-score, etc.

For example:

```
 ==================== LoRA_05B_dapt_work_aug_genInf_results.json ==============
Salary: 496 / 567 - 87.48%
Seniority: 355 / 689 - 51.52%
Work Arrangement: 58 / 99 - 58.59%

-------- sklearn metrics --------
Accuracy: 0.6708
Macro Recall: 0.6101
Macro F1-score: 0.6124

----- Salary Format Evaluation -----
Min Salary Accuracy: 75.49%
Max Salary Accuracy: 75.66%
Currency Accuracy: 77.25%
Frequency Accuracy: 74.78%
Overall Average Accuracy: 75.79%
---------------------------------------

 ==================== Claude_haiku_mulitAgent_results.json ====================
Salary: 471/567 - 83.07%
Seniority: 371/689 - 53.85%
Work Arrangement: 86/99 - 86.87%

-------- sklearn metrics --------
Accuracy: 0.6849
Macro Recall: 0.5897
Macro F1-score: 0.5857

----- Salary Format Evaluation -----
Min Salary Accuracy: 72.31%
Max Salary Accuracy: 71.25%
Currency Accuracy: 76.54%
Frequency Accuracy: 77.07%
Overall Average Accuracy: 74.29%
---------------------------------------
```

---

## Notes

- The project is divided into four stages:
  - **Stage 1**: Zero-shot and prompt-engineering baselines.
  - **Stage 2**: Lightweight fine-tuning via LoRA adapters and AutoGen pipeline.
  - **Stage 3**: Advanced methods like Domain-Adaptive Pretraining (DAPT) and POS/NER-enhanced prompt tuning.
  - **Stage 4**: General Inference with model.generate(), Inference with Langchain, and verification of output.
- File paths must be specified carefully to avoid errors.
- GPU with FP16 (half-precision) support is highly recommended.
- Preprocessing scripts are modular for easy modification and experimentation.
