# SEEK Industry project - AdSeek

This project focuses on processing, fine-tuning, and inference for job advertisement texts. It covers data cleaning, feature engineering, prompt tuning (Stage 1), LoRA-based lightweight fine-tuning (Stage 2), and advanced methods (Stage 3).

## 📂 Project Structure

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
├── Stage1_prompt_only/        # Stage 1: Prompt-only inference
│   ├── Claude_pe.ipynb
│   ├── GPT_pe.ipynb
│   ├── Qwen05B_p_eng_inference.ipynb
│   ├── Qwen25_inference.py     # Supports CLI-based batch inference
│
├── Stage2_LoRA_pt_agents/     # Stage 2: Lightweight fine-tuning with LoRA
│   ├── agent/                 # Synthetic data generation using Claude and GPT
│   │   ├── gen_claude_arr.ipynb
│   │   ├── gen_claude_salary.ipynb
│   │   ├── gen_claude_seniority.ipynb
│   │   └── gen_gpt.ipynb
│   ├── LoRA_qwen05.ipynb
│   ├── prompt_tuning_qwen05.ipynb
│   ├── Qwen05_LoRA.py          # Supports CLI-based LoRA fine-tuning
│   ├── qwen-0.5b-llm_kaggle.ipynb
│
├── Stage3_advanced/           # Stage 3: Advanced techniques (DAPT, POS/NER-based tuning)
│   ├── DAPT.ipynb
│   ├── LoRA_classification_head.ipynb
│   ├── POS_NER_prompt_tuning_1st.ipynb
│   ├── POS_NER_prompt_tuning_2nd.ipynb
│
├── eval.py                    # Evaluation script for inference results
├── README.md                   # Project overview and usage guide
└── requirements.txt            # Python package dependencies
```

---

## 🚀 Quick Start

### 1. Environment Setup

Install required packages:

```bash
pip install -r requirements.txt
```

Additionally, for `spaCy` models:

```bash
python -m spacy download en_core_web_sm
```

---

### 2. Batch Inference (Stage 1)

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

### 3. LoRA Fine-Tuning (Stage 2)

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

### 4. Evaluation

After inference or fine-tuning, evaluate results using `eval.py`:

```bash
python eval.py
```

Metrics include: Accuracy, Recall, Macro F1-score, etc.

---

## 📝 Notes

- The project is divided into three stages:
  - **Stage 1**: Zero-shot and prompt-engineering baselines.
  - **Stage 2**: Lightweight fine-tuning via LoRA adapters.
  - **Stage 3**: Advanced methods like Domain-Adaptive Pretraining (DAPT) and POS/NER-enhanced prompt tuning.
- File paths must be specified carefully to avoid errors.
- GPU with FP16 (half-precision) support is highly recommended.
- Preprocessing scripts are modular for easy modification and experimentation.
