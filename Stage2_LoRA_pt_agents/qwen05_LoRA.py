import os
import time
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)

def load_and_preprocess(data_path, tokenizer):
    dataset = load_dataset("json", data_files=data_path)
    dataset = dataset.filter(lambda x: len(x["prompt"]) > 0 and len(x["complete"]) > 0)

    def tokenize_function(examples):
        completions = [str(c) for c in examples["complete"]]
        texts = [p + c + tokenizer.eos_token for p, c in zip(examples["prompt"], completions)]
        tokenized = tokenizer(
            texts,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        prompt_tokenized = tokenizer(examples["prompt"], add_special_tokens=False)
        prompt_lengths = [len(ids) for ids in prompt_tokenized["input_ids"]]
        labels = tokenized["input_ids"].clone()
        for i, length in enumerate(prompt_lengths):
            if length >= 512:
                length = 511
            labels[i][:length] = -100
        tokenized["labels"] = labels
        return tokenized

    return dataset.map(tokenize_function, batched=True)

def train(json_path, model_save_path):
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to('cpu')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_and_preprocess(json_path, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./qwen_prompt_tuning_output",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=6,
        learning_rate=8e-4,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        dataloader_num_workers=4,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    start_time = time.time()
    trainer.train()
    print(f"Training finished: {time.time() - start_time:.2f}s")

    model.save_pretrained(model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, required=True)
    args = parser.parse_args()
    train(args.json_path, args.model_save_path)
