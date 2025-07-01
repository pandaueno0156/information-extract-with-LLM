import os
import time
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    PromptTuningConfig,
    LoraConfig,
    PeftModel,
    TaskType
)
import pandas as pd
from torch.cuda import max_memory_allocated, memory_allocated, memory_reserved
import json
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import pipeline
import re

global_re_attempts = 0

# Load the base model and tokenizer
model_path = "Qwen/Qwen2.5-0.5B-Instruct"
peft_model_path = "qwen_dapt_lora_augwork_final_model"  

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_fast=False
)


# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA model
model = PeftModel.from_pretrained(model, peft_model_path)

# Set model to evaluation mode
model.eval()


def get_task_prompt(prompt):
    if "[TASK: Salary]" in prompt:
        return "Salary"
    elif "[TASK: Seniority]" in prompt:
        return "Seniority"
    elif "[TASK: Work Arrangement]" in prompt:
        return "Work Arrangement"
    return "default" # This will return the default input prompt in case the task type cannot be identified

def validate_response(response, task_type):
    if task_type == "Salary":
        # Validate salary format: [min]-[max]-[currency]-[time_unit]

        parts = response.strip().split("-")
        if len(parts) != 4:
            return False
        try:
            # The response has 4 parts
            min_salary = int(parts[0])
            max_salary = int(parts[1])
            if min_salary < 0 or max_salary < 0:
                return False
            if min_salary > max_salary:
                return False
            if parts[2] not in ["AUD", "SGD", "HKD", "IDR", "THB", "NZD", "MYR", "PHP", "USD", "None"]:
                return False
            if parts[3] not in ["HOURLY", "DAILY", "WEEKLY", "MONTHLY", "ANNUAL", "None"]:
                return False
            # Pass all if statement, so it's valid response
            return True
        except ValueError:
            return False # The response is not a valid salary format
    elif task_type == "Seniority":
        # Validate seniority level
        response = response.strip()

        valid_levels = ["experienced", "intermediate", "senior", "entry level", "assistant", 
                "lead", "head", "junior", "graduate", "trainee", "associate", 
                "principal", "apprentice", "executive", "manager", "director", 
                "entry-level", "chief", "deputy", "mid-level", "specialist", 
                "experienced assistant", "supervisor", "qualified", "student", 
                "board", "graduate/junior", "senior associate", "mid-senior"]
        
        if response not in valid_levels:
            return False
        # Response in one of the valid_levels
        return True
    elif task_type == "Work Arrangement":
        # Validate work arrangement

        response = response.strip()
        valid_arrangements = ["Onsite", "Remote", "Hybrid"]

        if response not in valid_arrangements:
            return False
        return True

def clean_salary_response(response):
    
    output_parts = response
    if len(output_parts) > 1:
        # Get everything after "Output:"
        after_output = output_parts.strip()

        # First try: Check if the ouput after: matches the format
        match = re.search(r"(\d+-\d+-[A-Z]+-[A-Z]+|0-0-None-None)", after_output)

        # print(f"\nMatch: {match}")
        if match:
            return match.group(1)
        
    return "0-0-None-None"

def clean_seniority_response(response):
    
    output_parts = response
    if len(output_parts) > 1:
        # Get everything after "Output:"
        after_output = output_parts.strip().lower()
        valid_levels = {
                'experienced', 'intermediate', 'senior', 'entry level', 
                'assistant', 'lead', 'head', 'junior', 'graduate', 
                'trainee', 'associate', 'principal', 'apprentice', 
                'executive', 'manager', 'director', 'entry-level', 
                'chief', 'deputy', 'mid-level', 'specialist', 
                'experienced assistant', 'supervisor', 'qualified', 
                'student', 'board', 'graduate/junior', 'senior associate', 
                'mid-senior'
            }
        if after_output in valid_levels:
            return after_output

        for level in valid_levels:
            if level in after_output:
                return level
        
        if 'entry' in after_output and 'level' in after_output:
            return 'entry level'
        if 'senior' in after_output and 'associate' in after_output:
            return 'senior associate'
        if 'mid' in after_output and 'level' in after_output:
            return 'mid-level'
        if 'mid' in after_output and 'senior' in after_output:
            return 'mid-senior'
        if 'experienced' in after_output and 'assistant' in after_output:
            return 'experienced assistant'

    return "Unknown"

def clean_work_response(response):
    
    # output_parts = response.split("Output:")
    output_parts = response

    if len(output_parts) > 1:
        # Get everything after "Output:"
        after_output = output_parts.strip()

        # First try: Check if the ouput after: matches the extact answer
        if re.match(r"^(OnSite|Hybrid|Remote)$", after_output):
            return after_output
        
        # Second try: Check for case-insensitive matches
        match = re.search(r"(onsite|hybrid|remote)", after_output, re.IGNORECASE)
        if match:
            found_word = match.group(1).lower()
            if found_word == "onsite":
                return "OnSite"
            elif found_word == "hybrid":
                return "Hybrid"
            elif found_word == "remote":
                return "Remote"        
    return "Unknown"


# Generate response
def generate_response(prompt):
  with torch.no_grad():
      inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
      outputs = model.generate(
          input_ids=inputs["input_ids"],
          attention_mask=inputs["attention_mask"],
          max_new_tokens=10,
          temperature=0.5,
          do_sample=True,
          pad_token_id=tokenizer.eos_token_id
      )

      generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
      response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
      return response



def generate_response_with_verification(prompt, max_attempts=5):
    global global_re_attempts

    # Identify task type
    task_type = get_task_prompt(prompt)

    # Try to generate valid response
    for attempt in range(max_attempts):
        try:
            # Generate response - pass the original prompt
            raw_response = generate_response(prompt)

            # Clean the response - remove any prompt template text
            if task_type == 'Salary':
                cleaned_response = clean_salary_response(raw_response) # Use the new cleaning function
            elif task_type == 'Seniority':
                cleaned_response = clean_seniority_response(raw_response)
            elif task_type == 'Work Arrangement':
                cleaned_response = clean_work_response(raw_response)

            # Validate response
            if validate_response(cleaned_response, task_type):
                return cleaned_response
            else:
                global_re_attempts += 1
                # print(f"Task type: {task_type}")
                # print(f"Invalid response: {response}")
                # print(f"Invalid response on attempt {attempt + 1}. Retrying...")
        except Exception as e:
            global_re_attempts += 1
            print(f"Error on attempt {attempt + 1}: {e}")
            continue

    # Generate response
    # If all attemps failed, do the last attemp
    raw_response = generate_response(prompt)
    
    # Clean the response - remove any prompt template text
    if task_type == 'Salary':
        cleaned_response = clean_salary_response(raw_response) # Use the new cleaning function
    elif task_type == 'Seniority':
        cleaned_response = clean_seniority_response(raw_response)
    elif task_type == 'Work Arrangement':
        cleaned_response = clean_work_response(raw_response)

    return cleaned_response


# Load test data
test_path = "./dataset/testset/test_1355.json"

with open(test_path, 'r') as f: #Update with the path to your JSON File.
    data = json.load(f)


# Example usage in main loop:
print("\nStart inference...")

t0 = time.time()
answers = []
for i, item in enumerate(data):
    p = item['prompt']
    # Generate response with retry mechanism
    answer = generate_response_with_verification(p, max_attempts=5)
    answers.append(answer)
    
    if i % 50 == 0:
        print(f"Processed {i} items...")
        
t1 = time.time()
print("Inference time:", t1 - t0)

print("\nFinish inference...")

print(f'Total re-attempts: {global_re_attempts}')

# Appending answer to test datase†
df = pd.DataFrame(data)
df["y_pred"] = answers

# Output for eval.py to test performance
output_path = "results_collection/dapt_lora_output_langchain_inference.json"
df.to_json(output_path, orient='records', indent=4, force_ascii=False)

# Note: Inference time: 5419.779619216919 (Mac MAXm4)
# Total re-attempts: 232
