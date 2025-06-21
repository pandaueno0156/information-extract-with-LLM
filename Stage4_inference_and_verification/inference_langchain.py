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

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

# Create LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Define task-specific prompt
TASK_PROMPT = {
    "Salary": """Task: Extract salary information.
    Format: [min]-[max]-[currency]-[time_unit]

    Valid currency: ["AUD", "SGD", "HKD", "IDR", "THB", "NZD", "MYR", "PHP", "USD", "None"]
    Valid time_unit: ["HOURLY", "DAILY", "WEEKLY", "MONTHLY", "ANNUAL", "None"]
    If no salary is found, return: 0-0-None-None

    Input: {prompt}
    Output:""",
    
    "Seniority": """Task: Extract seniority level.
    Valid option: experienced, intermediate, senior, entry level, assistant, lead, head, junior, graduate, trainee, associate, principal, apprentice, executive, manager, director, entry-level, chief, deputy, mid-level, specialist, experienced assistant, supervisor, qualified, student, board, graduate/junior, senior associate, mid-senior
    
    IMPORTANT INSTRUCTIONS:
    1. Only extract the seniority level.
    2. ONLY CHOOSE output from Valid option list!
    3. NEVER INCLUDE ANY explanations or additional text


    Input: {prompt}
    Output:""",
    
    "Work Arrangement": """Task: Extract work arrangement.
    Options: OnSite, Remote, Hybrid
    IMPORTANT INSTRUCTIONS:
    1. The output must be capitalized!
    2. Output must be a single word!
    3. NEVER INCLUDE ANY explanations or additional text

    Input: {prompt}
    Output:"""
}

def get_task_prompt(prompt):
    prompt_lower = prompt.lower()
    if "salary" in prompt_lower:
        return "Salary"
    elif "seniority" in prompt_lower:
        return "Seniority"
    elif "work arrangement" in prompt_lower:
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
    
    output_parts = response.split("Output:")
    if len(output_parts) > 1:
        # Get everything after "Output:"
        after_output = output_parts[1].strip()

        # First try: Check if the ouput after: matches the format
        match = re.search(r"(\d+-\d+-[A-Z]+-[A-Z]+|0-0-None-None)", after_output)

        # print(f"\nMatch: {match}")
        if match:
            return match.group(1)
        
    return "0-0-None-None"

def clean_seniority_response(response):
    
    output_parts = response.split("Output:")
    if len(output_parts) > 1:
        # Get everything after "Output:"
        after_output = output_parts[1].strip().lower()
        valid_levels = {
                'experienced', 'intermediate', 'senior', 'entry level', 
                'assistant', 'lead', 'head', 'junior', 'graduate', 
                'trainee', 'associate', 'principal', 'apprentice', 
                'executive', 'manager', 'director', 'entry-level', 
                'chief', 'deputy', 'mid-level', 'specialist', 
                'experienced assistant', 'supervisor', 'qualified', 
                'student', 'board', 'graduate/junior', 'senior associate', 
                'mid-senior', 'unknown'
            }
        if after_output in valid_levels:
            return after_output

        for level in valid_levels:
            if level in after_output:
                return level
        
        if 'entry' in after_output and 'level' in after_output:
            return 'entry-level'
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
    
    output_parts = response.split("Output:")
    if len(output_parts) > 1:
        # Get everything after "Output:"
        after_output = output_parts[1].strip().upper()

        # First try: Check if the ouput after: matches the ALL UpperCase answer
        if re.match(r"^(ONSITE|HYBRID|REMOTE)$", after_output):
                return after_output.capitalize()
        # Second try: Check if the ouput after: matches the exatch word in the after output
        match = re.search(r"(OnSite|Hybrid|Remote)", after_output, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        
    return "Unknown"


def generate_response_with_chain(prompt, max_attempts=5):
    global global_re_attempts

    # Identify task type
    task_type = get_task_prompt(prompt)

    # Get prompt template
    template_string = TASK_PROMPT.get(task_type, "{prompt}")

    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template=template_string
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Try to generate valid response
    for attempt in range(max_attempts):
        try:
            # Generate response - pass the original prompt
            response = chain.run(prompt=prompt)

            # Clean the response - remove any prompt template text
            if task_type == 'Salary':
                response = clean_salary_response(response) # Use the new cleaning function
            elif task_type == 'Seniority':
                response = clean_seniority_response(response)
            elif task_type == 'Work Arrangement':
                response = clean_work_response(response)

            # Validate response
            if validate_response(response, task_type):
                return response
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
    response = chain.run(prompt=prompt)
    
    # Clean the response - remove any prompt template text
    if task_type == 'Salary':
        response = clean_salary_response(response) # Use the new cleaning function
    elif task_type == 'Seniority':
        response = clean_seniority_response(response)
    elif task_type == 'Work Arrangement':
        response = clean_work_response(response)

    return response


# Load test data
test_path = "./dataset/testset/test_1355.json"

with open(test_path, 'r') as f: #Update with the path to your JSON File.
    data = json.load(f)


# Example usage in your main loop:
print("\nStart inference...")

t0 = time.time()
answers = []
for i, item in enumerate(data):
    p = item['prompt']
    # Generate response with retry mechanism
    answer = generate_response_with_chain(p, max_attempts=5)
    answers.append(answer)
    
    # Optional: Print progress and validation status
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
