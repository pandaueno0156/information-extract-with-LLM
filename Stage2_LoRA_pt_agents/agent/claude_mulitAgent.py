import os
import autogen
import json
import pandas as pd
from dotenv import load_dotenv
import time  
import logging
import pathlib

logging.basicConfig(level=logging.ERROR)

load_dotenv()

llm_config = {
    "model": "claude-3-5-haiku-20241022",
    "api_key": os.getenv("ANTHROPIC_API_KEY"), 
    "base_url": "https://api.anthropic.com/",
    "api_type": "anthropic",
}

# GroupChat initiation agent
user_proxy_agent = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",  
    max_consecutive_auto_reply=3,  
    code_execution_config=False,
)

# Salary Agent
salary_agent = autogen.AssistantAgent(
    name="Salary",
    llm_config=llm_config,
    system_message="""
You specialize in extracting salary information from job advertisements. 

CRITICAL: Your response must ONLY be the exact format below. DO NOT include any explanations, reasoning, or additional text.

OUTPUT FORMAT: [minimum number]-[maximum number]-[currency]-[time unit]

Valid currencies: AUD, SGD, HKD, IDR, THB, NZD, MYR, PHP, USD, None
Valid time units: HOURLY, DAILY, WEEKLY, MONTHLY, ANNUAL, None

Examples of CORRECT responses:
32-45-AUD-HOURLY
80000-100000-USD-ANNUAL
0-0-None-None

Rules:
1. Round to whole numbers. If salary has decimal points (e.g., $32.50), round to nearest whole number (33)
2. Use only valid currencies and time units from the lists above
3. If no salary info: output 0-0-None-None
4. If single salary: use same number for min and max

STRICTLY FORBIDDEN:
- Do NOT write "Based on...", "The salary is...", "Explanation:", or any other text
- Do NOT provide reasoning or justification
- Do NOT use parentheses or additional comments
- Your response must be EXACTLY the format: number-number-currency-timeunit

RESPOND WITH THE FORMAT ONLY. NOTHING ELSE.
"""
)


# Seniority agent
seniority_agent = autogen.AssistantAgent(
    name="Seniority",
    llm_config=llm_config,
    system_message="""
You specialize in extracting seniority level information from job advertisements.

CRITICAL: Respond with ONLY one word from this exact list. NO explanations or additional text allowed.

Valid options:
experienced, intermediate, senior, entry level, assistant, lead, head, junior, graduate, trainee, associate, principal, apprentice, executive, manager, director, entry-level, chief, deputy, mid-level, specialist, experienced assistant, supervisor, qualified, student, board, graduate/junior, senior associate, mid-senior

Guidelines for selection:
- Entry level: 0-2 years, graduate, trainee, apprentice
- Intermediate/experienced: 3-7 years, specialist, qualified  
- Senior/lead: 5+ years, leadership responsibilities
- Management: manager, supervisor, head, director, executive

STRICTLY FORBIDDEN:
- Do NOT write full sentences
- Do NOT provide explanations or reasoning
- Do NOT use phrases like "The seniority level is..." or "Based on..."

RESPOND WITH ONE WORD ONLY FROM THE LIST ABOVE.
"""
)


# Arragement agent
arrangement_agent = autogen.AssistantAgent(
    name="Arrangement",
    llm_config=llm_config,
    system_message="""
CRITICAL: Respond with ONLY one word. NO explanations allowed.

Valid options: OnSite, Remote, Hybrid

Selection guidelines:
- OnSite: Physical location required, customer-facing, traditional industries
- Remote: Work from home, digital roles, location independent
- Hybrid: Flexible, mix of office and home, modern professional roles

STRICTLY FORBIDDEN:
- Do NOT write sentences or explanations
- Do NOT use phrases like "The work arrangement is..." or "Based on..."
- Do NOT provide reasoning

RESPOND WITH ONE WORD ONLY: OnSite, Remote, or Hybrid
"""
)


# Critic agent
critic_agent = autogen.AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="""
You verify whether outputs from other agents meet the required format.

CRITICAL: Respond with ONLY one of these options. NO explanations allowed.

Valid formats:
1. Salary: number-number-currency-time_unit (e.g., 50000-60000-USD-ANNUAL) or 0-0-None-None
2. Seniority: Must be one word from this exact list: experienced, intermediate, senior, entry level, assistant, lead, head, junior, graduate, trainee, associate, principal, apprentice, executive, manager, director, entry-level, chief, deputy, mid-level, specialist, experienced assistant, supervisor, qualified, student, board, graduate/junior, senior associate, mid-senior
3. Arrangement: Must be exactly one of: OnSite, Hybrid, Remote

RESPONSE RULES:
- If output is valid: respond with "TERMINATE"
- If salary format is wrong: respond with "@Salary"
- If seniority format is wrong: respond with "@Seniority"
- If arrangement format is wrong: respond with "@Arrangement"

STRICTLY FORBIDDEN:
- Do NOT write explanations like "The output format is incorrect because..."
- Do NOT provide reasoning or justification
- Do NOT use phrases like "Please fix..." or "The format should be..."

RESPOND WITH ONE WORD ONLY: Either "TERMINATE" or "@AgentName"
"""
)


def print_groupchat_log(groupchat):
    print("\n GroupChat Log:")
    for idx, msg in enumerate(groupchat.messages):
        name = msg.get("name", "Unknown")
        content = msg.get("content", "").strip()
        print(f"[{idx+1}] {name}: {content[:200]}{'...' if len(content) > 200 else ''}")

def custom_speaker_selection(last_speaker: autogen.Agent ,groupchat: autogen.GroupChat):

    messages = groupchat.messages
    # print(f"\nMessages: {messages}\n")
    # print(f"\nLast Speaker: {last_speaker}\n")
    
    # last message content from task agent
    last_message = messages[-1].get("content", "").strip()

    task_type = None

    # If user proxy just initiated, determine which task agent should respond
    if last_speaker.name == "User":  # User proxy initiated
        if '[TASK: Work Arrangement]' in last_message:
            task_type = "Work Arrangement"
            return arrangement_agent
        elif '[TASK: Salary]' in last_message:
            task_type = "Salary"
            return salary_agent
        elif '[TASK: Seniority]' in last_message:
            task_type = "Seniority"
            return seniority_agent

    # If last speaker was a task agent, let Critic agent to speak for evaluation
    if last_speaker in [salary_agent, seniority_agent, arrangement_agent]:
        return critic_agent
    
    # If last speaker was Critic, let Critic decide to send back to task agent or terminate
    if last_speaker == critic_agent:
        if "TERMINATE" in last_message:
            return None # None will flag Groupchat to terminate
        elif "@Salary" in last_message:
            return salary_agent
        elif "@Seniority" in last_message:
            return seniority_agent
        elif "@Arrangement" in last_message:
            return arrangement_agent
        else:
            # Critic agent failed to evaluate
            # force critic to go back to task agent to output again
            if task_type == "Work Arrangement":
                return arrangement_agent
            elif task_type == "Salary":
                return salary_agent
            elif task_type == "Seniority":
                return seniority_agent
    
    # Default fallback
    return None

def extract_output(groupchat, prompt):

    # Reset the groupchat to clear previous messages
    groupchat.reset()

    # User to begin the conversation
    user_proxy_agent.initiate_chat(
    manager,
    message=prompt,
    silent=False # Turn False if we want to see the Groupchat output
    )

    messages = groupchat.messages

    # print(f"\nMessages: {messages}\n")

    # Identify Task type in this input
    message_type = messages[0].get("content", "").strip()

    if '[TASK: Work Arrangement]' in message_type:
        task_agent = "Arrangement"
    elif '[TASK: Salary]' in message_type:
        task_agent = "Salary"
    elif '[TASK: Seniority]' in message_type:
        task_agent = "Seniority"

    for i in range(len(messages)-1, -1, -1):
        msg = messages[i]
        if msg.get("name") == "Critic" and "TERMINATE" in msg.get("content", "").strip():
            # Meaning we found the output
            for j in range(i-1, -1, -1):
                previous_msg = messages[j]
                if previous_msg.get("name") == task_agent:
                    # Final output
                    return previous_msg.get("content", "").strip()

    return 'Unknown' # If we cannot find the output, return Unknown


groupchat = autogen.GroupChat(
    agents=[user_proxy_agent, salary_agent, seniority_agent, arrangement_agent, critic_agent],
    messages=[],
    max_round=5,
    speaker_selection_method=custom_speaker_selection
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


# Test data
current_dir = pathlib.Path(__file__).parent
root_dir = current_dir.parent.parent
test_data_path = root_dir / "dataset" / "testset" / "test_1355.json"

answers = []
with open(test_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Start counting inference time...
t0 = time.time()

for i, item in enumerate(data):
    p = item['prompt']
    answer = extract_output(groupchat, p)
    answers.append(answer)
    if i % 50 == 0:
        print(f"Processed {i} items...")
        
t1 = time.time()
print("Inference time:", t1 - t0)

# Finish counting inference time...

save_path = root_dir / "results_collection" / "autogen" / "claude_haiku_mulitAgent_test_all_v1.json"

df = pd.DataFrame(data)
df['y_pred'] = answers
df.to_json(save_path, orient='records', indent=4, force_ascii=False)





# Testing only
# final_output = extract_output(groupchat)
# print(f"\nFinal Output: {final_output}\n")

# Single testing
# user_proxy_agent.initiate_chat(
#     manager,
#     message="Extract salary information from the job ad: 'The salary is $32-$45 per hour and the work location is USA.'",
# )

# user_proxy_agent.initiate_chat(
#     manager,
#     message="[TASK: Salary] Extract salary information from the job ad. \n\njob ad: \nCashier (Kota Tinggi) Bertanggungjawab sebagai cashier\nMengurus semua rekod mengenai cek yang diterima\nMenyediakan laporan yang diperlukan oleh HQ (Jabatan Akaun dan Jabatan Sumber Manusia)\nKiraan stok bulanan\nSemua kerja lain yang ditetapkan oleh pengurus cawangan dan supervisor pada bila-bila mengikut keperluan\nKeperluan\nBerkelulusan SPM \/ O Level \/ SKM Level 1 \/ SKM Level 2 \/ SKM Level 3 atau setaraf\nSedikit kemahiran tentang komputer\nMenepati masa\nKerja overtime (Jika diperlukan)\nGaji RM 1500 \u2013 1800++ Calon berminat boleh whatsapp 010-3938581\nSeng Li Marketing Sdn Bhd is a One-Stop Auto Parts Trading Company\nSalary :\nRM 1500 \u2013 1800 MY RM\u00a01,500 \u2013 RM\u00a01,800 per month\n\n",
# )

# Note:
# Test Sample Size: 1355
# Inference time: 4349.467544794083