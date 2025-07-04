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
    "model": "gpt-3.5-turbo",
    "api_key": os.getenv("OPENAI_API_KEY"), 
    "base_url": "https://api.openai.com/v1",
}

# Planner Agent
planner_agent = autogen.AssistantAgent(
    name="Planner",
    llm_config=llm_config,
    system_message="""
You are a task router. Based on the user's instruction, route the task as follows:

1. If the instruction contains "Extract salary information from the job ad", respond with SALARY. This task should be handled by the Salary Agent.
2. If the instruction contains "Extract seniority level from the job ad", respond with SENIORITY. This task should be handled by the Seniority Agent.
3. If the instruction contains "Extract work arrangement from the job ad", respond with ARRANGEMENT. This task should be handled by the Arrangement Agent.

Do not ask the user for clarification. Just respond with one keyword: SALARY, SENIORITY, or ARRANGEMENT.
If you receive 'TERMINATE' or the task is completed, end the conversation immediately.
"""
)

# Salary agent
salary_agent = autogen.AssistantAgent(
    name="Salary",
    llm_config=llm_config,
    system_message="""
You specialize in extracting salary information from job advertisements. 

Your output must follow the format below:
[minimum number]-[maximum number]-[currency]-[time unit]

**Valid currencies:** AUD, SGD, HKD, IDR, THB, NZD, MYR, PHP, USD, None
**Valid time units:** HOURLY, DAILY, WEEKLY, MONTHLY, ANNUAL, None

**Examples:**
- 32-45-AUD-HOURLY
- 80000-100000-USD-ANNUAL
- 25-30-SGD-HOURLY
- 4000-5000-NZD-MONTHLY

**Important rules:**
1. **Round to whole numbers**: If salary has decimal points (e.g., $32.50), round to nearest whole number (33)
2. **Use only valid currencies**: Must be from the list above, use "None" if currency not specified or not in list
3. **Use only valid time units**: Must be from the list above, use "None" if time unit not specified or not in list
4. **Missing information**: If salary information is incomplete or missing, output: 0-0-None-None
5. **Single salary**: If only one salary mentioned (e.g., "$50,000"), use it for both min and max: 50000-50000-AUD-ANNUAL

**Currency detection tips:**
- $ symbols: Determine currency from job location context (AUD for Australia, USD for US, SGD for Singapore, etc.)
- Explicit currency codes: Use as specified (AUD, USD, etc.)
- Location-based inference: Australian jobs = AUD, US jobs = USD, Singapore = SGD, etc.

**Time unit detection:**
- "per hour", "hourly", "/hr" = HOURLY
- "per day", "daily" = DAILY  
- "per week", "weekly" = WEEKLY
- "per month", "monthly" = MONTHLY
- "per year", "annually", "per annum", "yearly" = ANNUAL

**Conversion guidelines:**
- Always extract the salary as stated, do not convert between time units
- If multiple formats given (e.g., "$25/hour or $52,000/year"), choose the more specific one

If the information is incomplete or missing, output: 0-0-None-None

Only extract salary information. Do not respond to any other questions.
"""
)

# Seniority Agent
seniority_agent = autogen.AssistantAgent(
    name="Seniority",
    llm_config=llm_config,
    system_message="""
You specialize in extracting seniority level information from job advertisements.

You must choose the most appropriate seniority level from the following list:
experienced, intermediate, senior, entry level, assistant, lead, head, junior, graduate, trainee, associate, principal, apprentice, executive, manager, director, entry-level, chief, deputy, mid-level, specialist, experienced assistant, supervisor, qualified, student, board, graduate/junior, senior associate, mid-senior

IMPORTANT: You must always choose one option from the above list. Do not return UNKNOWN or None.

If the seniority level is not explicitly stated, make an educated guess based on these contextual clues:

**Entry Level Indicators:**
- "Graduate", "new graduate", "fresh graduate"
- "Entry level", "trainee", "apprentice"
- "No experience required", "will train"
- "Student", "intern", "junior" roles
- Basic or fundamental responsibilities
- Learning-focused language

**Junior/Associate Indicators:**
- "1-3 years experience"
- "Associate", "junior", "assistant" in title
- Support roles under supervision
- Basic technical skills required
- "Working under guidance"

**Intermediate/Mid-Level Indicators:**
- "3-7 years experience"
- "Specialist", "qualified", "experienced"
- Independent work expectations
- "Mid-level", "intermediate"
- Project ownership but not team leadership
- Advanced technical skills

**Senior Indicators:**
- "5+ years experience", "7+ years"
- "Senior" in title
- Leadership responsibilities
- Mentoring junior staff
- Strategic involvement
- "Lead" individual contributor roles

**Leadership/Management Indicators:**
- "Manager", "supervisor", "head", "director"
- "Team lead", "department head"
- Budget responsibility
- People management
- "Executive", "chief", "principal"
- Board-level responsibilities

**Default reasoning rules:**
1. If years of experience mentioned: 0-2 years = entry level, 3-5 years = intermediate, 5-8 years = senior, 8+ years = senior/lead
2. If job title contains leadership words (manager, director, head, chief) = choose appropriate leadership level
3. If heavy emphasis on learning/training = entry level or trainee
4. If independent complex work but no leadership = experienced or senior
5. If supervision/management mentioned = manager, supervisor, or lead
6. If C-level or board mentioned = executive, chief, or board
7. For unclear cases with professional responsibilities = intermediate or experienced

Do not output full sentences or explanations. 
The output must be exactly one word from the valid list only.
"""
)

# Arrangement Agent
arrangement_agent = autogen.AssistantAgent(
    name="Arrangement",
    llm_config=llm_config,
    system_message="""
You specialize in extracting work arrangement information from job advertisements. Your output must be one of the following options:
OnSite, Remote, Hybrid

IMPORTANT: You must always choose one of these three options. Do not return UNKNOWN.

If the work arrangement is not explicitly stated, make an educated guess based on these contextual clues:

**OnSite indicators:**
- Mentions specific physical location/address
- "On-site", "in-office", "at our facility"
- Physical tasks (kitchen hand, warehouse, retail, healthcare, manufacturing)
- Customer-facing roles (receptionist, cashier, server)
- Mentions of company facilities, equipment, or physical workspace
- Traditional industries (hospitality, retail, healthcare, construction)

**Remote indicators:**
- "Work from home", "remote", "telecommute"
- Digital/tech roles (software developer, data analyst, digital marketing)
- "Location independent", "anywhere"
- Emphasis on internet connection, home office setup

**Hybrid indicators:**
- "Flexible working", "mix of office and home"
- "2-3 days in office", mentions of specific office days
- Modern office environments with flexibility options
- Roles that could work both ways (consulting, project management)

**Default reasoning:**
- If the job requires physical presence (hands-on work, customer interaction, specialized equipment), choose OnSite
- If the job is entirely digital/computer-based with no location specifics, lean toward Remote
- If it's a modern professional role with some flexibility mentioned, choose Hybrid

Do not output full sentences or explanations. The output must be a single word only: OnSite, Remote, or Hybrid.
"""
)


# Critic Agent
critic_agent = autogen.AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="""
You are responsible for verifying whether the outputs from other agents meet the required format:

1. Salary output must be in the format: number-number-currency-time_unit, or 0-0-None-None (if salary information is not available).
2. Seniority output must be one of the following valid levels:
experienced, intermediate, senior, entry level, assistant, lead, head, junior, graduate, trainee, associate, principal, apprentice, executive, manager, director, entry-level, chief, deputy, mid-level, specialist, experienced assistant, supervisor, qualified, student, board, graduate/junior, senior associate, mid-senior
3. Arrangement output must be one of: OnSite, Hybrid, or Remote.

You do not need to extract information from the text — only check whether the output from the agents conforms to these formats.

If the output is valid, respond with 'TERMINATE'.

If the output is invalid, use the EXACT agent name with @ to request a retry:
- For salary issues: @Salary
- For seniority issues: @Seniority  
- For arrangement issues: @Arrangement

Only one agent should respond after your message.
Do not ask the planner or the user to take any action."""
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

    if len(messages) <= 1:
        return planner_agent
    
    # last message content from planner
    last_message = messages[-1].get("content", "").strip()

    # # If last speaker was User, let planner decide the task
    # if last_speaker == user_proxy_agent:
    #     return planner_agent
    
    # If last speaker was Planner, let Planner decide which task agent to speak next
    if last_speaker == planner_agent:
        if "SALARY" in last_message:
            return salary_agent
        elif "SENIORITY" in last_message:
            return seniority_agent
        elif "ARRANGEMENT" in last_message:
            return arrangement_agent
        else:
            return critic_agent
        
    # If last speaker was a task agent, let Critic agent to speak for evaluation
    if last_speaker in [salary_agent, seniority_agent, arrangement_agent]:
        return critic_agent
    
    # If last speaker was Critic, let Critic decide to send back to task agent or terminate
    if last_speaker == critic_agent:
        if "TERMINATE" in last_message:
            # Correct output condition met
            return None # None will flag Groupchat to terminate
        elif "@Salary" in last_message:
            return salary_agent
        elif "@Seniority" in last_message:
            return seniority_agent
        elif "@Arrangement" in last_message:
            return arrangement_agent
        else:
            # Critic agent failed to evaluate
            # We send back to planner to restart the task
            return planner_agent # If critic didn't specify, go back to planner to assign task
    
    # Default fallback
    return None

def extract_output(groupchat, prompt):
    # Reset the groupchat to clear previous messages
    groupchat.reset()

    planner_agent.initiate_chat(
    manager,
    message=prompt,
    silent=True # Turn False if we want to see the Groupchat output
    )

    messages = groupchat.messages

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
    agents=[planner_agent, salary_agent, seniority_agent, arrangement_agent, critic_agent],
    messages=[],
    max_round=9,
    speaker_selection_method=custom_speaker_selection
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)


# print_groupchat_log(groupchat)

# user_proxy_agent.initiate_chat(
#     manager,
#     message="Extract salary information from the job ad: 'The salary is $32-$45 per hour and the work location is USA.'",
# )

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

save_path = root_dir / "results_collection" / "autogen" / "gpt_mulitAgent_test_all_v2.json"

df = pd.DataFrame(data)
df['y_pred'] = answers
df.to_json(save_path, orient='records', indent=4, force_ascii=False)


# Testing only
# final_output = extract_output(groupchat)
# print(f"\nFinal Output: {final_output}\n")

# Note:
# Test Sample Size: 1355
# Inference time: 2404.9750740528107
# Inference time: 2426.4098238945007