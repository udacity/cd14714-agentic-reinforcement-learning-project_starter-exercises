# AGENTIC RL TRACES STARTER 
from dataclasses import dataclass, field
from datasets import Dataset
from datetime import datetime
import glob
import json
import numpy as np
import os
import pandas as pd
from peft import LoraConfig, PeftModel
import random
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from typing import Dict, Any, List, Optional, Any

from data_classes import  get_true_outcome, TimeSlot, ConferenceSimulator, PersonDescriptor, safe_extract_json

from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response
def calculate_reward(trace: Dict[str, Any]) -> float:
    final_rec_data = trace.get("final_recommendation_parsed")
    tools_used = trace.get("tools_used", [])
    completed_naturally = trace.get("completed_naturally", False)

    if not final_rec_data or 'recommendation' not in final_rec_data:
        return -1.0
    if not completed_naturally:
        return -0.75
    if not tools_used:
        return -0.5

    agent_outcome = final_rec_data.get('recommendation', 'FAIL').upper()
    if agent_outcome not in ["YES", "NO"]:
        return -0.25

    true_outcome = get_true_outcome(trace['ground_truth'])

    if agent_outcome == true_outcome:
        return 1.0
    else:
        return 0.1


### Core data classes
from data_classes import  TimeSlot, ConferenceSimulator, PersonDescriptor, safe_extract_json
system_prompt_configurations = [
    {
        "name": "Pax",
        "primary_directive": """You are a balanced meeting advisor.
Evaluate compatibility and potential for productive discussion, weighing both positive and
negative signals equally to provide a well-rounded recommendation. Use tools to gather information,
then reason through your decision step by step."""
    },
    {
        "name": "Vigil",
        "primary_directive": """You are a cautious meeting gatekeeper.
Your priority is to prevent unproductive meetings. Only recommend meetings if there is
extremely high confidence of success. Use tools to thoroughly analyze potential issues.
Avoid false positives at all costs."""
    },
    {
        "name": "Aura",
        "primary_directive": """You are an optimistic connector.
Your goal is to foster connections. Use tools to find potential synergies and common ground.
Err on the side of recommending meetings, even with moderate compatibility, to encourage networking."""
    },
    {
        "name": "Cortex",
        "primary_directive": """You are a data-driven analyst.
Use tools systematically to gather quantitative insights. Your recommendations must strictly 
follow the analytical data. Process each tool output carefully before proceeding."""
    },
    {
        "name": "Echo",
        "primary_directive": """You are an intuitive empath.
Use tools to understand the interpersonal dynamics, then rely heavily on your intuitive
assessment. Focus on reading between the lines of what the tools tell you."""
    },
    {
        "name": "Maven",
        "primary_directive": """You are a strategic prioritizer.
Use tools to understand potential synergies and strategic value. Prioritize meetings that
offer the highest strategic value or growth potential for participants."""
    },
    {
        "name": "Nexus",
        "primary_directive": """You are an efficiency expert.
Use tools to assess time efficiency and return on investment. Only recommend meetings
that represent highly efficient use of time for both parties."""
    }
]
# THE ABOVE ARE PROVIDED FOR SIMPLICITY. LEARNERS MAY TWEAK THESE AS THEY SEE FIT
# TO FURTHER IMPROVE THE QUALITY OF THE RESPONSES AND TRAINING DATA.



def extract_common_interests_and_topics(person_a_desc: str, person_b_desc: str) -> str:
   """
   Analyzes two person descriptions to identify common interests and conversation topics.
   
   Args:
       person_a_desc: Description of the first person
       person_b_desc: Description of the second person
       
   Returns:
       JSON string with topics, interests, compatibility score and explanation
   """
   prompt = f"""Analyze the two descriptions and identify common interests and potential conversation topics.

Person A: {person_a_desc}
Person B: {person_b_desc}

Return a single JSON object with:
- "topics": list of potential conversation topics
- "interests": list of common interests
- "compatibility_score": float from 0-1
- "explanation": string explaining the analysis

like so
""" + """

{
   'topics': ['topic1', 'topic2'],
   'interests': ['interest1', 'interest2'],
   'compatibility_score': 0.x,
   'explanation': '

}
"""
   response = get_llm_response(prompt, model='qwen3:0.6b', provider='ollama', format='json')
   json_out = response['response']

   return json_out

def assess_time_slot_fit_and_energy(person_a_desc: str, person_b_desc: str, time_slot: str) -> str:
    """
    Evaluates how well a time slot matches both people's energy levels and availability.
    
    Args:
        person_a_desc: Description of the first person
        person_b_desc: Description of the second person
        time_slot: The proposed meeting time slot
        
    Returns:
        JSON string with fit scores, energy levels, and potential issues
    """
    prompt = f"""Evaluate how well this time slot fits both people's energy levels and state.

    Person A: {person_a_desc}
    Person B: {person_b_desc}
    Time Slot: {time_slot}

    Return a JSON object with:
    - "fit_score": float from 0-1 
    - "person_a_energy": float from 0-1 (e.g., 0.2 for low, 0.8 for high)
    - "person_b_energy": float from 0-1 (e.g., 0.2 for low, 0.8 for high)
    - "summary": string summary
    - "red_flags": list of potential issues
    like so """+ """

    {
    'fit_score': 0.x, 
    'person_a_energy': 0.x, 
    'person_b_energy': 0.x, 
    'summary': 'string', 
    'red_flags':['item1', 'item2']
    }
    """
    response = get_llm_response(prompt, 
                                model='qwen3:0.6b',
                                provider='ollama', 
                                format='json')

    json_out = response['response']
    return json_out

def predict_follow_up_potential(person_a_desc: str, person_b_desc: str, time_slot: str) -> str:
    """
    Predicts the likelihood of productive follow-up interactions after a meeting.
    
    Args:
        person_a_desc: Description of the first person
        person_b_desc: Description of the second person
        time_slot: The meeting time slot
        
    Returns:
        JSON string with follow-up probability, relationship potential and suggestions
    """
    prompt = f"""Predict the follow-up potential for a meeting between these people.

    Person A: {person_a_desc}
    Person B: {person_b_desc}
    Time Slot: {time_slot}

    Return a JSON object with:
    - "follow_up_probability": float from 0-1
    - "relationship_potential": string (low/medium/high)
    - "business_potential": string (low/medium/high)
    - "rationale": string explanation
    - "suggested_next_steps": list of strings

    like: """ + """

    {
    'follow_up_probability': 0.x, 
    'relationship_potential': 'low/medium/high', 
    'business_potential': 'low/medium/high', 
    'rationale': 'string
    'suggested_next_steps': ['item1', 'item2']

    
    }
    """
    response = get_llm_response(prompt, model='qwen3:0.6b', provider='ollama', format='json')
    json_out = response['response']

    return json_out



### Set up your fine tuning predictor tool!
def predict_meeting_success_tool(person_a_desc: str, person_b_desc: str, time_slot: str) -> str:
    """
    Predicts the probability of a successful meeting between two people at a given time slot.
    
    Args:
        person_a_desc: Description of the first person including their personality and current state
        person_b_desc: Description of the second person including their personality and current state  
        time_slot: The time slot when the meeting would occur
        
    Returns:
        JSON string containing success probability and reasoning
    """
    model_path = "models/sft_prediction_model_gemma_270m"
    if not os.path.exists(model_path):
        return json.dumps({"status": "error", "message": "SFT model not found."})
    try:
        # Your code here
        # Should be two lines        
        #intermediate = ...
        #result = intermediate.some_method(...)

        if 'probability' not in result:
            return json.dumps({"status": "error", "message": "SFT prediction missing data."})
        return json.dumps({
            "status": "success",
            "probability": result['probability'],
            "reason": result.get('reason', 'N/A')
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": f"SFT model failed: {e}"})    
    
TOOLS = [
    predict_meeting_success_tool,
    extract_common_interests_and_topics,
    assess_time_slot_fit_and_energy,
    predict_follow_up_potential,
]


class AgentTraceCollector:
    def __init__(self):
        self.traces = []

    def record_trace(self, **kwargs):
        self.traces.append(kwargs)

    def save_traces_to_file(self, filename: str):
        if not self.traces:
            print("No traces to save.")
            return
        df_data = []
        for trace in self.traces:
            agent_outcome = trace.get("final_recommendation_parsed")
            row = {
                "system_prompt_name": trace.get("system_prompt_name"),
                "initial_user_prompt": trace.get("initial_user_prompt"),
                "tools_used": ",".join(trace.get("tools_used", [])),
                "total_iterations": trace.get("total_iterations"),
                "ground_truth_prob": trace.get("ground_truth"),
                "true_outcome": get_true_outcome(trace.get("ground_truth", 0.0)),
                "agent_outcome": agent_outcome,
                "reward": trace.get("reward"),
                "final_recommendation_reasoning": (trace.get("final_recommendation_parsed") or {}).get("reasoning"),
            }
            df_data.append(row)
        df = pd.DataFrame(df_data)
        df.to_csv(filename, index=False)
        print(f"Traces saved to {filename}")


class AgentToolLoop:
    # Here we use the npcpy tool use loop because it can auto call our tools for us and report back the results
    def __init__(self, 
                 agent: NPC, 
                 max_iterations: int = 'YOUR CODE HERE'):
        # the number of max iterations should be sufficient to allow the model to get somewhere
        # but shouldnt be so high that it is spending 20 some odd turns deciding what to do..

        self.agent = agent 
        self.max_iterations = max_iterations
    def _parse_final_json(self, 
                          text: str) -> Optional[Dict]:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match: 
                return None
            data = json.loads(match.group(0))
            return data
        except json.JSONDecodeError:
            return None

    def run_tool_loop(self, initial_prompt: str) -> Dict[str, Any]:
        enhanced_directive = self.agent.primary_directive + """

        Think carefully about how to address the user prompt. Call tools. Gather information. 

    """
        messages = [{"role": "system", "content": enhanced_directive}]
        raw_responses = []
        final_recommendation_data = None
        completed_naturally = False
        current_prompt = initial_prompt
        for i in range(self.max_iterations):
            print(f'itearation {i}')
            response_obj = self.agent.get_llm_response(
                current_prompt,
                messages=messages,
                auto_process_tool_calls=True
            )
            print(response_obj['response'][-500:])
        

            raw_responses.append(response_obj)
            messages = response_obj.get('messages') or messages
            last_assistant_content = messages[-1].get('content', '')

            final_recommendation_data = self._parse_final_json(last_assistant_content)
            
            if final_recommendation_data and 'recommendation' in final_recommendation_data and 'reasoning' in final_recommendation_data:
                 
                completed_naturally = True
                break
        
            current_prompt = "Are you finished? If so, provide your final JSON recommendation. If not, continue."
            if i ==1:
                messages[0]['content'] +='''Once you have sufficient information, 
                    provide your final recommendation as JSON:
                {
                "recommendation": "YES" or "NO", 
                "reasoning": "detailed explanation of your decision"
                }
                Do not include any other keys in your response. Your JSON must only be these two keys. It doesnt matter if you gathered infromation from tools and want to share it with the user, they will be able
                to see it through the inspection of the tool calling traces. It is importantly only your task to return this information very plainly once it is finished.
                '''

        return {
            "raw_responses": raw_responses,
            "final_recommendation": final_recommendation_data,
            "total_iterations": i + 1,
            "completed_naturally": completed_naturally
        }






def generate_agent_traces_for_training(num_scenarios_per_agent: int) -> List[Dict[str, Any]]:
    """
    Generates training traces by running agents through scenarios with different system prompts.
    
    Args:
        num_scenarios_per_agent: An integer for the number of scenarios to explore per system prompt
        
    Returns:
        List of trace dictionaries containing agent decisions and outcomes
    """
    traces_collector = AgentTraceCollector()
    descriptor = PersonDescriptor(temperature=0.8)
    for config in system_prompt_configurations:
        print(f"\n--- Generating traces for agent: {config['name']} ---")
        for j in range(num_scenarios_per_agent):
            current_agent = NPC(
                name=config["name"].lower(),
                primary_directive=config["primary_directive"],
                tools=TOOLS,
                model='qwen3:1.7b',
                provider='ollama',
            )
            tool_loop = AgentToolLoop(current_agent, max_iterations=8)
            simulator = ConferenceSimulator(num_attendees=50, seed=random.randint(0, 10000))
            p1_id, p2_id = random.sample(list(simulator.attendees.keys()), 2)
            person_a, person_b = simulator.attendees[p1_id], simulator.attendees[p2_id]
            time_slot = random.choice(list(TimeSlot))
            p1_desc = descriptor.generate_description(person_a, time_slot)
            p2_desc = descriptor.generate_description(person_b, time_slot)
            ts_str = time_slot.value.replace('_', ' ').title()
            _, gt_prob = simulator._calculate_meeting_success(person_a, person_b, time_slot)
            initial_prompt = f"""Your task is to decide if two people should meet. Use the available tools to gather information step-by-step. When you have enough information, stop using tools and provide your final answer as a single JSON object.

Person A: {p1_desc}
Person B: {p2_desc}
Time Slot: {ts_str}

Begin your analysis by calling a tool."""
            loop_result = tool_loop.run_tool_loop(initial_prompt)
            all_tools_used = []
            for resp in loop_result['raw_responses']:
                if resp and resp.get('tool_calls'):
                    for call in resp.get('tool_calls'):
                        all_tools_used.append(call['function']['name'])
            trace_data = {
                "system_prompt_name": config["name"],
                "initial_user_prompt": initial_prompt,
                "final_recommendation_parsed": loop_result['final_recommendation'],
                "tools_used": list(set(all_tools_used)),
                "total_iterations": loop_result['total_iterations'],
                "ground_truth": gt_prob,
                "completed_naturally": loop_result['completed_naturally']
            }
            trace_data['reward'] = calculate_reward(trace_data)
            traces_collector.record_trace(**trace_data)
            true_outcome_str = get_true_outcome(gt_prob)
            agent_decision = loop_result.get('final_recommendation')
            print(f"Scenario {j+1}/{num_scenarios_per_agent} ({config['name']}): True Outcome={true_outcome_str} (Prob={gt_prob:.2f}), Agent said='{agent_decision}', Reward={trace_data['reward']:.2f}")
    return traces_collector.traces





if __name__ == "__main__":
    traces_csv_file = None
    csv_pattern = "agent_traces_*.csv"
    existing_csvs = sorted(glob.glob(csv_pattern), key=os.path.getmtime, reverse=True)
    num_traces_per_agent = 'YOUR CODE HERE'
    generated_traces_list = generate_agent_traces_for_training(num_traces_per_agent)
    traces_csv_file = f"agent_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    temp_collector = AgentTraceCollector()
    temp_collector.traces = generated_traces_list
    temp_collector.save_traces_to_file(traces_csv_file)
    print(' With your traces saved, we can now start doing reinforcement learning with DPO! hop on over to starter_agentic_rlft.py to begin.')
