
# AGENTIC RL Fine Tuning with DPO
from datasets import Dataset
from datetime import datetime
import json
import glob
import os
import pandas as pd
from peft import LoraConfig
import random
import time 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional, Any
from starter_agentic_traces import TOOLS, system_prompt_configurations, AgentToolLoop
from data_classes import  get_true_outcome, calculate_accuracy_metrics, PersonDescriptor, TimeSlot, ConferenceSimulator
from npcpy.npc_compiler import NPC




def load_trained_model(base_model_id: str, adapter_path: Optional[str]):
    """
    Loads a model with optional LoRA adapter.
    
    Args:
        base_model_id: The base model identifier to load
        adapter_path: Optional path to LoRA adapter, if None loads base model only
        
    Returns:
        Tuple of (model, tokenizer)
    """    
    from peft import PeftModel
    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="auto",
        attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading and merging adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    else:
        print(f"No adapter found at {adapter_path}, using base model.")
    return model, tokenizer


def calculate_reward(trace: Dict[str, Any]) -> float:
    """
    Calculates the reward from the trace based on correctness and completion criteria.
    
    Args:
        trace: A dictionary storing trace information including recommendation, tools used, and ground truth
        
    Returns:
        Float reward value between -1.0 and 1.0
    """

    final_rec_data = trace.get("final_recommendation_parsed")
    tools_used = trace.get("tools_used", [])
    completed_naturally = trace.get("completed_naturally", False)

    if not final_rec_data or 'recommendation' not in final_rec_data:
        return 'YOUR CODE HERE'
    if not completed_naturally:
        return 'YOUR CODE HERE'
    if not tools_used:
        return 'YOUR CODE HERE'

    agent_outcome = final_rec_data.get('recommendation', 'FAIL').upper()
    if agent_outcome not in ["YES", "NO"]:
        return 'YOUR CODE HERE'

    true_outcome = get_true_outcome(trace['ground_truth'])

    if agent_outcome == true_outcome:
        return 'YOUR CODE HERE'
    else:
        return 'YOUR CODE HERE'


def create_preference_dataset_from_traces(csv_file_path: str) -> Optional[Dataset]:
    df = pd.read_csv(csv_file_path)
    df['reward'] = pd.to_numeric(df['reward'], errors='coerce')
    valid_df = df.dropna(subset=['reward', 'initial_user_prompt', 'final_recommendation_reasoning']).copy()
    valid_df = valid_df[valid_df['reward'] > -1.0]
    
    if len(valid_df) < 2:
        print("Not enough valid traces to create preference pairs.")
        return None
    
    valid_df = valid_df.sort_values(by='reward', ascending=False)
    
    # YOUR CODE HERE, YOU DECIDE HOW TO PAIR PREFERENCES, whether or not to use a reward gap criteria
    # and how to score them.


    filtered_pairs = []

    high_reward_traces = ...
    low_reward_traces = ...
        
    for _, high_trace in high_reward_traces.iterrows():
        for _, low_trace in low_reward_traces.iterrows():
            
            # YOUR CODE HERE, YOU DECIDE HOW TO PAIR PREFERENCES, EMPHASIZE REWARDS, ETC.
            # Just ensure your pairs represent diverse responses so the model does not only
            # learn the same dichotomies.
            # This double for loop is not strictly necessary, and you may choose to
            # structure this part however you like.

    if len(filtered_pairs) < 'YOUR_THRESHOLD_HERE':
        print(f"Only {len(filtered_pairs)} pairs with sufficient reward gap found. This may cause overfitting.")
    
    return Dataset.from_list(filtered_pairs)


def train_model_with_dpo(
    csv_file_path: str,
    base_model_id: str,
    new_adapter_path: str,
):
    """
    Trains a model using Direct Preference Optimization from trace data.
    
    Args:
        csv_file_path: Path to CSV file containing training traces
        base_model_id: Identifier for the base model to train
        new_adapter_path: Path where the new LoRA adapter will be saved

    """

    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    print("\n--- Starting DPO Training Process ---")
    print(f"Loading traces from {csv_file_path}...")
    preference_dataset = create_preference_dataset_from_traces(csv_file_path)
    if preference_dataset is None or len(preference_dataset) == 0:
        print("No valid preference pairs created. Cannot proceed with DPO training.")
        return

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    peft_config = LoraConfig(
        r='YOUR INT HERE',              
        lora_alpha='YOUR INT HERE',    
        lora_dropout='YOUR FLOAT FRAc HERE', 
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["YOUR LIST HERE"] 
    )

    from trl import DPOTrainer, DPOConfig
        

    training_args = DPOConfig(
        output_dir="./dpo_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate='your learning rate',
        max_steps='your max steps',
        weight_decay='your weight decay',
        beta='your beta',
        logging_steps=2,
        save_steps=10,
        remove_unused_columns=False,
        max_length=8192,
        max_prompt_length=8192,
        dataloader_num_workers=0,
        fp16=False, 
        bf16=False,
        optim="adamw_torch",
        warmup_steps=2,
        save_strategy="steps",
        save_total_limit=3,
    )

    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=preference_dataset,
        peft_config=peft_config,
    )
    print("Starting DPO training...")
    trainer.train()
    print("DPO training complete.")
    print(f"Saving new LoRA adapter to '{new_adapter_path}'...")
    trainer.save_model(new_adapter_path)
    print("Adapter saved successfully.")


def run_local_agent_evaluation(
   model_path: str,
   test_scenarios: List[Dict],
   model_type: str
) -> List[Dict]:
   """
   Evaluates the fine tuned agent's performance on test scenarios.
   
   Args:
       model_path: Path or identifier for the model to evaluate
       test_scenarios: List of test scenario dictionaries
       model_type: String identifier for the model type being evaluated
       
   Returns:
       List of evaluation result dictionaries
   """

   print(f"\n--- Running Full Local Evaluation for {model_type.upper()} Model ---")
   results = []
   
   for persona_idx, persona in enumerate(system_prompt_configurations):
       for scenario in test_scenarios:
           scenario_id = f"{scenario['scenario_id']}_p{persona_idx}"
           print(f"--- Scenario {scenario_id} ({model_type}, {persona['name']}) ---")
           true_outcome = get_true_outcome(scenario['ground_truth'])            
           
           current_agent = NPC(
               name=persona["name"].lower(),
               primary_directive=persona["primary_directive"],
               tools=TOOLS,
               model=model_path,
               provider="transformers"
           ) 
           tool_loop = AgentToolLoop(current_agent, max_iterations=8)
           
           initial_prompt = f"""Your task is to decide if two people should meet. Use the available tools to gather information step-by-step. 

Person A: {scenario['p1_desc']}
Person B: {scenario['p2_desc']}
Time Slot: {scenario['ts_str']}

Begin your analysis by calling a tool."""
           
           loop_result = tool_loop.run_tool_loop(initial_prompt)
           final_rec_data = loop_result['final_recommendation']
           
           agent_outcome = "FAIL"
           if final_rec_data and 'recommendation' in final_rec_data:
               agent_outcome = final_rec_data['recommendation'].upper()
           
           is_correct = 1 if agent_outcome == true_outcome else 0
           results.append({
               'scenario_id': scenario_id,
               'persona': persona['name'],
               'is_correct': is_correct,
               'agent_outcome': agent_outcome,
               'true_outcome': true_outcome,
           })
           print(f"Scenario {scenario_id} complete. GT={true_outcome}, Agent said={agent_outcome}, Correct={bool(is_correct)}")
   
   return results

def evaluate_model_performance(
   base_model_id: str,
   adapter_path: str,
   test_scenarios_count: int = 20
):
   print(f"Generating {test_scenarios_count} fresh evaluation scenarios...")
   test_scenarios = []
   eval_seed_rng = random.Random(37)
   for i in range(test_scenarios_count):
       simulator = ConferenceSimulator(num_attendees=2000, seed=eval_seed_rng.randint(20000, 30000))
       descriptor = PersonDescriptor(temperature=0.8)
       if len(simulator.attendees) < 2: continue
       p1_id, p2_id = eval_seed_rng.sample(list(simulator.attendees.keys()), 2)
       p1, p2 = simulator.attendees[p1_id], simulator.attendees[p2_id]
       ts_enum = eval_seed_rng.choice(list(TimeSlot))
       p1_desc, p2_desc = descriptor.generate_description(p1, ts_enum), descriptor.generate_description(p2, ts_enum)
       ts_str = ts_enum.value.replace('_', ' ').title()
       _, gt_prob = simulator._calculate_meeting_success(p1, p2, ts_enum)
       test_scenarios.append({
           'p1_desc': p1_desc, 'p2_desc': p2_desc, 'ts_str': ts_str,
           'ground_truth': gt_prob, 'scenario_id': i
       })
   
   baseline_results = run_local_agent_evaluation(base_model_id, test_scenarios, "baseline")
   trained_results = run_local_agent_evaluation(adapter_path, test_scenarios, "trained")
   
   baseline_metrics = calculate_accuracy_metrics(baseline_results)
   trained_metrics = calculate_accuracy_metrics(trained_results)
   improvement = trained_metrics['accuracy'] - baseline_metrics['accuracy']
   
   return {'improvement_percent': improvement}


if __name__ == "__main__":
    traces_csv_file = None
    csv_pattern = "agent_traces_*.csv"
    existing_csvs = sorted(glob.glob(csv_pattern), 
                           key=os.path.getmtime, 
                           reverse=True)
    most_recent_csv = existing_csvs[0]
    file_age_hours = (time.time() - os.path.getmtime(most_recent_csv)) / 3600
    print(f"Found existing trace file: {most_recent_csv} (created {file_age_hours:.1f} hours ago)")

    df = pd.read_csv(most_recent_csv)
    traces_csv_file = most_recent_csv
    
    base_model = "Qwen/Qwen3-0.6B"
    adapter_path = "./qwen3-dpo-adapter-v1"

    train_model_with_dpo(
        csv_file_path=traces_csv_file,
        base_model_id=base_model,
        new_adapter_path=adapter_path,
    )

    print("\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)

    evaluate_model_performance(
        base_model_id=base_model,
        adapter_path=adapter_path,
        test_scenarios_count=3
    )