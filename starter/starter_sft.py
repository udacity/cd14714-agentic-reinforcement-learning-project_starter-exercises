# Structured Fine Tuning STARTER 
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


from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response


### Core data classes
from data_classes import  TimeSlot, ConferenceSimulator, PersonDescriptor, safe_extract_json



# Ensure SFT Config is set up to train well. This may require some testing and tweaking and iterative solving to find a good set that works.

@dataclass
class SFTConfig:
    base_model_name: str = "google/gemma-3-270m-it"
    output_model_path: str = "models/sft_prediction_model_gemma_270m"
    sft_data_path: str='data/sft_training_data.csv'
    lora_r: int = 'YOUR CODE HERE'
    lora_alpha: int ='YOUR CODE HERE'
    use_4bit: bool = False
    fp16: bool = False
    bf16: bool = False
    lora_dropout: float = 'YOUR CODE HERE'
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    num_train_epochs: int = 'YOUR CODE HERE'
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 'YOUR CODE HERE'
    logging_steps: int = 10
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine_with_restarts"  
    weight_decay: float = 'YOUR CODE HERE'

def validate_model(predictor: MeetingPredictor, val_examples: List[Dict], num_samples: int = 20):
    predictions = []
    ground_truths = []
    
    for i, example in enumerate(val_examples[:num_samples]):
        try:
            # YOUR CODE HERE
            # LOOP THROUGH EXAMPLES TO VALIDATE YOUR FINE TUNED MODEL IS WORKING

            pred_result = ...
            predictions.append(...)
            ground_truths.append(...)
        except:
            continue
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)    
    correlation = np.corrcoef(predictions, ground_truths)[0, 1]
    mae = np.mean(np.abs(predictions - ground_truths))
    return correlation, mae, predictions, ground_truths

def run_sft_fine_tuning(config: SFTConfig, train_examples: List[Dict] = None, val_examples: List[Dict] = None):
    if train_examples is None:
        #
        dataset = load_dataset('csv', data_files=config.sft_data_path, split='train')
        
        formatted_examples = []
        for example in dataset:
            target_str = json.dumps(example['target_json_output'])
            text = f"<start_of_turn>user\n{example['input_text']}<end_of_turn>\n<start_of_turn>model\n{target_str}<end_of_turn>"
            formatted_examples.append({"text": text})
    else:
        print("--- Using provided training examples ---")
        formatted_examples = []
        for example in train_examples:
            target_str = json.dumps(example['target_json_output'])
            text = f"<start_of_turn>user\n{example['input_text']}<end_of_turn>\n<start_of_turn>model\n{target_str}<end_of_turn>"
            formatted_examples.append({"text": text})
    

    #set up the Model for training and train:

    dataset = Dataset.from_list(formatted_examples)
    print(f"Dataset created with {len(dataset)} examples")
    
    print(f"\n--- Configuring Model ('{config.base_model_name}') for SFT ---")
    
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name, 
                                                 trust_remote_code=True, 
                                                 attn_implementation="eager")
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, 
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(r=config.lora_r, 
                             lora_alpha=config.lora_alpha, 
                             lora_dropout=config.lora_dropout, 
                             target_modules=config.lora_target_modules,
                             bias="none", 
                             task_type="CAUSAL_LM")

    # Why is the above a Causal LM task? be sure to write about it in your report!
    training_args = TrainingArguments(
        output_dir=config.output_model_path, 
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size, 
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim, 
        logging_steps=config.logging_steps, 
        learning_rate=config.learning_rate,
        fp16=config.fp16, 
        bf16=config.bf16, 
        lr_scheduler_type=config.lr_scheduler_type, 
        group_by_length=True, 
        save_steps=50,
        weight_decay= config.weight_decay
    )

    trainer = SFTTrainer(model=model, 
                         train_dataset=dataset,
                         peft_config=peft_config, 
                         args=training_args)

    print("\n--- Starting Supervised Fine-Tuning ---")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    training_losses = []
    validation_correlations = []
    validation_maes = []
    
    for epoch in range(config.num_train_epochs):
        print(f"\n=== EPOCH {epoch + 1}/{config.num_train_epochs} ===")
        
        trainer.train(resume_from_checkpoint=False if epoch == 0 else True)
        
        current_loss = trainer.state.log_history[-1]['train_loss'] if trainer.state.log_history else 0
        training_losses.append(current_loss)
        print(f"Training Loss: {current_loss:.4f}")
    
        # Early stopping if loss becomes too low
        if current_loss <  'YOUR CODE HERE': 
            # what is a reasonable threshold for this training set? Overfitting will cause many issues so we want to be sure
            # we dont make a mistake here because that will affect the successfulness of the to model down the line.
            print("Loss dropped too low. Potential overfitting detected. Stopping training.")
            break

        if val_examples:
            print("Running validation...")
            val_predictions = []
            val_ground_truths = []
            
            trainer.model.eval()
            for i, val_example in enumerate(val_examples[:50]):

                test_prompt = f"<start_of_turn>user\n{val_example['input_text']}<end_of_turn>\n<start_of_turn>model\n"
                inputs = tokenizer(test_prompt, 
                                   return_tensors="pt", 
                                   truncation=True, 
                                   padding=True)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)

                with torch.no_grad():
                    outputs = trainer.model.generate(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        max_new_tokens='YOUR CODE HERE', # We want to make sure the max new tokens is small enough to avoid the model producing needless tokens, but not too small to restrict the outputs. 
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    response_text = tokenizer.decode(outputs[0], 
                                                     skip_special_tokens=True)                
                try:
                    parsed_json = safe_extract_json(response_text)
                    predicted_prob = float(parsed_json['probability'])
                    
                    val_predictions.append(predicted_prob)
                    val_ground_truths.append(val_example['ground_truth_prob'])
                    
                    if i % 10 ==0:  
                        print(f"Sample {i}: Pred={predicted_prob:.3f}, True={val_example['ground_truth_prob']:.3f}")                        
                except Exception as e:
                    print(f"VALIDATION FAILED on sample {i}: {str(e)}")
                    print(f"Raw response: {response_text}")
                    continue
            val_predictions = np.array(val_predictions)
            val_ground_truths = np.array(val_ground_truths)
            correlation = np.corrcoef(val_predictions, val_ground_truths)[0, 1] if len(val_predictions) > 1 else 0.0
            mae = np.mean(np.abs(val_predictions - val_ground_truths))
            validation_correlations.append(correlation)
            validation_maes.append(mae)
            print(f"Validation - Correlation: {correlation:.3f}, MAE: {mae:.3f}")
            print("Sample predictions:")
            for i in range(min(3, len(val_examples))):
                print(f"  Predicted: {val_predictions[i]:.3f}, True: {val_ground_truths[i]:.3f}")
    trainer.save_model(config.output_model_path)
    print(f"SFT model saved to '{config.output_model_path}'")
    

if __name__ == "__main__":
    print("--- STEP 1: Generating SFT Data for Prediction Model ---")
    sft_examples_to_generate = 2000
    
    os.makedirs("data", exist_ok=True)
    csv_path = "data/sft_training_data.csv"
    
    print(f"Found existing data at '{csv_path}', loading...")
    df = pd.read_csv(csv_path)
    sft_dataset_examples = []
    for _, row in df.iterrows():
        example = {
            'input_text': row['input_text'],
            'target_json_output': {
                'probability': row['target_probability'],
                'reason': row['target_reason']
            },
            'ground_truth_prob': row['ground_truth_prob']
        }
        sft_dataset_examples.append(example)
    print(f"Loaded {len(sft_dataset_examples)} existing examples")
    
    print(f"\n--- Sample Training Examples ---")
    for i in range(min(3, len(sft_dataset_examples))):
        example = sft_dataset_examples[i]
        print(f"\nExample {i + 1}:")
        print(f"Input: {example['input_text'][:300]}...")
        print(f"Target: {example['target_json_output']}")
        print(f"Ground Truth: {example['ground_truth_prob']:.3f}")

    print("\n--- STEP 2: Fine-Tuning the Gemma-270m Prediction Model ---")
    sft_config = SFTConfig()
            
    if torch.cuda.is_available():
        print("Using CUDA")
        sft_config.bf16 = True
        sft_config.fp16 = False
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        sft_config.bf16 = False  
        sft_config.fp16 = False  # MPS doesn't support fp16 mixed precision
        sft_config.per_device_train_batch_size = 2  
        sft_config.gradient_accumulation_steps = 4   
    else:
        print("*** Using CPU ***")
        sft_config.use_4bit = False
        sft_config.bf16 = False
        sft_config.fp16 = False

    split_idx = int(0.8 * len(sft_dataset_examples))
    train_examples = sft_dataset_examples[:split_idx]
    val_examples = sft_dataset_examples[split_idx:]
    
    print(f"Training on {len(train_examples)} examples, validating on {len(val_examples)}")

    run_sft_fine_tuning(sft_config, train_examples, val_examples)

    print("\n--- STAGE 2 would begin here ---")
    print("Next steps would involve:")
    print(' With your fine tuned model saved, we can now start building our agent traces')
    print(' We dont know if this SFT model will be helpful truly, so we need to explore many possible ways it could be used in different scenarios and compare its effectiveness to other methods and tools')
    print('be sure to report on your models training losses , how many epochs you used etc, in your report.')