# Starter Code Documentation

This directory contains the starter code for the Agentic Reinforcement Learning project. Each file has specific placeholders marked with `'YOUR CODE HERE'` that you need to complete.

## File Overview

### `data_classes.py`
Core data structures and simulation logic. This file is **complete** and should not be modified.
- `ConferenceSimulator`: Simulates conference meeting dynamics
- `Person`, `Meeting`, `TimeSlot`: Data models for the simulation
- `PersonDescriptor`: Generates natural language descriptions of attendees
- Helper functions for JSON parsing and metrics calculation

### `starter_sft.py`
**Stage 1: Supervised Fine-Tuning**

Placeholders to complete:
- **Lines 36-50**: `SFTConfig` parameters
  - `lora_r`: LoRA rank (try 8-16)
  - `lora_alpha`: LoRA scaling (try 16-32)
  - `lora_dropout`: Dropout for LoRA (try 0.05-0.1)
  - `num_train_epochs`: Training epochs (try 3-5)
  - `learning_rate`: Learning rate (try 2e-4 to 5e-4)
  - `weight_decay`: Weight decay (try 0.01)
- **Line 153**: Overfitting threshold (suggested: 0.01)
- **Line 179**: `max_new_tokens` for generation (try 50-100)
- **Lines 58-63**: Validation loop implementation

### `starter_agentic_traces.py`
**Stage 2: Agent Trace Collection**

This file contains the agent system with multiple personas. Key components:
- **Lines 49-97**: Agent persona definitions (can be modified to improve quality)
- **Lines 100+**: Tool functions that agents use to gather information
- `AgentToolLoop`: Manages the agent's tool execution
- `AgentTraceCollector`: Collects detailed execution traces

Most of this file is complete, but review the tool implementations for any missing logic.

### `starter_agentic_rlft.py`
**Stage 3: Reinforcement Learning Fine-Tuning**

Placeholders to complete:
- **Lines 69-84**: `calculate_reward()` function
  - Return -1.0 for missing recommendation
  - Return -0.75 for incomplete execution
  - Return -0.5 for no tools used
  - Return -0.25 for invalid outcomes
  - Return 1.0 for correct predictions
  - Return 0.1 for incorrect predictions
- **Lines 99-100+**: Preference pairing strategy
  - Create pairs with meaningful reward gaps
  - Balance chosen/rejected samples

## Execution Order

1. First, run SFT to create the base prediction model:
   ```bash
   python starter_sft.py
   ```

2. Generate agent traces with the trained model:
   ```bash
   python starter_agentic_traces.py
   ```

3. Run RLFT to improve the model using preferences:
   ```bash
   python starter_agentic_rlft.py
   ```

## Common Issues and Solutions

### Memory Issues
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Use fp16 or bf16 if supported by your hardware

### Training Instability
- Start with lower learning rates
- Increase warmup steps
- Monitor loss carefully for sudden spikes

### Poor Performance
- Ensure sufficient training data (2000+ examples recommended)
- Check that reward function properly distinguishes good/bad outcomes
- Verify agent tools are returning meaningful information

## Data Files Generated

- `data/sft_training_data.csv`: Training data for SFT (provided)
- `data/agent_traces_*.csv`: Collected agent execution traces
- `models/sft_prediction_model_gemma_270m/`: SFT model checkpoint
- `models/rlft_model/`: RLFT model checkpoint

## Debugging Tips

1. Add print statements in tool functions to trace execution
2. Check CSV files to verify trace quality
3. Plot training losses to detect overfitting
4. Test models interactively before full training runs
5. Save intermediate checkpoints frequently

## License
[License](../LICENSE.txt)
