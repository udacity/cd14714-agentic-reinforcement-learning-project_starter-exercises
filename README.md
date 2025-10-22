# Agentic Reinforcement Learning for Conference Meeting Optimization

This project implements a multi-stage training pipeline for optimizing meeting recommendations at conferences using agentic reinforcement learning techniques. You'll build an intelligent agent that learns to predict successful meeting outcomes and make optimal recommendations.

## Getting Started

This starter code provides the framework for implementing supervised fine-tuning (SFT) and reinforcement learning fine-tuning (RLFT) using Direct Preference Optimization (DPO).

### Dependencies

```bash
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
trl>=0.7.0
datasets>=2.14.0
pandas>=2.0.0
numpy>=1.24.0
npcpy>=0.1.0
```

### Installation

1. Clone this repository to your local machine
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install torch transformers peft trl datasets pandas numpy npcpy
```

4. Verify installation by checking imports:
```bash
python -c "import torch, transformers, peft, trl, npcpy; print('All dependencies installed successfully')"
```

## Project Structure

```
starter/
├── data_classes.py           # Core data models and simulation logic
├── starter_sft.py            # Stage 1: Supervised Fine-Tuning implementation
├── starter_agentic_traces.py # Stage 2: Agent trace collection
└── starter_agentic_rlft.py   # Stage 3: Reinforcement Learning Fine-Tuning
```

## Project Instructions

### Stage 1: Supervised Fine-Tuning (SFT)
1. Open `starter/starter_sft.py`
2. Complete the `SFTConfig` dataclass by filling in the `'YOUR CODE HERE'` placeholders:
   - Set appropriate LoRA parameters (r, alpha, dropout)
   - Configure training hyperparameters (epochs, learning rate, weight decay)
   - Determine overfitting threshold
   - Set max_new_tokens for generation
3. Run the SFT training:
   ```bash
   python starter/starter_sft.py
   ```
4. Monitor training loss and validation metrics

### Stage 2: Agent Trace Collection
1. Open `starter/starter_agentic_traces.py`
2. Review the agent personas and their decision-making strategies
3. Implement any missing tool functions (marked with `'YOUR CODE HERE'`)
4. Generate agent traces:
   ```bash
   python starter/starter_agentic_traces.py
   ```
5. Examine the generated CSV file with trace data

### Stage 3: Reinforcement Learning Fine-Tuning (RLFT)
1. Open `starter/starter_agentic_rlft.py`
2. Complete the `calculate_reward()` function with appropriate reward values
3. Implement the preference pairing strategy in `create_preference_dataset_from_traces()`
4. Run RLFT with DPO:
   ```bash
   python starter/starter_agentic_rlft.py
   ```
5. Evaluate the improved model performance

## Testing

### Unit Tests
Test individual components:
```python
# Test the reward function
from starter_agentic_rlft import calculate_reward
trace = {"final_recommendation_parsed": {"recommendation": "YES"},
         "tools_used": ["tool1"],
         "completed_naturally": True,
         "ground_truth": 0.8}
reward = calculate_reward(trace)
assert -1.0 <= reward <= 1.0
```

### Integration Testing
1. Verify SFT model saves correctly to `models/sft_prediction_model_gemma_270m`
2. Check agent traces are saved to CSV with all required fields
3. Confirm RLFT model shows improved accuracy over baseline

### Evaluation Metrics
- **SFT**: Correlation and MAE between predicted and ground truth probabilities
- **Agent Traces**: Completion rate, tool usage, and recommendation accuracy
- **RLFT**: Final accuracy improvement over SFT baseline

## Deliverables

1. **Completed Code**: All `'YOUR CODE HERE'` placeholders filled with working implementations
2. **Trained Models**: SFT and RLFT model checkpoints
3. **Training Report**: Document including:
   - Chosen hyperparameters and justification
   - Training curves (loss, validation metrics)
   - Performance comparison between stages
   - Analysis of agent behavior patterns
   - Lessons learned and challenges faced

## Tips for Success

- Start with conservative hyperparameters to avoid overfitting
- Monitor training loss carefully - stop if it drops too low (< 0.01 suggested)
- Experiment with different reward structures in RLFT
- Use smaller batch sizes if running on limited hardware
- Save checkpoints frequently during long training runs

## Built With

* [PyTorch](https://pytorch.org/) - Deep learning framework
* [Transformers](https://huggingface.co/transformers/) - Pre-trained models and training utilities
* [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
* [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
* [NPCPy](https://github.com/npcpy/npcpy) - Agent framework for NPC interactions

## License

[License](LICENSE.txt)
