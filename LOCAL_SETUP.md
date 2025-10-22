# Local Setup Instructions

## System Requirements

### Minimum Requirements
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space for models and datasets
- **Python**: 3.8 or higher

### GPU Support (Optional but Recommended)
- **NVIDIA GPU**: CUDA 11.7+ with at least 4GB VRAM
- **Apple Silicon**: M1/M2/M3 with Metal Performance Shaders (MPS)
- **CPU Fallback**: Works but significantly slower

## Installation Steps

### 1. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n rl-agent python=3.9
conda activate rl-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. GPU-Specific Setup

#### For NVIDIA GPUs
```bash
# Install CUDA-enabled PyTorch (if not already installed)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### For Apple Silicon (M1/M2/M3)
```bash
# PyTorch with MPS support (usually included in standard install)
pip install torch
```

### 4. Download Models

The Gemma model will be downloaded automatically on first run, but you can pre-download it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# This will download ~550MB for Gemma-2B model
model_name = "google/gemma-2b"
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Model downloaded successfully!")
```

**Note**: First-time download requires:
- Hugging Face account (free): https://huggingface.co/join
- Accept Gemma model terms: https://huggingface.co/google/gemma-2b
- Login via CLI: `huggingface-cli login`

## Troubleshooting

### Memory Issues
If you encounter out-of-memory errors:

1. **Reduce batch size** in training scripts:
   ```python
   per_device_train_batch_size=1  # Instead of 4
   gradient_accumulation_steps=4  # To maintain effective batch size
   ```

2. **Use gradient checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Use 8-bit quantization** (requires `bitsandbytes`):
   ```bash
   pip install bitsandbytes
   ```
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       load_in_8bit=True,
       device_map="auto"
   )
   ```

### CUDA Errors
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Clear cache if needed: `torch.cuda.empty_cache()`

### Apple Silicon Issues
- Ensure using ARM64 Python: `python -c "import platform; print(platform.machine())"`
- Should output `arm64`, not `x86_64`
- If MPS not working: `export PYTORCH_ENABLE_MPS_FALLBACK=1`

### Slow Training on CPU
If only CPU is available:
- Expect 10-20x slower training
- Consider using Google Colab (free GPU): https://colab.research.google.com
- Or use smaller model variants if available

## Testing Your Setup

Run this test script to verify everything is working:

```python
# test_setup.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_setup():
    print("Testing setup...")

    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")

    # Check device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU")
    else:
        device = "cpu"
        print("Using CPU (training will be slow)")

    # Test model loading
    print("\nTesting model loading...")
    try:
        model_name = "google/gemma-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto"
        )
        print("✓ Model loaded successfully!")

        # Test inference
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20)
        print("✓ Inference test passed!")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure to:")
        print("1. Run: huggingface-cli login")
        print("2. Accept terms at: https://huggingface.co/google/gemma-2b")
        return False

    print("\n✅ All tests passed! You're ready to start training.")
    return True

if __name__ == "__main__":
    test_setup()
```

## Resource Estimation

### Training Time Estimates (per epoch)
- **GPU (RTX 3060)**: ~5-10 minutes
- **Apple M1/M2**: ~10-20 minutes
- **CPU (i7/Ryzen 7)**: ~1-2 hours

### Memory Usage
- **Model**: ~2GB for Gemma-2B
- **Training**: Additional 4-6GB depending on batch size
- **Inference**: ~3GB total

## Alternative: Cloud Options

If local resources are insufficient:

1. **Google Colab** (Free)
   - Free GPU (T4) for up to 12 hours
   - Upload notebook and install requirements

## Getting Help

- Check GPU memory: `nvidia-smi` (NVIDIA) or Activity Monitor (Mac)
- Monitor training: Use `tqdm` progress bars in code
- Debug imports: `python -c "import [package]; print([package].__version__)"`
- Clear Python cache: `find . -type d -name __pycache__ -exec rm -r {} +`