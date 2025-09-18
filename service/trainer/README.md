# Trainer Service

Neural network training and evaluation service for boolean formula trajectory generation using GPT-2 based models.

## Overview

The Trainer service provides a complete pipeline for training transformer-based models on boolean formula trajectories. It includes custom GPT-2 architectures, specialized data handling, training utilities, and comprehensive evaluation metrics focusing on avgQ (average satisfaction) optimization.

## Features

- **Custom GPT-2 Architecture**: Modified GPT-2 model with formula-specific embeddings and multi-component loss
- **Trajectory Dataset Management**: Efficient loading from local files or warehouse service
- **Data Augmentation**: Permutation-based augmentation for improved generalization
- **Custom Training Loop**: Extended HuggingFace Trainer with detailed loss component tracking
- **Comprehensive Evaluation**: avgQ-based metrics and distribution analysis
- **Data Cleaning Utilities**: Pre-processing tools for trajectory quality control
- **TRL Integration**: Reward function for reinforcement learning fine-tuning

## Installation

1. Install the longshot library:
```bash
cd ../../library
pip install -e .
cd ../service/trainer
```

2. Install service dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure warehouse service is running if downloading trajectories (see main README).

## Quick Start

### Training a Model

Basic training with default settings:
```bash
python train.py
```

The training script expects data in `./data/n{num_vars}w{width}.json` format.

### Evaluating a Model

Evaluate a trained model:
```bash
python eval.py --model-path ./models/n3w2-2025-09-13T17:04:40
```

### Data Cleaning

Clean and prepare trajectory data:
```bash
python data_cleaning.py \
    --num-vars 4 \
    --width 3 \
    --avgq-threshold 0.5 \
    --output cleaned_data.json
```

## Component Documentation

### 1. Model Architecture (`model.py`)

#### GPT2ForLongshot

Custom GPT-2 model with specialized components for formula generation:

**Key Features:**
- Variable and clause embeddings for formula structure
- Semantic embeddings with optional weight sharing
- Multi-component loss function (generation + avgQ regression + diversity)
- HuggingFace-compatible serialization

**Configuration Parameters:**
```python
GPT2ForLongshotConfig(
    num_vars=3,           # Number of boolean variables
    width=2,              # Maximum formula width
    n_embed_lit=32,       # Literal embedding dimension
    ub_q=3.0,            # Upper bound for Q values
    alpha=1.0,           # Generation loss weight
    beta=40.0,           # AvgQ loss weight
    gamma=0.2,           # Diversity loss weight
    share_semantic=True,  # Share semantic embeddings
    universal=False,      # Use universal quantification
    gpt2_config=...      # Standard GPT2Config
)
```

### 2. Training Script (`train.py`)

Main training pipeline with configurable parameters:

**Default Configuration:**
- Model: 18-layer GPT-2 with 256 embedding dimensions
- Training: 350 epochs, batch size 32, learning rate 2e-5
- Data split: 80% train, 20% eval
- Augmentation: Input permutation enabled
- Device: CUDA (GPU 1 by default)

**Key Settings to Tune:**
```python
# Model architecture
n_positions=64        # Maximum sequence length
n_embd=256           # Embedding dimension
n_layer=18           # Number of transformer layers
n_head=8             # Number of attention heads

# Training hyperparameters
learning_rate=2e-5
per_device_train_batch_size=32
num_train_epochs=350
weight_decay=0.00
```

**Output:**
- Checkpoints saved to `./output/`
- TensorBoard logs in `./log/{timestamp}/`
- Final model in `./models/n{n}w{w}-{timestamp}/`

### 3. Evaluation Script (`eval.py`)

Comprehensive model evaluation with generation quality metrics:

```bash
python eval.py [OPTIONS]
```

**Key Options:**
- `--model-path PATH`: Path to trained model (required)
- `--dataset-path PATH`: Local dataset file (optional)
- `--num-vars N`: Number of variables (auto-detected from model)
- `--width N`: Formula width (auto-detected from model)
- `--num-sequences N`: Sequences per prompt (default: 5)
- `--num-prompts N`: Number of test prompts (default: 10)
- `--prompt-length N`: Prompt token count (default: 5)
- `--max-length N`: Max generation length (default: 64)
- `--temperature F`: Sampling temperature (default: 1.0)
- `--top-k N`: Top-k sampling (default: 50)
- `--top-p F`: Nucleus sampling (default: 0.95)
- `--output PATH`: Save results to file

**Metrics Computed:**
- avgQ distributions (prompt vs generated)
- Improvement statistics (mean, std, percentage)
- Best/worst case analysis
- Cross-entropy loss on test data

### 4. Dataset Module (`dataset.py`)

Flexible trajectory dataset with multiple data sources:

```python
TrajectoryDataset(
    num_vars=4,              # Filter by variables
    width=3,                 # Filter by width
    local_file="data.json",  # Or use local file
    warehouse_host="localhost",
    warehouse_port=8000,
    cache_dir="./cache",     # Optional caching
    force_download=False,    # Force refresh
    timeout=30.0
)
```

**Features:**
- Automatic warehouse connection and download
- Local file caching for efficiency
- Format validation and error handling
- Compatible with PyTorch DataLoader

### 5. Data Collator (`collator.py`)

Batching and augmentation for trajectory data:

```python
TrajectoryCollator(
    num_vars=3,
    permute_input=True,      # Enable augmentation
    mask_probability=0.0,     # Token masking (experimental)
    device="cpu"
)
```

**Augmentation Strategy:**
- Random variable permutation preserving formula semantics
- Consistent permutation within batch
- Optional token masking for robustness

### 6. Data Cleaning (`data_cleaning.py`)

Pre-processing utilities for trajectory quality control:

```bash
python data_cleaning.py \
    --host localhost \
    --port 8000 \
    --num-vars 4 \
    --width 3 \
    --avgq-threshold 0.5 \
    --max-avgq-only \
    --truncate \
    --output cleaned.json
```

**Cleaning Operations:**
1. **Filter by avgQ threshold**: Remove low-quality trajectories
2. **Truncate to max avgQ**: Keep only improving portions
3. **Remove duplicates**: Deduplicate trajectory dataset
4. **Format validation**: Ensure correct step structure

**Statistics Reported:**
- Original vs cleaned trajectory counts
- avgQ distribution before/after
- Removed trajectory analysis

### 7. Reward Function (`reward_function.py`)

Integration with TRL for reinforcement learning:

```python
reward_fn = LongshotRewardFunction(
    num_vars=4,
    width=3,
    penalty=-10.0,           # Invalid token penalty
    reward_scale=1.0,        # Reward scaling factor
    use_differential=False   # Absolute vs differential rewards
)

# Compute rewards for generated sequences
rewards = reward_fn.compute_rewards(token_ids)
```

**Use Cases:**
- PPO fine-tuning with TRL
- RLHF alignment
- Reward-weighted supervised learning

### 8. Custom Trainer (`custom_trainer.py`)

Extended HuggingFace Trainer with enhanced logging:

**Additional Features:**
- Individual loss component tracking (generation, avgQ, diversity)
- TensorBoard integration for detailed metrics
- Gradient norm monitoring
- Custom evaluation metrics

## Training Pipeline

### 1. Data Preparation

```bash
# Download and clean trajectories
python data_cleaning.py \
    --num-vars 3 \
    --width 2 \
    --avgq-threshold 0.3 \
    --output ./data/n3w2.json
```

### 2. Model Training

```bash
# Modify train.py parameters as needed
python train.py
```

Monitor with TensorBoard:
```bash
tensorboard --logdir ./log/
```

### 3. Model Evaluation

```bash
# Evaluate on test set
python eval.py \
    --model-path ./models/n3w2-latest \
    --num-prompts 100 \
    --output evaluation_results.json
```

### 4. Model Deployment

Models are saved in HuggingFace format and can be:
- Loaded with `GPT2ForLongshot.from_pretrained()`
- Uploaded to HuggingFace Hub
- Used for inference in production

## Advanced Usage

### Custom Training Configuration

Edit `train.py` to modify:
```python
# Different model size
gpt2_config = GPT2Config(
    n_positions=128,     # Longer sequences
    n_embd=512,         # Larger embeddings
    n_layer=24,         # Deeper model
    n_head=16           # More attention heads
)

# Different loss weights
model_config = GPT2ForLongshotConfig(
    alpha=0.5,          # Less weight on generation
    beta=60.0,          # More weight on avgQ
    gamma=0.5           # More diversity emphasis
)
```

### Distributed Training

For multi-GPU training, use HuggingFace Accelerate:
```bash
accelerate config
accelerate launch train.py
```

### Hyperparameter Tuning

Key parameters to optimize:
- Learning rate schedule (linear, cosine, polynomial)
- Batch size vs gradient accumulation
- Loss component weights (α, β, γ)
- Model architecture (layers, heads, embedding dim)
- Augmentation strategies

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient accumulation
   - Use mixed precision training
   - Reduce model size

2. **Poor avgQ Performance**
   - Increase beta weight in loss
   - Use better data cleaning thresholds
   - Increase training epochs
   - Try different learning rates

3. **Overfitting**
   - Enable data augmentation
   - Add weight decay
   - Reduce model size
   - Use dropout

4. **Slow Training**
   - Check GPU utilization
   - Increase batch size
   - Use compiled models (torch.compile)
   - Enable mixed precision

## Architecture

```
trainer/
├── train.py              # Main training script
├── eval.py              # Evaluation pipeline
├── model.py             # GPT2ForLongshot implementation
├── dataset.py           # Trajectory dataset
├── collator.py          # Data collator with augmentation
├── custom_trainer.py    # Extended trainer
├── data_cleaning.py     # Pre-processing utilities
├── reward_function.py   # TRL integration
└── test/               # Test suite
```

## Testing

Run the test suite:
```bash
pytest test/ -v
```

Key test files:
- `test_model_save_load.py`: Model serialization
- `test_eval.py`: Evaluation metrics
- `test_data_cleaning.py`: Cleaning functions
- `test_trajectory_dataset.py`: Dataset operations
- `test_collator_augmentation.py`: Augmentation logic

## Integration with Other Services

- **Warehouse Service**: Source of trajectory data
- **Clusterbomb Service**: Generates trajectories for training
- **MAP-Elites**: Provides diverse training data

## Performance Benchmarks

Typical training metrics (n=3, w=2):
- Training time: ~2-4 hours on V100 GPU
- Final avgQ: 0.7-0.85
- Improvement rate: 60-75% of generated sequences
- Model size: ~100MB

## Future Improvements

- [ ] Multi-task learning across different (n,w) configurations
- [ ] Curriculum learning with progressive difficulty
- [ ] Online learning from new trajectories
- [ ] Transformer architecture variants (BERT, T5)
- [ ] Reinforcement learning fine-tuning