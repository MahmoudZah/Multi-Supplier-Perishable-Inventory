# RL Training for Perishable Inventory MDP

This directory contains everything needed to train Deep RL agents on Google Colab.

## Files

- **`config.json`**: Environment and training configuration
- **`gym_wrapper.py`**: Gymnasium wrapper for the inventory environment
- **`train_rl.py`**: Main training script
- **`training_guide.md`**: Best practices and research advice

## Quick Start on Google Colab

### 1. Upload to Drive
Upload the entire `Multi-Supplier-Perishable-Inventory` folder to Google Drive.

### 2. Open Colab Notebook
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Multi-Supplier-Perishable-Inventory
```

### 3. Install Dependencies
```bash
!pip install -q gymnasium==0.29.1 stable-baselines3 shimmy tensorboard
```

**Note**: We use `gymnasium==0.29.1` to match Stable Baselines3's requirement.

### 4. Configure
Edit `colab_training/config.json` to set:
- MDP parameters (suppliers, demand, costs)
- Training hyperparameters (timesteps, learning rate)
- Evaluation frequency

### 5. Train
```bash
!python colab_training/train_rl.py
```

### 6. Monitor with TensorBoard
```python
%load_ext tensorboard
%tensorboard --logdir logs/tensorboard
```

## Configuration

The `config.json` file uses the following structure:

```json
{
  "mdp_params": {
    "shelf_life": 5,
    "suppliers": [...],
    "demand": {"type": "poisson", "mean": 10.0},
    "costs": {...}
  },
  "training_params": {
    "algorithm": "PPO",
    "total_timesteps": 100000,
    "learning_rate": 0.0003,
    "seed": 42,
    "n_envs": 4
  },
  "eval_params": {
    "eval_freq": 10000,
    "n_eval_episodes": 10
  }
}
```

## Next Steps

1. **Train multiple seeds**: Run training with different seeds (42, 43, 44, etc.)
2. **Compare with baselines**: Use the existing heuristic policies (TBS, Base Stock)
3. **Test robustness**: Evaluate on different demand patterns
4. **Read training_guide.md**: For detailed best practices

## Important Notes

- The trained model is saved as `final_model.zip`
- Best model during training is saved in `logs/best_model/`
- TensorBoard logs are in `logs/tensorboard/`
