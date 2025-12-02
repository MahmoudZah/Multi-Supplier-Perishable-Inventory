# RL Training Guide & Research Best Practices

This guide explains how to train the RL model on Google Colab and provides advice for conducting robust, unbiased research.

## 1. Running on Google Colab

### Setup
1.  **Upload Code**: Upload the entire `Multi-Supplier-Perishable-Inventory` folder to your Google Drive.
2.  **Mount Drive**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Multi-Supplier-Perishable-Inventory
    ```
3.  **Install Dependencies**:
    ```bash
    !pip install -r requirements.txt
    !pip install gymnasium stable-baselines3 shimmy
    ```

### Training
1.  **Configure**: Edit `colab_training/config.json` to set your desired parameters.
2.  **Run**:
    ```bash
    !python colab_training/train_rl.py
    ```
3.  **Monitor**: TensorBoard logs are saved to `logs/tensorboard`.
    ```bash
    %load_ext tensorboard
    %tensorboard --logdir logs/tensorboard
    ```

## 2. Achieving Best Results

### Normalization is Key
Neural networks struggle with unscaled inputs (e.g., inventory=100 vs backorders=0).
-   **Observation Normalization**: The `gym_wrapper.py` should ideally normalize inputs to roughly `[-1, 1]` or `[0, 1]`.
    -   *Current Implementation*: Does not auto-normalize. Consider using `VecNormalize` from Stable Baselines3 in `train_rl.py`.
-   **Reward Scaling**: Rewards can be large negative numbers (e.g., -1000). Scaling them by `1/100` or `1/mean_demand` helps gradient stability.

### Hyperparameter Tuning
Default PPO parameters are decent but rarely optimal.
-   **Use Optuna**: Use the [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) or write a simple Optuna script to tune:
    -   `learning_rate`
    -   `n_steps` (buffer size)
    -   `batch_size`
    -   `gamma` (discount factor)
    -   `ent_coef` (entropy coefficient for exploration)

### Algorithm Choice
-   **PPO**: Robust, easy to tune, good default choice.
-   **SAC/TD3**: Sample efficient, often better for continuous control, but harder to tune.
-   **Recurrent Policies (LSTM)**: If the state is partially observable (e.g., you don't see the full pipeline), use `MlpLstmPolicy` (requires `sb3-contrib`).

## 3. Robust & Unbiased Research

To ensure your research is high quality:

### 1. Multiple Random Seeds
**Never report results from a single training run.** RL is stochastic.
-   Train **5-10 separate agents** with different seeds (e.g., 42, 43, 44...).
-   Report the **mean** and **standard deviation** (or confidence intervals) of their performance.
-   The provided `train_rl.py` takes a seed from config; you should write a loop to run it multiple times.

### 2. Separate Training and Testing Environments
-   **Training**: Train on a standard demand pattern.
-   **Testing**: Evaluate on:
    -   **In-distribution**: Same parameters, different random seed.
    -   **Out-of-distribution (OOD)**: Change parameters to stress-test.
        -   Increase demand variance (higher coefficient of variation).
        -   Change lead times.
        -   Add demand shocks/spikes.

### 3. Strong Baselines
Compare your RL agent against:
-   **Optimal/Heuristic**: The `TailoredBaseSurgePolicy` (TBS) provided in the repo is a very strong baseline for this specific problem.
-   **Simple**: Base Stock Policy.
-   **Naive**: Constant Order.

### 4. Ablation Studies
If you modify the state space or reward function, test if those changes actually help.
-   *Example*: "Does adding the 'exogenous state' to the observation actually improve performance?" -> Train with and without it.

### 5. Proper Evaluation Metric
-   Don't just look at "Total Reward" (which is abstract).
-   Report business metrics: **Fill Rate**, **Spoilage Rate**, **Average Inventory**, **Service Level**.
