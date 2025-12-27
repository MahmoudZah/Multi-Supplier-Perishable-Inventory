# 🏥 Multi-Supplier Perishable Inventory MDP

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green)](https://stable-baselines3.readthedocs.io/)

A **Deep Reinforcement Learning framework** for pharmaceutical supply chain optimization, specifically designed to address the dual challenge of **drug shortages** and **inventory wastage** in emerging markets like Egypt. This project implements a comprehensive Markov Decision Process (MDP) with multi-supplier dynamics, perishability constraints, and crisis modeling—benchmarked against 7 classical inventory policies.

> 📄 **Research Paper**: See [Visioneers_Report.pdf](Docs/Visioneers_Report.pdf) for the complete academic paper and experimental results.

---

## 🌍 Motivation: The Egyptian Pharmaceutical Challenge

The pharmaceutical sector in Egypt faces critical structural vulnerabilities:

- **65%** of finished pharmaceuticals are imported
- **90%** of raw materials sourced from abroad  
- **800+ essential medications** reported unavailable in July 2024
- **17 million expired drug packages** withdrawn in the same year

This creates a **dual optimization problem**: minimize stockouts to protect patient health while reducing spoilage to ensure economic viability. Traditional heuristic-based inventory management fails to balance these conflicting objectives under volatile, non-stationary market conditions.

---

## 🧠 Our Solution: Curriculum-PPO Framework

We employ **Proximal Policy Optimization (PPO)** with a three-stage **curriculum learning** strategy:

```
Simple → Moderate → Complex → Extreme
(Stationary)  (Seasonal)  (Spikes)   (Crisis)
```

### Key Results (vs. Traditional Policies)

| Metric | RL Agent | Best Heuristic | Improvement |
|--------|----------|----------------|-------------|
| **Cost Reduction** | 263.4 | 1909.9 (BaseStock) | **86%** lower |
| **Spoilage Rate** | 7.35% | 26.27% (VectorBS) | **72%** reduction |
| **Fill Rate** | 98.76% | 99.66% (VectorBS) | Comparable |
| **Cost Variance** | Low | High | Eliminates catastrophic events |

---

## 🚀 Key Features

### 🔬 Advanced MDP Formulation

- **Survival-Adjusted Inventory Position**: Weights stock by probability of consumption before expiry
- **FIFO Depletion**: Oldest inventory consumed first, realistic for pharmaceuticals
- **Age-Dependent Holding Costs**: Older stock incurs higher carrying costs
- **Crisis Dynamics**: Demand spikes and supplier disruptions modeled via exogenous state

### 📊 Comprehensive Policy Suite

We implement and benchmark **7 classical inventory policies**:

| Policy | Type | Description |
|--------|------|-------------|
| **TBS** | Dual-Sourcing | Tailored Base-Surge with slow/fast supplier allocation |
| **BaseStock** | Single-Supplier | Order-up-to target inventory position |
| **PIL** | Periodic Review | Projected Inventory Level policy |
| **DIP** | Dual-Index | Separate inventory positions for each supplier |
| **PEIP** | Perishability-Aware | Projected Effective Inventory with shelf-life adjustment |
| **VectorBS** | Age-Weighted | Base-stock with multi-dimensional inventory tracking |
| **DoNothing** | Baseline | No ordering (lower bound) |

### 🌐 Environment Suite

- **105+ unique environments** across 4 complexity tiers
- **Configurable demand**: Poisson, Negative Binomial, Seasonal, Composite with spikes
- **Multi-supplier support**: Up to 15 suppliers with MOQ constraints
- **Stochastic lead times**: Bernoulli and Markovian delay models

### 🎮 Gymnasium-Compatible Wrapper

```python
from colab_training.gym_env import create_gym_env

env = create_gym_env(
    shelf_life=5,
    mean_demand=20.0,
    enable_crisis=True,
    fast_lead_time=1,
    slow_lead_time=4
)

# Works with Stable-Baselines3, Ray RLlib, etc.
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

---

## 📐 Mathematical Model

### State Space

The state at time $t$ comprises:

$$X_t = \left( I_t, \{P_t^{(s)}\}_{s \in S}, B_t, Z_t \right)$$

| Component | Description | Dimension |
|-----------|-------------|-----------|
| $I_t$ | On-hand inventory by remaining shelf-life | $N$ (shelf-life buckets) |
| $P_t^{(s)}$ | Pipeline orders from supplier $s$ | $L_s$ (lead time) |
| $B_t$ | Backlog (unfulfilled demand) | 1 |
| $Z_t$ | Exogenous state (season, crisis level) | $d_z$ |

### Cost Structure

Single-period cost:
$$c(X_t, a_t) = C^{purch} + C^{hold} + C^{short} + C^{spoil}$$

With age-dependent holding cost:
$$h_n = h_{base} + h_{prem} \cdot \frac{N - n}{N}$$

### Reward Shaping

$$R_{shaped} = -\alpha \cdot C^{purch} - \beta \cdot (C^{hold} + C^{spoil}) - \zeta \cdot C^{short} + \delta \cdot \text{bonus}$$

Default weights: $(\alpha, \beta, \zeta, \delta) = (0.5, 0.3, 0.2, 0.1)$

> 📘 **Full mathematical formulation**: See [perishable_inventory_mdp_formulation.tex](Docs/perishable_inventory_mdp_formulation.tex)

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/MahmoudZah/Multi-Supplier-Perishable-Inventory.git
cd Multi-Supplier-Perishable-Inventory

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy>=1.20.0` | Numerical operations |
| `scipy>=1.7.0` | Statistical distributions |
| `gymnasium>=0.29.0` | RL environment interface |
| `stable-baselines3>=2.2.0` | PPO implementation |
| `torch>=2.0.0` | Neural network backend |
| `tensorboard>=2.15.0` | Training visualization |

---

## 🏃‍♂️ Quick Start

### Training an RL Agent

```bash
# Train with curriculum learning (5M steps, 8 parallel envs)
python colab_training/train_rl.py

# Quick test mode
python colab_training/train_rl.py --test-mode
```

### Training on Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Multi-Supplier-Perishable-Inventory
!pip install gymnasium stable-baselines3 shimmy tensorboard
!python colab_training/train_rl.py
```

### Benchmarking Policies

```python
from colab_training.benchmark import evaluate_policy, get_all_policies_for_env
from colab_training.gym_env import create_gym_env

env = create_gym_env(shelf_life=5, mean_demand=15.0)
policies = get_all_policies_for_env(env)

for name, policy in policies.items():
    result = evaluate_policy(policy, env, n_episodes=10)
    print(f"{name}: Cost={result.mean_cost:.2f}, Fill Rate={result.mean_fill_rate:.2%}")
```

---

## 📂 Project Structure

```
.
├── perishable_inventory_mdp/       # Core MDP Implementation
│   ├── environment.py              # Main MDP logic with transition dynamics
│   ├── state.py                    # InventoryState with FIFO aging
│   ├── demand.py                   # Poisson, NegBin, Seasonal, Composite demand
│   ├── policies.py                 # TBS, PIL, DIP, PEIP, VectorBS policies
│   ├── costs.py                    # Cost parameter structures
│   ├── crisis.py                   # Supply chain disruption modeling
│   ├── contracts.py                # Supplier contracts and MOQ
│   └── solver.py                   # Value iteration (small spaces)
│
├── colab_training/                 # RL Training Infrastructure
│   ├── gym_env.py                  # Gymnasium wrapper with reward shaping
│   ├── train_rl.py                 # PPO training script with curriculum
│   ├── benchmark.py                # Policy evaluation and comparison
│   ├── environment_suite.py        # 105+ benchmark environments
│   ├── callbacks.py                # Training callbacks and monitoring
│   ├── config.json                 # Training hyperparameters
│   ├── policy_benchmark.ipynb      # Comprehensive benchmark notebook
│   └── training_guide.md           # Best practices for research
│
├── models/                         # Pre-trained RL Models
│   ├── model_complex/              # Trained on Simple, Moderate & Complex environments
│   └── model_extreme/              # Trained on Extreme (crisis) scenarios
│
├── Docs/                           # Documentation
│   ├── Visioneers_Report.pdf       # Research paper with full results
│   └── perishable_inventory_mdp_formulation.tex  # Mathematical formulation
│
├── tests/                          # Comprehensive Test Suite (17 test files)
├── examples/                       # Example simulations
└── requirements.txt
```

---

## 📊 Benchmarking Results

### Performance by Complexity Level

| Complexity | RL Cost | VectorBS Cost | TBS Cost | BaseStock Cost |
|------------|---------|---------------|----------|----------------|
| **Simple** | ~100 | ~95 | ~110 | ~120 |
| **Moderate** | ~150 | ~140 | ~180 | ~300 |
| **Complex** | ~250 | ~200 | ~400 | ~1200 |
| **Extreme** | ~400 | ~350 | ~800 | ~3000+ |

### Key Insights

1. **Simple environments**: All policies perform comparably since demand is stationary
2. **Moderate/Complex**: RL learns to pre-position inventory and balance suppliers
3. **Extreme**: RL eliminates the "long tail" of catastrophic cost events
4. **VectorBS**: Strongest heuristic, but higher spoilage than RL (26% vs 7%)

---

## 🔬 Research Contributions

1. **Novel MDP Formulation**: First comprehensive model combining multi-supplier dynamics, perishability, and crisis scenarios for pharmaceutical inventory
2. **Curriculum Learning for Inventory**: Demonstrated effectiveness of progressive complexity training for supply chain RL
3. **Comprehensive Baseline Suite**: Implementation of 7 classical inventory policies for fair comparison
4. **Emerging Market Focus**: Specifically designed for pharmaceutical challenges in developing economies

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{visioneers2024pharmaceutical,
  title={Addressing the Dual Challenge of Drug Shortages and Expiry in Egypt: A Predictive Modeling Approach},
  author={Anan,and Zahran, Mahmoud and Ellaithy, Ammar and Raafat, Marize and Wael, Joyce and Sheref, Mohamed and Eleshary, Fady N. and Islam, Noureldin and Mostafa, Bassel and Amr, Seif Eldin},
  journal={TCCD 14 Research Day},
  year={2024},
  note={Mentored by Dr. Samah ElTantawy, Cairo University}
}
```

---

## 🤝 Acknowledgments

- **Dr. Samah ElTantawy** (Cairo University) - Research mentor and advisor
- **Dr. Aliaa Rehan** (Cairo University) - Domain expertise in healthcare
- **Eng. Alaa Tarek & Eng. Amira Omar** (Cairo University) - Technical guidance
- **Dr. Sherif AbuElmagd Awad** (ADWIA Pharmaceuticals) - Industry insights

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔮 Future Work

- **Observation Standardization**: Universal wrapper for better generalization
- **Transformer Architectures**: Self-attention for temporal dependencies
- **Multi-Agent Extension**: Coordination between multiple pharmacies/hospitals
- **Quantum RL**: Exploring quantum variational circuits for large state spaces
