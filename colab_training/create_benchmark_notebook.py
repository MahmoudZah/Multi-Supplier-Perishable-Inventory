#!/usr/bin/env python3
"""
Generates a comprehensive benchmarking Jupyter notebook.

This script creates a notebook that:
1. Supports 2 RL models (levels 1-3 and extreme)
2. Fairly compares ALL policies in the SAME environment
3. Provides comprehensive visualizations
4. Non-biased evaluation methodology
"""

import json

def create_notebook():
    """Create the benchmark notebook structure."""
    
    cells = []
    
    # ==================== HEADER ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "header"},
        "source": [
            "# 📊 Comprehensive Policy Benchmarking\n",
            "\n",
            "This notebook performs **fair, unbiased comparisons** between all policies:\n",
            "\n",
            "| Policy Type | Policies |\n",
            "|-------------|----------|\n",
            "| **RL Models** | Model 1 (Levels 1-3), Model 2 (Extreme) |\n",
            "| **Classical** | TBS, BaseStock, DoNothing |\n",
            "| **Advanced** | PIL, DIP, PEIP, VectorBS |\n",
            "\n",
            "**Methodology:**\n",
            "- All policies evaluated on the **exact same environment instances**\n",
            "- Same random seeds for fair demand/spoilage realization\n",
            "- Comprehensive metrics: cost, fill rate, spoilage, orders\n",
            "\n",
            "---"
        ]
    })
    
    # ==================== SETUP ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "setup-header"},
        "source": ["## 1️⃣ Setup & Imports"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "install"},
        "outputs": [],
        "source": [
            "# Install dependencies (run once)\n",
            "# !pip install stable-baselines3 gymnasium numpy pandas matplotlib seaborn scipy"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "imports"},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "from typing import Dict, List, Any, Optional\n",
            "from dataclasses import dataclass, field\n",
            "import json\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# RL imports\n",
            "from stable_baselines3 import PPO\n",
            "from stable_baselines3.common.monitor import Monitor\n",
            "from gymnasium.wrappers import TimeLimit\n",
            "\n",
            "# Project imports\n",
            "import sys\n",
            "sys.path.insert(0, '..')\n",
            "\n",
            "from colab_training.gym_env import PerishableInventoryGymWrapper, RewardConfig\n",
            "from colab_training.environment_suite import (\n",
            "    get_canonical_suite, build_environment_from_config\n",
            ")\n",
            "from colab_training.benchmark import (\n",
            "    get_tbs_policy_for_env, get_basestock_policy_for_env,\n",
            "    get_pil_policy_for_env, get_dip_policy_for_env,\n",
            "    get_peip_policy_for_env, get_vector_bs_policy_for_env,\n",
            "    AVAILABLE_BASELINES\n",
            ")\n",
            "from perishable_inventory_mdp.policies import DoNothingPolicy\n",
            "\n",
            "# Visualization settings\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "sns.set_palette('husl')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)\n",
            "plt.rcParams['font.size'] = 11\n",
            "\n",
            "print('✅ All imports successful!')\n",
            "print(f'Available baselines: {AVAILABLE_BASELINES}')"
        ]
    })
    
    # ==================== CONFIGURATION ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "config-header"},
        "source": ["## 2️⃣ Configuration\n", "\n", "Configure your RL model paths and evaluation parameters."]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "config"},
        "outputs": [],
        "source": [
            "# ============================================================\n",
            "# CONFIGURATION - EDIT THESE PATHS TO YOUR MODELS\n",
            "# ============================================================\n",
            "\n",
            "# RL Model paths\n",
            "RL_MODEL_LEVELS_1_3 = 'logs/best_model/best_model.zip'  # For simple, moderate, complex\n",
            "RL_MODEL_EXTREME = 'logs/extreme_model/best_model.zip'  # For extreme level\n",
            "\n",
            "# Evaluation parameters\n",
            "N_EPISODES = 10          # Episodes per environment (increase for more accuracy)\n",
            "N_ENVS_PER_LEVEL = 10    # Environments per complexity level\n",
            "MAX_STEPS = 500          # Max steps per episode\n",
            "RANDOM_SEED = 42         # For reproducibility\n",
            "\n",
            "# Output directory\n",
            "OUTPUT_DIR = Path('benchmark_results')\n",
            "OUTPUT_DIR.mkdir(exist_ok=True)\n",
            "\n",
            "print(f'📁 Results will be saved to: {OUTPUT_DIR.absolute()}')"
        ]
    })
    
    # ==================== LOAD MODELS ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "models-header"},
        "source": ["## 3️⃣ Load RL Models"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "load-models"},
        "outputs": [],
        "source": [
            "# Load RL models\n",
            "print('📦 Loading RL Models...')\n",
            "\n",
            "try:\n",
            "    model_levels_1_3 = PPO.load(RL_MODEL_LEVELS_1_3)\n",
            "    print(f'  ✅ Model (Levels 1-3): {RL_MODEL_LEVELS_1_3}')\n",
            "except FileNotFoundError:\n",
            "    model_levels_1_3 = None\n",
            "    print(f'  ⚠️ Model not found: {RL_MODEL_LEVELS_1_3}')\n",
            "\n",
            "try:\n",
            "    model_extreme = PPO.load(RL_MODEL_EXTREME)\n",
            "    print(f'  ✅ Model (Extreme): {RL_MODEL_EXTREME}')\n",
            "except FileNotFoundError:\n",
            "    model_extreme = None\n",
            "    print(f'  ⚠️ Model not found: {RL_MODEL_EXTREME}')\n",
            "\n",
            "if model_levels_1_3 is None and model_extreme is None:\n",
            "    print('\\n❌ ERROR: No RL models found! Please check paths in Configuration.')"
        ]
    })
    
    # ==================== ENVIRONMENT SETUP ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "env-header"},
        "source": ["## 4️⃣ Environment Setup"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "env-setup"},
        "outputs": [],
        "source": [
            "# Load environment suite\n",
            "suite = get_canonical_suite()\n",
            "print(f'📊 Loaded {len(suite)} environments')\n",
            "\n",
            "# Display complexity distribution\n",
            "complexity_counts = {}\n",
            "for level in ['simple', 'moderate', 'complex', 'extreme']:\n",
            "    configs = suite.get_by_complexity(level)\n",
            "    complexity_counts[level] = len(configs)\n",
            "    print(f'  {level.upper()}: {len(configs)} environments')\n",
            "\n",
            "# Reward configuration\n",
            "reward_config = RewardConfig()\n",
            "\n",
            "def create_env(config, seed=None):\n",
            "    '''Create gym environment from config with optional seed.'''\n",
            "    mdp = build_environment_from_config(config)\n",
            "    env = PerishableInventoryGymWrapper(mdp=mdp, reward_config=reward_config)\n",
            "    env = TimeLimit(env, max_episode_steps=MAX_STEPS)\n",
            "    env = Monitor(env)\n",
            "    if seed is not None:\n",
            "        env.reset(seed=seed)\n",
            "    return env"
        ]
    })
    
    # ==================== EVALUATION FUNCTIONS ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "eval-header"},
        "source": ["## 5️⃣ Evaluation Framework\n", "\n", "Fair evaluation: all policies tested on **identical** environment instances."]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "eval-functions"},
        "outputs": [],
        "source": [
            "@dataclass\n",
            "class EpisodeMetrics:\n",
            "    '''Metrics from a single episode.'''\n",
            "    total_cost: float\n",
            "    holding_cost: float\n",
            "    backorder_cost: float\n",
            "    spoilage_cost: float\n",
            "    ordering_cost: float\n",
            "    total_demand: float\n",
            "    total_sales: float\n",
            "    total_spoilage: float\n",
            "    total_orders: float\n",
            "    fill_rate: float\n",
            "    avg_inventory: float\n",
            "\n",
            "@dataclass\n",
            "class PolicyResult:\n",
            "    '''Aggregated results for a policy.'''\n",
            "    policy_name: str\n",
            "    env_id: str\n",
            "    complexity: str\n",
            "    episodes: List[EpisodeMetrics] = field(default_factory=list)\n",
            "    \n",
            "    @property\n",
            "    def mean_cost(self):\n",
            "        return np.mean([e.total_cost for e in self.episodes])\n",
            "    \n",
            "    @property\n",
            "    def std_cost(self):\n",
            "        return np.std([e.total_cost for e in self.episodes])\n",
            "    \n",
            "    @property\n",
            "    def mean_fill_rate(self):\n",
            "        return np.mean([e.fill_rate for e in self.episodes])\n",
            "    \n",
            "    @property\n",
            "    def mean_spoilage(self):\n",
            "        return np.mean([e.total_spoilage for e in self.episodes])\n",
            "\n",
            "def run_episode(policy, env, is_rl_model=False):\n",
            "    '''Run single episode and collect metrics.'''\n",
            "    obs, info = env.reset()\n",
            "    done = False\n",
            "    truncated = False\n",
            "    \n",
            "    total_cost = 0\n",
            "    holding_cost = 0\n",
            "    backorder_cost = 0\n",
            "    spoilage_cost = 0\n",
            "    ordering_cost = 0\n",
            "    total_demand = 0\n",
            "    total_sales = 0\n",
            "    total_spoilage = 0\n",
            "    total_orders = 0\n",
            "    inventory_sum = 0\n",
            "    steps = 0\n",
            "    \n",
            "    while not done and not truncated:\n",
            "        # Get action\n",
            "        if is_rl_model:\n",
            "            action, _ = policy.predict(obs, deterministic=True)\n",
            "        else:\n",
            "            # Baseline policy - get MDP state\n",
            "            mdp = env.unwrapped.mdp if hasattr(env.unwrapped, 'mdp') else env.mdp\n",
            "            state = mdp.current_state\n",
            "            action_dict = policy.get_action(state, mdp)\n",
            "            # Convert to gym action\n",
            "            action = env.unwrapped._action_dict_to_array(action_dict) if hasattr(env.unwrapped, '_action_dict_to_array') else np.zeros(env.action_space.shape[0])\n",
            "        \n",
            "        obs, reward, done, truncated, info = env.step(action)\n",
            "        \n",
            "        # Accumulate metrics from info\n",
            "        if 'holding_cost' in info:\n",
            "            holding_cost += info.get('holding_cost', 0)\n",
            "        if 'backorder_cost' in info:\n",
            "            backorder_cost += info.get('backorder_cost', 0)\n",
            "        if 'spoilage_cost' in info:\n",
            "            spoilage_cost += info.get('spoilage_cost', 0)\n",
            "        if 'ordering_cost' in info:\n",
            "            ordering_cost += info.get('ordering_cost', 0)\n",
            "        if 'demand' in info:\n",
            "            total_demand += info.get('demand', 0)\n",
            "        if 'sales' in info:\n",
            "            total_sales += info.get('sales', 0)\n",
            "        if 'spoilage' in info:\n",
            "            total_spoilage += info.get('spoilage', 0)\n",
            "        if 'total_order' in info:\n",
            "            total_orders += info.get('total_order', 0)\n",
            "        if 'inventory' in info:\n",
            "            inventory_sum += info.get('inventory', 0)\n",
            "        \n",
            "        total_cost -= reward  # reward is negative cost\n",
            "        steps += 1\n",
            "    \n",
            "    fill_rate = total_sales / max(total_demand, 1e-6)\n",
            "    avg_inventory = inventory_sum / max(steps, 1)\n",
            "    \n",
            "    return EpisodeMetrics(\n",
            "        total_cost=total_cost,\n",
            "        holding_cost=holding_cost,\n",
            "        backorder_cost=backorder_cost,\n",
            "        spoilage_cost=spoilage_cost,\n",
            "        ordering_cost=ordering_cost,\n",
            "        total_demand=total_demand,\n",
            "        total_sales=total_sales,\n",
            "        total_spoilage=total_spoilage,\n",
            "        total_orders=total_orders,\n",
            "        fill_rate=fill_rate,\n",
            "        avg_inventory=avg_inventory\n",
            "    )\n",
            "\n",
            "print('✅ Evaluation framework ready')"
        ]
    })
    
    # ==================== FAIR COMPARISON FUNCTION ====================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "fair-comparison"},
        "outputs": [],
        "source": [
            "def evaluate_all_policies_fair(\n",
            "    config,\n",
            "    rl_model,\n",
            "    n_episodes: int = 10,\n",
            "    seed: int = 42\n",
            ") -> Dict[str, PolicyResult]:\n",
            "    '''\n",
            "    Evaluate ALL policies on the SAME environment with SAME seeds.\n",
            "    \n",
            "    This ensures completely fair comparison:\n",
            "    - Same demand realizations\n",
            "    - Same spoilage outcomes\n",
            "    - Same initial conditions\n",
            "    '''\n",
            "    results = {}\n",
            "    env_id = config.env_id\n",
            "    complexity = config.complexity\n",
            "    \n",
            "    # Create base environment for policy creation\n",
            "    base_env = create_env(config, seed=seed)\n",
            "    \n",
            "    # Get all available policies for this environment\n",
            "    policies = {'RL': (rl_model, True)} if rl_model else {}\n",
            "    \n",
            "    # Add baseline policies\n",
            "    try:\n",
            "        policies['TBS'] = (get_tbs_policy_for_env(base_env), False)\n",
            "    except: pass\n",
            "    \n",
            "    try:\n",
            "        policies['BaseStock'] = (get_basestock_policy_for_env(base_env), False)\n",
            "    except: pass\n",
            "    \n",
            "    policies['DoNothing'] = (DoNothingPolicy(), False)\n",
            "    \n",
            "    try:\n",
            "        policies['PIL'] = (get_pil_policy_for_env(base_env), False)\n",
            "    except: pass\n",
            "    \n",
            "    try:\n",
            "        policies['DIP'] = (get_dip_policy_for_env(base_env), False)\n",
            "    except: pass\n",
            "    \n",
            "    try:\n",
            "        policies['PEIP'] = (get_peip_policy_for_env(base_env), False)\n",
            "    except: pass\n",
            "    \n",
            "    try:\n",
            "        policies['VectorBS'] = (get_vector_bs_policy_for_env(base_env), False)\n",
            "    except: pass\n",
            "    \n",
            "    base_env.close()\n",
            "    \n",
            "    # Evaluate each policy with SAME seeds for each episode\n",
            "    for policy_name, (policy, is_rl) in policies.items():\n",
            "        result = PolicyResult(policy_name=policy_name, env_id=env_id, complexity=complexity)\n",
            "        \n",
            "        for ep in range(n_episodes):\n",
            "            # Use same seed for this episode across all policies\n",
            "            episode_seed = seed + ep\n",
            "            env = create_env(config, seed=episode_seed)\n",
            "            \n",
            "            try:\n",
            "                metrics = run_episode(policy, env, is_rl_model=is_rl)\n",
            "                result.episodes.append(metrics)\n",
            "            except Exception as e:\n",
            "                print(f'    ⚠️ {policy_name} failed episode {ep}: {e}')\n",
            "            finally:\n",
            "                env.close()\n",
            "        \n",
            "        results[policy_name] = result\n",
            "    \n",
            "    return results\n",
            "\n",
            "print('✅ Fair comparison function ready')"
        ]
    })
    
    # ==================== RUN BENCHMARKS ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "benchmark-header"},
        "source": ["## 6️⃣ Run Comprehensive Benchmarks"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "run-benchmarks"},
        "outputs": [],
        "source": [
            "# Run comprehensive benchmarks\n",
            "all_results = []\n",
            "\n",
            "print('=' * 70)\n",
            "print('📊 COMPREHENSIVE POLICY BENCHMARKING')\n",
            "print('=' * 70)\n",
            "print(f'Episodes per env: {N_EPISODES}')\n",
            "print(f'Environments per level: {N_ENVS_PER_LEVEL}')\n",
            "print('=' * 70)\n",
            "\n",
            "for complexity in ['simple', 'moderate', 'complex', 'extreme']:\n",
            "    configs = suite.get_by_complexity(complexity)\n",
            "    n_envs = min(N_ENVS_PER_LEVEL, len(configs))\n",
            "    \n",
            "    if n_envs == 0:\n",
            "        continue\n",
            "    \n",
            "    # Select appropriate RL model\n",
            "    if complexity == 'extreme':\n",
            "        rl_model = model_extreme\n",
            "        model_label = 'RL (Extreme)'\n",
            "    else:\n",
            "        rl_model = model_levels_1_3\n",
            "        model_label = 'RL (L1-3)'\n",
            "    \n",
            "    print(f'\\n🔍 {complexity.upper()} ({n_envs} environments)')\n",
            "    print(f'   Using: {model_label}')\n",
            "    print('-' * 50)\n",
            "    \n",
            "    for i, config in enumerate(configs[:n_envs]):\n",
            "        print(f'  [{i+1}/{n_envs}] {config.env_id}...', end=' ')\n",
            "        \n",
            "        try:\n",
            "            results = evaluate_all_policies_fair(\n",
            "                config=config,\n",
            "                rl_model=rl_model,\n",
            "                n_episodes=N_EPISODES,\n",
            "                seed=RANDOM_SEED\n",
            "            )\n",
            "            \n",
            "            for policy_name, result in results.items():\n",
            "                all_results.append({\n",
            "                    'policy': policy_name,\n",
            "                    'env_id': config.env_id,\n",
            "                    'complexity': complexity,\n",
            "                    'mean_cost': result.mean_cost,\n",
            "                    'std_cost': result.std_cost,\n",
            "                    'fill_rate': result.mean_fill_rate,\n",
            "                    'spoilage': result.mean_spoilage,\n",
            "                    'n_episodes': len(result.episodes)\n",
            "                })\n",
            "            \n",
            "            # Print RL cost for this env\n",
            "            if 'RL' in results:\n",
            "                print(f'RL cost={results[\"RL\"].mean_cost:.1f}')\n",
            "            else:\n",
            "                print('done')\n",
            "        except Exception as e:\n",
            "            print(f'FAILED: {e}')\n",
            "\n",
            "print('\\n' + '=' * 70)\n",
            "print('✅ BENCHMARKING COMPLETE')\n",
            "print('=' * 70)\n",
            "\n",
            "# Create DataFrame\n",
            "df = pd.DataFrame(all_results)\n",
            "print(f'\\nTotal evaluations: {len(df)}')\n",
            "df.head(10)"
        ]
    })
    
    # ==================== SAVE RESULTS ====================
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "save-results"},
        "outputs": [],
        "source": [
            "# Save results\n",
            "results_path = OUTPUT_DIR / 'benchmark_results.csv'\n",
            "df.to_csv(results_path, index=False)\n",
            "print(f'💾 Results saved to: {results_path}')\n",
            "\n",
            "# Save as JSON too\n",
            "df.to_json(OUTPUT_DIR / 'benchmark_results.json', orient='records', indent=2)"
        ]
    })
    
    # ==================== VISUALIZATION SECTION ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "viz-header"},
        "source": ["## 7️⃣ Comprehensive Visualizations"]
    })
    
    # Cost comparison by complexity
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "viz-cost-complexity"},
        "outputs": [],
        "source": [
            "# 1. Mean Cost by Policy and Complexity\n",
            "fig, ax = plt.subplots(figsize=(14, 7))\n",
            "\n",
            "pivot = df.pivot_table(values='mean_cost', index='policy', columns='complexity', aggfunc='mean')\n",
            "# Reorder columns\n",
            "cols = ['simple', 'moderate', 'complex', 'extreme']\n",
            "pivot = pivot[[c for c in cols if c in pivot.columns]]\n",
            "\n",
            "pivot.plot(kind='bar', ax=ax, width=0.8, edgecolor='white', linewidth=1.5)\n",
            "ax.set_xlabel('Policy', fontsize=12, fontweight='bold')\n",
            "ax.set_ylabel('Mean Total Cost', fontsize=12, fontweight='bold')\n",
            "ax.set_title('📊 Mean Cost by Policy and Complexity Level', fontsize=14, fontweight='bold')\n",
            "ax.legend(title='Complexity', bbox_to_anchor=(1.02, 1), loc='upper left')\n",
            "ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')\n",
            "\n",
            "# Add value labels\n",
            "for container in ax.containers:\n",
            "    ax.bar_label(container, fmt='%.0f', fontsize=8, padding=2)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'cost_by_complexity.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    })
    
    # Fill rate comparison
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "viz-fillrate"},
        "outputs": [],
        "source": [
            "# 2. Fill Rate by Policy and Complexity  \n",
            "fig, ax = plt.subplots(figsize=(14, 7))\n",
            "\n",
            "pivot_fr = df.pivot_table(values='fill_rate', index='policy', columns='complexity', aggfunc='mean')\n",
            "pivot_fr = pivot_fr[[c for c in cols if c in pivot_fr.columns]]\n",
            "\n",
            "pivot_fr.plot(kind='bar', ax=ax, width=0.8, edgecolor='white', linewidth=1.5)\n",
            "ax.set_xlabel('Policy', fontsize=12, fontweight='bold')\n",
            "ax.set_ylabel('Mean Fill Rate', fontsize=12, fontweight='bold')\n",
            "ax.set_title('📦 Fill Rate by Policy and Complexity Level', fontsize=14, fontweight='bold')\n",
            "ax.legend(title='Complexity', bbox_to_anchor=(1.02, 1), loc='upper left')\n",
            "ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')\n",
            "ax.set_ylim(0, 1.1)\n",
            "ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% Target')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'fillrate_by_complexity.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    })
    
    # Box plot of costs
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "viz-boxplot"},
        "outputs": [],
        "source": [
            "# 3. Box Plot of Cost Distribution\n",
            "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
            "\n",
            "for idx, complexity in enumerate(['simple', 'moderate', 'complex', 'extreme']):\n",
            "    ax = axes[idx // 2, idx % 2]\n",
            "    subset = df[df['complexity'] == complexity]\n",
            "    \n",
            "    if len(subset) > 0:\n",
            "        subset.boxplot(column='mean_cost', by='policy', ax=ax)\n",
            "        ax.set_title(f'{complexity.upper()}', fontsize=12, fontweight='bold')\n",
            "        ax.set_xlabel('Policy')\n",
            "        ax.set_ylabel('Mean Cost')\n",
            "        plt.sca(ax)\n",
            "        plt.xticks(rotation=45, ha='right')\n",
            "\n",
            "plt.suptitle('📈 Cost Distribution by Complexity Level', fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'cost_boxplots.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    })
    
    # Heatmap
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "viz-heatmap"},
        "outputs": [],
        "source": [
            "# 4. Heatmap of Policy Performance\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
            "\n",
            "# Cost heatmap\n",
            "pivot_cost = df.pivot_table(values='mean_cost', index='policy', columns='complexity', aggfunc='mean')\n",
            "sns.heatmap(pivot_cost, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=axes[0])\n",
            "axes[0].set_title('Mean Cost (lower is better)', fontweight='bold')\n",
            "\n",
            "# Fill rate heatmap\n",
            "pivot_fr = df.pivot_table(values='fill_rate', index='policy', columns='complexity', aggfunc='mean')\n",
            "sns.heatmap(pivot_fr, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1])\n",
            "axes[1].set_title('Fill Rate (higher is better)', fontweight='bold')\n",
            "\n",
            "plt.suptitle('🔥 Policy Performance Heatmaps', fontsize=14, fontweight='bold', y=1.02)\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'performance_heatmaps.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    })
    
    # RL vs Best Baseline
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "viz-rl-vs-baseline"},
        "outputs": [],
        "source": [
            "# 5. RL vs Best Baseline Comparison\n",
            "summary = df.groupby(['complexity', 'policy'])['mean_cost'].mean().reset_index()\n",
            "\n",
            "comparison_data = []\n",
            "for complexity in ['simple', 'moderate', 'complex', 'extreme']:\n",
            "    subset = summary[summary['complexity'] == complexity]\n",
            "    if len(subset) == 0:\n",
            "        continue\n",
            "    \n",
            "    rl_cost = subset[subset['policy'] == 'RL']['mean_cost'].values\n",
            "    rl_cost = rl_cost[0] if len(rl_cost) > 0 else None\n",
            "    \n",
            "    baselines = subset[subset['policy'] != 'RL']\n",
            "    if len(baselines) > 0:\n",
            "        best_baseline = baselines.loc[baselines['mean_cost'].idxmin()]\n",
            "        comparison_data.append({\n",
            "            'complexity': complexity,\n",
            "            'RL_cost': rl_cost,\n",
            "            'best_baseline': best_baseline['policy'],\n",
            "            'best_baseline_cost': best_baseline['mean_cost'],\n",
            "            'improvement': (best_baseline['mean_cost'] - rl_cost) / best_baseline['mean_cost'] * 100 if rl_cost else None\n",
            "        })\n",
            "\n",
            "comparison_df = pd.DataFrame(comparison_data)\n",
            "print('\\n📊 RL vs Best Baseline Summary:')\n",
            "print('=' * 70)\n",
            "print(comparison_df.to_string(index=False))\n",
            "\n",
            "# Plot\n",
            "if len(comparison_df) > 0:\n",
            "    fig, ax = plt.subplots(figsize=(12, 6))\n",
            "    \n",
            "    x = np.arange(len(comparison_df))\n",
            "    width = 0.35\n",
            "    \n",
            "    bars1 = ax.bar(x - width/2, comparison_df['RL_cost'].fillna(0), width, label='RL', color='#2ecc71')\n",
            "    bars2 = ax.bar(x + width/2, comparison_df['best_baseline_cost'], width, label='Best Baseline', color='#3498db')\n",
            "    \n",
            "    ax.set_xlabel('Complexity Level', fontsize=12, fontweight='bold')\n",
            "    ax.set_ylabel('Mean Cost', fontsize=12, fontweight='bold')\n",
            "    ax.set_title('🏆 RL vs Best Baseline by Complexity', fontsize=14, fontweight='bold')\n",
            "    ax.set_xticks(x)\n",
            "    ax.set_xticklabels(comparison_df['complexity'].str.upper())\n",
            "    ax.legend()\n",
            "    \n",
            "    # Add improvement percentage\n",
            "    for i, (_, row) in enumerate(comparison_df.iterrows()):\n",
            "        if row['improvement']:\n",
            "            color = 'green' if row['improvement'] > 0 else 'red'\n",
            "            ax.annotate(f\"{row['improvement']:.1f}%\", \n",
            "                       xy=(i, max(row['RL_cost'] or 0, row['best_baseline_cost']) + 50),\n",
            "                       ha='center', fontsize=10, color=color, fontweight='bold')\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig(OUTPUT_DIR / 'rl_vs_baseline.png', dpi=150, bbox_inches='tight')\n",
            "    plt.show()"
        ]
    })
    
    # Policy ranking
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "viz-ranking"},
        "outputs": [],
        "source": [
            "# 6. Overall Policy Ranking\n",
            "ranking = df.groupby('policy').agg({\n",
            "    'mean_cost': ['mean', 'std'],\n",
            "    'fill_rate': 'mean',\n",
            "    'spoilage': 'mean'\n",
            "}).round(2)\n",
            "\n",
            "ranking.columns = ['Mean Cost', 'Std Cost', 'Fill Rate', 'Spoilage']\n",
            "ranking = ranking.sort_values('Mean Cost')\n",
            "\n",
            "print('\\n🏆 OVERALL POLICY RANKING (by Mean Cost)')\n",
            "print('=' * 70)\n",
            "print(ranking.to_string())\n",
            "\n",
            "# Visualize ranking\n",
            "fig, ax = plt.subplots(figsize=(12, 6))\n",
            "colors = plt.cm.viridis(np.linspace(0, 0.8, len(ranking)))\n",
            "\n",
            "bars = ax.barh(ranking.index, ranking['Mean Cost'], color=colors, edgecolor='white', linewidth=1.5)\n",
            "ax.set_xlabel('Mean Cost', fontsize=12, fontweight='bold')\n",
            "ax.set_ylabel('Policy', fontsize=12, fontweight='bold')\n",
            "ax.set_title('🏆 Overall Policy Ranking (Lower Cost = Better)', fontsize=14, fontweight='bold')\n",
            "\n",
            "# Add value labels\n",
            "for bar, cost in zip(bars, ranking['Mean Cost']):\n",
            "    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, \n",
            "            f'{cost:.0f}', va='center', fontsize=10)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'policy_ranking.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    })
    
    # Radar chart
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "viz-radar"},
        "outputs": [],
        "source": [
            "# 7. Radar Chart of Multi-Metric Performance\n",
            "from math import pi\n",
            "\n",
            "# Prepare data\n",
            "metrics = ['fill_rate', 'mean_cost', 'spoilage']\n",
            "policy_means = df.groupby('policy')[metrics].mean()\n",
            "\n",
            "# Normalize metrics (invert cost and spoilage so higher is better)\n",
            "normalized = policy_means.copy()\n",
            "normalized['mean_cost'] = 1 - (normalized['mean_cost'] - normalized['mean_cost'].min()) / (normalized['mean_cost'].max() - normalized['mean_cost'].min() + 1e-6)\n",
            "normalized['spoilage'] = 1 - (normalized['spoilage'] - normalized['spoilage'].min()) / (normalized['spoilage'].max() - normalized['spoilage'].min() + 1e-6)\n",
            "\n",
            "# Create radar chart\n",
            "categories = ['Fill Rate', 'Cost Efficiency', 'Low Spoilage']\n",
            "N = len(categories)\n",
            "angles = [n / float(N) * 2 * pi for n in range(N)]\n",
            "angles += angles[:1]\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))\n",
            "\n",
            "colors = plt.cm.tab10(np.linspace(0, 1, len(normalized)))\n",
            "\n",
            "for idx, (policy, row) in enumerate(normalized.iterrows()):\n",
            "    values = [row['fill_rate'], row['mean_cost'], row['spoilage']]\n",
            "    values += values[:1]\n",
            "    ax.plot(angles, values, 'o-', linewidth=2, label=policy, color=colors[idx])\n",
            "    ax.fill(angles, values, alpha=0.1, color=colors[idx])\n",
            "\n",
            "ax.set_xticks(angles[:-1])\n",
            "ax.set_xticklabels(categories, fontsize=11)\n",
            "ax.set_title('🎯 Multi-Metric Policy Comparison\\n(Higher = Better)', fontsize=14, fontweight='bold', y=1.08)\n",
            "ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(OUTPUT_DIR / 'radar_comparison.png', dpi=150, bbox_inches='tight')\n",
            "plt.show()"
        ]
    })
    
    # ==================== STATISTICAL ANALYSIS ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "stats-header"},
        "source": ["## 8️⃣ Statistical Analysis"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "stats"},
        "outputs": [],
        "source": [
            "# Statistical summary\n",
            "from scipy import stats as scipy_stats\n",
            "\n",
            "print('📊 STATISTICAL SUMMARY')\n",
            "print('=' * 70)\n",
            "\n",
            "# Summary statistics\n",
            "summary_stats = df.groupby('policy').agg({\n",
            "    'mean_cost': ['count', 'mean', 'std', 'min', 'max'],\n",
            "    'fill_rate': ['mean', 'std']\n",
            "}).round(3)\n",
            "\n",
            "print('\\n📈 Summary Statistics:')\n",
            "print(summary_stats.to_string())\n",
            "\n",
            "# Pairwise comparisons with RL\n",
            "print('\\n\\n📉 RL vs Baselines Statistical Tests:')\n",
            "print('-' * 70)\n",
            "\n",
            "rl_costs = df[df['policy'] == 'RL']['mean_cost'].values\n",
            "\n",
            "if len(rl_costs) > 0:\n",
            "    for policy in df['policy'].unique():\n",
            "        if policy == 'RL':\n",
            "            continue\n",
            "        \n",
            "        baseline_costs = df[df['policy'] == policy]['mean_cost'].values\n",
            "        \n",
            "        if len(baseline_costs) > 1 and len(rl_costs) > 1:\n",
            "            # Paired t-test\n",
            "            min_len = min(len(rl_costs), len(baseline_costs))\n",
            "            t_stat, p_value = scipy_stats.ttest_ind(rl_costs[:min_len], baseline_costs[:min_len])\n",
            "            \n",
            "            rl_mean = np.mean(rl_costs)\n",
            "            baseline_mean = np.mean(baseline_costs)\n",
            "            improvement = (baseline_mean - rl_mean) / baseline_mean * 100\n",
            "            \n",
            "            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''\n",
            "            \n",
            "            print(f'{policy:12} | RL: {rl_mean:.1f} vs {baseline_mean:.1f} | '\n",
            "                  f'Diff: {improvement:+.1f}% | p={p_value:.4f} {sig}')\n",
            "else:\n",
            "    print('No RL results available for comparison')\n",
            "\n",
            "print('\\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001')"
        ]
    })
    
    # ==================== FINAL SUMMARY ====================
    cells.append({
        "cell_type": "markdown",
        "metadata": {"id": "summary-header"},
        "source": ["## 9️⃣ Final Summary & Export"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "final-summary"},
        "outputs": [],
        "source": [
            "# Final summary\n",
            "print('\\n' + '=' * 70)\n",
            "print('📋 BENCHMARK SUMMARY REPORT')\n",
            "print('=' * 70)\n",
            "\n",
            "print(f'\\n📊 Evaluation Statistics:')\n",
            "print(f'   Total evaluations: {len(df)}')\n",
            "print(f'   Policies tested: {df[\"policy\"].nunique()}')\n",
            "print(f'   Environments tested: {df[\"env_id\"].nunique()}')\n",
            "print(f'   Episodes per evaluation: {N_EPISODES}')\n",
            "\n",
            "print(f'\\n🏆 Policy Rankings (by Mean Cost):')\n",
            "rankings = df.groupby('policy')['mean_cost'].mean().sort_values()\n",
            "for i, (policy, cost) in enumerate(rankings.items(), 1):\n",
            "    print(f'   {i}. {policy}: {cost:.1f}')\n",
            "\n",
            "print(f'\\n📁 Output Files:')\n",
            "for f in OUTPUT_DIR.glob('*'):\n",
            "    print(f'   {f}')\n",
            "\n",
            "print('\\n' + '=' * 70)\n",
            "print('✅ BENCHMARKING COMPLETE!')\n",
            "print('=' * 70)"
        ]
    })
    
    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

if __name__ == "__main__":
    notebook = create_notebook()
    output_path = "colab_training/policy_benchmark.ipynb"
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Notebook created: {output_path}")
