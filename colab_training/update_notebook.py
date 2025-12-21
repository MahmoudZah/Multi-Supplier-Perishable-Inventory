
import json
import os

NOTEBOOK_PATH = "f:/Multi-Supplier-Perishable-Inventory/colab_training/policy_benchmark.ipynb"

def update_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    
    # 1. Update Imports
    print("Updating imports...")
    import_cell = None
    for cell in cells:
        if cell.get('metadata', {}).get('id') == 'imports':
            import_cell = cell
            break
    
    if import_cell:
        source = "".join(import_cell['source'])
        if "from perishable_inventory_mdp import plotting" not in source:
            # Add it after other local imports
            if "from colab_training.benchmark import" in source:
                new_source = source.replace(
                    "from colab_training.benchmark import (\n",
                    "from perishable_inventory_mdp import plotting\nfrom colab_training.benchmark import (\n"
                )
            else:
                new_source = source + "\nfrom perishable_inventory_mdp import plotting\n"
            
            # Split back into lines for JSON format
            import_cell['source'] = [line + "\n" for line in new_source.split('\n')][:-1] 
            # (simple split might add extra newline, careful)
            # Actually, let's just append neatly
            import_cell['source'].append("\n")
            import_cell['source'].append("from perishable_inventory_mdp import plotting\n")
            
    # 2. Add Paid Cost Comparison Plot
    print("Adding Paid Cost Comparison cell...")
    cost_viz_idx = -1
    for i, cell in enumerate(cells):
        if cell.get('metadata', {}).get('id') == 'viz-cost-complexity':
            cost_viz_idx = i
            break
            
    if cost_viz_idx != -1:
        # Create new cell
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "viz-paid-cost"},
            "outputs": [],
            "source": [
                "# Paid Cost Comparison (Purchase vs Holding)\n",
                "print('Generating Paid Cost Comparison Plot...')\n",
                "\n",
                "if 'df' in locals() and len(df) > 0:\n",
                "    # Convert to list of dicts for the plotting function\n",
                "    results_list = df.to_dict('records')\n",
                "    \n",
                "    # Generate Paid Cost Plot\n",
                "    fig_cost = plotting.plot_paid_cost_comparison(results_list)\n",
                "    plt.show()\n",
                "else:\n",
                "    print('Warning: DataFrame df not found or empty. Run benchmarks first.')\n"
            ]
        }
        # Insert after the cost complexity plot
        cells.insert(cost_viz_idx + 1, new_cell)

    # 3. Update Simulation Trace Plot
    print("Updating Simulation Trace cell...")
    trace_cell = None
    for cell in cells:
        if cell.get('metadata', {}).get('id') == 'timeseries-plot':
            trace_cell = cell
            break
            
    if trace_cell:
        trace_cell['source'] = [
            "# Simulation Trace Visualization (Updated)\n",
            "print('Generating Simulation Trace...')\n",
            "\n",
            "# Select a specific policy for trace (e.g., RL or TBS)\n",
            "# Using a complex environment configuration for demonstration\n",
            "from colab_training.environment_suite import build_environment_from_config\n",
            "\n",
            "trace_env_config = {\n",
            "    'n_suppliers': 2,\n",
            "    'max_inventory': 50,\n",
            "    'max_order_quantity': 20,\n",
            "    'demand_params': {'mean': 8},\n",
            "    'lead_times': [1, 3],  # Fast, Slow\n",
            "    'unit_costs': [10, 5], # Expensive, Cheap\n",
            "    'holding_cost': 1,\n",
            "    'shortage_cost': 20,\n",
            "    'spoilage_cost': 5,\n",
            "    'shelf_life': 5,\n",
            "    'random_seed': 42\n",
            "}\n",
            "\n",
            "# Create environment\n",
            "trace_env = PerishableInventoryGymWrapper(build_environment_from_config(trace_env_config, complexity_level=1))\n",
            "\n",
            "# Use TBS as a reliable baseline for trace if RL is not available or behaves erratically\n",
            "trace_policy = get_tbs_policy_for_env(trace_env)\n",
            "policy_name = 'TBS Policy'\n",
            "\n",
            "# Run a short episode\n",
            "obs, _ = trace_env.reset(seed=123)\n",
            "trace_data = []\n",
            "current_inv = 0 \n",
            "\n",
            "for _ in range(30): # 30 periods trace\n",
            "    if hasattr(trace_policy, 'predict'):\n",
            "        action, _ = trace_policy.predict(obs, deterministic=True)\n",
            "    else:\n",
            "        action = trace_policy.act(obs)\n",
            "        \n",
            "    obs, reward, terminated, truncated, info = trace_env.step(action)\n",
            "    \n",
            "    # Post-process info for plotting\n",
            "    # Separate orders if possible (assuming 2 suppliers: 0=slow/cheap, 1=fast/expensive usually sorted by cost?)\n",
            "    # gym_env V2 sorts suppliers. Usually index 0 is cheap/slow, index -1 is fast/expensive.\n",
            "    # But let's check action shape. 'orders' in info is just the array.\n",
            "    orders = info.get('orders', [0])\n",
            "    \n",
            "    # Attempt to split orders into slow/fast for visualization\n",
            "    if len(orders) >= 2:\n",
            "        # Assuming 0 is slow/cheap, 1 is fast/expensive based on standard setup\n",
            "        info['order_slow'] = orders[0]\n",
            "        info['order_fast'] = orders[1]\n",
            "    else:\n",
            "        info['order_slow'] = orders[0]\n",
            "        info['order_fast'] = 0\n",
            "\n",
            "    # Ensure inventory position is present\n",
            "    info['period'] = info.get('time_step')\n",
            "    info['inventory_total'] = info.get('inventory')\n",
            "    info['ip_total'] = info.get('inventory_position')\n",
            "    info['cost'] = info.get('total_cost')\n",
            "    \n",
            "    # Add params for reference lines if TBS\n",
            "    if hasattr(trace_policy, 'base_stock_level'):\n",
            "        info['base_stock'] = trace_policy.base_stock_level\n",
            "    if hasattr(trace_policy, 'reorder_point'):\n",
            "        info['reorder_point'] = trace_policy.reorder_point\n",
            "        \n",
            "    trace_data.append(info)\n",
            "    \n",
            "    if terminated or truncated:\n",
            "        break\n",
            "\n",
            "# Generate Plot\n",
            "plotting.plot_simulation_trace(trace_data)\n",
            "plt.show()\n"
        ]

    # Save updated notebook
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    
    print("Notebook updated successfully.")

if __name__ == "__main__":
    update_notebook()
