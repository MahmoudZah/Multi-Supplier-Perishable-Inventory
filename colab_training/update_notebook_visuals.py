import json
import os

NOTEBOOK_PATH = r'f:\Multi-Supplier-Perishable-Inventory\colab_training\policy_benchmark.ipynb'

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update viz-spoilage-rate
    spoilage_found = False
    for cell in nb['cells']:
        if cell.get('metadata', {}).get('id') == 'viz-spoilage-rate':
            print("Updating viz-spoilage-rate cell...")
            cell['source'] = [
                "# 3. Spoilage Rate by Policy and Complexity (Updated)\n",
                "fig, ax = plt.subplots(figsize=(10, 6))\n",
                "plotting.plot_metric_comparison(\n",
                "    df, \n",
                "    metric_col='spoilage_rate', \n",
                "    title='Spoilage Rate by Policy and Complexity (Log Scale)', \n",
                "    ylabel='Mean Spoilage Rate', \n",
                "    ax=ax, \n",
                "    log_scale=True\n",
                ")\n",
                "plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='5% Target')\n",
                "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
            spoilage_found = True
            break
    
    if not spoilage_found:
        print("Warning: viz-spoilage-rate cell not found.")

    # 2. Add new metrics visualizations after viz-ranking or at end
    # Find viz-ranking index
    insert_idx = -1
    for idx, cell in enumerate(nb['cells']):
        if cell.get('metadata', {}).get('id') == 'viz-ranking':
            insert_idx = idx + 1
            break
    
    if insert_idx == -1:
        # If not found, append to end before the last cell (which might be download)
        insert_idx = len(nb['cells']) - 1

    print(f"Inserting new visualization cells at index {insert_idx}...")

    # Fill Rate Cell
    fill_rate_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "id": "viz-fill-rate"
        },
        "outputs": [],
        "source": [
            "# 7. Fill Rate Comparison (NEW)\n",
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "plotting.plot_metric_comparison(\n",
            "    df, \n",
            "    metric_col='fill_rate', \n",
            "    title='Fill Rate by Policy and Complexity', \n",
            "    ylabel='Mean Fill Rate', \n",
            "    ax=ax\n",
            ")\n",
            "plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% Target')\n",
            "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }

    # Total Cost Cell
    total_cost_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "id": "viz-total-cost"
        },
        "outputs": [],
        "source": [
            "# 8. Total Cost Comparison (NEW)\n",
            "fig, ax = plt.subplots(figsize=(10, 6))\n",
            "plotting.plot_metric_comparison(\n",
            "    df, \n",
            "    metric_col='total_cost', \n",
            "    title='Total Cost by Policy and Complexity (Log Scale)', \n",
            "    ylabel='Mean Total Cost', \n",
            "    ax=ax,\n",
            "    log_scale=True\n",
            ")\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }

    # Insert cells
    nb['cells'].insert(insert_idx, total_cost_cell)
    nb['cells'].insert(insert_idx, fill_rate_cell)

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    
    print("Notebook updated successfully.")

if __name__ == "__main__":
    update_notebook()
