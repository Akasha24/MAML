"""
Compute average MAML and baseline errors for 5-shot and 20-shot scenarios.

- Uses existing `results/maml_model.pt` (trained meta-model)
- Uses `test.load_test_tasks()` for 5-shot (uses first 5 support samples)
- Generates 20 new tasks with `WirelessTaskGenerator` for 20-shot

Prints results so we can paste them into README.
"""

import numpy as np
import torch
from pathlib import Path

# Import helper functions from test.py and generate_data.py
import test as test_module
from generate_data import WirelessTaskGenerator


def evaluate_few_shot_on_existing_tasks(model, tasks, k_shot=5, inner_lr=0.01, inner_steps=5, baseline_steps=200, device='cpu'):
    maml_losses = []
    baseline_losses = []

    for task in tasks:
        # Use first k support samples
        support_subset = {'X': task['X_support'][:k_shot], 'Y': task['Y_support'][:k_shot]}
        query = {'X': task['X_query'], 'Y': task['Y_query']}

        # MAML adaptation
        adapted_model, _ = test_module.adapt_on_support(model, support_subset, inner_lr=inner_lr, num_steps=inner_steps, device=device)
        qloss, _, _ = test_module.evaluate_on_query(adapted_model, query, device=device)
        maml_losses.append(qloss)

        # Baseline: train fresh on the support subset
        baseline_loss = test_module.compute_baseline(support_subset, query, device=device, steps_per_task=baseline_steps)
        baseline_losses.append(baseline_loss)

    return np.mean(maml_losses), np.mean(baseline_losses)


def evaluate_on_generated_tasks(model, n_tasks=20, n_support=20, n_query=64, inner_lr=0.01, inner_steps=5, baseline_steps=200, device='cpu', seed=123):
    generator = WirelessTaskGenerator(input_dim=4, output_dim=1, random_seed=seed)
    tasks = generator.generate_task_distribution(n_tasks=n_tasks, n_support=n_support, n_query=n_query)

    maml_losses = []
    baseline_losses = []

    for task in tasks:
        support = {'X': task['X_support'], 'Y': task['Y_support']}
        query = {'X': task['X_query'], 'Y': task['Y_query']}

        adapted_model, _ = test_module.adapt_on_support(model, support, inner_lr=inner_lr, num_steps=inner_steps, device=device)
        qloss, _, _ = test_module.evaluate_on_query(adapted_model, query, device=device)
        maml_losses.append(qloss)

        baseline_loss = test_module.compute_baseline(support, query, device=device, steps_per_task=baseline_steps)
        baseline_losses.append(baseline_loss)

    return np.mean(maml_losses), np.mean(baseline_losses)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load trained meta-model
    model = test_module.load_trained_model(device=device)

    # Load existing test tasks (these have support size 8) and evaluate 5-shot using first 5 samples
    test_tasks = test_module.load_test_tasks()
    maml_5, base_5 = evaluate_few_shot_on_existing_tasks(model, test_tasks, k_shot=5, device=device)

    # Generate 20-shot tasks and evaluate
    maml_20, base_20 = evaluate_on_generated_tasks(model, n_tasks=20, n_support=20, n_query=64, device=device)

    print('\n=== Few-shot Results ===')
    print(f'5-shot average MAML query loss:    {maml_5:.6f}')
    print(f'5-shot average Baseline query loss:{base_5:.6f}')

    print('\n=== Many-shot Results ===')
    print(f'20-shot average MAML query loss:    {maml_20:.6f}')
    print(f'20-shot average Baseline query loss:{base_20:.6f}')

    # Print a concise table-like line for README
    print('\nTABLE_LINE: {:.6f},{:.6f},{:.6f},{:.6f}'.format(maml_5, base_5, maml_20, base_20))
