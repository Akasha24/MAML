"""
Test and evaluate MAML, Reptile, and Baseline models for wireless channel estimation.

This script:
1. Loads trained MAML and Reptile meta-models
2. Evaluates on test tasks (adapt on support, measure on query)
3. Compares against baseline (train from scratch)
4. Generates plots using ACTUAL training history and evaluation data
5. Computes few-shot results table
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

try:
    from generate_data import WirelessTaskGenerator
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False


class ChannelEstimationNetwork(nn.Module):
    """
    Neural network for channel estimation.
    Same architecture as used in training.
    """
    
    def __init__(self, input_dim=4, output_dim=1, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def clone(self):
        """Create a deep copy of the network."""
        return deepcopy(self)


# ========== REAL PLOTTING FUNCTIONS ==========

def plot_training_loss_curve(output_path='results/plot_loss.png'):
    """
    Plot ACTUAL training loss curves from saved training history.
    
    Loads results/training_history.npz saved by train.py and plots
    the real MAML, Reptile, and Baseline query/support loss over iterations.
    
    Args:
        output_path: Where to save the PNG file
    """
    history_path = Path('results') / 'training_history.npz'
    
    if not history_path.exists():
        print(f"[WARNING] Training history not found at {history_path}")
        print("  Run 'python train.py' first to generate training history.")
        return
    
    data = np.load(history_path)
    
    maml_query = data['maml_query']
    maml_support = data['maml_support']
    reptile_query = data['reptile_query']
    reptile_support = data['reptile_support']
    baseline_query = data['baseline_query']
    baseline_support = data['baseline_support']
    
    iterations = np.arange(1, len(maml_query) + 1)
    
    # Compute smoothed curves (moving average)
    window = max(1, len(iterations) // 20)
    
    def smooth(arr, w):
        if w <= 1:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode='valid')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Query loss (left panel) ---
    ax = axes[0]
    alpha_raw = 0.25
    ax.plot(iterations, maml_query, color='#2196F3', alpha=alpha_raw, linewidth=0.8)
    ax.plot(iterations, reptile_query, color='#4CAF50', alpha=alpha_raw, linewidth=0.8)
    ax.plot(iterations, baseline_query, color='#F44336', alpha=alpha_raw, linewidth=0.8)
    
    # Smoothed lines
    sm_iters = iterations[window-1:]
    ax.plot(sm_iters, smooth(maml_query, window), color='#2196F3', linewidth=2.5, label='MAML')
    ax.plot(sm_iters, smooth(reptile_query, window), color='#4CAF50', linewidth=2.5, label='Reptile')
    ax.plot(sm_iters, smooth(baseline_query, window), color='#F44336', linewidth=2.5, linestyle='--', label='Baseline')
    
    ax.set_xlabel('Meta-Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Query Loss Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    # --- Support loss (right panel) ---
    ax = axes[1]
    ax.plot(iterations, maml_support, color='#2196F3', alpha=alpha_raw, linewidth=0.8)
    ax.plot(iterations, reptile_support, color='#4CAF50', alpha=alpha_raw, linewidth=0.8)
    ax.plot(iterations, baseline_support, color='#F44336', alpha=alpha_raw, linewidth=0.8)
    
    ax.plot(sm_iters, smooth(maml_support, window), color='#2196F3', linewidth=2.5, label='MAML')
    ax.plot(sm_iters, smooth(reptile_support, window), color='#4CAF50', linewidth=2.5, label='Reptile')
    ax.plot(sm_iters, smooth(baseline_support, window), color='#F44336', linewidth=2.5, linestyle='--', label='Baseline')
    
    ax.set_xlabel('Meta-Training Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Support Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Support Loss Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Training loss curve saved: {output_file}")
    plt.close(fig)


def plot_few_shot_comparison(maml_model, reptile_model, test_tasks,
                             inner_lr=0.01, inner_steps=5, baseline_steps=200,
                             device='cpu', output_path='results/plot_comparison.png'):
    """
    Plot ACTUAL few-shot performance comparison using real model evaluation.
    
    Evaluates MAML, Reptile, and Baseline at different support set sizes
    using actual model predictions.
    
    Args:
        maml_model: Trained MAML model
        reptile_model: Trained Reptile model
        test_tasks: List of test task dicts
        inner_lr: Adaptation learning rate
        inner_steps: Adaptation gradient steps
        baseline_steps: Baseline training steps per task
        device: 'cpu' or 'cuda'
        output_path: Where to save the PNG file
    """
    shot_counts = [2, 4, 6, 8]  # Must be <= support set size (8)
    
    maml_errors = []
    reptile_errors = []
    baseline_errors = []
    
    print("  Evaluating at different shot counts...")
    for k in shot_counts:
        maml_losses = []
        reptile_losses = []
        base_losses = []
        
        for task in test_tasks:
            support_k = {'X': task['X_support'][:k], 'Y': task['Y_support'][:k]}
            query = {'X': task['X_query'], 'Y': task['Y_query']}
            
            # MAML adaptation
            adapted_maml, _ = adapt_on_support(maml_model, support_k,
                                                inner_lr=inner_lr, num_steps=inner_steps, device=device)
            ml, _, _ = evaluate_on_query(adapted_maml, query, device=device)
            maml_losses.append(ml)
            
            # Reptile adaptation
            adapted_reptile, _ = adapt_on_support(reptile_model, support_k,
                                                   inner_lr=inner_lr, num_steps=inner_steps, device=device)
            rl, _, _ = evaluate_on_query(adapted_reptile, query, device=device)
            reptile_losses.append(rl)
            
            # Baseline
            bl = compute_baseline(support_k, query, device=device, steps_per_task=baseline_steps)
            base_losses.append(bl)
        
        maml_errors.append(np.mean(maml_losses))
        reptile_errors.append(np.mean(reptile_losses))
        baseline_errors.append(np.mean(base_losses))
        print(f"    {k}-shot: MAML={maml_errors[-1]:.4f}  Reptile={reptile_errors[-1]:.4f}  Baseline={baseline_errors[-1]:.4f}")
    
    maml_errors = np.array(maml_errors)
    reptile_errors = np.array(reptile_errors)
    baseline_errors = np.array(baseline_errors)
    shot_counts = np.array(shot_counts)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(shot_counts, maml_errors, 'o-', linewidth=2.5, markersize=9, color='#2196F3',
            label='MAML (Meta-Learned Init)', markerfacecolor='#BBDEFB',
            markeredgewidth=2, markeredgecolor='#1565C0')
    
    ax.plot(shot_counts, reptile_errors, 's-', linewidth=2.5, markersize=9, color='#4CAF50',
            label='Reptile (First-Order Meta)', markerfacecolor='#C8E6C9',
            markeredgewidth=2, markeredgecolor='#2E7D32')
    
    ax.plot(shot_counts, baseline_errors, 'D--', linewidth=2.5, markersize=9, color='#F44336',
            label='Baseline (Random Init)', markerfacecolor='#FFCDD2',
            markeredgewidth=2, markeredgecolor='#C62828')
    
    # Shade MAML advantage region
    worst_meta = np.maximum(maml_errors, reptile_errors)
    mask = baseline_errors > worst_meta
    if mask.any():
        ax.fill_between(shot_counts, worst_meta, baseline_errors,
                         alpha=0.12, color='green', label='Meta-Learning Advantage')
    
    ax.set_xlabel('Number of Support Samples (Shots)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query MSE Loss', fontsize=12, fontweight='bold')
    ax.set_title('Few-Shot Adaptation: MAML vs Reptile vs Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)
    ax.set_xticks(shot_counts)
    
    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Few-shot comparison plot saved: {output_file}")
    plt.close(fig)


# ========== EVALUATION HELPER FUNCTIONS ==========

def adapt_on_support(model, support_data, inner_lr=0.01, num_steps=5, device='cpu'):
    """
    Perform task-specific adaptation using support set.
    
    Takes several gradient steps on the support set to adapt the model
    to this specific task's channel characteristics.
    
    Args:
        model: Neural network (ChannelEstimationNetwork)
        support_data: dict with 'X' (n_support, input_dim) and 'Y' (n_support, output_dim)
        inner_lr: Learning rate for adaptation steps
        num_steps: Number of gradient steps to take
        device: 'cpu' or 'cuda'
        
    Returns:
        adapted_model: Model with task-specific weights
        adaptation_loss: Loss on support set after adaptation (diagnostic info)
    """
    # Clone the model to avoid modifying original
    adapted_model = model.clone()
    adapted_model.to(device)
    
    # Optimizer for inner loop
    optimizer = optim.SGD(adapted_model.parameters(), lr=inner_lr)
    loss_fn = nn.MSELoss()
    
    # Prepare data
    X_support = torch.from_numpy(support_data['X']).float().to(device)
    Y_support = torch.from_numpy(support_data['Y']).float().to(device)
    
    # Take gradient steps on support set
    adaptation_loss = 0.0
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass
        preds = adapted_model(X_support)
        loss = loss_fn(preds, Y_support)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record final loss
        if step == num_steps - 1:
            adaptation_loss = loss.item()
    
    return adapted_model, adaptation_loss


def evaluate_on_query(model, query_data, device='cpu'):
    """
    Evaluate model on query set (without updating weights).
    
    Args:
        model: Neural network
        query_data: dict with 'X' (n_query, input_dim) and 'Y' (n_query, output_dim)
        device: 'cpu' or 'cuda'
        
    Returns:
        query_loss: MSE loss on query set
        predictions: Model predictions on query set
        targets: Ground truth query targets
    """
    loss_fn = nn.MSELoss()
    
    # Prepare data
    X_query = torch.from_numpy(query_data['X']).float().to(device)
    Y_query = torch.from_numpy(query_data['Y']).float().to(device)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_query)
        loss = loss_fn(preds, Y_query).item()
    
    return loss, preds.cpu().numpy(), Y_query.cpu().numpy()


def compute_baseline(support_data, query_data, device='cpu', steps_per_task=200):
    """
    Compute baseline error: train model from scratch on the support set and
    evaluate on the query set.

    Args:
        support_data: dict with 'X' and 'Y' for the support set
        query_data: dict with 'X' and 'Y' for the query set
        device: 'cpu' or 'cuda'
        steps_per_task: Number of training steps on the support set

    Returns:
        baseline_loss: MSE loss on the query set after training on support
    """
    # Create fresh model (no meta-learning benefit)
    baseline_model = ChannelEstimationNetwork().to(device)
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Prepare data
    X_support = torch.from_numpy(support_data['X']).float().to(device)
    Y_support = torch.from_numpy(support_data['Y']).float().to(device)
    X_query = torch.from_numpy(query_data['X']).float().to(device)
    Y_query = torch.from_numpy(query_data['Y']).float().to(device)

    # Train on support set
    for _ in range(steps_per_task):
        optimizer.zero_grad()
        preds = baseline_model(X_support)
        loss = loss_fn(preds, Y_support)
        loss.backward()
        optimizer.step()

    # Evaluate on query set
    baseline_model.eval()
    with torch.no_grad():
        preds = baseline_model(X_query)
        final_loss = loss_fn(preds, Y_query).item()

    return final_loss


def load_test_tasks(data_path='results'):
    """
    Load test tasks from NPZ file.
    
    Args:
        data_path: Directory containing test_tasks.npz
        
    Returns:
        List of task dicts with X_support, Y_support, X_query, Y_query
    """
    data_file = Path(data_path) / 'test_tasks.npz'
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {data_file}\n"
            "Run 'python generate_data.py' first to generate the dataset."
        )
    
    data = np.load(data_file)
    
    # Convert stacked arrays to list of tasks
    n_tasks = data['X_support'].shape[0]
    tasks = []
    
    for i in range(n_tasks):
        tasks.append({
            'X_support': data['X_support'][i],
            'Y_support': data['Y_support'][i],
            'X_query': data['X_query'][i],
            'Y_query': data['Y_query'][i],
            'snr': data['snr'][i],
            'num_paths': data['num_paths'][i],
            'noise_scale': data['noise_scale'][i],
        })
    
    return tasks


def load_trained_model(model_path, device='cpu'):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model weights (.pt file)
        device: 'cpu' or 'cuda'
        
    Returns:
        model: Loaded neural network
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_file}\n"
            "Run 'python train.py' first to train the model."
        )
    
    model = ChannelEstimationNetwork(input_dim=4, output_dim=1)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    return model


def evaluate_few_shot_on_existing_tasks(model, tasks, k_shot=5, inner_lr=0.01, inner_steps=5, baseline_steps=200, device='cpu'):
    """
    Evaluate model on existing tasks with k-shot learning.
    Uses first k support samples from each task.
    """
    maml_losses = []
    baseline_losses = []

    for task in tasks:
        # Use first k support samples
        support_subset = {'X': task['X_support'][:k_shot], 'Y': task['Y_support'][:k_shot]}
        query = {'X': task['X_query'], 'Y': task['Y_query']}

        # Model adaptation
        adapted_model, _ = adapt_on_support(model, support_subset, inner_lr=inner_lr, num_steps=inner_steps, device=device)
        qloss, _, _ = evaluate_on_query(adapted_model, query, device=device)
        maml_losses.append(qloss)

        # Baseline: train fresh on the support subset
        baseline_loss = compute_baseline(support_subset, query, device=device, steps_per_task=baseline_steps)
        baseline_losses.append(baseline_loss)

    return np.mean(maml_losses), np.mean(baseline_losses)


def evaluate_on_generated_tasks(model, n_tasks=20, n_support=20, n_query=64, inner_lr=0.01, inner_steps=5, baseline_steps=200, device='cpu', seed=123):
    """
    Evaluate model on newly generated tasks with specific shot count.
    """
    if not HAS_GENERATOR:
        print("[WARNING] WirelessTaskGenerator not available. Skipping generated-task evaluation.")
        return None, None
        
    generator = WirelessTaskGenerator(input_dim=4, output_dim=1, random_seed=seed)
    tasks = generator.generate_task_distribution(n_tasks=n_tasks, n_support=n_support, n_query=n_query)

    model_losses = []
    baseline_losses = []

    for task in tasks:
        support = {'X': task['X_support'], 'Y': task['Y_support']}
        query = {'X': task['X_query'], 'Y': task['Y_query']}

        adapted_model, _ = adapt_on_support(model, support, inner_lr=inner_lr, num_steps=inner_steps, device=device)
        qloss, _, _ = evaluate_on_query(adapted_model, query, device=device)
        model_losses.append(qloss)

        baseline_loss = compute_baseline(support, query, device=device, steps_per_task=baseline_steps)
        baseline_losses.append(baseline_loss)

    return np.mean(model_losses), np.mean(baseline_losses)


def compute_few_shot_table(maml_model, reptile_model, test_tasks, inner_lr=0.01, inner_steps=10, baseline_steps=50, device='cpu'):
    """
    Compute few-shot results for MAML, Reptile, and Baseline.
    """
    print("\n=== Computing Few-Shot Results ===")
    
    # 5-shot evaluation
    maml_5, base_5_m = evaluate_few_shot_on_existing_tasks(
        maml_model, test_tasks, k_shot=5, 
        inner_lr=inner_lr, inner_steps=inner_steps, 
        baseline_steps=baseline_steps, device=device
    )
    reptile_5, base_5_r = evaluate_few_shot_on_existing_tasks(
        reptile_model, test_tasks, k_shot=5, 
        inner_lr=inner_lr, inner_steps=inner_steps, 
        baseline_steps=baseline_steps, device=device
    )
    base_5 = (base_5_m + base_5_r) / 2  # Average baseline across runs
    
    # 20-shot evaluation (generate new tasks with 20 support samples)
    maml_20, base_20_m = evaluate_on_generated_tasks(
        maml_model, n_tasks=20, n_support=20, n_query=64, 
        inner_lr=inner_lr, inner_steps=inner_steps, 
        baseline_steps=baseline_steps, device=device
    )
    reptile_20, base_20_r = evaluate_on_generated_tasks(
        reptile_model, n_tasks=20, n_support=20, n_query=64, 
        inner_lr=inner_lr, inner_steps=inner_steps, 
        baseline_steps=baseline_steps, device=device, seed=124
    )
    
    print('\n=== Few-shot Results (5-shot) ===')
    print(f'  MAML query loss:     {maml_5:.6f}')
    print(f'  Reptile query loss:  {reptile_5:.6f}')
    print(f'  Baseline query loss: {base_5:.6f}')

    if maml_20 is not None and reptile_20 is not None:
        base_20 = (base_20_m + base_20_r) / 2
        print('\n=== Many-shot Results (20-shot) ===')
        print(f'  MAML query loss:     {maml_20:.6f}')
        print(f'  Reptile query loss:  {reptile_20:.6f}')
        print(f'  Baseline query loss: {base_20:.6f}')
        
        print('\n+----------------------------+--------------------+---------------------+')
        print('| Method                     | 5-shot Error (MSE) | 20-shot Error (MSE) |')
        print('+----------------------------+--------------------+---------------------+')
        print(f'| Baseline (from scratch)    | {base_5:>18.6f} | {base_20:>19.6f} |')
        print(f'| MAML (meta-learned init)   | {maml_5:>18.6f} | {maml_20:>19.6f} |')
        print(f'| Reptile (first-order meta) | {reptile_5:>18.6f} | {reptile_20:>19.6f} |')
        print('+----------------------------+--------------------+---------------------+')
        return maml_5, reptile_5, base_5, maml_20, reptile_20, base_20
    else:
        return maml_5, reptile_5, base_5, None, None, None


def print_results_table(results):
    """
    Pretty-print evaluation results.
    
    Args:
        results: List of dicts with task evaluation details
    """
    print()
    print("=" * 110)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 110)
    print()
    print(f"{'Task':<6} {'SNR (dB)':<12} {'Paths':<8} {'MAML':<14} {'Reptile':<14} {'Baseline':<14} {'Best Method':<14}")
    print("-" * 110)
    
    for r in results:
        losses = {
            'MAML': r['maml_loss'],
            'Reptile': r['reptile_loss'],
            'Baseline': r['baseline_loss']
        }
        best = min(losses, key=losses.get)
        
        print(
            f"{r['task_id']:<6} "
            f"{r['snr']:<12.1f} "
            f"{r['paths']:<8} "
            f"{r['maml_loss']:<14.6f} "
            f"{r['reptile_loss']:<14.6f} "
            f"{r['baseline_loss']:<14.6f} "
            f"{best:<14}"
        )
    
    print("-" * 110)


def print_metric_interpretation():
    """Explain what the evaluation metrics mean."""
    print()
    print("=" * 90)
    print("UNDERSTANDING THE METRICS")
    print("=" * 90)
    print()
    print("LOSS VALUES (Lower is Better)")
    print("-" * 90)
    print("- MAML Loss:     MSE after inner-loop adaptation from MAML initialization")
    print("- Reptile Loss:  MSE after inner-loop adaptation from Reptile initialization")
    print("- Baseline Loss: MSE training a fresh model from scratch for 200 steps")
    print()
    print("Interpretation:")
    print("  - If Meta < Baseline: [OK] Meta-learning helps! (faster adaptation)")
    print("  - If Meta > Baseline: [FAIL] Adaptation insufficient (needs tuning)")
    print()
    print("EXPECTED TREND by SNR:")
    print("-" * 90)
    print("  Higher SNR -> Less noise -> Easier estimation -> Lower error")
    print("  Lower SNR  -> More noise -> Harder estimation -> Higher error")
    print()
    print("=" * 90)
    print()


def main():
    print("=" * 90)
    print("MAML + Reptile Test Evaluation")
    print("=" * 90)
    print()
    
    # Settings (must match train.py)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inner_lr = 0.01
    inner_steps = 10
    baseline_steps = 50
    
    print(f"Device: {device}")
    print(f"Adaptation learning rate: {inner_lr}")
    print(f"Adaptation steps: {inner_steps}")
    print(f"Baseline training steps: {baseline_steps}")
    print()
    
    # Load models and dataset
    print("Loading trained models and test dataset...")
    maml_model = load_trained_model('results/maml_model.pt', device=device)
    
    reptile_path = Path('results/reptile_model.pt')
    if reptile_path.exists():
        reptile_model = load_trained_model('results/reptile_model.pt', device=device)
        has_reptile = True
        print("[OK] Loaded MAML and Reptile models")
    else:
        reptile_model = None
        has_reptile = False
        print("[OK] Loaded MAML model (Reptile model not found, skipping)")
    
    test_tasks = load_test_tasks()
    print(f"[OK] Loaded {len(test_tasks)} test tasks")
    print()
    
    # Evaluate each task
    print("Evaluating on test tasks...")
    print("-" * 90)
    
    results = []
    maml_losses = []
    reptile_losses = []
    baseline_losses = []
    
    maml_model.eval()
    if has_reptile:
        reptile_model.eval()
    
    for i, task in enumerate(test_tasks):
        support_data = {'X': task['X_support'], 'Y': task['Y_support']}
        query_data = {'X': task['X_query'], 'Y': task['Y_query']}
        
        # MAML adaptation
        adapted_maml, _ = adapt_on_support(
            maml_model, support_data, inner_lr=inner_lr, num_steps=inner_steps, device=device
        )
        maml_loss, _, _ = evaluate_on_query(adapted_maml, query_data, device=device)
        
        # Reptile adaptation
        if has_reptile:
            adapted_reptile, _ = adapt_on_support(
                reptile_model, support_data, inner_lr=inner_lr, num_steps=inner_steps, device=device
            )
            reptile_loss, _, _ = evaluate_on_query(adapted_reptile, query_data, device=device)
        else:
            reptile_loss = float('inf')
        
        # Baseline
        baseline_loss = compute_baseline(
            support_data, query_data, device=device, steps_per_task=baseline_steps
        )
        
        maml_losses.append(maml_loss)
        reptile_losses.append(reptile_loss)
        baseline_losses.append(baseline_loss)
        
        results.append({
            'task_id': i,
            'snr': task['snr'],
            'paths': task['num_paths'],
            'maml_loss': maml_loss,
            'reptile_loss': reptile_loss,
            'baseline_loss': baseline_loss,
        })
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{len(test_tasks)} tasks")
    
    print()
    
    # Print results table
    print_results_table(results)
    
    # Summary statistics
    avg_maml = np.mean(maml_losses)
    avg_reptile = np.mean(reptile_losses) if has_reptile else float('inf')
    avg_baseline = np.mean(baseline_losses)
    
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"Average Query Loss (MAML):       {avg_maml:.6f}")
    if has_reptile:
        print(f"Average Query Loss (Reptile):    {avg_reptile:.6f}")
    print(f"Average Query Loss (Baseline):   {avg_baseline:.6f}")
    print()
    
    maml_impr = (avg_baseline - avg_maml) / avg_baseline * 100
    print(f"MAML improvement over baseline:    {maml_impr:>6.1f}%")
    if has_reptile:
        reptile_impr = (avg_baseline - avg_reptile) / avg_baseline * 100
        print(f"Reptile improvement over baseline: {reptile_impr:>6.1f}%")
    
    print()
    
    # Determine winner
    if has_reptile:
        best_method = 'MAML' if avg_maml <= avg_reptile else 'Reptile'
        best_loss = min(avg_maml, avg_reptile)
    else:
        best_method = 'MAML'
        best_loss = avg_maml
    
    if best_loss < avg_baseline:
        print(f"[OK] {best_method} outperforms Baseline ({best_loss:.6f} vs {avg_baseline:.6f})")
    else:
        print(f"[INFO] Baseline outperforms meta-learners — may need more meta-training iterations")
    
    print()
    print("=" * 90)
    print()
    
    # Per-SNR analysis
    snrs = sorted(set([float(r['snr']) for r in results]))
    print("PERFORMANCE BY SNR LEVEL")
    print("-" * 90)
    
    for snr in snrs:
        snr_results = [r for r in results if abs(r['snr'] - snr) < 0.1]
        if snr_results:
            avg_m = np.mean([r['maml_loss'] for r in snr_results])
            avg_r = np.mean([r['reptile_loss'] for r in snr_results]) if has_reptile else 0
            avg_b = np.mean([r['baseline_loss'] for r in snr_results])
            
            line = f"SNR = {snr:>5.1f} dB: MAML={avg_m:.6f}  "
            if has_reptile:
                line += f"Reptile={avg_r:.6f}  "
            line += f"Baseline={avg_b:.6f}"
            print(line)
    
    print()
    
    # Print metric interpretation guide
    print_metric_interpretation()
    
    # Generate plots from real data
    print("Generating training loss curve from actual training history...")
    plot_training_loss_curve(output_path='results/plot_loss.png')
    print()
    
    if has_reptile:
        print("Generating few-shot comparison plot from actual model evaluation...")
        plot_few_shot_comparison(
            maml_model, reptile_model, test_tasks,
            inner_lr=inner_lr, inner_steps=inner_steps,
            baseline_steps=baseline_steps, device=device,
            output_path='results/plot_comparison.png'
        )
        print()
    
    # Compute few-shot table
    if has_reptile:
        compute_few_shot_table(
            maml_model, reptile_model, test_tasks,
            inner_lr=inner_lr, inner_steps=inner_steps,
            baseline_steps=baseline_steps, device=device
        )
    
    print()
    print("=" * 90)
    print("[OK] EVALUATION COMPLETE")
    print("=" * 90)
    print(f"Results saved to: {Path('results').absolute()}")
    print(f"  - results/plot_loss.png (training loss curve)")
    print(f"  - results/plot_comparison.png (MAML vs Reptile vs Baseline comparison)")
    print("=" * 90)
    print()


if __name__ == '__main__':
    main()
