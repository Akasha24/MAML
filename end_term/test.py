"""
Evaluate a trained MAML model on test tasks.

For each test task:
1. Adapt the model using support set (5 gradient steps)
2. Evaluate on query set
3. Compare against baseline (no adaptation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from copy import deepcopy


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

    This matches the baseline used in `train.py` where a fresh model is
    trained on the task's support set for `steps_per_task` steps.

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


def load_trained_model(model_path='results/maml_model.pt', device='cpu'):
    """
    Load trained MAML model.
    
    Args:
        model_path: Path to saved model weights
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


def print_results_table(results):
    """
    Pretty-print evaluation results.
    
    Args:
        results: List of dicts with 'task_id', 'snr', 'paths', 'adapted_loss',
                 'baseline_loss', 'improvement'
    """
    print()
    print("=" * 90)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 90)
    print()
    print(f"{'Task':<6} {'SNR (dB)':<12} {'Paths':<8} {'Adapted':<14} {'Baseline':<14} {'Improvement':<12}")
    print("-" * 90)
    
    for r in results:
        improvement_pct = (r['baseline_loss'] - r['adapted_loss']) / r['baseline_loss'] * 100
        print(
            f"{r['task_id']:<6} "
            f"{r['snr']:<12.1f} "
            f"{r['paths']:<8} "
            f"{r['adapted_loss']:<14.6f} "
            f"{r['baseline_loss']:<14.6f} "
            f"{improvement_pct:>10.1f}%"
        )
    
    print("-" * 90)


def main():
    print("=" * 90)
    print("MAML Test Evaluation")
    print("=" * 90)
    print()
    
    # Settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inner_lr = 0.01
    inner_steps = 5
    baseline_steps = 200
    
    print(f"Device: {device}")
    print(f"Adaptation learning rate: {inner_lr}")
    print(f"Adaptation steps: {inner_steps}")
    print(f"Baseline training steps: {baseline_steps}")
    print()
    
    # Load model and dataset
    print("Loading trained model and test dataset...")
    model = load_trained_model(device=device)
    test_tasks = load_test_tasks()
    print(f"✓ Loaded model and {len(test_tasks)} test tasks")
    print()
    
    # Evaluate each task
    print("Evaluating on test tasks...")
    print("-" * 90)
    
    results = []
    adapted_losses = []
    baseline_losses = []
    
    model.eval()  # Set to eval mode (no gradients for main model)
    
    for i, task in enumerate(test_tasks):
        # Prepare support and query sets
        support_data = {'X': task['X_support'], 'Y': task['Y_support']}
        query_data = {'X': task['X_query'], 'Y': task['Y_query']}
        
        # Step 1: Adapt model on support set
        adapted_model, adapt_loss = adapt_on_support(
            model, support_data, 
            inner_lr=inner_lr, 
            num_steps=inner_steps,
            device=device
        )
        
        # Step 2: Evaluate adapted model on query set
        query_loss, preds, targets = evaluate_on_query(
            adapted_model, query_data, device=device
        )
        
        # Step 3: Compute baseline (train fresh on support set)
        baseline_loss = compute_baseline(
            support_data, query_data,
            device=device,
            steps_per_task=baseline_steps
        )
        
        # Store results
        adapted_losses.append(query_loss)
        baseline_losses.append(baseline_loss)
        
        results.append({
            'task_id': i,
            'snr': task['snr'],
            'paths': task['num_paths'],
            'adapted_loss': query_loss,
            'baseline_loss': baseline_loss,
            'improvement': (baseline_loss - query_loss) / baseline_loss * 100
        })
        
        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{len(test_tasks)} tasks")
    
    print()
    
    # Print results table
    print_results_table(results)
    
    # Summary statistics
    avg_adapted = np.mean(adapted_losses)
    avg_baseline = np.mean(baseline_losses)
    avg_improvement = np.mean([r['improvement'] for r in results])
    
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"Average Query Loss (MAML with adaptation):       {avg_adapted:.6f}")
    print(f"Average Query Loss (Baseline without MAML):      {avg_baseline:.6f}")
    print()
    print(f"Average Improvement:                             {avg_improvement:>6.1f}%")
    print()
    
    # Interpretation
    if avg_improvement > 0:
        print("✓ MAML shows BETTER performance than baseline")
        print("  → Meta-learning initialization helps fast adaptation")
    elif avg_improvement < -50:
        print("✗ MAML shows WORSE performance than baseline")
        print("  → May need hyperparameter tuning or more meta-training iterations")
    else:
        print("○ MAML shows comparable performance to baseline")
        print("  → Results depend on task diversity and training iterations")
    
    print()
    print("=" * 90)
    print()
    
    # Optional: Per-SNR analysis
    snrs = sorted(set([r['snr'] for r in results]))
    print("PERFORMANCE BY SNR LEVEL")
    print("-" * 90)
    
    for snr in snrs:
        snr_results = [r for r in results if abs(r['snr'] - snr) < 0.1]
        if snr_results:
            avg_adapted_snr = np.mean([r['adapted_loss'] for r in snr_results])
            avg_baseline_snr = np.mean([r['baseline_loss'] for r in snr_results])
            improvement_snr = np.mean([r['improvement'] for r in snr_results])
            
            print(f"SNR = {snr:>5.1f} dB: "
                  f"Adapted={avg_adapted_snr:.6f}  "
                  f"Baseline={avg_baseline_snr:.6f}  "
                  f"Improvement={improvement_snr:>6.1f}%")
    
    print()


if __name__ == '__main__':
    main()
