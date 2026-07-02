"""
Meta-learning training for wireless channel estimation using MAML and Reptile.
Compares MAML, Reptile (first-order MAML variant), and Baseline (train from scratch).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from collections import defaultdict


class ChannelEstimationNetwork(nn.Module):
    """
    Simple neural network for channel estimation.
    
    Architecture:
    Input(input_dim) -> Hidden(64) -> Hidden(32) -> Output(output_dim)
    
    This network learns to estimate channel coefficients from pilot signals.
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


class MAMLTrainer:
    """
    Model-Agnostic Meta-Learning (MAML) trainer for channel estimation.
    
    MAML Algorithm (plain English):
    1. Start with shared weights θ
    2. For each task in batch:
       a) Take few gradient steps on support set -> get adapted weights θ'
       b) Compute loss on query set using θ'
    3. Update θ based on query losses (meta-gradient)
    4. Repeat
    
    This creates an initialization that adapts quickly to new tasks.
    """
    
    def __init__(self, model, device='cpu', inner_lr=0.01, outer_lr=0.001, 
                 inner_steps=5):
        """
        Initialize MAML trainer.
        
        Args:
            model: Neural network to train
            device: 'cpu' or 'cuda'
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-update)
            inner_steps: Number of gradient steps per task
                   """
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=outer_lr)
        self.loss_fn = nn.MSELoss()
    
    def inner_loop(self, x_support, y_support):
        """
        Adapt to a single task (inner loop).
        
        Takes few gradient steps on support set and returns adapted model.
        
        Args:
            x_support: Support inputs (n_support, input_dim)
            y_support: Support labels (n_support, output_dim)
            
        Returns:
            adapted_model: Network with updated weights
            support_loss: Loss on support set after adaptation
        """
        adapted_model = self.model.clone()
        adapted_optim = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Take inner_steps gradient steps on support set
        for _ in range(self.inner_steps):
            adapted_optim.zero_grad()
            support_pred = adapted_model(x_support)
            support_loss = self.loss_fn(support_pred, y_support)
            support_loss.backward()
            adapted_optim.step()
        
        # Clear any gradients left on the adapted model before query evaluation
        adapted_optim.zero_grad()
        
        # Final support loss
        with torch.no_grad():
            support_pred = adapted_model(x_support)
            final_support_loss = self.loss_fn(support_pred, y_support).item()
        
        return adapted_model, final_support_loss
    
    def outer_loop(self, tasks_batch):
        """
        Meta-update step (outer loop).
        
        For each task:
        1. Adapt on support set
        2. Compute loss on query set using adapted model
        3. Accumulate query gradient
        
        Then update meta-weights.
        
        Args:
            tasks_batch: List of task dicts with X_support, Y_support, X_query, Y_query
            
        Returns:
            dict with metrics (avg query loss, etc.)
        """
        self.meta_optimizer.zero_grad()
        
        total_query_loss = 0.0
        total_support_loss = 0.0
        num_tasks = len(tasks_batch)
        
        # Process each task
        for task in tasks_batch:
            # Prepare data
            x_support = torch.from_numpy(task['X_support']).float().to(self.device)
            y_support = torch.from_numpy(task['Y_support']).float().to(self.device)
            x_query = torch.from_numpy(task['X_query']).float().to(self.device)
            y_query = torch.from_numpy(task['Y_query']).float().to(self.device)
            
            # Inner loop: adapt to this task
            adapted_model, support_loss = self.inner_loop(x_support, y_support)
            total_support_loss += support_loss
            
            # Compute loss on query set with adapted model
            query_pred = adapted_model(x_query)
            query_loss = self.loss_fn(query_pred, y_query)
            
            # First-order MAML: compute query gradients on adapted parameters,
            # then transfer them to the base model without second-order terms.
            query_loss.backward()
            for base_param, adapted_param in zip(self.model.parameters(), adapted_model.parameters()):
                if adapted_param.grad is None:
                    continue
                if base_param.grad is None:
                    base_param.grad = adapted_param.grad.detach().clone()
                else:
                    base_param.grad += adapted_param.grad.detach()
            total_query_loss += query_loss.item()
        
        # Meta-update: update base model weights using accumulated task gradients
        self.meta_optimizer.step()
        
        return {
            'avg_query_loss': total_query_loss / num_tasks,
            'avg_support_loss': total_support_loss / num_tasks,
            'num_tasks': num_tasks
        }
    
    def evaluate(self, tasks_batch):
        """
        Evaluate model on a batch of tasks.
        
        Args:
            tasks_batch: List of task dicts
            
        Returns:
            dict with evaluation metrics
        """
        was_training = self.model.training
        self.model.eval()
        
        total_query_loss = 0.0
        total_support_loss = 0.0
        num_tasks = len(tasks_batch)
        
        for task in tasks_batch:
            x_support = torch.from_numpy(task['X_support']).float().to(self.device)
            y_support = torch.from_numpy(task['Y_support']).float().to(self.device)
            x_query = torch.from_numpy(task['X_query']).float().to(self.device)
            y_query = torch.from_numpy(task['Y_query']).float().to(self.device)
            
            # Adapt on support (need gradients for inner loop)
            adapted_model, support_loss = self.inner_loop(x_support, y_support)
            total_support_loss += support_loss
            
            # Evaluate on query
            with torch.no_grad():
                query_pred = adapted_model(x_query)
                query_loss = self.loss_fn(query_pred, y_query).item()
            total_query_loss += query_loss
        
        self.model.train(was_training)
        
        return {
            'avg_query_loss': total_query_loss / num_tasks,
            'avg_support_loss': total_support_loss / num_tasks,
            'num_tasks': num_tasks
        }


class ReptileTrainer:
    """
    Reptile meta-learning trainer for channel estimation.
    
    Reptile Algorithm (Nichol et al., 2018):
    1. Start with shared weights θ
    2. For each task in batch:
       a) Clone model -> take k SGD steps on support set -> get adapted weights θ'
       b) Compute direction: Δ = θ' - θ
    3. Update θ by moving toward the average adapted weights:
       θ ← θ + ε * mean(Δ)
    
    Key differences from MAML:
    - No query-set gradient computation during training
    - No second-order gradients needed
    - Simpler, more memory-efficient
    - Often more stable in practice
    """
    
    def __init__(self, model, device='cpu', inner_lr=0.01, reptile_lr=0.001,
                 inner_steps=5):
        """
        Initialize Reptile trainer.
        
        Args:
            model: Neural network to train
            device: 'cpu' or 'cuda'
            inner_lr: Learning rate for inner loop (task adaptation via SGD)
            reptile_lr: Outer step size (Adam learning rate)
            inner_steps: Number of gradient steps per task in inner loop
        """
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.reptile_lr = reptile_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=reptile_lr)
        self.loss_fn = nn.MSELoss()
    
    def inner_loop(self, x_support, y_support):
        """
        Adapt to a single task (inner loop).
        
        Takes several SGD steps on the support set.
        
        Args:
            x_support: Support inputs
            y_support: Support labels
            
        Returns:
            adapted_model: Model with task-adapted weights
            final_support_loss: Loss after adaptation
        """
        adapted_model = self.model.clone()
        adapted_optim = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.inner_steps):
            adapted_optim.zero_grad()
            pred = adapted_model(x_support)
            loss = self.loss_fn(pred, y_support)
            loss.backward()
            adapted_optim.step()
        
        # Final support loss
        with torch.no_grad():
            pred = adapted_model(x_support)
            final_loss = self.loss_fn(pred, y_support).item()
        
        return adapted_model, final_loss
    
    def outer_loop(self, tasks_batch):
        """
        Reptile meta-update step (outer loop).
        
        For each task:
        1. Clone model and adapt on support set for k steps
        2. Compute weight difference (θ' - θ)
        
        Then move base weights toward average adapted weights using Adam optimizer.
        
        Args:
            tasks_batch: List of task dicts
            
        Returns:
            dict with metrics
        """
        total_query_loss = 0.0
        total_support_loss = 0.0
        num_tasks = len(tasks_batch)
        
        # Accumulate weight differences across tasks
        weight_diffs = None
        
        for task in tasks_batch:
            x_support = torch.from_numpy(task['X_support']).float().to(self.device)
            y_support = torch.from_numpy(task['Y_support']).float().to(self.device)
            x_query = torch.from_numpy(task['X_query']).float().to(self.device)
            y_query = torch.from_numpy(task['Y_query']).float().to(self.device)
            
            # Inner loop: adapt to this task
            adapted_model, support_loss = self.inner_loop(x_support, y_support)
            total_support_loss += support_loss
            
            # Evaluate on query for monitoring (not used for gradient)
            with torch.no_grad():
                query_pred = adapted_model(x_query)
                query_loss = self.loss_fn(query_pred, y_query).item()
            total_query_loss += query_loss
            
            # Compute weight difference: θ' - θ
            if weight_diffs is None:
                weight_diffs = [
                    (adapted_p.data - base_p.data).clone()
                    for base_p, adapted_p in zip(self.model.parameters(), adapted_model.parameters())
                ]
            else:
                for i, (base_p, adapted_p) in enumerate(
                    zip(self.model.parameters(), adapted_model.parameters())
                ):
                    weight_diffs[i] += (adapted_p.data - base_p.data)
        
        # Adam-accelerated Reptile update: θ ← θ - Adam(-mean(θ' - θ))
        self.meta_optimizer.zero_grad()
        for param, diff in zip(self.model.parameters(), weight_diffs):
            # Gradient is the negative of the weight differences (moving toward them)
            param.grad = -diff / num_tasks
        self.meta_optimizer.step()
        
        return {
            'avg_query_loss': total_query_loss / num_tasks,
            'avg_support_loss': total_support_loss / num_tasks,
            'num_tasks': num_tasks
        }
    
    def evaluate(self, tasks_batch):
        """
        Evaluate Reptile model on a batch of tasks.
        
        Uses the same inner-loop adaptation as training, then evaluates on query set.
        
        Args:
            tasks_batch: List of task dicts
            
        Returns:
            dict with evaluation metrics
        """
        was_training = self.model.training
        self.model.eval()
        
        total_query_loss = 0.0
        total_support_loss = 0.0
        num_tasks = len(tasks_batch)
        
        for task in tasks_batch:
            x_support = torch.from_numpy(task['X_support']).float().to(self.device)
            y_support = torch.from_numpy(task['Y_support']).float().to(self.device)
            x_query = torch.from_numpy(task['X_query']).float().to(self.device)
            y_query = torch.from_numpy(task['Y_query']).float().to(self.device)
            
            # Adapt on support
            adapted_model, support_loss = self.inner_loop(x_support, y_support)
            total_support_loss += support_loss
            
            # Evaluate on query
            with torch.no_grad():
                query_pred = adapted_model(x_query)
                query_loss = self.loss_fn(query_pred, y_query).item()
            total_query_loss += query_loss
        
        self.model.train(was_training)
        
        return {
            'avg_query_loss': total_query_loss / num_tasks,
            'avg_support_loss': total_support_loss / num_tasks,
            'num_tasks': num_tasks
        }


class BaselineTrainer:
    """
    Baseline: Train a fresh model from scratch on each task's support set.
    
    For comparison:
    - Train on task's support set for 200 steps
    - Evaluate on task's query set
    - Do this independently for each task
    
    Good baseline captures "how well can a model learn from scratch in limited data"
    """
    
    def __init__(self, input_dim=4, output_dim=1, device='cpu', 
                 steps_per_task=200):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.steps_per_task = steps_per_task
        self.loss_fn = nn.MSELoss()
    
    def train_on_task(self, x_support, y_support, x_query, y_query):
        """
        Train a single model from scratch and evaluate.
        
        Args:
            x_support, y_support: Support set
            x_query, y_query: Query set
            
        Returns:
            dict with support_loss, query_loss
        """
        # Create fresh model
        model = ChannelEstimationNetwork(self.input_dim, self.output_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        x_support_t = torch.from_numpy(x_support).float().to(self.device)
        y_support_t = torch.from_numpy(y_support).float().to(self.device)
        x_query_t = torch.from_numpy(x_query).float().to(self.device)
        y_query_t = torch.from_numpy(y_query).float().to(self.device)
        
        # Train on support set
        for _ in range(self.steps_per_task):
            optimizer.zero_grad()
            pred = model(x_support_t)
            loss = self.loss_fn(pred, y_support_t)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            support_pred = model(x_support_t)
            support_loss = self.loss_fn(support_pred, y_support_t).item()
            
            query_pred = model(x_query_t)
            query_loss = self.loss_fn(query_pred, y_query_t).item()
        
        return {
            'support_loss': support_loss,
            'query_loss': query_loss
        }
    
    def evaluate(self, tasks_batch):
        """
        Evaluate baseline on a batch of tasks.
        
        Args:
            tasks_batch: List of task dicts
            
        Returns:
            dict with average metrics
        """
        total_support_loss = 0.0
        total_query_loss = 0.0
        
        for task in tasks_batch:
            result = self.train_on_task(
                task['X_support'], task['Y_support'],
                task['X_query'], task['Y_query']
            )
            total_support_loss += result['support_loss']
            total_query_loss += result['query_loss']
        
        return {
            'avg_support_loss': total_support_loss / len(tasks_batch),
            'avg_query_loss': total_query_loss / len(tasks_batch),
            'num_tasks': len(tasks_batch)
        }


def load_dataset(data_path='results'):
    """
    Load training and test tasks from NPZ files.
    
    Args:
        data_path: Directory containing train_tasks.npz and test_tasks.npz
        
    Returns:
        (train_tasks, test_tasks): Lists of task dictionaries
    """
    data_dir = Path(data_path)
    
    train_data = np.load(data_dir / 'train_tasks.npz')
    test_data = np.load(data_dir / 'test_tasks.npz')
    
    # Convert stacked arrays back to list of tasks
    def stack_to_tasks(data):
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
    
    train_tasks = stack_to_tasks(train_data)
    test_tasks = stack_to_tasks(test_data)
    
    return train_tasks, test_tasks


def plot_results(history, output_path='results/plot_loss.png'):
    """
    Plot training curves comparing MAML, Reptile, and Baseline.
    
    Args:
        history: Dict with 'maml_query', 'maml_support', 'reptile_query',
                 'reptile_support', 'baseline_query', 'baseline_support'
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Query loss
    axes[0].plot(history['maml_query'], label='MAML Query', linewidth=2, color='#2196F3')
    axes[0].plot(history['reptile_query'], label='Reptile Query', linewidth=2, color='#4CAF50')
    axes[0].plot(history['baseline_query'], label='Baseline Query', linewidth=2, color='#F44336', linestyle='--')
    axes[0].set_xlabel('Iteration', fontsize=11)
    axes[0].set_ylabel('Query Loss (MSE)', fontsize=11)
    axes[0].set_title('Query Loss: MAML vs Reptile vs Baseline', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Support loss
    axes[1].plot(history['maml_support'], label='MAML Support', linewidth=2, color='#2196F3')
    axes[1].plot(history['reptile_support'], label='Reptile Support', linewidth=2, color='#4CAF50')
    axes[1].plot(history['baseline_support'], label='Baseline Support', linewidth=2, color='#F44336', linestyle='--')
    axes[1].set_xlabel('Iteration', fontsize=11)
    axes[1].set_ylabel('Support Loss (MSE)', fontsize=11)
    axes[1].set_title('Support Loss: MAML vs Reptile vs Baseline', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[OK] Plot saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("MAML + Reptile Training for Wireless Channel Estimation")
    print("=" * 70)
    print()
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_iterations = 1000
    batch_size = 4  # Tasks per batch
    inner_lr = 0.01
    inner_steps = 10
    outer_lr = 0.001
    
    print(f"Device: {device}")
    print(f"Meta-learning iterations: {num_iterations}")
    print(f"Batch size: {batch_size} tasks")
    print(f"Inner LR: {inner_lr}, Inner steps: {inner_steps}")
    print(f"Outer LR (MAML): {outer_lr}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    train_tasks, test_tasks = load_dataset('results')
    print(f"  Training tasks: {len(train_tasks)}")
    print(f"  Test tasks: {len(test_tasks)}")
    print()
    
    # Initialize models and trainers
    print("Initializing models...")
    
    # MAML model
    maml_model = ChannelEstimationNetwork(input_dim=4, output_dim=1)
    maml_trainer = MAMLTrainer(
        model=maml_model,
        device=device,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps
    )
    
    # Reptile model (separate initialization)
    reptile_model = ChannelEstimationNetwork(input_dim=4, output_dim=1)
    reptile_trainer = ReptileTrainer(
        model=reptile_model,
        device=device,
        inner_lr=inner_lr,
        reptile_lr=0.003,
        inner_steps=inner_steps
    )
    
    # Baseline trainer
    baseline_trainer = BaselineTrainer(device=device, steps_per_task=50)
    print()
    
    # Training loop
    print("Training...")
    print("-" * 70)
    
    history = defaultdict(list)
    
    for iteration in range(num_iterations):
        # Sample batch of training tasks (same batch for fair comparison)
        batch_indices = np.random.choice(len(train_tasks), size=batch_size, replace=False)
        train_batch = [train_tasks[i] for i in batch_indices]
        
        # MAML meta-update
        maml_metrics = maml_trainer.outer_loop(train_batch)
        history['maml_query'].append(maml_metrics['avg_query_loss'])
        history['maml_support'].append(maml_metrics['avg_support_loss'])
        
        # Reptile meta-update
        reptile_metrics = reptile_trainer.outer_loop(train_batch)
        history['reptile_query'].append(reptile_metrics['avg_query_loss'])
        history['reptile_support'].append(reptile_metrics['avg_support_loss'])
        
        # Baseline evaluation (on same batch for fair comparison)
        baseline_metrics = baseline_trainer.evaluate(train_batch)
        history['baseline_query'].append(baseline_metrics['avg_query_loss'])
        history['baseline_support'].append(baseline_metrics['avg_support_loss'])
        
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"  MAML     - Query Loss: {maml_metrics['avg_query_loss']:.6f}, "
                  f"Support Loss: {maml_metrics['avg_support_loss']:.6f}")
            print(f"  Reptile  - Query Loss: {reptile_metrics['avg_query_loss']:.6f}, "
                  f"Support Loss: {reptile_metrics['avg_support_loss']:.6f}")
            print(f"  Baseline - Query Loss: {baseline_metrics['avg_query_loss']:.6f}, "
                  f"Support Loss: {baseline_metrics['avg_support_loss']:.6f}")
            print()
    
    print("-" * 70)
    print()
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    maml_test = maml_trainer.evaluate(test_tasks)
    reptile_test = reptile_trainer.evaluate(test_tasks)
    baseline_test = baseline_trainer.evaluate(test_tasks)
    
    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"MAML:")
    print(f"  Query Loss:   {maml_test['avg_query_loss']:.6f}")
    print(f"  Support Loss: {maml_test['avg_support_loss']:.6f}")
    print()
    print(f"Reptile:")
    print(f"  Query Loss:   {reptile_test['avg_query_loss']:.6f}")
    print(f"  Support Loss: {reptile_test['avg_support_loss']:.6f}")
    print()
    print(f"Baseline (train from scratch):")
    print(f"  Query Loss:   {baseline_test['avg_query_loss']:.6f}")
    print(f"  Support Loss: {baseline_test['avg_support_loss']:.6f}")
    print()
    
    maml_improvement = (baseline_test['avg_query_loss'] - maml_test['avg_query_loss']) / baseline_test['avg_query_loss'] * 100
    reptile_improvement = (baseline_test['avg_query_loss'] - reptile_test['avg_query_loss']) / baseline_test['avg_query_loss'] * 100
    print(f"MAML improvement over baseline:    {maml_improvement:.1f}%")
    print(f"Reptile improvement over baseline:  {reptile_improvement:.1f}%")
    print("=" * 70)
    print()
    
    # Plot results
    plot_results(history)
    
    # Save training history for test.py to use real data in plots
    history_path = Path('results') / 'training_history.npz'
    np.savez_compressed(
        history_path,
        maml_query=np.array(history['maml_query']),
        maml_support=np.array(history['maml_support']),
        reptile_query=np.array(history['reptile_query']),
        reptile_support=np.array(history['reptile_support']),
        baseline_query=np.array(history['baseline_query']),
        baseline_support=np.array(history['baseline_support']),
    )
    print(f"[OK] Training history saved: {history_path}")
    
    # Save models
    torch.save(maml_trainer.model.state_dict(), 'results/maml_model.pt')
    print("[OK] MAML model saved: results/maml_model.pt")
    
    torch.save(reptile_trainer.model.state_dict(), 'results/reptile_model.pt')
    print("[OK] Reptile model saved: results/reptile_model.pt")
    print()


if __name__ == '__main__':
    main()
