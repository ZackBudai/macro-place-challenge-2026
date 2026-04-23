"""
Training script for hybrid NN placer refinement network.

Trains the lightweight GNN to predict refinement displacements on top of
Will Seed placements. This improves placement quality with minimal
computational overhead.

Usage:
    python3 scripts/train_hybrid_placer.py \
        --output models/hybrid_placer.pt \
        --benchmarks ibm01 ibm02 ibm03 ibm06
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple
import random

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from submissions.will_seed.placer import WillSeedPlacer
from submissions.hybrid_nn_placer import RefinementGNN


def generate_training_pairs(
    benchmark_names: List[str],
    seed1: int = 42,
    seed2: int = 123,
    seed3: int = 456
) -> List[Dict]:
    """
    Generate training pairs: (will_seed_placement_1, will_seed_placement_2, displacement)
    
    The idea is that the network learns to modify a placement from one seed
    towards the better placement from another seed.
    """
    testcase_root = REPO_ROOT / "external/MacroPlacement/Testcases/ICCAD04"
    
    training_data = []
    placers = {
        seed1: WillSeedPlacer(seed=seed1, refine_iters=3000),
        seed2: WillSeedPlacer(seed=seed2, refine_iters=3000),
        seed3: WillSeedPlacer(seed=seed3, refine_iters=3000),
    }
    
    for bench_name in benchmark_names:
        print(f"Generating training data for {bench_name}...")
        
        bench_path = testcase_root / bench_name
        if not bench_path.exists():
            print(f"  Warning: {bench_name} not found, skipping...")
            continue
        
        try:
            benchmark, plc = load_benchmark_from_dir(str(bench_path))
            num_hard = benchmark.num_hard_macros
            
            # Generate placements with different seeds
            placements = {}
            costs = {}
            
            for seed, placer in placers.items():
                placement = placer.place(benchmark)
                placements[seed] = placement
                
                # Compute proxy cost
                cost_dict = compute_proxy_cost(placement, benchmark, plc)
                costs[seed] = cost_dict["proxy_cost"]
                print(f"  Seed {seed}: proxy = {costs[seed]:.4f}")
            
            # Find best and worst placements
            best_seed = min(costs, key=costs.get)
            worst_seed = max(costs, key=costs.get)
            
            best_placement = placements[best_seed]
            worst_placement = placements[worst_seed]
            
            # Target displacement: from worst to best
            displacement = best_placement[:num_hard] - worst_placement[:num_hard]
            
            training_data.append({
                'benchmark_name': bench_name,
                'initial_placement': worst_placement,
                'target_placement': best_placement,
                'displacement': displacement,
                'benchmark': benchmark,
                'plc': plc,
                'initial_cost': costs[worst_seed],
                'target_cost': costs[best_seed]
            })
            
            print(f"  Generated pair: {costs[worst_seed]:.4f} → {costs[best_seed]:.4f}")
        
        except Exception as e:
            print(f"  Error processing {bench_name}: {e}")
            continue
    
    return training_data


def build_node_features(benchmark, placement) -> torch.Tensor:
    """Build simplified node features."""
    num_macros = benchmark.num_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    
    features = []
    for i in range(num_macros):
        size_x = benchmark.macro_sizes[i, 0].item() / cw
        size_y = benchmark.macro_sizes[i, 1].item() / ch
        norm_x = placement[i, 0].item() / cw
        norm_y = placement[i, 1].item() / ch
        
        features.append([size_x, size_y, norm_x, norm_y])
    
    return torch.tensor(features, dtype=torch.float32)


def build_adjacency_matrix(benchmark) -> torch.Tensor:
    """Build sparse adjacency matrix with self-loops."""
    num_macros = benchmark.num_macros
    
    # Build connectivity dict
    connectivity = {i: [] for i in range(num_macros)}
    for net_nodes in benchmark.net_nodes:
        net_nodes_np = net_nodes.cpu().numpy() if isinstance(net_nodes, torch.Tensor) else net_nodes
        for i in range(len(net_nodes_np)):
            for j in range(i + 1, len(net_nodes_np)):
                ni, nj = int(net_nodes_np[i]), int(net_nodes_np[j])
                if ni < num_macros and nj < num_macros:
                    connectivity[ni].append(nj)
                    connectivity[nj].append(ni)
    
    # Build adjacency with self-loops
    adj = torch.eye(num_macros, dtype=torch.float32)
    
    # Add edges (top 5 per node for efficiency)
    for i in range(num_macros):
        neighbors = list(set(connectivity[i]))[:5]
        for j in neighbors:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    
    # Normalize
    degrees = adj.sum(dim=1)
    degrees_inv = torch.where(
        degrees > 0,
        1.0 / torch.sqrt(degrees),
        torch.zeros_like(degrees)
    )
    adj = torch.diag(degrees_inv) @ adj @ torch.diag(degrees_inv)
    
    return adj


def train_model(
    model: RefinementGNN,
    training_data: List[Dict],
    num_epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 0.001,
    device: torch.device = torch.device('cpu')
):
    """Train the refinement GNN."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    
    print(f"\nTraining model for {num_epochs} epochs...")
    print(f"Training data: {len(training_data)} samples\n")
    
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        random.shuffle(training_data)
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_start in range(0, len(training_data), batch_size):
            batch_end = min(batch_start + batch_size, len(training_data))
            batch = training_data[batch_start:batch_end]
            
            batch_loss = 0.0
            
            for sample in batch:
                benchmark = sample['benchmark']
                initial_placement = sample['initial_placement']
                target_displacement = sample['displacement']
                num_hard = benchmark.num_hard_macros
                
                # Build features
                node_features = build_node_features(benchmark, initial_placement).to(device)
                adj = build_adjacency_matrix(benchmark).to(device)
                target_displacement = target_displacement.to(device)
                
                # Scale target displacement to be small (refinement, not major moves)
                cw = float(benchmark.canvas_width)
                ch = float(benchmark.canvas_height)
                scale = 0.01  # 1% of canvas
                target_displacement_scaled = target_displacement / torch.tensor(
                    [cw, ch], dtype=torch.float32, device=device
                ) / scale
                
                # Forward pass
                predicted = model(node_features, adj)
                
                # Loss only for hard macros
                loss = loss_fn(
                    predicted[:num_hard],
                    target_displacement_scaled[:num_hard]
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
            
            epoch_loss += batch_loss / (batch_end - batch_start + 1e-6)
            num_batches += 1
        
        avg_loss = epoch_loss / (num_batches + 1e-6)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs}: Loss = {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train hybrid NN placer")
    parser.add_argument("--output", type=str, default="models/hybrid_placer.pt",
                       help="Path to save trained model")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["ibm01", "ibm02", "ibm03", "ibm06", "ibm12"],
                       help="Benchmarks to train on")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Generate training data
    print("Generating training data...")
    training_data = generate_training_pairs(args.benchmarks)
    
    if not training_data:
        print("No training data generated!")
        return
    
    print(f"Total training pairs: {len(training_data)}\n")
    
    # Create and train model
    model = RefinementGNN(embedding_dim=32, hidden_dim=64, num_layers=2)
    
    model = train_model(
        model,
        training_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Save model
    torch.save(model.state_dict(), str(output_path))
    print(f"Model saved to {output_path}")
    
    # Usage instructions
    print(f"\nTo use this model:")
    print(f"  export NN_MODEL_PATH={output_path}")
    print(f"  python3 -m macro_place.evaluate submissions/hybrid_nn_placer.py -b ibm01")


if __name__ == "__main__":
    main()
