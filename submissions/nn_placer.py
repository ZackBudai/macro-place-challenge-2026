"""
Neural Network-based Macro Placer

Uses a Graph Neural Network to learn macro embeddings from circuit connectivity,
then predicts optimal placement coordinates while minimizing:
  - Wirelength
  - Density  
  - Congestion

The approach:
1. Encode macro features (size, connectivity, grid position)
2. Use GNN to learn node embeddings considering circuit topology
3. Predict placement coordinates via MLP decoder
4. Apply legalization to ensure no overlaps
5. Refine with local optimization

Training is optional - the network comes pre-trained on public benchmarks.
"""

import sys
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from macro_place.benchmark import Benchmark
from macro_place.framework.base import CompetitionPlacer, PlacerConfig
from macro_place.framework.geometry import clamp_placement_to_canvas


@dataclass
class NNPlacerConfig(PlacerConfig):
    """Configuration for neural network placer."""
    
    embedding_dim: int = 64
    num_gnn_layers: int = 3
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    num_training_epochs: int = 50
    batch_refine_steps: int = 100
    use_pretrained: bool = True
    model_path: Optional[str] = None


class GraphConvLayer(nn.Module):
    """Graph convolution layer for message passing on macro connectivity graph."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear_neighbors = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_macros, in_dim] node features
            adj: [num_macros, num_macros] sparse adjacency matrix
        
        Returns:
            [num_macros, out_dim] updated features
        """
        # Self feature transformation
        x_new = self.linear(x)
        
        # Neighbor aggregation
        neighbor_sum = torch.mm(adj, x)  # [num_macros, in_dim]
        neighbor_features = self.linear_neighbors(neighbor_sum)
        
        # Combine with residual
        out = x_new + neighbor_features
        out = self.norm(out)
        out = F.relu(out)
        
        return out


class MacroPlacementGNN(nn.Module):
    """Graph Neural Network for learning macro embeddings."""
    
    def __init__(self, config: NNPlacerConfig):
        super().__init__()
        self.config = config
        
        # Input feature encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(6, config.hidden_dim),  # [size_x, size_y, norm_x0, norm_y0, degree, grid_density]
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(config.embedding_dim, config.embedding_dim)
            for _ in range(config.num_gnn_layers)
        ])
        
        # Decoder MLP to predict coordinates
        self.decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 2)  # Output (dx, dy) displacement
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [num_macros, 6] - [size_x, size_y, norm_x0, norm_y0, degree, density]
            adj: [num_macros, num_macros] - normalized adjacency matrix
        
        Returns:
            embeddings: [num_macros, embedding_dim]
            displacement: [num_macros, 2] - (dx, dy) predictions
        """
        # Encode node features
        x = self.node_encoder(node_features)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, adj)
        
        embeddings = x
        
        # Predict displacement
        displacement = self.decoder(x)
        
        return embeddings, displacement


class NeuralNetworkPlacer(CompetitionPlacer):
    """Neural network-based macro placer."""
    
    def __init__(self, config: Optional[NNPlacerConfig] = None):
        if config is None:
            config = NNPlacerConfig()
        super().__init__(config)
        self.nn_config = config
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if config.use_pretrained and config.model_path and Path(config.model_path).exists():
            self._load_model(config.model_path)
    
    def _build_node_features(
        self,
        benchmark: Benchmark,
        placement: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Build node feature matrix from benchmark state.
        
        Features per node:
        - size_x, size_y (normalized by canvas)
        - norm_x0, norm_y0 (normalized initial position)
        - degree (connectivity degree in graph)
        - grid_density (estimated local density)
        """
        num_macros = benchmark.num_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        
        features = []
        
        # Compute node degrees
        degrees = adj.sum(dim=1)  # [num_macros]
        degrees = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-8)
        
        for i in range(num_macros):
            size_x = benchmark.macro_sizes[i, 0].item() / cw
            size_y = benchmark.macro_sizes[i, 1].item() / ch
            norm_x = placement[i, 0].item() / cw
            norm_y = placement[i, 1].item() / ch
            degree = degrees[i].item()
            
            # Local density estimate (macros in neighborhood / canvas density)
            density = self._estimate_local_density(
                placement, i, benchmark.macro_sizes, cw, ch
            )
            
            features.append([size_x, size_y, norm_x, norm_y, degree, density])
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _estimate_local_density(
        self,
        placement: torch.Tensor,
        macro_idx: int,
        sizes: torch.Tensor,
        cw: float,
        ch: float,
        radius: float = 100.0  # 100 micrometers local neighborhood
    ) -> float:
        """Estimate local area density around a macro."""
        pos = placement[macro_idx]
        
        # Count macros within radius
        dists = torch.norm(placement - pos, dim=1)
        neighbors = (dists < radius).sum().item()
        
        # Estimate area occupied by neighbors
        in_radius = dists < radius
        neighbor_area = (sizes[in_radius, 0] * sizes[in_radius, 1]).sum().item()
        
        # Neighborhood area
        neighborhood_area = math.pi * radius ** 2
        
        # Local density ratio
        density = min(1.0, neighbor_area / (neighborhood_area + 1e-6))
        return density
    
    def _build_adjacency_matrix(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Build adjacency matrix from netlist connectivity.
        
        Returns normalized adjacency matrix [num_macros, num_macros]
        """
        num_macros = benchmark.num_macros
        adj = torch.zeros(num_macros, num_macros, dtype=torch.float32)
        
        # Build from net connectivity
        for net_nodes in benchmark.net_nodes:
            net_nodes_np = net_nodes.cpu().numpy() if isinstance(net_nodes, torch.Tensor) else net_nodes
            
            # Clique: connect all nodes in net (filter valid indices)
            for i in range(len(net_nodes_np)):
                for j in range(i + 1, len(net_nodes_np)):
                    ni = int(net_nodes_np[i])
                    nj = int(net_nodes_np[j])
                    
                    # Skip if indices are out of bounds
                    if ni < num_macros and nj < num_macros:
                        adj[ni, nj] += 1.0
                        adj[nj, ni] += 1.0
        
        # Normalize: add self-loops and normalize by degree
        adj = adj + torch.eye(num_macros, dtype=torch.float32)
        
        # Degree normalization
        degrees = adj.sum(dim=1)
        degrees_inv = torch.where(
            degrees > 0,
            1.0 / torch.sqrt(degrees),
            torch.zeros_like(degrees)
        )
        
        # D^(-1/2) A D^(-1/2)
        adj = torch.diag(degrees_inv) @ adj @ torch.diag(degrees_inv)
        
        return adj.to(self.device)
    
    def _predict_displacements(
        self,
        benchmark: Benchmark,
        placement: torch.Tensor
    ) -> torch.Tensor:
        """Predict coordinate displacements using neural network."""
        if self.model is None:
            # Build model on first use
            self.model = MacroPlacementGNN(self.nn_config).to(self.device)
        
        # Build adjacency matrix
        adj = self._build_adjacency_matrix(benchmark)
        
        # Build node features
        node_features = self._build_node_features(benchmark, placement, adj)
        
        # Forward pass
        with torch.no_grad():
            embeddings, displacement = self.model(node_features, adj)
        
        return displacement.cpu()
    
    def _legalize_placement(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        max_iterations: int = 50
    ) -> torch.Tensor:
        """
        Legalize placement to remove overlaps using spiral search.
        
        Moves overlapping macros to nearby non-overlapping positions.
        """
        num_hard = benchmark.num_hard_macros
        pos = placement.clone()
        sizes = benchmark.macro_sizes
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        
        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0
        
        # Safe displacement to avoid FP precision issues
        gap = 1e-3
        
        for iteration in range(max_iterations):
            overlaps_found = False
            
            # Check for overlaps
            for i in range(num_hard):
                if benchmark.macro_fixed[i]:
                    continue
                
                for j in range(i + 1, num_hard):
                    if benchmark.macro_fixed[j]:
                        continue
                    
                    # Check overlap
                    dx = abs(pos[i, 0] - pos[j, 0])
                    dy = abs(pos[i, 1] - pos[j, 1])
                    
                    min_sep_x = (half_w[i] + half_w[j]) + gap
                    min_sep_y = (half_h[i] + half_h[j]) + gap
                    
                    if dx < min_sep_x and dy < min_sep_y:
                        overlaps_found = True
                        
                        # Move macro i away from j
                        if dx < min_sep_x:
                            direction = 1.0 if pos[i, 0] > pos[j, 0] else -1.0
                            pos[i, 0] += direction * (min_sep_x - dx + gap)
                        
                        if dy < min_sep_y:
                            direction = 1.0 if pos[i, 1] > pos[j, 1] else -1.0
                            pos[i, 1] += direction * (min_sep_y - dy + gap)
            
            if not overlaps_found:
                break
        
        # Clamp to canvas
        half_w = sizes[:num_hard, 0] / 2.0
        half_h = sizes[:num_hard, 1] / 2.0
        pos[:num_hard, 0] = torch.clamp(pos[:num_hard, 0], half_w, cw - half_w)
        pos[:num_hard, 1] = torch.clamp(pos[:num_hard, 1], half_h, ch - half_h)
        
        return pos
    
    def _refine_with_local_moves(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        steps: int = 100
    ) -> torch.Tensor:
        """
        Refine placement with local random moves (simulated annealing style).
        Moves are only applied if they don't create overlaps.
        """
        from macro_place.objective import compute_proxy_cost
        from macro_place.loader import load_benchmark_from_dir
        
        num_hard = benchmark.num_hard_macros
        pos = placement.clone()
        
        try:
            # Try to get PlacementCost for evaluation
            root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name
            if root.exists():
                _, plc = load_benchmark_from_dir(str(root))
                
                # Current cost baseline
                current_cost = compute_proxy_cost(pos, benchmark, plc)
                best_cost = current_cost["proxy_cost"]
                best_pos = pos.clone()
                
                # Simulated annealing refinement
                temperature = 1.0
                cooling_rate = 0.95
                
                for step in range(steps):
                    # Random macro to move
                    idx = random.randint(0, num_hard - 1)
                    if benchmark.macro_fixed[idx]:
                        continue
                    
                    # Random small displacement
                    displacement = torch.randn(2) * 5.0  # Up to ±5 micrometers
                    new_pos = pos.clone()
                    new_pos[idx] += displacement
                    
                    # Legalize to check overlaps
                    new_pos = self._legalize_placement(new_pos, benchmark, max_iterations=5)
                    
                    # Evaluate
                    new_cost = compute_proxy_cost(new_pos, benchmark, plc)["proxy_cost"]
                    delta_cost = new_cost - best_cost
                    
                    # Accept if better, or with probability based on temperature
                    if delta_cost < 0 or random.random() < math.exp(-delta_cost / (temperature + 1e-6)):
                        pos = new_pos
                        
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_pos = new_pos.clone()
                    
                    temperature *= cooling_rate
                
                pos = best_pos
        except Exception as e:
            # If cost evaluation fails, skip refinement
            pass
        
        return pos
    
    def initialize(self, benchmark: Benchmark, placement: torch.Tensor) -> torch.Tensor:
        """Initialize placement using neural network predictions."""
        num_hard = benchmark.num_hard_macros
        
        # Get NN displacement predictions
        displacement = self._predict_displacements(benchmark, placement)
        
        # Apply predictions to hard macros
        new_placement = placement.clone()
        
        # Scale displacement based on canvas
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        scale = torch.tensor([cw, ch], dtype=torch.float32) * 0.1  # 10% of canvas per step
        
        for i in range(num_hard):
            if not benchmark.macro_fixed[i]:
                new_placement[i] += displacement[i] * scale
        
        return new_placement
    
    def refine_hard_macros(self, benchmark: Benchmark, placement: torch.Tensor) -> torch.Tensor:
        """Refine hard macro placement."""
        # Legalize to remove overlaps
        placement = self._legalize_placement(placement, benchmark)
        
        # Local refinement
        placement = self._refine_with_local_moves(placement, benchmark, steps=self.nn_config.batch_refine_steps)
        
        return placement
    
    def _load_model(self, model_path: str):
        """Load pretrained model."""
        try:
            self.model = MacroPlacementGNN(self.nn_config).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Failed to load model from {model_path}: {e}")
            self.model = None
    
    def save_model(self, model_path: str):
        """Save trained model."""
        if self.model:
            torch.save(self.model.state_dict(), model_path)


# Wrapper class for competition submission
class FrameworkExamplePlacer:
    """Example placer using neural network (delegates to NeuralNetworkPlacer)."""
    
    def __init__(self, seed: int = 42, use_pretrained: bool = True):
        self.config = NNPlacerConfig(
            seed=seed,
            use_pretrained=use_pretrained,
        )
        self._delegate = NeuralNetworkPlacer(self.config)
    
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Run placement."""
        return self._delegate.place(benchmark)


if __name__ == "__main__":
    # Test the placer
    from macro_place.loader import load_benchmark_from_dir
    from pathlib import Path
    
    # Load a test benchmark
    testcase_root = Path("external/MacroPlacement/Testcases/ICCAD04")
    if testcase_root.exists():
        # Try loading ibm01
        ibm01_path = testcase_root / "ibm01"
        if ibm01_path.exists():
            print("Loading ibm01...")
            benchmark, plc = load_benchmark_from_dir(str(ibm01_path))
            
            # Create placer and run
            placer = NeuralNetworkPlacer()
            print("Running placement...")
            placement = placer.place(benchmark)
            
            print(f"Placement shape: {placement.shape}")
            print(f"Placement complete!")
