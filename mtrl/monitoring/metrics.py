import chex
import jax.numpy as jnp
from jaxtyping import Array, Float

from functools import partial

from mtrl.types import Intermediates, LayerActivationsDict, LogDict
import jax
import jax.numpy as jnp
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from umap import UMAP


@partial(jax.jit, static_argnames=['grid_size'])
def compute_grid_coverage_jax(projection_2d: jnp.ndarray, grid_size: int = 50) -> float:
    """
    JAX-compiled function to compute coverage from 2D projections.
    
    Args:
        projection_2d: Array of shape (N, 2) with values in [0, 1)
        grid_size: Size of the grid (G in the paper)
    
    Returns:
        coverage: Fraction of grid cells occupied
    """
    # Compute grid indices
    grid_indices = jnp.minimum(
        jnp.floor(projection_2d * grid_size).astype(jnp.int32),
        grid_size - 1
    )
    
    # Flatten 2D indices to 1D
    flat_indices = grid_indices[:, 0] * grid_size + grid_indices[:, 1]
    
    # Count unique cells using histogram
    occupied = jnp.bincount(flat_indices, length=grid_size**2) > 0
    num_occupied = jnp.sum(occupied)
    
    coverage = num_occupied / (grid_size ** 2)
    return coverage


def compute_grid_coverage_numpy(projection_2d: np.ndarray, grid_size: int = 50) -> float:
    """
    NumPy version of grid coverage computation.
    
    Args:
        projection_2d: Array of shape (N, 2) with values in [0, 1)
        grid_size: Size of the grid
    
    Returns:
        coverage: Fraction of grid cells occupied
    """
    grid_indices = np.floor(projection_2d * grid_size).astype(int)
    grid_indices = np.minimum(grid_indices, grid_size - 1)
    
    # Get unique cells
    unique_cells = np.unique(grid_indices, axis=0)
    coverage = len(unique_cells) / (grid_size ** 2)
    
    return coverage


# ============================================================================
# LSH-based Coverage Tracker (Online, Fast)
# ============================================================================

class LSHVisitationCounter:
    """
    Locality-Sensitive Hashing for fast online state coverage estimation.
    
    Uses random projections to hash high-dimensional continuous states into
    discrete bins, allowing efficient tracking of visited states.
    """
    
    def __init__(self, state_dim: int, n_hash_functions: int = 15, bin_width: float = 0.05):
        """
        Args:
            state_dim: Dimensionality of state space (39 for Meta-World)
            n_hash_functions: Number of hash functions (more = finer discrimination)
            bin_width: Width of discretization bins (smaller = finer granularity)
        """
        self.state_dim = state_dim
        self.n_hash_functions = n_hash_functions
        self.bin_width = bin_width
        
        # Random projection vectors for hashing (fixed after initialization)
        self.projection_vectors = [
            np.random.randn(state_dim) for _ in range(n_hash_functions)
        ]
        
        # Store visitation counts for each hash
        self.visitation_counts = defaultdict(int)
        
    def hash_state(self, state: np.ndarray) -> tuple:
        """
        Convert continuous state to discrete hash key.
        
        Args:
            state: State vector of shape (state_dim,)
        
        Returns:
            hash_key: Tuple of hash values
        """
        hash_values = []
        for proj_vec in self.projection_vectors:
            # Project state onto random direction
            projection = np.dot(state, proj_vec)
            # Discretize projection
            bin_id = int(projection / self.bin_width)
            hash_values.append(bin_id)
        
        return tuple(hash_values)
    
    def record_visit(self, state: np.ndarray) -> None:
        """
        Record a visit to a state.
        
        Args:
            state: State vector of shape (state_dim,)
        """
        hash_key = self.hash_state(state)
        self.visitation_counts[hash_key] += 1
    
    def record_visits_batch(self, states: np.ndarray) -> None:
        """
        Record visits to multiple states (more efficient).
        
        Args:
            states: State vectors of shape (N, state_dim)
        """
        for state in states:
            self.record_visit(state)
    
    def get_visit_count(self, state: np.ndarray) -> int:
        """
        Get number of times a state (or similar states) has been visited.
        
        Args:
            state: State vector of shape (state_dim,)
        
        Returns:
            count: Number of visits
        """
        hash_key = self.hash_state(state)
        return self.visitation_counts[hash_key]
    
    def get_novelty_score(self, state: np.ndarray) -> float:
        """
        Get novelty score for a state (higher = more novel).
        Can be used as exploration bonus.
        
        Args:
            state: State vector of shape (state_dim,)
        
        Returns:
            novelty: Score in (0, 1], where 1 is completely novel
        """
        count = self.get_visit_count(state)
        return 1.0 / (1.0 + count)
    
    def get_num_unique_states(self) -> int:
        """Get number of unique state bins visited."""
        return len(self.visitation_counts)
    
    def reset(self) -> None:
        """Clear all visitation counts."""
        self.visitation_counts.clear()


# ============================================================================
# UMAP-based Coverage Tracker (Offline, Visualization)
# ============================================================================

class UMAPCoverageTracker:
    """
    UMAP-based coverage tracker for visualization and global analysis.
    
    Projects high-dimensional states to 2D using UMAP, then computes
    coverage as the fraction of a grid that contains at least one state.
    """
    
    def __init__(self, grid_size: int = 50, use_jax: bool = True):
        """
        Args:
            grid_size: Size of the 2D grid (G x G)
            use_jax: Whether to use JAX for coverage computation (faster)
        """
        self.grid_size = grid_size
        self.use_jax = use_jax
        
        # Storage
        self.states = []
        self.labels = []
        
        # UMAP model
        self.umap_model = None
        self.projection = None
        
        # Track when data has been added since last fit
        self._data_changed = False
        self._num_states_at_last_fit = 0
    
    def add_data(self, states: np.ndarray, labels: Optional[List[str]] = None) -> None:
        """
        Add states to the tracker.
        
        Args:
            states: Array of shape (N, state_dim)
            labels: Optional list of task names for each state (or single task name)
        """
        states = np.array(states)
        self.states.append(states)
        
        if labels is not None:
            # Handle both list of labels and single label string
            if isinstance(labels, str):
                labels = [labels] * len(states)
            elif isinstance(labels, list) and len(labels) == 1 and len(states) > 1:
                labels = labels * len(states)
            
            # Ensure labels match number of states
            if len(labels) != len(states):
                raise ValueError(f"Number of labels ({len(labels)}) must match number of states ({len(states)})")
            
            self.labels.extend(labels)
        else:
            # No labels provided
            self.labels.extend([None] * len(states))
        
        # Mark that data has changed
        self._data_changed = True
    
    def _get_all_states(self) -> np.ndarray:
        """Get concatenated array of all states."""
        if not self.states:
            raise ValueError("No states added yet!")
        return np.concatenate(self.states, axis=0)
    
    def fit_umap(self, n_neighbors: int = 15, min_dist: float = 0.1, 
                 random_state: int = 42, force: bool = False) -> np.ndarray:
        """
        Fit UMAP model on collected states.
        
        Args:
            n_neighbors: UMAP n_neighbors parameter (local vs global structure)
            min_dist: UMAP min_dist parameter (tightness of clustering)
            random_state: Random seed for reproducibility
            force: Force refitting even if data hasn't changed
        
        Returns:
            projection: 2D projection of shape (N, 2)
        """
        all_states = self._get_all_states()
        
        # Check if we need to refit
        if not force and not self._data_changed and self.projection is not None:
            if len(all_states) == self._num_states_at_last_fit:
                print("UMAP already fitted and data unchanged. Skipping refit.")
                return self.projection
        
        # Verify labels match
        if len(self.labels) != len(all_states):
            raise ValueError(
                f"Label/state mismatch! States: {len(all_states)}, Labels: {len(self.labels)}. "
                f"This is a bug - please report it."
            )
        
        print(f"Fitting UMAP on {len(all_states)} states of dimension {all_states.shape[1]}...")
        
        # Fit UMAP
        n_neighbors_actual = min(n_neighbors, len(all_states) - 1)
        self.umap_model = UMAP(
            n_components=2,
            n_neighbors=n_neighbors_actual,
            min_dist=min_dist,
            #random_state=random_state,
            verbose=False
        )
        
        self.projection = self.umap_model.fit_transform(all_states)
        
        # Normalize to [0, 1)
        self.projection = self._normalize_projection(self.projection)
        
        # Update tracking
        self._data_changed = False
        self._num_states_at_last_fit = len(all_states)
        
        print(f"UMAP fitted. Projection shape: {self.projection.shape}")
        
        return self.projection
    
    def transform_states(self, states: np.ndarray) -> np.ndarray:
        """
        Transform new states using fitted UMAP model.
        
        Args:
            states: Array of shape (N, state_dim)
        
        Returns:
            projection: 2D projection of shape (N, 2)
        """
        if self.umap_model is None:
            raise ValueError("Must fit UMAP first using fit_umap()")
        
        projection = self.umap_model.transform(states)
        projection = self._normalize_projection(projection)
        
        return projection
    
    def _normalize_projection(self, projection: np.ndarray) -> np.ndarray:
        """Normalize projection to [0, 1)."""
        projection = (projection - projection.min(axis=0)) / \
                     (projection.max(axis=0) - projection.min(axis=0) + 1e-8)
        return np.clip(projection, 0, 0.9999)
    
    def compute_coverage(self) -> float:
        """
        Compute coverage metric.
        Will automatically refit UMAP if new data has been added.
        
        Returns:
            coverage: Fraction of grid cells occupied, in [0, 1]
        """
        # Always refit if data has changed
        if self._data_changed or self.projection is None:
            self.fit_umap()
        
        if self.use_jax:
            coverage = float(compute_grid_coverage_jax(
                jnp.array(self.projection), 
                self.grid_size
            ))
        else:
            coverage = compute_grid_coverage_numpy(
                self.projection, 
                self.grid_size
            )
        
        return coverage
    
    def compute_coverage_by_label(self) -> Dict[str, float]:
        """
        Compute coverage for each label separately.
        Will automatically refit UMAP if new data has been added.
        
        Returns:
            coverage_dict: Dictionary mapping label to coverage
        """
        # Always refit if data has changed
        if self._data_changed or self.projection is None:
            self.fit_umap()
        
        # Safety check after fitting
        all_states = self._get_all_states()
        if len(self.labels) != len(all_states) or len(self.projection) != len(all_states):
            raise ValueError(
                f"Mismatch after fitting! States: {len(all_states)}, "
                f"Labels: {len(self.labels)}, Projection: {len(self.projection)}"
            )
        
        if not self.labels or all(l is None for l in self.labels):
            return {}
        
        coverage_dict = {}
        unique_labels = set(l for l in self.labels if l is not None)
        
        for label in unique_labels:
            # Get indices for this label
            indices = [i for i, l in enumerate(self.labels) if l == label]
            
            if not indices:
                continue
            
            # Additional safety check
            max_idx = max(indices)
            if max_idx >= len(self.projection):
                raise ValueError(
                    f"Index error for label {label}: max index {max_idx}, "
                    f"but projection only has {len(self.projection)} elements"
                )
            
            label_projection = self.projection[indices]
            
            if self.use_jax:
                coverage = float(compute_grid_coverage_jax(
                    jnp.array(label_projection), 
                    self.grid_size
                ))
            else:
                coverage = compute_grid_coverage_numpy(
                    label_projection, 
                    self.grid_size
                )
            
            coverage_dict[label] = coverage
        
        return coverage_dict
    
    def visualize(self, save_path: Optional[str] = None, 
                  color_by_label: bool = True,
                  figsize: Tuple[int, int] = (12, 10),
                  show_grid: bool = True) -> None:
        """
        Visualize the UMAP projection with grid.
        Will automatically refit UMAP if new data has been added.
        
        Args:
            save_path: Path to save figure (if None, just display)
            color_by_label: Whether to color points by label
            figsize: Figure size
            show_grid: Whether to show the grid overlay
        """
        # Always refit if data has changed
        if self._data_changed or self.projection is None:
            self.fit_umap()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot points
        valid_labels = [l for l in self.labels if l is not None]
        if color_by_label and valid_labels:
            unique_labels = sorted(set(valid_labels))
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                indices = [j for j, l in enumerate(self.labels) if l == label]
                
                if not indices:
                    continue
                
                ax.scatter(
                    self.projection[indices, 0],
                    self.projection[indices, 1],
                    c=[colors[i]],
                    label=label,
                    alpha=0.6,
                    s=5
                )
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        else:
            ax.scatter(
                self.projection[:, 0],
                self.projection[:, 1],
                alpha=0.5,
                s=5,
                c='blue'
            )
        
        # Draw grid
        if show_grid:
            for i in range(self.grid_size + 1):
                ax.axhline(i/self.grid_size, color='gray', alpha=0.2, linewidth=0.5)
                ax.axvline(i/self.grid_size, color='gray', alpha=0.2, linewidth=0.5)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        
        # Compute and display coverage
        coverage = self.compute_coverage()
        num_states = len(self._get_all_states())
        ax.set_title(f'State Space Coverage: {coverage:.4f} ({num_states:,} states)', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def get_num_states(self) -> int:
        """Get total number of states stored."""
        if not self.states:
            return 0
        return sum(len(s) for s in self.states)
    
    def reset(self) -> None:
        """Clear all stored data."""
        self.states = []
        self.labels = []
        self.umap_model = None
        self.projection = None
        self._data_changed = False
        self._num_states_at_last_fit = 0

# ============================================================================
# Combined Multi-Task Coverage Analyzer
# ============================================================================

class MultiTaskCoverageAnalyzer:
    """
    Combines UMAP and LSH for comprehensive coverage analysis.
    
    - UMAP: For visualization and global understanding (run periodically)
    - LSH: For fast online metrics and exploration bonuses (run every step)
    """
    
    def __init__(self, task_names: List[str], state_dim: int = 39,
                 grid_size: int = 50, n_hash_functions: int = 15,
                 bin_width: float = 0.05):
        """
        Args:
            task_names: List of task names
            state_dim: Dimensionality of state space
            grid_size: Grid size for UMAP coverage
            n_hash_functions: Number of hash functions for LSH
            bin_width: Bin width for LSH
        """
        self.task_names = task_names
        self.state_dim = state_dim
        
        # UMAP tracker (global)
        self.umap_tracker = UMAPCoverageTracker(grid_size=grid_size, use_jax=False)
        
        # LSH trackers (one per task)
        self.lsh_trackers = {
            task: LSHVisitationCounter(
                state_dim=state_dim,
                n_hash_functions=n_hash_functions,
                bin_width=bin_width
            )
            for task in task_names
        }
        
        # Global LSH tracker (all tasks combined)
        self.lsh_global = LSHVisitationCounter(
            state_dim=state_dim,
            n_hash_functions=n_hash_functions,
            bin_width=bin_width
        )
    
    def add_trajectory(self, task_name: str, observations: np.ndarray) -> None:
        """
        Add a trajectory of observations.
        
        Args:
            task_name: Name of the task
            observations: Array of shape (horizon, state_dim)
        """
        observations = np.array(observations)
        
        # Update UMAP tracker
        labels = [task_name] * len(observations)
        self.umap_tracker.add_data(observations, labels)
        
        # Update LSH trackers
        self.lsh_trackers[task_name].record_visits_batch(observations)
        self.lsh_global.record_visits_batch(observations)
    
    def add_trajectories_batch(self, task_name: str, 
                               trajectories: List[np.ndarray]) -> None:
        """
        Add multiple trajectories at once.
        
        Args:
            task_name: Name of the task
            trajectories: List of observation arrays, each of shape (horizon, state_dim)
        """
        all_obs = np.concatenate(trajectories, axis=0)
        self.add_trajectory(task_name, all_obs)
    
    def get_novelty_bonus(self, task_name: str, state: np.ndarray) -> float:
        """
        Get exploration bonus for a state (for online training).
        
        Args:
            task_name: Name of the task
            state: State vector of shape (state_dim,)
        
        Returns:
            novelty: Novelty score in (0, 1]
        """
        return self.lsh_trackers[task_name].get_novelty_score(state)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all coverage metrics.
        
        Returns:
            metrics: Dictionary of all metrics
        """
        metrics = {}
        
        # UMAP-based coverage
        print("Computing UMAP coverage...")
        umap_global = self.umap_tracker.compute_coverage()
        metrics['umap_coverage_global'] = umap_global
        
        umap_by_task = self.umap_tracker.compute_coverage_by_label()
        for task, coverage in umap_by_task.items():
            metrics[f'umap_coverage_{task}'] = coverage
        
        # LSH-based unique state counts
        metrics['lsh_unique_states_global'] = self.lsh_global.get_num_unique_states()
        
        for task_name in self.task_names:
            num_unique = self.lsh_trackers[task_name].get_num_unique_states()
            metrics[f'lsh_unique_states_{task_name}'] = num_unique
        
        return metrics
    
    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of UMAP coverage.
        
        Args:
            save_path: Path to save figure
        """
        self.umap_tracker.visualize(save_path=save_path, color_by_label=True)
    
    def reset(self) -> None:
        """Reset all trackers."""
        self.umap_tracker.reset()
        for tracker in self.lsh_trackers.values():
            tracker.reset()
        self.lsh_global.reset()


def compute_srank(
    feature_matrix: Float[Array, "num_features feature_dim"], delta: float = 0.01
) -> Float[Array, ""]:
    """Compute effective rank (srank) of a feature matrix.

    Args:
        feature_matrix: Matrix of shape [num_features, feature_dim]
        delta: Threshold parameter (default: 0.01)

    Returns:
        Effective rank (srank) value
    """
    s = jnp.linalg.svd(feature_matrix, compute_uv=False)
    cumsum = jnp.cumsum(s)
    total = jnp.sum(s)
    ratios = cumsum / total
    mask = ratios >= (1.0 - delta)
    srank = jnp.argmax(mask) + 1
    return srank


def extract_activations(network_dict: Intermediates) -> LayerActivationsDict:
    def recursive_extract(
        d: Intermediates, current_path: list[str] = []
    ) -> LayerActivationsDict:
        activations = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    sub_activations = recursive_extract(v, current_path + [k])
                    activations.update(sub_activations)
                else:
                    assert isinstance(v, tuple)
                    # HACK: assuming every module only has 1 output
                    activations[k] = v[0]
        return activations

    return recursive_extract(network_dict)


def get_dormant_neuron_logs(
    layer_activations: LayerActivationsDict, dormant_neuron_threshold: float = 0.1
) -> LogDict:
    """Compute the dormant neuron ratio per layer using Equation 1 from
    "The Dormant Neuron Phenomenon in Deep Reinforcement Learning" (Sokar et al., 2023; https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf).

    Adapted from https://github.com/google/dopamine/blob/master/dopamine/labs/redo/tfagents/sac_train_eval.py#L563"""

    all_layers_score: LayerActivationsDict = {}
    dormant_neurons = {}  # To store both mask and count for each layer

    for act_key, act_value in layer_activations.items():
        chex.assert_rank(act_value, 2)
        neurons_score = jnp.mean(jnp.abs(act_value), axis=0)
        neurons_score = neurons_score / (jnp.mean(neurons_score) + 1e-9)
        all_layers_score[act_key] = neurons_score

        mask = jnp.where(
            neurons_score <= dormant_neuron_threshold,
            jnp.ones_like(neurons_score, dtype=jnp.int32),
            jnp.zeros_like(neurons_score, dtype=jnp.int32),
        )
        num_dormant_neurons = jnp.sum(mask)

        dormant_neurons[act_key] = {"mask": mask, "count": num_dormant_neurons}

    logs = {}

    total_dead_neurons = 0
    total_hidden_count = 0
    for layer_name, layer_score in all_layers_score.items():
        num_dormant_neurons = dormant_neurons[layer_name]["count"]
        logs[f"{layer_name}_ratio"] = (num_dormant_neurons / layer_score.shape[0]) * 100
        logs[f"{layer_name}_count"] = num_dormant_neurons
        total_dead_neurons += num_dormant_neurons
        total_hidden_count += layer_score.shape[0]

    logs.update(
        {
            "total_ratio": jnp.array((total_dead_neurons / total_hidden_count) * 100),
            "total_count": total_dead_neurons,
        }
    )

    return logs

