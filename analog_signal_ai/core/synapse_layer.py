"""
SynapseLayer - Stores sparse connectivity and routes spikes with delays.

Implements sparse synaptic connections using scipy.sparse matrices for
efficient memory usage and computation. Routes spikes from presynaptic
to postsynaptic neurons with configurable delays.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from typing import List, Tuple, Optional


class SynapseLayer:
    """
    Sparse synaptic connection layer.
    
    Uses CSR (Compressed Sparse Row) format for efficient:
    - Memory: O(n_connections) vs O(n_pre * n_post)
    - Forward propagation: O(n_connections)
    - Sparse weight updates
    
    Attributes:
        weights: Sparse weight matrix, shape (n_post, n_pre)
        delays: Transmission delays in ms, shape (n_post, n_pre)
        n_pre: Number of presynaptic neurons
        n_post: Number of postsynaptic neurons
    """
    
    def __init__(
        self, 
        n_pre: int, 
        n_post: int, 
        default_delay: float = 1.0
    ):
        """
        Initialize synaptic layer with sparse connectivity.
        
        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            default_delay: Default transmission delay in ms
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.default_delay = default_delay
        
        # Initialize sparse weight matrix (LIL format for construction)
        self.weights = lil_matrix((n_post, n_pre), dtype=np.float32)
        # Delay matrix (can also be sparse)
        self.delays = lil_matrix((n_post, n_pre), dtype=np.float32)
        
    def add_connection(
        self, 
        pre_id: int, 
        post_id: int, 
        weight: float, 
        delay: Optional[float] = None
    ):
        """
        Add a single synaptic connection.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            weight: Synaptic weight
            delay: Transmission delay in ms (uses default if None)
        """
        if delay is None:
            delay = self.default_delay
            
        self.weights[post_id, pre_id] = weight
        self.delays[post_id, pre_id] = delay
        
    def add_random_connections(
        self, 
        connection_prob: float = 0.1, 
        weight_range: Tuple[float, float] = (0.5, 1.5),
        seed: Optional[int] = None
    ):
        """
        Add random sparse connections.
        
        Args:
            connection_prob: Probability of connection between any two neurons
            weight_range: (min_weight, max_weight) for random weights
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        min_w, max_w = weight_range
        
        for pre in range(self.n_pre):
            for post in range(self.n_post):
                if np.random.rand() < connection_prob:
                    weight = np.random.uniform(min_w, max_w)
                    self.add_connection(pre, post, weight)
                    
    def propagate(
        self, 
        pre_id: int, 
        current_time: float
    ) -> List[Tuple[float, int, float]]:
        """
        Route spike from presynaptic neuron to targets.
        
        Args:
            pre_id: ID of firing presynaptic neuron
            current_time: Current simulation time in ms
            
        Returns:
            List of (arrival_time, post_id, weight) tuples
        """
        # Get weights and delays for this presynaptic neuron
        # Using CSC format would be more efficient for this operation
        weights_csc = self.weights.tocsc()
        delays_csc = self.delays.tocsc()
        
        # Get all postsynaptic targets
        col = weights_csc.getcol(pre_id)
        target_ids = col.indices
        target_weights = col.data
        target_delays = delays_csc.getcol(pre_id).data
        
        # Generate events for each target
        events = []
        for post_id, weight, delay in zip(target_ids, target_weights, target_delays):
            arrival_time = current_time + delay
            events.append((arrival_time, post_id, weight))
            
        return events
    
    def update_weight(
        self, 
        pre_id: int, 
        post_id: int, 
        delta_w: float,
        min_weight: float = 0.0,
        max_weight: float = 5.0
    ):
        """
        Update synaptic weight with STDP.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            delta_w: Weight change
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        old_weight = self.weights[post_id, pre_id]
        new_weight = np.clip(old_weight + delta_w, min_weight, max_weight)
        self.weights[post_id, pre_id] = new_weight
        
    def finalize(self):
        """Convert to CSR format for efficient computation."""
        self.weights = self.weights.tocsr()
        self.delays = self.delays.tocsr()
        
    def get_weights(self) -> csr_matrix:
        """Return weight matrix in CSR format."""
        return self.weights.tocsr()
    
    def get_connectivity_stats(self) -> dict:
        """Return statistics about connectivity."""
        n_connections = self.weights.nnz
        density = n_connections / (self.n_pre * self.n_post)
        return {
            'n_connections': n_connections,
            'density': density,
            'n_pre': self.n_pre,
            'n_post': self.n_post
        }
    
    def __repr__(self):
        stats = self.get_connectivity_stats()
        return (f"SynapseLayer(n_pre={stats['n_pre']}, n_post={stats['n_post']}, "
                f"n_connections={stats['n_connections']}, density={stats['density']:.3f})")
