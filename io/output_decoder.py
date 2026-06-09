"""
OutputDecoder - Converts spike trains to output predictions.

Implements multiple decoding strategies:
- Rate decoding: Count spikes over time window
- Temporal decoding: Decode from spike timing
- Population coding: Decode from population activity
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict


class OutputDecoder:
    """
    Decodes spike trains into output predictions.
    
    Supports multiple decoding strategies:
    - rate: Count spikes per neuron over time window
    - temporal: Decode from first spike timing
    - population: Decode from population activity pattern
    
    Attributes:
        n_outputs: Number of output channels
        decoding: Decoding strategy ('rate', 'temporal', 'population')
        window: Time window for rate decoding (ms)
    """
    
    def __init__(
        self, 
        n_outputs: int, 
        decoding: str = 'rate',
        window: float = 100.0
    ):
        """
        Initialize output decoder.
        
        Args:
            n_outputs: Number of output channels
            decoding: Decoding strategy
            window: Time window for rate decoding (ms)
        """
        self.n_outputs = n_outputs
        self.decoding = decoding
        self.window = window
        
        # Accumulator for spikes
        self.spike_counts = np.zeros(n_outputs, dtype=np.int32)
        self.first_spike_times = np.full(n_outputs, np.inf, dtype=np.float32)
        self.spike_history: Dict[int, List[float]] = defaultdict(list)
        
    def record_spike(self, neuron_id: int, time: float):
        """
        Record a spike from an output neuron.
        
        Args:
            neuron_id: ID of the firing neuron
            time: Spike time in ms
        """
        if neuron_id < self.n_outputs:
            self.spike_counts[neuron_id] += 1
            self.first_spike_times[neuron_id] = min(self.first_spike_times[neuron_id], time)
            self.spike_history[neuron_id].append(time)
            
    def decode(self) -> np.ndarray:
        """
        Decode recorded spikes into output values.
        
        Returns:
            Output array, shape (n_outputs,)
        """
        if self.decoding == 'rate':
            return self._decode_rate()
        elif self.decoding == 'temporal':
            return self._decode_temporal()
        elif self.decoding == 'population':
            return self._decode_population()
        else:
            raise ValueError(f"Unknown decoding: {self.decoding}")
            
    def _decode_rate(self) -> np.ndarray:
        """
        Rate decoding: spike counts → output values.
        
        Normalize spike counts to [0, 1] range.
        """
        max_count = max(self.spike_counts.max(), 1)
        return self.spike_counts.astype(np.float32) / max_count
    
    def _decode_temporal(self) -> np.ndarray:
        """
        Temporal decoding: first spike time → output value.
        
        Earlier spikes → higher output values.
        """
        # Normalize to [0, 1]: earlier spikes get higher values
        # inf (no spike) → 0, earliest spike → 1
        min_time = self.first_spike_times.min()
        max_time = self.first_spike_times.max()
        
        if max_time == np.inf:
            return np.zeros(self.n_outputs, dtype=np.float32)
            
        # Invert: earlier → higher
        output = np.zeros(self.n_outputs, dtype=np.float32)
        valid = self.first_spike_times < np.inf
        
        if valid.any():
            time_range = max_time - min_time if max_time > min_time else 1.0
            output[valid] = 1.0 - (self.first_spike_times[valid] - min_time) / time_range
            
        return output
    
    def _decode_population(self) -> np.ndarray:
        """
        Population decoding: normalize across population.
        
        Softmax-like normalization of spike counts.
        """
        counts = self.spike_counts.astype(np.float32)
        
        if counts.sum() == 0:
            return np.zeros(self.n_outputs, dtype=np.float32)
            
        # Softmax normalization
        exp_counts = np.exp(counts - counts.max())
        return exp_counts / exp_counts.sum()
    
    def reset(self):
        """Clear all accumulated spike data."""
        self.spike_counts.fill(0)
        self.first_spike_times.fill(np.inf)
        self.spike_history.clear()
        
    def get_spike_counts(self) -> np.ndarray:
        """Return current spike counts."""
        return self.spike_counts.copy()
    
    def get_spike_history(self, neuron_id: int) -> List[float]:
        """
        Get spike history for a specific neuron.
        
        Args:
            neuron_id: Neuron ID
            
        Returns:
            List of spike times
        """
        return self.spike_history.get(neuron_id, [])
    
    def __repr__(self):
        return f"OutputDecoder(n_outputs={self.n_outputs}, decoding='{self.decoding}')"
