"""
STDPLearner - Implements Spike-Timing-Dependent Plasticity learning rule.

STDP is a biologically-inspired learning rule where synaptic strength changes
based on the relative timing of pre- and post-synaptic spikes:
- If post fires after pre (Δt > 0): Long-term potentiation (LTP)
- If pre fires after post (Δt < 0): Long-term depression (LTD)

Mathematical formulation:
    Δw = {
        A+ * exp(-Δt / τ+)  if Δt > 0
        -A- * exp(Δt / τ-)  if Δt < 0
    }

where:
    - Δt = t_post - t_pre
    - A+, A-: learning rate parameters
    - τ+, τ-: time constants
"""

import numpy as np
from typing import Dict, Tuple, Optional
from ..core.synapse_layer import SynapseLayer


class STDPLearner:
    """
    STDP learning rule implementation.
    
    Tracks spike timing for each neuron and computes weight updates
    based on the spike-timing-dependent plasticity rule.
    
    Attributes:
        tau_plus: Time constant for LTP (post after pre)
        tau_minus: Time constant for LTD (pre after post)
        A_plus: Learning rate for LTP
        A_minus: Learning rate for LTD
        last_spike_times: Dict mapping neuron_id -> last spike time
    """
    
    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        A_plus: float = 0.1,
        A_minus: float = 0.1
    ):
        """
        Initialize STDP learner.
        
        Args:
            tau_plus: Time constant for potentiation (ms)
            tau_minus: Time constant for depression (ms)
            A_plus: Learning rate for LTP
            A_minus: Learning rate for LTD
        """
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        
        # Track last spike time for each neuron
        self.last_spike_times: Dict[int, float] = {}
        
    def record_spike(self, neuron_id: int, time: float):
        """
        Record a spike event for timing tracking.
        
        Args:
            neuron_id: ID of the neuron that fired
            time: Spike time in ms
        """
        self.last_spike_times[neuron_id] = time
        
    def compute_weight_update(self, delta_t: float) -> float:
        """
        Compute weight change based on spike timing difference.
        
        Args:
            delta_t: t_post - t_pre in ms
            
        Returns:
            Weight change Δw
        """
        if delta_t > 0:
            # Post after pre → LTP (potentiation)
            return self.A_plus * np.exp(-delta_t / self.tau_plus)
        elif delta_t < 0:
            # Pre after post → LTD (depression)
            return -self.A_minus * np.exp(delta_t / self.tau_minus)
        else:
            # Simultaneous spikes → no change
            return 0.0
    
    def update_synapse(
        self,
        synapse_layer: SynapseLayer,
        pre_id: int,
        post_id: int,
        min_weight: float = 0.0,
        max_weight: float = 5.0
    ) -> float:
        """
        Update synaptic weight based on STDP rule.
        
        Checks timing of pre and post spikes and applies
        the appropriate weight change.
        
        Args:
            synapse_layer: SynapseLayer to update
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            
        Returns:
            Weight change applied (0 if no update)
        """
        # Check if we have spike times for both neurons
        if pre_id not in self.last_spike_times or post_id not in self.last_spike_times:
            return 0.0
            
        t_pre = self.last_spike_times[pre_id]
        t_post = self.last_spike_times[post_id]
        
        delta_t = t_post - t_pre
        delta_w = self.compute_weight_update(delta_t)
        
        if abs(delta_w) > 1e-6:  # Only update if significant
            synapse_layer.update_weight(pre_id, post_id, delta_w, min_weight, max_weight)
            
        return delta_w
    
    def process_spike_pair(
        self,
        synapse_layer: SynapseLayer,
        pre_id: int,
        post_id: int,
        current_time: float,
        window: float = 100.0
    ):
        """
        Process a spike pair and apply STDP update.
        
        This method handles the common case where a post-synaptic neuron
        fires and we need to update its incoming synapses based on recent
        pre-synaptic activity.
        
        Args:
            synapse_layer: SynapseLayer to update
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            current_time: Current simulation time
            window: Time window for STDP (ms)
        """
        # Record the spike
        self.record_spike(post_id, current_time)
        
        # Check if pre neuron fired recently
        if pre_id in self.last_spike_times:
            t_pre = self.last_spike_times[pre_id]
            delta_t = current_time - t_pre
            
            # Only update if within STDP window
            if abs(delta_t) < window:
                self.update_synapse(synapse_layer, pre_id, post_id)
                
    def reset(self):
        """Clear all recorded spike times."""
        self.last_spike_times.clear()
        
    def get_parameters(self) -> dict:
        """Return STDP parameters."""
        return {
            'tau_plus': self.tau_plus,
            'tau_minus': self.tau_minus,
            'A_plus': self.A_plus,
            'A_minus': self.A_minus
        }
    
    def __repr__(self):
        return (f"STDPLearner(tau_plus={self.tau_plus}, tau_minus={self.tau_minus}, "
                f"A_plus={self.A_plus}, A_minus={self.A_minus})")
