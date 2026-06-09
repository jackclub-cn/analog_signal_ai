"""
NeuronLayer - Simulates analog membrane potential integration and spike generation.

Implements the core computation of the analog signal AI framework:
- Exponential decay of membrane potential (leakage)
- Integration of incoming inputs
- Threshold-based spike generation
- Refractory period enforcement
"""

import numpy as np
from typing import List


class NeuronLayer:
    """
    Simulates a layer of neurons with analog membrane potential dynamics.
    
    The membrane potential V(t) evolves according to:
        V(t+dt) = α^dt * V(t) + I(t)
    
    where:
        - α: decay factor (leakage)
        - I(t): input current at time t
        - dt: time step
    
    Spikes are generated when V > threshold and the neuron is not in refractory period.
    
    Attributes:
        V: Membrane potentials, shape (n_neurons,)
        threshold: Spike threshold (default 1.0)
        decay: Leakage factor α ∈ (0, 1) (default 0.9)
        refractory: Refractory period in ms (default 2.0)
    """
    
    def __init__(
        self, 
        n_neurons: int, 
        threshold: float = 1.0, 
        decay: float = 0.9, 
        refractory: float = 2.0
    ):
        """
        Initialize neuron layer.
        
        Args:
            n_neurons: Number of neurons in the layer
            threshold: Spike threshold voltage
            decay: Membrane potential decay factor (0 < decay < 1)
            refractory: Refractory period duration in ms
        """
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.decay = decay
        self.refractory = refractory
        
        # State variables
        self.V = np.zeros(n_neurons, dtype=np.float32)
        self.last_fire_times = np.full(n_neurons, -np.inf, dtype=np.float32)
        self.refractory_until = np.zeros(n_neurons, dtype=np.float32)
        
    def integrate(
        self, 
        inputs: np.ndarray, 
        current_time: float, 
        dt: float = 1.0
    ) -> List[int]:
        """
        Update membrane potentials and generate spikes.
        
        Args:
            inputs: Weighted input currents, shape (n_neurons,)
            current_time: Current simulation time in ms
            dt: Time step in ms
            
        Returns:
            List of neuron IDs that fired this step
        """
        # 1. Apply exponential decay
        self.V *= self.decay ** dt
        
        # 2. Add inputs
        self.V += inputs
        
        # 3. Check which neurons are not in refractory
        active = current_time >= self.refractory_until
        
        # 4. Threshold check: fire if V > threshold and not in refractory
        firing = (self.V > self.threshold) & active
        firing_ids = np.where(firing)[0]
        
        # 5. Reset firing neurons and update refractory times
        if len(firing_ids) > 0:
            self.V[firing_ids] = 0.0
            self.refractory_until[firing_ids] = current_time + self.refractory
            self.last_fire_times[firing_ids] = current_time
        
        return firing_ids.tolist()
    
    def reset(self):
        """Reset all neuron states to initial values."""
        self.V.fill(0.0)
        self.last_fire_times.fill(-np.inf)
        self.refractory_until.fill(0.0)
        
    def get_potentials(self) -> np.ndarray:
        """Return current membrane potentials."""
        return self.V.copy()
    
    def __repr__(self):
        return (f"NeuronLayer(n_neurons={self.n_neurons}, "
                f"threshold={self.threshold}, decay={self.decay}, "
                f"refractory={self.refractory})")
