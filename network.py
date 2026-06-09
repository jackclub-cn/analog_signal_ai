"""
AnalogNetwork - Main network class integrating all components.

Combines the three-layer architecture:
- Digital Interface Layer: InputEncoder, OutputDecoder
- Analog Processing Layer: NeuronLayer, SynapseLayer, EventQueue
- Learning Layer: STDPLearner

Implements the main API:
- forward(): Run inference
- train(): Train with STDP
- reset_states(): Clear network state
- save/load(): Persistence
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import json
import pickle
from pathlib import Path

from .core.neuron_layer import NeuronLayer
from .core.event_queue import EventQueue
from .core.synapse_layer import SynapseLayer
from .learning.stdp_learner import STDPLearner
from .io.input_encoder import InputEncoder
from .io.output_decoder import OutputDecoder


class AnalogNetwork:
    """
    Main network class for Analog Signal AI Framework.
    
    Implements a hybrid three-layer architecture combining digital interfaces
    with analog-inspired processing and local learning rules.
    
    Key Features:
    - Event-driven sparse activation
    - Numerical simulation of analog integration
    - STDP-based local learning
    - Runs on standard CPU without specialized hardware
    
    Attributes:
        n_inputs: Number of input neurons
        n_hidden: Number of hidden neurons
        n_outputs: Number of output neurons
        dt: Simulation time step (ms)
        
    Example:
        >>> net = AnalogNetwork(n_inputs=784, n_hidden=256, n_outputs=10)
        >>> output = net.forward(input_data, steps=100)
        >>> net.train(inputs, rewards, epochs=10)
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_hidden: int,
        n_outputs: int,
        dt: float = 1.0,
        neuron_threshold: float = 1.0,
        neuron_decay: float = 0.9,
        neuron_refractory: float = 2.0,
        connection_prob: float = 0.1,
        stdp_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize analog network.
        
        Args:
            n_inputs: Number of input neurons
            n_hidden: Number of hidden neurons
            n_outputs: Number of output neurons
            dt: Simulation time step in ms
            neuron_threshold: Spike threshold for neurons
            neuron_decay: Membrane potential decay factor
            neuron_refractory: Refractory period in ms
            connection_prob: Probability of synaptic connection
            stdp_params: STDP parameters (tau_plus, tau_minus, A_plus, A_minus)
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.dt = dt
        
        # Initialize components
        # Input encoder
        self.input_encoder = InputEncoder(n_inputs, encoding='rate')
        
        # Hidden layer
        self.hidden_neurons = NeuronLayer(
            n_hidden,
            threshold=neuron_threshold,
            decay=neuron_decay,
            refractory=neuron_refractory
        )
        
        # Output layer
        self.output_neurons = NeuronLayer(
            n_outputs,
            threshold=neuron_threshold,
            decay=neuron_decay,
            refractory=neuron_refractory
        )
        
        # Synaptic connections
        # Input → Hidden
        self.input_hidden_synapses = SynapseLayer(n_inputs, n_hidden)
        self.input_hidden_synapses.add_random_connections(connection_prob)
        self.input_hidden_synapses.finalize()
        
        # Hidden → Output
        self.hidden_output_synapses = SynapseLayer(n_hidden, n_outputs)
        self.hidden_output_synapses.add_random_connections(connection_prob)
        self.hidden_output_synapses.finalize()
        
        # Event queue
        self.event_queue = EventQueue()
        
        # STDP learner
        if stdp_params is None:
            stdp_params = {
                'tau_plus': 20.0,
                'tau_minus': 20.0,
                'A_plus': 0.1,
                'A_minus': 0.1
            }
        self.stdp_learner = STDPLearner(**stdp_params)
        
        # Output decoder
        self.output_decoder = OutputDecoder(n_outputs, decoding='rate')
        
        # Statistics
        self.stats = {
            'total_spikes': 0,
            'total_time_steps': 0,
            'inference_calls': 0,
            'training_calls': 0
        }
        
    def forward(
        self,
        inputs: np.ndarray,
        steps: int = 100,
        return_spikes: bool = False
    ) -> np.ndarray:
        """
        Run forward inference.
        
        Args:
            inputs: Input data, shape (n_inputs,) or (batch, n_inputs)
            steps: Number of simulation steps
            return_spikes: If True, return spike history
            
        Returns:
            Output predictions, shape (n_outputs,)
            If return_spikes=True, also returns spike history dict
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            
        # Reset states
        self.reset_states()
        
        # Encode inputs to spikes
        self.input_encoder.encode(inputs[0], self.event_queue, start_time=0.0)
        
        # Run simulation
        spike_history = {'hidden': [], 'output': []}
        
        for step in range(steps):
            current_time = step * self.dt
            
            # Process events at current time
            events = self.event_queue.pop_events_at_time(current_time)
            
            if events:
                # Accumulate inputs to hidden layer
                hidden_inputs = np.zeros(self.n_hidden, dtype=np.float32)
                for neuron_id, weight in events:
                    if neuron_id < self.n_inputs:
                        # Input → Hidden
                        propagated = self.input_hidden_synapses.propagate(neuron_id, current_time)
                        for arrival_time, post_id, w in propagated:
                            if abs(arrival_time - current_time) < self.dt:
                                hidden_inputs[post_id] += w
                                
                # Update hidden neurons
                hidden_spikes = self.hidden_neurons.integrate(hidden_inputs, current_time, self.dt)
                
                # Propagate hidden spikes to output
                output_inputs = np.zeros(self.n_outputs, dtype=np.float32)
                for pre_id in hidden_spikes:
                    propagated = self.hidden_output_synapses.propagate(pre_id, current_time)
                    for arrival_time, post_id, w in propagated:
                        if abs(arrival_time - current_time) < self.dt:
                            output_inputs[post_id] += w
                            
                # Update output neurons
                output_spikes = self.output_neurons.integrate(output_inputs, current_time, self.dt)
                
                # Record spikes
                spike_history['hidden'].extend([(current_time, spike_id) for spike_id in hidden_spikes])
                spike_history['output'].extend([(current_time, spike_id) for spike_id in output_spikes])
                
                # Record to decoder
                for spike_id in output_spikes:
                    self.output_decoder.record_spike(spike_id, current_time)
                    
                self.stats['total_spikes'] += len(hidden_spikes) + len(output_spikes)
                
            self.stats['total_time_steps'] += 1
            
        # Decode output
        output = self.output_decoder.decode()
        
        self.stats['inference_calls'] += 1
        
        if return_spikes:
            return output, spike_history
        return output
    
    def train(
        self,
        inputs: np.ndarray,
        rewards: Optional[np.ndarray] = None,
        epochs: int = 1,
        steps_per_epoch: int = 100
    ) -> Dict[str, float]:
        """
        Train network with STDP learning.
        
        Args:
            inputs: Training inputs, shape (n_samples, n_inputs)
            rewards: Optional reward signals, shape (n_samples,)
            epochs: Number of training epochs
            steps_per_epoch: Simulation steps per sample
            
        Returns:
            Training statistics dict
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
            
        if rewards is None:
            rewards = np.ones(len(inputs))
            
        total_weight_updates = 0
        
        for epoch in range(epochs):
            for i, (input_data, reward) in enumerate(zip(inputs, rewards)):
                # Run forward pass
                output, spike_history = self.forward(
                    input_data, 
                    steps=steps_per_epoch, 
                    return_spikes=True
                )
                
                # Apply STDP learning
                # Update hidden→output synapses
                for time, post_id in spike_history['output']:
                    self.stdp_learner.record_spike(post_id, time)
                    # Find presynaptic hidden neurons that fired recently
                    for h_time, pre_id in spike_history['hidden']:
                        if abs(h_time - time) < 100:  # STDP window
                            delta_w = self.stdp_learner.compute_weight_update(time - h_time)
                            if abs(delta_w) > 1e-6:
                                delta_w *= reward  # Reward modulation
                                self.hidden_output_synapses.update_weight(pre_id, post_id, delta_w)
                                total_weight_updates += 1
                                
                # Clear STDP traces for next sample
                self.stdp_learner.reset()
                
        self.stats['training_calls'] += 1
        
        return {
            'epochs': epochs,
            'samples': len(inputs),
            'weight_updates': total_weight_updates
        }
    
    def reset_states(self):
        """Reset all network states."""
        self.hidden_neurons.reset()
        self.output_neurons.reset()
        self.event_queue.clear()
        self.output_decoder.reset()
        self.stdp_learner.reset()
        
    def save(self, path: str):
        """
        Save network to file.
        
        Args:
            path: File path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle (simpler for numpy arrays)
        with open(path, 'wb') as f:
            pickle.dump({
                'config': {
                    'n_inputs': self.n_inputs,
                    'n_hidden': self.n_hidden,
                    'n_outputs': self.n_outputs,
                    'dt': self.dt
                },
                'weights': {
                    'input_hidden': self.input_hidden_synapses.weights,
                    'hidden_output': self.hidden_output_synapses.weights
                },
                'stats': self.stats
            }, f)
            
    def load(self, path: str):
        """
        Load network from file.
        
        Args:
            path: File path to load from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        # Verify config
        assert data['config']['n_inputs'] == self.n_inputs
        assert data['config']['n_hidden'] == self.n_hidden
        assert data['config']['n_outputs'] == self.n_outputs
        
        # Restore weights
        self.input_hidden_synapses.weights = data['weights']['input_hidden']
        self.hidden_output_synapses.weights = data['weights']['hidden_output']
        self.stats = data.get('stats', self.stats)
        
    def get_stats(self) -> Dict[str, Any]:
        """Return network statistics."""
        return {
            **self.stats,
            'active_ratio': self.stats['total_spikes'] / max(self.stats['total_time_steps'], 1),
            'input_connections': self.input_hidden_synapses.get_connectivity_stats(),
            'output_connections': self.hidden_output_synapses.get_connectivity_stats()
        }
    
    def __repr__(self):
        return (f"AnalogNetwork(n_inputs={self.n_inputs}, "
                f"n_hidden={self.n_hidden}, n_outputs={self.n_outputs})")
