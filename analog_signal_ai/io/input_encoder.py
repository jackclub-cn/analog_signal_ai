"""
InputEncoder - Converts input data to spike trains.

Implements multiple encoding strategies:
- Rate coding: Encode values as spike rates
- Temporal coding: Encode values as spike timing
- Direct mapping: Direct threshold-based spike generation
"""

import numpy as np
from typing import List, Tuple
from ..core.event_queue import EventQueue


class InputEncoder:
    """
    Encodes input data into spike events.
    
    Supports multiple encoding strategies for different types of data:
    - rate: Higher values → more spikes over time
    - temporal: Higher values → earlier spike times
    - direct: Direct threshold comparison
    - poisson: Poisson spike generation with rate proportional to input
    
    Attributes:
        n_inputs: Number of input channels
        encoding: Encoding strategy ('rate', 'temporal', 'direct', 'poisson')
    """
    
    def __init__(
        self, 
        n_inputs: int, 
        encoding: str = 'rate',
        max_rate: float = 100.0,  # Hz
        duration: float = 100.0   # ms
    ):
        """
        Initialize input encoder.
        
        Args:
            n_inputs: Number of input channels
            encoding: Encoding strategy
            max_rate: Maximum spike rate for rate coding (Hz)
            duration: Encoding duration for rate coding (ms)
        """
        self.n_inputs = n_inputs
        self.encoding = encoding
        self.max_rate = max_rate
        self.duration = duration
        
    def encode(
        self, 
        data: np.ndarray, 
        event_queue: EventQueue,
        start_time: float = 0.0
    ):
        """
        Encode input data as spike events.
        
        Args:
            data: Input data, shape (n_inputs,) or (n_samples, n_inputs)
            event_queue: EventQueue to push spike events to
            start_time: Starting time for spike events (ms)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        if self.encoding == 'rate':
            self._encode_rate(data, event_queue, start_time)
        elif self.encoding == 'temporal':
            self._encode_temporal(data, event_queue, start_time)
        elif self.encoding == 'direct':
            self._encode_direct(data, event_queue, start_time)
        elif self.encoding == 'poisson':
            self._encode_poisson(data, event_queue, start_time)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
            
    def _encode_rate(
        self, 
        data: np.ndarray, 
        event_queue: EventQueue,
        start_time: float
    ):
        """
        Rate coding: values → spike rates.
        
        Higher input values produce more spikes over the duration.
        """
        dt = 1.0  # 1ms time step
        
        for i, value in enumerate(data[0]):
            if value > 0:
                # Convert value to spike rate
                rate = value * self.max_rate
                # Expected number of spikes
                n_spikes = int(rate * self.duration / 1000.0)
                
                # Distribute spikes evenly over duration
                if n_spikes > 0:
                    interval = self.duration / n_spikes
                    for j in range(n_spikes):
                        spike_time = start_time + j * interval
                        event_queue.push(spike_time, i, 1.0)
                        
    def _encode_temporal(
        self, 
        data: np.ndarray, 
        event_queue: EventQueue,
        start_time: float
    ):
        """
        Temporal coding: values → spike timing.
        
        Higher input values produce earlier spike times.
        """
        for i, value in enumerate(data[0]):
            if value > 0:
                # Map value to spike time (higher value → earlier spike)
                # value=1 → spike at start_time, value→0 → spike at start_time+duration
                spike_time = start_time + self.duration * (1.0 - value)
                event_queue.push(spike_time, i, 1.0)
                
    def _encode_direct(
        self, 
        data: np.ndarray, 
        event_queue: EventQueue,
        start_time: float
    ):
        """
        Direct mapping: threshold-based spike generation.
        
        Generate a spike immediately if value exceeds threshold.
        """
        threshold = 0.5
        for i, value in enumerate(data[0]):
            if value > threshold:
                event_queue.push(start_time, i, value)
                
    def _encode_poisson(
        self, 
        data: np.ndarray, 
        event_queue: EventQueue,
        start_time: float
    ):
        """
        Poisson spike generation.
        
        Generate spikes according to Poisson process with rate proportional to input.
        """
        dt = 1.0  # 1ms time step
        n_steps = int(self.duration / dt)
        
        for i, value in enumerate(data[0]):
            if value > 0:
                rate = value * self.max_rate
                prob = rate * dt / 1000.0  # Probability of spike per time step
                
                for step in range(n_steps):
                    if np.random.rand() < prob:
                        spike_time = start_time + step * dt
                        event_queue.push(spike_time, i, 1.0)
                        
    def reset(self):
        """Reset encoder state (no-op for stateless encoders)."""
        pass
    
    def __repr__(self):
        return f"InputEncoder(n_inputs={self.n_inputs}, encoding='{self.encoding}')"
