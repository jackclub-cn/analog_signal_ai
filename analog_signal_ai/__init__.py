"""
Analog Signal AI Framework

A software model framework implementing analog-inspired neural networks
with event-driven sparse activation and STDP learning.

Key Features:
- Numerical simulation of analog integrators
- Event-driven sparse activation (1-5% active neurons)
- STDP (Spike-Timing-Dependent Plasticity) learning
- Runs on standard CPU without specialized hardware

Performance Target:
- 10-50x efficiency improvement over Transformer architectures
- Memory efficient: O(active_neurons) vs O(N²)
"""

__version__ = "0.1.0"
__author__ = "Analog Signal AI Team"

from .core.neuron_layer import NeuronLayer
from .core.event_queue import EventQueue
from .core.synapse_layer import SynapseLayer
from .learning.stdp_learner import STDPLearner
from .io.input_encoder import InputEncoder
from .io.output_decoder import OutputDecoder
from .network import AnalogNetwork

__all__ = [
    "NeuronLayer",
    "EventQueue", 
    "SynapseLayer",
    "STDPLearner",
    "InputEncoder",
    "OutputDecoder",
    "AnalogNetwork",
]
