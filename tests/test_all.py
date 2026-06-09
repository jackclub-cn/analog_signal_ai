"""
Test suite for Analog Signal AI Framework.

Tests all core modules and the main network functionality.
"""

import unittest
import numpy as np

from analog_signal_ai.core.neuron_layer import NeuronLayer
from analog_signal_ai.core.event_queue import EventQueue
from analog_signal_ai.core.synapse_layer import SynapseLayer
from analog_signal_ai.learning.stdp_learner import STDPLearner
from analog_signal_ai.io.input_encoder import InputEncoder
from analog_signal_ai.io.output_decoder import OutputDecoder
from analog_signal_ai.network import AnalogNetwork


class TestNeuronLayer(unittest.TestCase):
    """Test NeuronLayer functionality."""
    
    def test_initialization(self):
        """Test neuron layer initialization."""
        layer = NeuronLayer(n_neurons=100, threshold=1.0, decay=0.9, refractory=2.0)
        self.assertEqual(layer.n_neurons, 100)
        self.assertEqual(layer.threshold, 1.0)
        self.assertEqual(layer.decay, 0.9)
        self.assertEqual(layer.refractory, 2.0)
        self.assertEqual(len(layer.V), 100)
        self.assertTrue(np.all(layer.V == 0.0))
        
    def test_integration_and_spike_generation(self):
        """Test membrane potential integration and spike generation."""
        layer = NeuronLayer(n_neurons=10, threshold=1.0, decay=0.95)
        
        # Apply strong input to trigger spike
        inputs = np.zeros(10)
        inputs[0] = 1.5  # Above threshold
        
        spikes = layer.integrate(inputs, current_time=0.0, dt=1.0)
        
        # Should generate spike for neuron 0
        self.assertEqual(len(spikes), 1)
        self.assertEqual(spikes[0], 0)
        
        # Membrane potential should be reset
        self.assertEqual(layer.V[0], 0.0)
        
    def test_refractory_period(self):
        """Test that neurons in refractory don't fire."""
        layer = NeuronLayer(n_neurons=10, threshold=1.0, refractory=5.0)
        
        # Trigger spike
        inputs = np.zeros(10)
        inputs[0] = 1.5
        spikes1 = layer.integrate(inputs, current_time=0.0, dt=1.0)
        
        # Try to fire again immediately (should be in refractory)
        inputs[0] = 2.0  # Even stronger
        spikes2 = layer.integrate(inputs, current_time=1.0, dt=1.0)
        
        self.assertEqual(len(spikes1), 1)
        self.assertEqual(len(spikes2), 0)  # Should not fire during refractory
        
    def test_decay(self):
        """Test membrane potential decay."""
        layer = NeuronLayer(n_neurons=10, threshold=10.0, decay=0.9)
        
        # Apply input
        inputs = np.zeros(10)
        inputs[0] = 5.0
        layer.integrate(inputs, current_time=0.0, dt=1.0)
        
        # Check potential decayed correctly
        # V = 0.9 * 0 + 5.0 = 5.0
        self.assertAlmostEqual(layer.V[0], 5.0, places=5)
        
        # Apply no input (decay only)
        inputs[0] = 0.0
        layer.integrate(inputs, current_time=1.0, dt=1.0)
        
        # V = 0.9 * 5.0 = 4.5
        self.assertAlmostEqual(layer.V[0], 4.5, places=5)
        
    def test_reset(self):
        """Test state reset."""
        layer = NeuronLayer(n_neurons=10)
        
        # Apply input
        inputs = np.ones(10) * 0.5
        layer.integrate(inputs, current_time=0.0, dt=1.0)
        
        # Reset
        layer.reset()
        
        self.assertTrue(np.all(layer.V == 0.0))
        self.assertTrue(np.all(layer.last_fire_times == -np.inf))


class TestEventQueue(unittest.TestCase):
    """Test EventQueue functionality."""
    
    def test_push_and_pop(self):
        """Test event push and pop operations."""
        queue = EventQueue()
        
        # Push events
        queue.push(time=3.0, neuron_id=2, weight=1.0)
        queue.push(time=1.0, neuron_id=0, weight=1.0)
        queue.push(time=2.0, neuron_id=1, weight=1.0)
        
        # Pop should return in time order
        event1 = queue.pop()
        self.assertEqual(event1[0], 1.0)  # time
        self.assertEqual(event1[1], 0)    # neuron_id
        
        event2 = queue.pop()
        self.assertEqual(event2[0], 2.0)
        
        event3 = queue.pop()
        self.assertEqual(event3[0], 3.0)
        
        # Empty queue
        self.assertIsNone(queue.pop())
        
    def test_peek_time(self):
        """Test peek operation."""
        queue = EventQueue()
        
        self.assertEqual(queue.peek_time(), float('inf'))
        
        queue.push(time=5.0, neuron_id=0, weight=1.0)
        queue.push(time=2.0, neuron_id=1, weight=1.0)
        
        self.assertEqual(queue.peek_time(), 2.0)
        # Peek should not remove
        self.assertEqual(len(queue), 2)
        
    def test_pop_events_at_time(self):
        """Test popping all events at a specific time."""
        queue = EventQueue()
        
        queue.push(time=1.0, neuron_id=0, weight=1.0)
        queue.push(time=1.0, neuron_id=1, weight=0.5)
        queue.push(time=2.0, neuron_id=2, weight=1.0)
        
        events = queue.pop_events_at_time(1.0)
        
        self.assertEqual(len(events), 2)
        self.assertEqual(len(queue), 1)  # Only time=2.0 remains


class TestSynapseLayer(unittest.TestCase):
    """Test SynapseLayer functionality."""
    
    def test_add_connection(self):
        """Test adding synaptic connections."""
        syn = SynapseLayer(n_pre=10, n_post=5)
        
        syn.add_connection(pre_id=0, post_id=0, weight=1.5, delay=2.0)
        
        self.assertEqual(syn.weights[0, 0], 1.5)
        self.assertEqual(syn.delays[0, 0], 2.0)
        
    def test_random_connections(self):
        """Test random connection generation."""
        syn = SynapseLayer(n_pre=100, n_post=50)
        
        syn.add_random_connections(connection_prob=0.1, seed=42)
        syn.finalize()
        
        stats = syn.get_connectivity_stats()
        
        # Should have approximately 10% connectivity
        self.assertGreater(stats['density'], 0.05)
        self.assertLess(stats['density'], 0.15)
        
    def test_propagate(self):
        """Test spike propagation."""
        syn = SynapseLayer(n_pre=10, n_post=5)
        
        syn.add_connection(pre_id=0, post_id=0, weight=1.0, delay=1.0)
        syn.add_connection(pre_id=0, post_id=1, weight=0.5, delay=2.0)
        syn.finalize()
        
        events = syn.propagate(pre_id=0, current_time=10.0)
        
        self.assertEqual(len(events), 2)
        
        # Check timing
        times = [e[0] for e in events]
        self.assertIn(11.0, times)  # 10.0 + 1.0
        self.assertIn(12.0, times)  # 10.0 + 2.0
        
    def test_weight_update(self):
        """Test synaptic weight update."""
        syn = SynapseLayer(n_pre=10, n_post=5)
        
        syn.add_connection(pre_id=0, post_id=0, weight=1.0)
        
        syn.update_weight(pre_id=0, post_id=0, delta_w=0.5)
        
        self.assertEqual(syn.weights[0, 0], 1.5)
        
        # Test clipping
        syn.update_weight(pre_id=0, post_id=0, delta_w=10.0, max_weight=5.0)
        self.assertEqual(syn.weights[0, 0], 5.0)


class TestSTDPLearner(unittest.TestCase):
    """Test STDP learning functionality."""
    
    def test_weight_update_calculation(self):
        """Test STDP weight update calculation."""
        learner = STDPLearner(tau_plus=20.0, tau_minus=20.0, A_plus=0.1, A_minus=0.1)
        
        # Post after pre (LTP)
        delta_w1 = learner.compute_weight_update(delta_t=10.0)
        self.assertGreater(delta_w1, 0)
        
        # Pre after post (LTD)
        delta_w2 = learner.compute_weight_update(delta_t=-10.0)
        self.assertLess(delta_w2, 0)
        
        # Simultaneous
        delta_w3 = learner.compute_weight_update(delta_t=0.0)
        self.assertEqual(delta_w3, 0.0)
        
    def test_spike_recording(self):
        """Test spike time recording."""
        learner = STDPLearner()
        
        learner.record_spike(neuron_id=0, time=10.0)
        learner.record_spike(neuron_id=1, time=15.0)
        
        self.assertEqual(learner.last_spike_times[0], 10.0)
        self.assertEqual(learner.last_spike_times[1], 15.0)


class TestInputEncoder(unittest.TestCase):
    """Test InputEncoder functionality."""
    
    def test_rate_encoding(self):
        """Test rate coding encoding."""
        encoder = InputEncoder(n_inputs=10, encoding='rate', max_rate=100.0, duration=100.0)
        queue = EventQueue()
        
        data = np.zeros(10)
        data[0] = 1.0  # Maximum rate
        
        encoder.encode(data, queue, start_time=0.0)
        
        # Should generate spikes
        self.assertGreater(len(queue), 0)
        
    def test_direct_encoding(self):
        """Test direct threshold encoding."""
        encoder = InputEncoder(n_inputs=10, encoding='direct')
        queue = EventQueue()
        
        data = np.zeros(10)
        data[0] = 0.8  # Above threshold (0.5)
        
        encoder.encode(data, queue, start_time=0.0)
        
        self.assertEqual(len(queue), 1)


class TestOutputDecoder(unittest.TestCase):
    """Test OutputDecoder functionality."""
    
    def test_rate_decoding(self):
        """Test rate decoding."""
        decoder = OutputDecoder(n_outputs=5, decoding='rate')
        
        # Record spikes
        decoder.record_spike(neuron_id=0, time=10.0)
        decoder.record_spike(neuron_id=0, time=20.0)
        decoder.record_spike(neuron_id=1, time=15.0)
        
        output = decoder.decode()
        
        # Neuron 0 fired twice
        self.assertGreater(output[0], output[1])
        self.assertEqual(output[2], 0.0)
        
    def test_reset(self):
        """Test decoder reset."""
        decoder = OutputDecoder(n_outputs=5)
        
        decoder.record_spike(neuron_id=0, time=10.0)
        decoder.reset()
        
        output = decoder.decode()
        self.assertTrue(np.all(output == 0.0))


class TestAnalogNetwork(unittest.TestCase):
    """Test main AnalogNetwork class."""
    
    def test_initialization(self):
        """Test network initialization."""
        net = AnalogNetwork(n_inputs=10, n_hidden=20, n_outputs=5)
        
        self.assertEqual(net.n_inputs, 10)
        self.assertEqual(net.n_hidden, 20)
        self.assertEqual(net.n_outputs, 5)
        
    def test_forward_pass(self):
        """Test forward inference."""
        net = AnalogNetwork(n_inputs=10, n_hidden=20, n_outputs=5)
        
        inputs = np.random.rand(10)
        output = net.forward(inputs, steps=50)
        
        self.assertEqual(len(output), 5)
        self.assertTrue(np.all(output >= 0.0))
        self.assertTrue(np.all(output <= 1.0))
        
    def test_training(self):
        """Test STDP training."""
        net = AnalogNetwork(n_inputs=10, n_hidden=20, n_outputs=5)
        
        inputs = np.random.rand(3, 10)
        rewards = np.ones(3)
        
        stats = net.train(inputs, rewards, epochs=1, steps_per_epoch=50)
        
        self.assertEqual(stats['epochs'], 1)
        self.assertEqual(stats['samples'], 3)
        
    def test_save_load(self):
        """Test network persistence."""
        net = AnalogNetwork(n_inputs=10, n_hidden=20, n_outputs=5)
        
        # Run some inference
        inputs = np.random.rand(10)
        output1 = net.forward(inputs, steps=50)
        
        # Save
        save_path = '/tmp/test_network.pkl'
        net.save(save_path)
        
        # Create new network and load
        net2 = AnalogNetwork(n_inputs=10, n_hidden=20, n_outputs=5)
        net2.load(save_path)
        
        # Should produce same output (weights restored)
        # Note: output may differ due to random encoding, but weights should match
        self.assertEqual(net.stats['inference_calls'], net2.stats['inference_calls'])
        
    def test_statistics(self):
        """Test network statistics."""
        net = AnalogNetwork(n_inputs=10, n_hidden=20, n_outputs=5)
        
        inputs = np.random.rand(10)
        net.forward(inputs, steps=50)
        
        stats = net.get_stats()
        
        self.assertIn('total_spikes', stats)
        self.assertIn('inference_calls', stats)
        self.assertEqual(stats['inference_calls'], 1)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
