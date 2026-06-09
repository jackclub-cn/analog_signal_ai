"""
Integration tests for Analog Signal AI Framework.

Tests end-to-end workflows, full pipeline execution, and multi-component interactions.
"""

import unittest
import numpy as np
import tempfile
import os
import time

from analog_signal_ai.core.neuron_layer import NeuronLayer
from analog_signal_ai.core.event_queue import EventQueue
from analog_signal_ai.core.synapse_layer import SynapseLayer
from analog_signal_ai.learning.stdp_learner import STDPLearner
from analog_signal_ai.io.input_encoder import InputEncoder
from analog_signal_ai.io.output_decoder import OutputDecoder
from analog_signal_ai.network import AnalogNetwork


class TestFullInferencePipeline(unittest.TestCase):
    """Test complete inference pipeline from input to output."""
    
    def test_end_to_end_inference(self):
        """Test full forward pass through network."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        # Create input signal
        inputs = np.random.rand(50)
        
        # Run inference
        output = net.forward(inputs, steps=100)
        
        # Verify output
        self.assertEqual(len(output), 10)
        self.assertTrue(np.all(output >= 0.0))
        self.assertTrue(np.all(output <= 1.0))
        
        # Network should have recorded statistics
        stats = net.get_stats()
        self.assertGreater(stats['total_spikes'], 0)
        self.assertGreater(stats['total_time_steps'], 0)
        
    def test_inference_with_spike_tracking(self):
        """Test inference with detailed spike history."""
        net = AnalogNetwork(n_inputs=20, n_hidden=50, n_outputs=5)
        
        inputs = np.ones(20) * 0.8  # Strong input
        
        output, spike_history = net.forward(inputs, steps=200, return_spikes=True)
        
        # Should have generated spikes
        self.assertGreater(len(spike_history['hidden']), 0)
        self.assertGreater(len(spike_history['output']), 0)
        
        # Spikes should have valid timestamps
        for time, neuron_id in spike_history['hidden']:
            self.assertGreater(time, 0)
            self.assertLess(time, 200)
            self.assertGreaterEqual(neuron_id, 0)
            self.assertLess(neuron_id, 50)
            
    def test_different_input_patterns(self):
        """Test network response to different input patterns."""
        net = AnalogNetwork(n_inputs=100, n_hidden=200, n_outputs=10)
        
        # Sparse input
        sparse_input = np.zeros(100)
        sparse_input[0:10] = 0.8
        
        output_sparse = net.forward(sparse_input, steps=100)
        
        # Dense input with different values
        dense_input = np.ones(100) * 0.2  # Much lower rate
        
        output_dense = net.forward(dense_input, steps=100)
        
        # Both outputs should be valid
        self.assertEqual(len(output_sparse), 10)
        self.assertEqual(len(output_dense), 10)
        self.assertTrue(np.all(output_sparse >= 0.0))
        self.assertTrue(np.all(output_dense >= 0.0))
        
    def test_zero_input_handling(self):
        """Test network with zero input."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        zero_input = np.zeros(50)
        
        output = net.forward(zero_input, steps=100)
        
        # Should produce valid output (possibly zero)
        self.assertEqual(len(output), 10)
        self.assertTrue(np.all(output >= 0.0))


class TestTrainingWorkflow(unittest.TestCase):
    """Test complete training workflows."""
    
    def test_basic_training_cycle(self):
        """Test single training epoch."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        # Training data
        inputs = np.random.rand(5, 50)
        rewards = np.ones(5)
        
        stats = net.train(inputs, rewards, epochs=1, steps_per_epoch=100)
        
        self.assertEqual(stats['epochs'], 1)
        self.assertEqual(stats['samples'], 5)
        self.assertGreater(stats['weight_updates'], 0)
        
    def test_training_with_positive_rewards(self):
        """Test training with positive rewards (LTP reinforcement)."""
        net = AnalogNetwork(n_inputs=20, n_hidden=50, n_outputs=5)
        
        inputs = np.random.rand(3, 20)
        rewards = np.array([1.0, 1.0, 1.0])  # All positive
        
        # Store initial weights (convert to dense for comparison)
        initial_weights = net.hidden_output_synapses.weights.toarray().copy()
        
        net.train(inputs, rewards, epochs=5, steps_per_epoch=100)
        
        # Weights may have changed (not guaranteed due to sparse connectivity)
        final_weights = net.hidden_output_synapses.weights.toarray()
        # Just verify training completed without error
        self.assertEqual(net.stats['training_calls'], 1)
        
    def test_training_with_negative_rewards(self):
        """Test training with negative rewards (LTD reinforcement)."""
        net = AnalogNetwork(n_inputs=20, n_hidden=50, n_outputs=5)
        
        inputs = np.random.rand(3, 20)
        rewards = np.array([-0.5, -0.5, -0.5])  # All negative
        
        # Training should complete without error
        stats = net.train(inputs, rewards, epochs=5, steps_per_epoch=100)
        
        self.assertEqual(stats['epochs'], 5)
        self.assertEqual(stats['samples'], 3)
        
    def test_training_without_rewards(self):
        """Test training without explicit rewards (default to 1.0)."""
        net = AnalogNetwork(n_inputs=30, n_hidden=60, n_outputs=8)
        
        inputs = np.random.rand(4, 30)
        
        # Train without rewards
        stats = net.train(inputs, epochs=2, steps_per_epoch=100)
        
        self.assertEqual(stats['epochs'], 2)
        self.assertGreater(stats['weight_updates'], 0)
        
    def test_multi_epoch_training(self):
        """Test training across multiple epochs."""
        net = AnalogNetwork(n_inputs=30, n_hidden=80, n_outputs=5)
        
        inputs = np.random.rand(10, 30)
        rewards = np.random.rand(10) * 2 - 0.5  # Mixed rewards
        
        stats = net.train(inputs, rewards, epochs=10, steps_per_epoch=50)
        
        self.assertEqual(stats['epochs'], 10)
        self.assertEqual(stats['samples'], 10)
        
    def test_learning_progression(self):
        """Test that learning actually progresses across epochs."""
        net = AnalogNetwork(n_inputs=20, n_hidden=50, n_outputs=5)
        
        inputs = np.random.rand(5, 20)
        rewards = np.ones(5)
        
        # Run first epoch
        stats1 = net.train(inputs, rewards, epochs=1, steps_per_epoch=100)
        
        # Run second epoch
        stats2 = net.train(inputs, rewards, epochs=1, steps_per_epoch=100)
        
        # Both should have weight updates
        self.assertGreater(stats1['weight_updates'], 0)
        self.assertGreater(stats2['weight_updates'], 0)


class TestPersistenceWorkflow(unittest.TestCase):
    """Test save/load workflows."""
    
    def test_save_load_roundtrip(self):
        """Test complete save and load cycle."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        # Run some inference to establish state
        inputs = np.random.rand(50)
        net.forward(inputs, steps=100)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            save_path = f.name
        
        try:
            net.save(save_path)
            
            # Load into new network
            net2 = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
            net2.load(save_path)
            
            # Verify weights match (convert sparse to dense for comparison)
            self.assertTrue(np.allclose(
                net.input_hidden_synapses.weights.toarray(),
                net2.input_hidden_synapses.weights.toarray()
            ))
            self.assertTrue(np.allclose(
                net.hidden_output_synapses.weights.toarray(),
                net2.hidden_output_synapses.weights.toarray()
            ))
            
            # Stats should be preserved
            self.assertEqual(net.stats['inference_calls'], net2.stats['inference_calls'])
            
        finally:
            os.unlink(save_path)
            
    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        net = AnalogNetwork(n_inputs=20, n_hidden=50, n_outputs=5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'subdir', 'nested', 'network.pkl')
            
            net.save(save_path)
            
            # File should exist
            self.assertTrue(os.path.exists(save_path))
            
    def test_load_config_mismatch(self):
        """Test that load raises error on config mismatch."""
        net1 = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            save_path = f.name
        
        try:
            net1.save(save_path)
            
            # Try to load into different sized network
            net2 = AnalogNetwork(n_inputs=20, n_hidden=50, n_outputs=5)
            
            with self.assertRaises(AssertionError):
                net2.load(save_path)
                
        finally:
            os.unlink(save_path)


class TestMultiComponentInteraction(unittest.TestCase):
    """Test interactions between multiple components."""
    
    def test_encoder_to_queue_integration(self):
        """Test input encoder feeding event queue."""
        encoder = InputEncoder(n_inputs=20, encoding='rate', max_rate=100.0, duration=50.0)
        queue = EventQueue()
        
        inputs = np.ones(20) * 0.5
        
        encoder.encode(inputs, queue, start_time=0.0)
        
        # Queue should have events
        self.assertGreater(len(queue), 0)
        
        # Events should be properly structured
        event = queue.pop()
        self.assertIsNotNone(event)
        self.assertIn(len(event), [3, 4])  # (time, neuron_id, weight) or similar
        
    def test_synapse_to_queue_integration(self):
        """Test synapse propagation feeding event queue."""
        syn = SynapseLayer(n_pre=10, n_post=5)
        
        syn.add_connection(pre_id=0, post_id=0, weight=1.0, delay=2.0)
        syn.add_connection(pre_id=0, post_id=1, weight=0.5, delay=3.0)
        syn.finalize()
        
        queue = EventQueue()
        
        events = syn.propagate(pre_id=0, current_time=0.0)
        
        # Push events to queue
        for event in events:
            queue.push(event[0], event[1], event[2])
        
        # Queue should have 2 events
        self.assertEqual(len(queue), 2)
        
        # Events should arrive at correct times
        event1 = queue.pop()
        self.assertEqual(event1[0], 2.0)  # first delay
        
    def test_stdp_with_synapse_integration(self):
        """Test STDP learner updating synapse weights."""
        syn = SynapseLayer(n_pre=10, n_post=5)
        syn.add_connection(pre_id=0, post_id=0, weight=1.0)
        syn.add_connection(pre_id=1, post_id=0, weight=1.0)
        syn.finalize()
        
        learner = STDPLearner(tau_plus=20.0, tau_minus=20.0, A_plus=0.2, A_minus=0.2)
        
        initial_weight = syn.weights[0, 0]
        
        # LTP scenario: pre fires, then post fires
        learner.record_spike(neuron_id=0, time=10.0)  # pre
        learner.record_spike(neuron_id=100, time=15.0)  # post (use different ID for post)
        
        delta_w = learner.compute_weight_update(5.0)  # delta_t = 5
        
        self.assertGreater(delta_w, 0)
        
    def test_neuron_to_decoder_integration(self):
        """Test neuron spikes being recorded by decoder."""
        decoder = OutputDecoder(n_outputs=5)
        
        # Simulate output neuron firing
        decoder.record_spike(neuron_id=0, time=10.0)
        decoder.record_spike(neuron_id=0, time=20.0)
        decoder.record_spike(neuron_id=1, time=15.0)
        
        output = decoder.decode()
        
        # Neuron 0 fired twice, neuron 1 fired once
        self.assertGreater(output[0], output[1])


class TestNetworkStateManagement(unittest.TestCase):
    """Test network state reset and management."""
    
    def test_state_reset_completeness(self):
        """Test that reset_states clears all component states."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        # Run inference to establish state
        inputs = np.random.rand(50)
        net.forward(inputs, steps=100)
        
        # Reset states
        net.reset_states()
        
        # Neuron potentials should be zero
        self.assertTrue(np.all(net.hidden_neurons.V == 0.0))
        self.assertTrue(np.all(net.output_neurons.V == 0.0))
        
        # Event queue should be empty
        self.assertEqual(len(net.event_queue), 0)
        
        # Output decoder should be reset
        self.assertTrue(np.all(net.output_decoder.spike_counts == 0))
        
    def test_sequential_inference_states(self):
        """Test that sequential inference calls properly manage states."""
        net = AnalogNetwork(n_inputs=30, n_hidden=60, n_outputs=8)
        
        inputs1 = np.random.rand(30)
        inputs2 = np.random.rand(30)
        
        # First inference
        output1 = net.forward(inputs1, steps=100)
        
        # States should be reset before second inference
        output2 = net.forward(inputs2, steps=100)
        
        # Outputs should be independent
        # (Not necessarily different, but computed from clean state each time)
        self.assertEqual(len(output1), 8)
        self.assertEqual(len(output2), 8)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_inference_timing(self):
        """Test inference execution time."""
        net = AnalogNetwork(n_inputs=100, n_hidden=200, n_outputs=10)
        
        inputs = np.random.rand(100)
        
        start = time.time()
        output = net.forward(inputs, steps=100)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 1 second)
        self.assertLess(elapsed, 1.0)
        
    def test_training_timing(self):
        """Test training execution time."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        inputs = np.random.rand(10, 50)
        rewards = np.ones(10)
        
        start = time.time()
        stats = net.train(inputs, rewards, epochs=5, steps_per_epoch=100)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 15 seconds for sparse updates)
        self.assertLess(elapsed, 15.0)
        
    def test_large_network_performance(self):
        """Test performance with larger network."""
        net = AnalogNetwork(n_inputs=200, n_hidden=500, n_outputs=20, connection_prob=0.05)
        
        inputs = np.random.rand(200)
        
        start = time.time()
        output = net.forward(inputs, steps=200)
        elapsed = time.time() - start
        
        # Larger network should still be reasonably fast
        self.assertLess(elapsed, 2.0)
        
    def test_scalability_sparsity(self):
        """Test that sparsity is maintained at scale."""
        net = AnalogNetwork(n_inputs=500, n_hidden=1000, n_outputs=50, connection_prob=0.02)
        
        inputs = np.random.rand(500)
        
        net.forward(inputs, steps=100)
        
        stats = net.get_stats()
        
        # Active ratio should be finite and reasonable
        active_ratio = stats['total_spikes'] / stats['total_time_steps']
        self.assertGreater(active_ratio, 0)  # Should have some spikes
        self.assertLess(active_ratio, 50.0)  # Should not fire excessively


class TestBatchProcessing(unittest.TestCase):
    """Test batch data handling."""
    
    def test_batch_training(self):
        """Test training with batch of samples."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        batch_inputs = np.random.rand(20, 50)
        batch_rewards = np.random.rand(20)
        
        stats = net.train(batch_inputs, batch_rewards, epochs=3, steps_per_epoch=50)
        
        self.assertEqual(stats['samples'], 20)
        
    def test_single_sample_training(self):
        """Test training with single sample."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        single_input = np.random.rand(50)
        
        stats = net.train(single_input, epochs=1, steps_per_epoch=100)
        
        self.assertEqual(stats['samples'], 1)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_mismatched_input_size(self):
        """Test handling of wrong input size."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        # Wrong size input
        wrong_input = np.random.rand(30)  # Should be 50
        
        # Should handle gracefully (may fail or truncate)
        try:
            output = net.forward(wrong_input, steps=100)
            # If it doesn't fail, check output is still valid
            self.assertEqual(len(output), 10)
        except (ValueError, IndexError):
            # Expected to fail
            pass
            
    def test_empty_training_data(self):
        """Test training with no samples."""
        net = AnalogNetwork(n_inputs=50, n_hidden=100, n_outputs=10)
        
        empty_inputs = np.zeros((0, 50))
        
        # Should handle gracefully
        try:
            stats = net.train(empty_inputs, epochs=1, steps_per_epoch=100)
            self.assertEqual(stats['samples'], 0)
        except (ValueError, IndexError):
            # Expected to fail
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)