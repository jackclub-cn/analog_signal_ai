"""
Basic usage example for Analog Signal AI Framework.

Demonstrates:
1. Network creation and configuration
2. Forward inference
3. STDP training
4. Performance comparison
"""

import sys
sys.path.insert(0, '/root/.hermes/kanban/workspaces/t_557d8b04')

import numpy as np
from analog_signal_ai import AnalogNetwork


def example_basic_inference():
    """Example 1: Basic forward inference."""
    print("=" * 60)
    print("Example 1: Basic Forward Inference")
    print("=" * 60)
    
    # Create network
    net = AnalogNetwork(
        n_inputs=10,
        n_hidden=20,
        n_outputs=5,
        neuron_threshold=1.0,
        neuron_decay=0.9,
        connection_prob=0.15
    )
    
    # Generate random input
    inputs = np.random.rand(10)
    print(f"\nInput shape: {inputs.shape}")
    print(f"Input values: {inputs[:5]}...")
    
    # Run inference
    output = net.forward(inputs, steps=100)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output values: {output}")
    print(f"Predicted class: {np.argmax(output)}")
    
    # Check statistics
    stats = net.get_stats()
    print(f"\nNetwork statistics:")
    print(f"  Total spikes: {stats['total_spikes']}")
    print(f"  Time steps: {stats['total_time_steps']}")
    print(f"  Active ratio: {stats['active_ratio']:.3f}")
    print(f"  Hidden connections: {stats['input_connections']['n_connections']}")
    print(f"  Output connections: {stats['output_connections']['n_connections']}")
    

def example_pattern_classification():
    """Example 2: Simple pattern classification."""
    print("\n" + "=" * 60)
    print("Example 2: Pattern Classification")
    print("=" * 60)
    
    # Create larger network for pattern recognition
    net = AnalogNetwork(
        n_inputs=25,    # 5x5 patterns
        n_hidden=50,
        n_outputs=3,    # 3 pattern classes
        connection_prob=0.2
    )
    
    # Create simple patterns
    patterns = [
        np.zeros(25),  # Pattern 0: all zeros
        np.ones(25),   # Pattern 1: all ones
        np.array([1 if i % 2 == 0 else 0 for i in range(25)])  # Pattern 2: alternating
    ]
    
    print("\nTesting pattern classification:")
    for i, pattern in enumerate(patterns):
        output = net.forward(pattern, steps=100)
        predicted = np.argmax(output)
        print(f"  Pattern {i}: predicted class {predicted}, confidence: {output[predicted]:.3f}")
        

def example_training():
    """Example 3: STDP training."""
    print("\n" + "=" * 60)
    print("Example 3: STDP Training")
    print("=" * 60)
    
    # Create network
    net = AnalogNetwork(
        n_inputs=10,
        n_hidden=20,
        n_outputs=3,
        connection_prob=0.15
    )
    
    # Generate training data
    n_samples = 10
    inputs = np.random.rand(n_samples, 10)
    rewards = np.random.rand(n_samples)  # Random rewards
    
    print(f"\nTraining on {n_samples} samples...")
    
    # Train network
    stats = net.train(
        inputs=inputs,
        rewards=rewards,
        epochs=2,
        steps_per_epoch=50
    )
    
    print(f"Training complete:")
    print(f"  Epochs: {stats['epochs']}")
    print(f"  Samples: {stats['samples']}")
    print(f"  Weight updates: {stats['weight_updates']}")
    
    # Test after training
    test_input = np.random.rand(10)
    output = net.forward(test_input, steps=100)
    print(f"\nTest output after training: {output}")
    

def example_save_load():
    """Example 4: Save and load network."""
    print("\n" + "=" * 60)
    print("Example 4: Save and Load Network")
    print("=" * 60)
    
    # Create and train network
    net = AnalogNetwork(n_inputs=10, n_hidden=20, n_outputs=5)
    
    # Run some inference
    inputs = np.random.rand(10)
    output1 = net.forward(inputs, steps=50)
    
    # Save network
    save_path = '/tmp/demo_network.pkl'
    net.save(save_path)
    print(f"\nNetwork saved to: {save_path}")
    
    # Load into new network
    net2 = AnalogNetwork(n_inputs=10, n_hidden=20, n_outputs=5)
    net2.load(save_path)
    print("Network loaded successfully")
    
    # Verify stats
    print(f"Original inference calls: {net.stats['inference_calls']}")
    print(f"Loaded inference calls: {net2.stats['inference_calls']}")
    

def example_performance_comparison():
    """Example 5: Performance comparison."""
    print("\n" + "=" * 60)
    print("Example 5: Performance Analysis")
    print("=" * 60)
    
    import time
    
    # Compare different network sizes
    sizes = [
        (100, 50, 10),
        (500, 200, 20),
        (1000, 500, 50)
    ]
    
    print("\nPerformance comparison for different network sizes:")
    print(f"{'Size':<20} {'Time (ms)':<15} {'Spikes':<10} {'Active %':<10}")
    print("-" * 60)
    
    for n_in, n_hidden, n_out in sizes:
        net = AnalogNetwork(
            n_inputs=n_in,
            n_hidden=n_hidden,
            n_outputs=n_out,
            connection_prob=0.1
        )
        
        inputs = np.random.rand(n_in)
        
        # Time the inference
        start = time.time()
        output = net.forward(inputs, steps=50)
        elapsed = (time.time() - start) * 1000
        
        stats = net.get_stats()
        active_pct = (stats['total_spikes'] / (stats['total_time_steps'] * (n_hidden + n_out))) * 100
        
        print(f"{n_in}-{n_hidden}-{n_out:<10} {elapsed:<15.2f} {stats['total_spikes']:<10} {active_pct:<10.2f}%")
        

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("ANALOG SIGNAL AI FRAMEWORK - EXAMPLES")
    print("=" * 60)
    
    example_basic_inference()
    example_pattern_classification()
    example_training()
    example_save_load()
    example_performance_comparison()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
