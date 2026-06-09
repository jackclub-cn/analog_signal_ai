"""
EventQueue - Manages time-ordered spike events for event-driven simulation.

Uses a priority queue (heap) to efficiently manage spike events:
- Events are ordered by time
- Supports O(log K) insertion and extraction
- Enables sparse, event-driven computation
"""

import heapq
from typing import List, Tuple, Optional


class EventQueue:
    """
    Priority queue for managing spike events in temporal order.
    
    Each event is a tuple: (time, neuron_id, weight)
    - time: when the event should be processed
    - neuron_id: target neuron receiving the spike
    - weight: synaptic weight of the connection
    
    The queue maintains events sorted by time, enabling efficient
    event-driven simulation where we only process active events.
    
    Complexity:
        - push: O(log K) where K = events in queue
        - pop: O(log K)
        - peek: O(1)
    """
    
    def __init__(self):
        """Initialize empty event queue."""
        self.queue: List[Tuple[float, int, float]] = []
        
    def push(self, time: float, neuron_id: int, weight: float):
        """
        Schedule a new event.
        
        Args:
            time: Event time in ms
            neuron_id: Target neuron ID
            weight: Synaptic weight
        """
        heapq.heappush(self.queue, (time, neuron_id, weight))
        
    def pop(self) -> Optional[Tuple[float, int, float]]:
        """
        Get and remove the next event by time.
        
        Returns:
            Tuple (time, neuron_id, weight) or None if queue is empty
        """
        if self.queue:
            return heapq.heappop(self.queue)
        return None
    
    def peek_time(self) -> float:
        """
        Check the time of the next event without removing it.
        
        Returns:
            Time of next event, or inf if queue is empty
        """
        return self.queue[0][0] if self.queue else float('inf')
    
    def pop_events_at_time(self, time: float) -> List[Tuple[int, float]]:
        """
        Get all events at a specific time.
        
        Args:
            time: Target time in ms
            
        Returns:
            List of (neuron_id, weight) tuples at this time
        """
        events = []
        while self.queue and abs(self.queue[0][0] - time) < 1e-6:
            _, neuron_id, weight = heapq.heappop(self.queue)
            events.append((neuron_id, weight))
        return events
    
    def clear(self):
        """Remove all events from queue."""
        self.queue.clear()
        
    def __len__(self) -> int:
        """Return number of events in queue."""
        return len(self.queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.queue) == 0
    
    def __repr__(self):
        return f"EventQueue(n_events={len(self.queue)})"
