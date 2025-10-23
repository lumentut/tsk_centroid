import time
from typing import TypedDict
from collections import defaultdict
from functools import wraps


class TimingStats(TypedDict):
    count: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float


def execution_time(func):
    """Decorator to measure execution time and store all results."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Initialize timing storage if needed
        if not hasattr(self, 'timing_results'):
            self.timing_results = defaultdict(list)
        
        # Store execution time in a list
        self.timing_results[func.__name__].append(execution_time)
        
        return result
    return wrapper


class Timing:
    """Mixin class to provide timing functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_results = {}

    def get_avg_execution_time(self, method_name: str) -> float:
        """Get the average execution time."""
        if not hasattr(self, "timing_results") or not self.timing_results:
            return 0.0
        return sum(self.timing_results.get(method_name, [])) / len(
            self.timing_results.get(method_name, [])
        )

    def get_execution_time_stats(self, method_name):
        """Get timing statistics for a method."""
        if method_name not in self.timing_results:
            return None
        
        times = self.timing_results[method_name]
        
        # Handle case where times might be a single float (defensive programming)
        if isinstance(times, (int, float)):
            times = [times]  # Convert single value to list
        
        if not times:  # Empty list
            return None
        
        return TimingStats(
            count=len(times),
            total_time=sum(times),
            average_time=sum(times) / len(times),
            min_time=min(times),
            max_time=max(times),
        )
