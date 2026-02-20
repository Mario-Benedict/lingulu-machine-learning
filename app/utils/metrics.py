"""
Simple in-memory metrics tracking for API latency monitoring.
Tracks p50, p90, p99 percentiles for performance monitoring.
"""
import time
import threading
from collections import deque
from typing import Dict, List
import statistics


class MetricsTracker:
    """
    Thread-safe metrics tracker for API latency monitoring.
    Maintains a sliding window of recent latencies.
    """
    
    def __init__(self, max_samples: int = 1000):
        """
        Initialize metrics tracker.
        
        Args:
            max_samples: Maximum number of samples to keep in memory
        """
        self.max_samples = max_samples
        self.latencies = deque(maxlen=max_samples)
        self.lock = threading.Lock()
        self.total_requests = 0
        self.total_errors = 0
        
    def record_latency(self, latency_seconds: float):
        """Record a latency measurement."""
        with self.lock:
            self.latencies.append(latency_seconds)
            self.total_requests += 1
    
    def record_error(self):
        """Record an error occurrence."""
        with self.lock:
            self.total_errors += 1
    
    def get_metrics(self) -> Dict:
        """
        Get current metrics including percentiles.
        
        Returns:
            Dictionary containing p50, p90, p99 and other metrics
        """
        with self.lock:
            if not self.latencies:
                return {
                    "total_requests": self.total_requests,
                    "total_errors": self.total_errors,
                    "error_rate": 0.0,
                    "samples_count": 0,
                    "latency_p50_ms": 0.0,
                    "latency_p90_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "latency_mean_ms": 0.0,
                    "latency_min_ms": 0.0,
                    "latency_max_ms": 0.0
                }
            
            sorted_latencies = sorted(self.latencies)
            count = len(sorted_latencies)
            
            # Calculate percentiles
            p50 = self._percentile(sorted_latencies, 50)
            p90 = self._percentile(sorted_latencies, 90)
            p99 = self._percentile(sorted_latencies, 99)
            
            # Convert to milliseconds
            p50_ms = p50 * 1000
            p90_ms = p90 * 1000
            p99_ms = p99 * 1000
            mean_ms = statistics.mean(sorted_latencies) * 1000
            min_ms = min(sorted_latencies) * 1000
            max_ms = max(sorted_latencies) * 1000
            
            error_rate = (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0.0
            
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": round(error_rate, 2),
                "samples_count": count,
                "latency_p50_ms": round(p50_ms, 2),
                "latency_p90_ms": round(p90_ms, 2),
                "latency_p99_ms": round(p99_ms, 2),
                "latency_mean_ms": round(mean_ms, 2),
                "latency_min_ms": round(min_ms, 2),
                "latency_max_ms": round(max_ms, 2)
            }
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self.lock:
            self.latencies.clear()
            self.total_requests = 0
            self.total_errors = 0
    
    @staticmethod
    def _percentile(sorted_data: List[float], percentile: int) -> float:
        """
        Calculate percentile from sorted data.
        
        Args:
            sorted_data: Sorted list of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not sorted_data:
            return 0.0
        
        k = (len(sorted_data) - 1) * (percentile / 100)
        f = int(k)
        c = f + 1
        
        if c >= len(sorted_data):
            return sorted_data[-1]
        
        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)
        
        return d0 + d1


# Global metrics tracker instance
_metrics_tracker = None


def get_metrics_tracker() -> MetricsTracker:
    """Get or create the global metrics tracker instance."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker


def track_latency(func):
    """
    Decorator to track function execution latency.
    
    Usage:
        @track_latency
        def my_endpoint():
            ...
    """
    def wrapper(*args, **kwargs):
        tracker = get_metrics_tracker()
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            latency = time.time() - start_time
            tracker.record_latency(latency)
            return result
        except Exception as e:
            tracker.record_error()
            raise e
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
