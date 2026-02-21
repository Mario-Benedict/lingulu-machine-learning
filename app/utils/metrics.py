"""
Simple in-memory metrics tracking for API latency monitoring.
Tracks p50, p90, p99 percentiles for performance monitoring.
"""
import time
import threading
import psutil
from collections import deque
from typing import Dict, List, Tuple, Optional
import statistics
from datetime import datetime


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
        self.inference_latencies = deque(maxlen=max_samples)  # Only inference time
        self.lock = threading.Lock()
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()
        
        # System metrics caching (to avoid frequent polling)
        self._system_metrics_cache = None
        self._system_metrics_cache_time = 0
        self._system_metrics_cache_ttl = 2.0  # Cache for 2 seconds
        
        # Check GPU availability
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            self.gpu_name = torch.cuda.get_device_name(0) if self.has_gpu else None
        except Exception:
            self.has_gpu = False
            self.gpu_name = None
        
    def record_inference_latency(self, latency_seconds: float):
        """Record inference latency measurement (model inference only)."""
        with self.lock:
            self.inference_latencies.append(latency_seconds)
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
            if not self.inference_latencies:
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
                    "latency_max_ms": 0.0,
                    "uptime_seconds": time.time() - self.start_time
                }
            
            sorted_latencies = sorted(self.inference_latencies)
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
                "latency_max_ms": round(max_ms, 2),
                "uptime_seconds": round(time.time() - self.start_time, 2)
            }
    
    def get_latencies_list(self) -> List[float]:
        """Get recorded latencies as list (in milliseconds).
        
        Returns:
            List of latency values in ms
        """
        with self.lock:
            return [lat * 1000 for lat in self.inference_latencies]
    
    def get_system_metrics(self) -> Dict:
        """Get system resource usage (CPU, RAM, GPU).
        Cached for 2 seconds to avoid frequent polling.
        """
        with self.lock:
            current_time = time.time()
            
            # Return cached result if still valid
            if (self._system_metrics_cache is not None and 
                current_time - self._system_metrics_cache_time < self._system_metrics_cache_ttl):
                return self._system_metrics_cache
        
        # Cache expired or not initialized, fetch new data
        try:
            cpu_percent = psutil.cpu_percent(interval=0)
            memory = psutil.virtual_memory()
            ram_percent = memory.percent
            ram_used_gb = memory.used / (1024**3)
            ram_total_gb = memory.total / (1024**3)
            
            result = {
                "cpu_percent": round(cpu_percent, 1),
                "ram_percent": round(ram_percent, 1),
                "ram_used_gb": round(ram_used_gb, 2),
                "ram_total_gb": round(ram_total_gb, 2),
                "has_gpu": self.has_gpu,
                "gpu_name": self.gpu_name
            }
            
            # Add GPU metrics if available
            if self.has_gpu:
                try:
                    import torch
                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    result.update({
                        "gpu_memory_allocated_gb": round(gpu_memory_allocated, 2),
                        "gpu_memory_reserved_gb": round(gpu_memory_reserved, 2),
                        "gpu_memory_total_gb": round(gpu_memory_total, 2),
                        "gpu_memory_percent": round((gpu_memory_reserved / gpu_memory_total) * 100, 1)
                    })
                except Exception:
                    pass
            
            # Update cache
            with self.lock:
                self._system_metrics_cache = result
                self._system_metrics_cache_time = current_time
            
            return result
        except Exception:
            return {
                "cpu_percent": 0,
                "ram_percent": 0,
                "ram_used_gb": 0,
                "ram_total_gb": 0,
                "has_gpu": False,
                "gpu_name": None
            }
    
    def reset_metrics(self):
        """Reset all metrics including latency data.
        Note: uptime (start_time) is NOT reset to maintain accurate uptime tracking.
        """
        with self.lock:
            self.inference_latencies.clear()
            self.total_requests = 0
            self.total_errors = 0
            # DO NOT reset start_time - uptime should continue running
    
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
    """Decorator to track function execution latency.
    Note: This tracks total request time. Use model's internal tracking for inference time.
    """
    def wrapper(*args, **kwargs):
        # Just pass through, actual tracking happens in model.predict()
        return func(*args, **kwargs)
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
