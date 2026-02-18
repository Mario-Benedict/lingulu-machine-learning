"""
AWS CloudWatch metrics middleware for tracking application performance.
"""
import time
from functools import wraps
from typing import List, Callable
import boto3
from flask import request

from app.utils.logger import get_logger

logger = get_logger(__name__)


class CloudWatchMetrics:
    """CloudWatch metrics handler for tracking latency and performance."""
    
    def __init__(self, namespace: str, enabled: bool = True):
        """
        Initialize CloudWatch metrics handler.
        
        Args:
            namespace: CloudWatch namespace for metrics
            enabled: Enable/disable metrics collection
        """
        self.namespace = namespace
        self.enabled = enabled
        self.latencies: List[float] = []
        
        if self.enabled:
            try:
                self.cloudwatch = boto3.client('cloudwatch')
                logger.info(f"CloudWatch metrics initialized with namespace: {namespace}")
            except Exception as e:
                logger.warning(f"Failed to initialize CloudWatch client: {e}")
                self.enabled = False
        else:
            logger.info("CloudWatch metrics disabled")
    
    def track_latency(self, endpoint: str) -> Callable:
        """
        Decorator to track endpoint latency and send to CloudWatch.
        
        Args:
            endpoint: Name of the endpoint being tracked
            
        Returns:
            Decorated function
        """
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    latency = time.time() - start_time
                    self._record_latency(endpoint, latency)
            
            return wrapper
        return decorator
    
    def _record_latency(self, endpoint: str, latency: float) -> None:
        """
        Record latency for an endpoint.
        
        Args:
            endpoint: Name of the endpoint
            latency: Latency in seconds
        """
        if not self.enabled:
            return
        
        try:
            # Store latency for batch calculation
            self.latencies.append(latency)
            
            # Calculate percentiles if we have enough data
            if len(self.latencies) >= 100:
                self._send_percentile_metrics(endpoint)
                self.latencies.clear()
            
            # Send individual latency metric
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        'MetricName': 'RequestLatency',
                        'Value': latency,
                        'Unit': 'Seconds',
                        'Dimensions': [
                            {'Name': 'Endpoint', 'Value': endpoint},
                            {'Name': 'Method', 'Value': request.method}
                        ]
                    }
                ]
            )
            
            logger.debug(f"Recorded latency for {endpoint}: {latency:.3f}s")
            
        except Exception as e:
            logger.error(f"Failed to send CloudWatch metrics: {e}")
    
    def _send_percentile_metrics(self, endpoint: str) -> None:
        """
        Calculate and send percentile metrics to CloudWatch.
        
        Args:
            endpoint: Name of the endpoint
        """
        if not self.latencies:
            return
        
        try:
            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)
            
            # Calculate percentiles
            p50 = sorted_latencies[int(n * 0.50)]
            p90 = sorted_latencies[int(n * 0.90)]
            p99 = sorted_latencies[int(n * 0.99)]
            
            # Send percentile metrics
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        'MetricName': 'LatencyP50',
                        'Value': p50,
                        'Unit': 'Seconds',
                        'Dimensions': [{'Name': 'Endpoint', 'Value': endpoint}]
                    },
                    {
                        'MetricName': 'LatencyP90',
                        'Value': p90,
                        'Unit': 'Seconds',
                        'Dimensions': [{'Name': 'Endpoint', 'Value': endpoint}]
                    },
                    {
                        'MetricName': 'LatencyP99',
                        'Value': p99,
                        'Unit': 'Seconds',
                        'Dimensions': [{'Name': 'Endpoint', 'Value': endpoint}]
                    }
                ]
            )
            
            logger.info(
                f"Sent percentile metrics for {endpoint}: "
                f"P50={p50:.3f}s, P90={p90:.3f}s, P99={p99:.3f}s"
            )
            
        except Exception as e:
            logger.error(f"Failed to send percentile metrics: {e}")
    
    def record_custom_metric(
        self, 
        metric_name: str, 
        value: float, 
        unit: str = 'None',
        dimensions: dict = None
    ) -> None:
        """
        Record a custom metric to CloudWatch.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: CloudWatch unit (e.g., 'Count', 'Seconds', 'Bytes')
            dimensions: Additional dimensions for the metric
        """
        if not self.enabled:
            return
        
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )
            
            logger.debug(f"Recorded custom metric {metric_name}: {value}")
            
        except Exception as e:
            logger.error(f"Failed to send custom metric: {e}")


def setup_cloudwatch_metrics(namespace: str, enabled: bool = True) -> CloudWatchMetrics:
    """
    Setup CloudWatch metrics for the application.
    
    Args:
        namespace: CloudWatch namespace
        enabled: Enable/disable metrics
        
    Returns:
        CloudWatchMetrics instance
    """
    return CloudWatchMetrics(namespace=namespace, enabled=enabled)
