"""
Prometheus metrics configuration and middleware.
"""
from prometheus_flask_exporter import PrometheusMetrics
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)


def setup_metrics(app, buckets: List[float] = None):
    """
    Set up Prometheus metrics for the Flask application.
    
    Args:
        app: Flask application instance
        buckets: Histogram buckets for latency measurements
        
    Returns:
        Tuple of (metrics, model_latency_metric)
    """
    if buckets is None:
        buckets = [1.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, float("inf")]
    
    logger.info("Initializing Prometheus metrics")
    
    # Initialize Prometheus metrics with endpoint grouping
    metrics = PrometheusMetrics(app, group_by='endpoint')
    
    # Create custom histogram for model latency
    # This allows tracking p50, p90, p99 percentiles
    model_latency = metrics.histogram(
        'model_latency_seconds',
        'Wav2Vec2 Inference Latency',
        labels={'model': 'wav2vec2-pronunciation'},
        buckets=buckets
    )
    
    logger.info(
        f"Prometheus metrics initialized with buckets: {buckets}"
    )
    
    return metrics, model_latency
