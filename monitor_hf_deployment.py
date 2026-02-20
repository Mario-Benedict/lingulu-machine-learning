"""
Monitoring script for Lingulu ML API deployed on Hugging Face Spaces.
Continuously monitors API metrics and displays latency statistics.

Usage:
    python monitor_hf_deployment.py <your-space-url>
    
Example:
    python monitor_hf_deployment.py https://mario-benedict-lingulu-ml-api.hf.space
"""
import sys
import time
import requests
from datetime import datetime
from typing import Dict, Optional


class APIMonitor:
    """Monitor API metrics and display statistics."""
    
    def __init__(self, base_url: str):
        """
        Initialize API monitor.
        
        Args:
            base_url: Base URL of the deployed API
        """
        self.base_url = base_url.rstrip('/')
        self.metrics_url = f"{self.base_url}/api/metrics"
        self.health_url = f"{self.base_url}/api/model/health"
        
    def check_health(self) -> bool:
        """
        Check if API is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = requests.get(self.health_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'healthy'
        except Exception as e:
            print(f"❌ Health check failed: {e}")
        return False
    
    def get_metrics(self) -> Optional[Dict]:
        """
        Fetch current metrics from API.
        
        Returns:
            Metrics dictionary or None if failed
        """
        try:
            response = requests.get(self.metrics_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('metrics')
        except Exception as e:
            print(f"❌ Failed to fetch metrics: {e}")
        return None
    
    def display_metrics(self, metrics: Dict):
        """
        Display metrics in a formatted way.
        
        Args:
            metrics: Metrics dictionary
        """
        print("\n" + "="*60)
        print(f"📊 Metrics at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Request Statistics
        print("\n📈 Request Statistics:")
        print(f"  Total Requests:  {metrics['total_requests']:,}")
        print(f"  Total Errors:    {metrics['total_errors']:,}")
        print(f"  Error Rate:      {metrics['error_rate']:.2f}%")
        print(f"  Samples Tracked: {metrics['samples_count']:,}")
        
        # Latency Statistics
        print("\n⚡ Latency Statistics (milliseconds):")
        print(f"  P50 (Median):    {metrics['latency_p50_ms']:>8.2f} ms")
        print(f"  P90:             {metrics['latency_p90_ms']:>8.2f} ms")
        print(f"  P99:             {metrics['latency_p99_ms']:>8.2f} ms")
        print(f"  Mean:            {metrics['latency_mean_ms']:>8.2f} ms")
        print(f"  Min:             {metrics['latency_min_ms']:>8.2f} ms")
        print(f"  Max:             {metrics['latency_max_ms']:>8.2f} ms")
        
        # Alerts
        print("\n🚨 Alerts:")
        alerts = []
        
        if metrics['error_rate'] > 5:
            alerts.append(f"⚠️  High error rate: {metrics['error_rate']:.2f}%")
        
        if metrics['latency_p99_ms'] > 1000:
            alerts.append(f"⚠️  P99 latency exceeds 1 second: {metrics['latency_p99_ms']:.2f}ms")
        
        if metrics['latency_p50_ms'] > 500:
            alerts.append(f"⚠️  P50 latency exceeds 500ms: {metrics['latency_p50_ms']:.2f}ms")
        
        if not alerts:
            print("  ✅ All metrics look good!")
        else:
            for alert in alerts:
                print(f"  {alert}")
        
        print("="*60)
    
    def run(self, interval: int = 60):
        """
        Run monitoring loop.
        
        Args:
            interval: Seconds between metric checks
        """
        print(f"🚀 Starting API Monitor")
        print(f"📍 Target: {self.base_url}")
        print(f"⏱️  Check interval: {interval} seconds")
        print(f"⌨️  Press Ctrl+C to stop\n")
        
        # Initial health check
        if not self.check_health():
            print("❌ API is not healthy. Please check deployment.")
            return
        
        print("✅ API is healthy. Starting monitoring...\n")
        
        try:
            while True:
                metrics = self.get_metrics()
                
                if metrics:
                    self.display_metrics(metrics)
                else:
                    print(f"⚠️  Could not fetch metrics at {datetime.now()}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python monitor_hf_deployment.py <space-url>")
        print("\nExample:")
        print("  python monitor_hf_deployment.py https://mario-benedict-lingulu-ml-api.hf.space")
        sys.exit(1)
    
    base_url = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    monitor = APIMonitor(base_url)
    monitor.run(interval)


if __name__ == "__main__":
    main()
