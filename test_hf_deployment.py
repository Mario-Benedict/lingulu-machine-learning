"""
Testing script for Lingulu ML API deployed on Hugging Face Spaces.
Tests all endpoints and verifies functionality.

Usage:
    python test_hf_deployment.py <your-space-url> [audio-file]
    
Example:
    python test_hf_deployment.py https://mario-benedict-lingulu-ml-api.hf.space test_audio.wav
"""
import sys
import requests
import time
from pathlib import Path


class APITester:
    """Test deployed API endpoints."""
    
    def __init__(self, base_url: str):
        """
        Initialize API tester.
        
        Args:
            base_url: Base URL of the deployed API
        """
        self.base_url = base_url.rstrip('/')
        self.health_url = f"{self.base_url}/api/model/health"
        self.predict_url = f"{self.base_url}/api/model/predict"
        self.metrics_url = f"{self.base_url}/api/metrics"
        
    def test_health(self) -> bool:
        """
        Test health endpoint.
        
        Returns:
            True if test passed
        """
        print("\n🧪 Testing Health Endpoint...")
        print(f"   URL: {self.health_url}")
        
        try:
            start = time.time()
            response = requests.get(self.health_url, timeout=30)
            latency = (time.time() - start) * 1000
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Latency: {latency:.2f}ms")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Response: {data}")
                
                if data.get('status') == 'healthy':
                    print("   ✅ Health check PASSED")
                    return True
                else:
                    print("   ❌ API is not healthy")
                    return False
            else:
                print(f"   ❌ Unexpected status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Health check FAILED: {e}")
            return False
    
    def test_predict(self, audio_file: str = None, text: str = None) -> bool:
        """
        Test prediction endpoint.
        
        Args:
            audio_file: Path to audio file (optional)
            text: Reference text for GOP (optional)
            
        Returns:
            True if test passed
        """
        print("\n🧪 Testing Predict Endpoint...")
        print(f"   URL: {self.predict_url}")
        
        if not audio_file:
            print("   ⚠️  No audio file provided, skipping predict test")
            return True
        
        audio_path = Path(audio_file)
        if not audio_path.exists():
            print(f"   ❌ Audio file not found: {audio_file}")
            return False
        
        try:
            print(f"   Audio File: {audio_file}")
            if text:
                print(f"   Reference Text: '{text}'")
            
            files = {'file': open(audio_path, 'rb')}
            data = {'text': text} if text else {}
            
            print("   Sending request...")
            start = time.time()
            response = requests.post(
                self.predict_url,
                files=files,
                data=data,
                timeout=300  # 5 minutes for GPU processing
            )
            latency = (time.time() - start) * 1000
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Latency: {latency:.2f}ms")
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n   📝 Response:")
                print(f"      Status: {data.get('status')}")
                print(f"      Filename: {data.get('filename')}")
                print(f"      Transcription: {data.get('transcription')}")
                print(f"      Model Latency: {data.get('latency_seconds')}s")
                
                if 'pronunciation_assessment' in data:
                    assessment = data['pronunciation_assessment']
                    print(f"\n      🎯 Pronunciation Assessment:")
                    print(f"         Overall Score: {assessment['overall_score']:.2f}")
                    print(f"         Accuracy: {assessment['accuracy_score']:.2f}")
                
                print("   ✅ Prediction test PASSED")
                return True
            else:
                print(f"   ❌ Prediction FAILED: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Prediction test FAILED: {e}")
            return False
    
    def test_metrics(self) -> bool:
        """
        Test metrics endpoint.
        
        Returns:
            True if test passed
        """
        print("\n🧪 Testing Metrics Endpoint...")
        print(f"   URL: {self.metrics_url}")
        
        try:
            start = time.time()
            response = requests.get(self.metrics_url, timeout=10)
            latency = (time.time() - start) * 1000
            
            print(f"   Status Code: {response.status_code}")
            print(f"   Latency: {latency:.2f}ms")
            
            if response.status_code == 200:
                data = response.json()
                metrics = data.get('metrics', {})
                
                print(f"\n   📊 Current Metrics:")
                print(f"      Total Requests: {metrics.get('total_requests', 0)}")
                print(f"      Error Rate: {metrics.get('error_rate', 0):.2f}%")
                print(f"      P50 Latency: {metrics.get('latency_p50_ms', 0):.2f}ms")
                print(f"      P90 Latency: {metrics.get('latency_p90_ms', 0):.2f}ms")
                print(f"      P99 Latency: {metrics.get('latency_p99_ms', 0):.2f}ms")
                
                print("   ✅ Metrics test PASSED")
                return True
            else:
                print(f"   ❌ Unexpected status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"   ❌ Metrics test FAILED: {e}")
            return False
    
    def run_all_tests(self, audio_file: str = None, text: str = None):
        """
        Run all tests.
        
        Args:
            audio_file: Path to audio file for testing
            text: Reference text for GOP testing
        """
        print("="*70)
        print("🚀 Starting API Tests")
        print(f"📍 Target: {self.base_url}")
        print("="*70)
        
        results = []
        
        # Test 1: Health
        results.append(("Health Check", self.test_health()))
        
        # Test 2: Prediction (if audio provided)
        if audio_file:
            results.append(("Prediction", self.test_predict(audio_file, text)))
        
        # Test 3: Metrics
        results.append(("Metrics", self.test_metrics()))
        
        # Summary
        print("\n" + "="*70)
        print("📋 Test Summary")
        print("="*70)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name:<20} {status}")
        
        print("="*70)
        print(f"   Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("   🎉 All tests PASSED!")
        else:
            print("   ⚠️  Some tests FAILED")
        
        print("="*70)
        
        return passed == total


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python test_hf_deployment.py <space-url> [audio-file] [reference-text]")
        print("\nExamples:")
        print("  # Test health and metrics only")
        print("  python test_hf_deployment.py https://mario-benedict-lingulu-ml-api.hf.space")
        print("\n  # Test with audio file")
        print("  python test_hf_deployment.py https://mario-benedict-lingulu-ml-api.hf.space test.wav")
        print("\n  # Test with audio and pronunciation assessment")
        print("  python test_hf_deployment.py https://mario-benedict-lingulu-ml-api.hf.space test.wav \"hello world\"")
        sys.exit(1)
    
    base_url = sys.argv[1]
    audio_file = sys.argv[2] if len(sys.argv) > 2 else None
    text = sys.argv[3] if len(sys.argv) > 3 else None
    
    tester = APITester(base_url)
    success = tester.run_all_tests(audio_file, text)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
