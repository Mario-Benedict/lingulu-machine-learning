"""
Metrics endpoint for monitoring API performance.
Provides p50, p90, p99 latency metrics and dashboard.
"""
from flask import Blueprint, jsonify, render_template_string

from app.utils.metrics import get_metrics_tracker
from app.utils.logger import get_logger

logger = get_logger(__name__)

metrics_bp = Blueprint('metrics', __name__, url_prefix='/api')


def create_metrics_routes():
    """
    Create metrics monitoring routes.
    
    Returns:
        Blueprint with registered routes
    """
    
    @metrics_bp.route('/metrics', methods=['GET'])
    def get_metrics():
        """
        Get current API metrics including latency percentiles.
        
        Returns:
            JSON with p50, p90, p99 latency and other performance metrics
        """
        tracker = get_metrics_tracker()
        metrics = tracker.get_metrics()
        
        logger.debug("Metrics requested")
        
        return jsonify({
            "status": "success",
            "metrics": metrics
        }), 200
    
    @metrics_bp.route('/metrics/reset', methods=['POST'])
    def reset_metrics():
        """
        Reset all metrics (for testing/debugging).
        
        Returns:
            JSON confirmation
        """
        tracker = get_metrics_tracker()
        tracker.reset_metrics()
        
        logger.info("Metrics reset")
        
        return jsonify({
            "status": "success",
            "message": "Metrics have been reset"
        }), 200
    
    @metrics_bp.route('/metrics/history', methods=['GET'])
    def get_metrics_history():
        """
        Get latency history for graphing.
        
        Returns:
            JSON with timestamp and latency pairs
        """
        tracker = get_metrics_tracker()
        history = tracker.get_latency_history()
        
        data = {
            'timestamps': [int(ts * 1000) for ts, _ in history],
            'latencies': [lat for _, lat in history]
        }
        return jsonify(data), 200
    
    @metrics_bp.route('/metrics/system', methods=['GET'])
    def get_system_metrics():
        """
        Get system resource usage (CPU, RAM, GPU).
        
        Returns:
            JSON with system metrics
        """
        tracker = get_metrics_tracker()
        system_metrics = tracker.get_system_metrics()
        return jsonify(system_metrics), 200
    
    @metrics_bp.route('/dashboard', methods=['GET'])
    def dashboard():
        """
        Performance monitoring dashboard - classic elegant design.
        """
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lingulu ML Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            background: #f5f5f0;
            color: #2c2c2c;
            padding: 30px;
            line-height: 1.6;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        .header {
            background: linear-gradient(135deg, #1a1a1a 0%, #3a3a3a 100%);
            color: #f5f5f0;
            padding: 40px;
            border-radius: 2px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .header h1 {
            font-size: 2.2em;
            font-weight: 400;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .header .subtitle {
            font-size: 1em;
            opacity: 0.8;
            font-family: 'Arial', sans-serif;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            background: #2d7a2d;
            color: white;
            border-radius: 2px;
            font-size: 0.85em;
            font-family: 'Arial', sans-serif;
            margin-left: 15px;
        }
        .section-title {
            font-size: 1.4em;
            font-weight: 400;
            margin: 30px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #2c2c2c;
        }
        .system-info {
            background: white;
            border: 1px solid #d0d0d0;
            padding: 20px 30px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .system-info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .system-info-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px dotted #d0d0d0;
        }
        .system-info-label {
            font-size: 0.9em;
            color: #666;
            font-family: 'Arial', sans-serif;
        }
        .system-info-value {
            font-weight: 600;
            color: #2c2c2c;
            font-family: 'Arial', sans-serif;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            border: 1px solid #d0d0d0;
            padding: 25px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            transition: box-shadow 0.3s;
        }
        .metric-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        }
        .metric-label {
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 10px;
            font-family: 'Arial', sans-serif;
        }
        .metric-value {
            font-size: 2.2em;
            font-weight: 300;
            color: #1a1a1a;
        }
        .metric-unit {
            font-size: 0.45em;
            color: #999;
            font-weight: 400;
        }
        .chart-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            background: white;
            border: 1px solid #d0d0d0;
            padding: 30px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        .chart-container.full-width {
            grid-column: 1 / -1;
        }
        .chart-title {
            font-size: 1.1em;
            font-weight: 400;
            color: #2c2c2c;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #d0d0d0;
            font-size: 0.9em;
            font-family: 'Arial', sans-serif;
        }
        @media (max-width: 1024px) {
            .chart-row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Lingulu ML Performance Dashboard</h1>
            <p class="subtitle">Real-time Inference Monitoring<span class="status-badge">ACTIVE</span></p>
        </div>
        
        <div class="system-info">
            <div class="system-info-grid">
                <div class="system-info-item">
                    <span class="system-info-label">Device</span>
                    <span class="system-info-value" id="device-name">Loading...</span>
                </div>
                <div class="system-info-item">
                    <span class="system-info-label">Uptime</span>
                    <span class="system-info-value" id="uptime">--</span>
                </div>
                <div class="system-info-item">
                    <span class="system-info-label">CPU Usage</span>
                    <span class="system-info-value" id="cpu-usage">--%</span>
                </div>
                <div class="system-info-item">
                    <span class="system-info-label">RAM Usage</span>
                    <span class="system-info-value" id="ram-usage">--</span>
                </div>
                <div class="system-info-item" id="gpu-usage-item" style="display:none;">
                    <span class="system-info-label">GPU Memory</span>
                    <span class="system-info-value" id="gpu-usage">--</span>
                </div>
            </div>
        </div>
        
        <h2 class="section-title">Inference Latency Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">P50 Latency</div>
                <div class="metric-value" id="p50">--<span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">P90 Latency</div>
                <div class="metric-value" id="p90">--<span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">P99 Latency</div>
                <div class="metric-value" id="p99">--<span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Latency</div>
                <div class="metric-value" id="mean">--<span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Requests</div>
                <div class="metric-value" id="total-requests">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Error Rate</div>
                <div class="metric-value" id="error-rate">--<span class="metric-unit">%</span></div>
            </div>
        </div>
        
        <h2 class="section-title">Performance Analysis</h2>
        <div class="chart-container full-width">
            <div class="chart-title">Inference Latency Timeline</div>
            <canvas id="latencyChart" height="80"></canvas>
        </div>
        
        <div class="chart-row">
            <div class="chart-container">
                <div class="chart-title">Latency Percentiles</div>
                <canvas id="distributionChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">System Resources</div>
                <canvas id="resourceChart"></canvas>
            </div>
        </div>
        
        <div class="footer">
            Dashboard auto-refreshes every 3 seconds | Lingulu Machine Learning &copy; 2026
        </div>
    </div>
    
    <script>
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        const distributionCtx = document.getElementById('distributionChart').getContext('2d');
        const resourceCtx = document.getElementById('resourceChart').getContext('2d');
        
        const latencyChart = new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Latency',
                    data: [],
                    borderColor: '#2c2c2c',
                    backgroundColor: 'rgba(44, 44, 44, 0.05)',
                    tension: 0.3,
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: (ctx) => 'Latency: ' + ctx.parsed.y.toFixed(2) + ' ms'
                        }
                    }
                },
                scales: {
                    x: { display: false },
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Milliseconds', font: { family: 'Arial', size: 11 } },
                        grid: { color: '#e0e0e0' }
                    }
                }
            }
        });
        
        const distributionChart = new Chart(distributionCtx, {
            type: 'bar',
            data: {
                labels: ['P50', 'P90', 'P99', 'Mean'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: ['#4a4a4a', '#6a6a6a', '#8a8a8a', '#5a5a5a'],
                    borderColor: '#2c2c2c',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Milliseconds', font: { family: 'Arial', size: 11 } },
                        grid: { color: '#e0e0e0' }
                    },
                    x: { grid: { display: false } }
                }
            }
        });
        
        const resourceChart = new Chart(resourceCtx, {
            type: 'doughnut',
            data: {
                labels: ['CPU', 'RAM', 'GPU'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#4a4a4a', '#6a6a6a', '#8a8a8a'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { font: { family: 'Arial' } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ctx.label + ': ' + ctx.parsed.toFixed(1) + '%'
                        }
                    }
                }
            }
        });
        
        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return hours + 'h ' + minutes + 'm';
        }
        
        async function updateDashboard() {
            try {
                const [metricsRes, historyRes, systemRes] = await Promise.all([
                    fetch('/api/metrics'),
                    fetch('/api/metrics/history'),
                    fetch('/api/metrics/system')
                ]);
                
                const metricsData = await metricsRes.json();
                const metrics = metricsData.metrics;
                const history = await historyRes.json();
                const system = await systemRes.json();
                
                // Update metric cards
                document.getElementById('p50').innerHTML = metrics.latency_p50_ms.toFixed(1) + '<span class="metric-unit">ms</span>';
                document.getElementById('p90').innerHTML = metrics.latency_p90_ms.toFixed(1) + '<span class="metric-unit">ms</span>';
                document.getElementById('p99').innerHTML = metrics.latency_p99_ms.toFixed(1) + '<span class="metric-unit">ms</span>';
                document.getElementById('mean').innerHTML = metrics.latency_mean_ms.toFixed(1) + '<span class="metric-unit">ms</span>';
                document.getElementById('total-requests').textContent = metrics.total_requests;
                document.getElementById('error-rate').innerHTML = metrics.error_rate.toFixed(1) + '<span class="metric-unit">%</span>';
                
                // Update system info
                document.getElementById('uptime').textContent = formatUptime(metrics.uptime_seconds);
                document.getElementById('cpu-usage').textContent = system.cpu_percent + '%';
                document.getElementById('ram-usage').textContent = system.ram_used_gb.toFixed(1) + ' / ' + system.ram_total_gb.toFixed(1) + ' GB';
                
                if (system.has_gpu) {
                    document.getElementById('gpu-usage-item').style.display = 'flex';
                    document.getElementById('device-name').textContent = system.gpu_name;
                    document.getElementById('gpu-usage').textContent = 
                        system.gpu_memory_reserved_gb.toFixed(1) + ' / ' + system.gpu_memory_total_gb.toFixed(1) + ' GB';
                    
                    resourceChart.data.datasets[0].data = [
                        system.cpu_percent,
                        system.ram_percent,
                        system.gpu_memory_percent || 0
                    ];
                } else {
                    document.getElementById('device-name').textContent = 'CPU Only';
                    resourceChart.data.labels = ['CPU', 'RAM'];
                    resourceChart.data.datasets[0].data = [system.cpu_percent, system.ram_percent];
                }
                resourceChart.update('none');
                
                // Update distribution chart
                distributionChart.data.datasets[0].data = [
                    metrics.latency_p50_ms,
                    metrics.latency_p90_ms,
                    metrics.latency_p99_ms,
                    metrics.latency_mean_ms
                ];
                distributionChart.update('none');
                
                // Update latency timeline
                const displayCount = Math.min(50, history.latencies.length);
                latencyChart.data.labels = Array(displayCount).fill('');
                latencyChart.data.datasets[0].data = history.latencies.slice(-displayCount);
                latencyChart.update('none');
                
            } catch (error) {
                console.error('Failed to update dashboard:', error);
            }
        }
        
        updateDashboard();
        setInterval(updateDashboard, 3000);
    </script>
</body>
</html>"""
        return render_template_string(html)
    
    return metrics_bp
