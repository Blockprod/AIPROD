#!/usr/bin/env python3
"""
Script de monitoring pour AIPROD
Récupère et affiche les métriques en temps réel
"""

import os
import time
import requests
from typing import Dict, Any

# Configuration
API_BASE_URL = os.getenv("AIPROD_API_URL", "http://localhost:8000")
REFRESH_INTERVAL = int(os.getenv("MONITOR_REFRESH", "5"))


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_health() -> Dict[str, Any]:
    """Get health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_metrics() -> Dict[str, Any]:
    """Get pipeline metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_alerts() -> Dict[str, Any]:
    """Get active alerts"""
    try:
        response = requests.get(f"{API_BASE_URL}/alerts", timeout=5)
        return response.json()
    except Exception as e:
        return {"alerts": [], "error": str(e)}


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics for display"""
    if "error" in metrics:
        return f"❌ Error fetching metrics: {metrics['error']}"
    
    output = []
    output.append("📊 PIPELINE METRICS")
    output.append("=" * 50)
    
    total = metrics.get("total_executions", 0)
    output.append(f"Total Executions: {total}")
    
    if total > 0:
        avg_latency = metrics.get("average_latency", 0)
        avg_cost = metrics.get("average_cost", 0)
        avg_quality = metrics.get("average_quality", 0)
        
        output.append(f"Average Latency: {avg_latency:.2f}s")
        output.append(f"Average Cost: ${avg_cost:.2f}")
        output.append(f"Average Quality: {avg_quality:.2f}")
    else:
        output.append("No executions yet")
    
    return "\n".join(output)


def format_alerts(alerts: Dict[str, Any]) -> str:
    """Format alerts for display"""
    output = []
    output.append("\n🚨 ACTIVE ALERTS")
    output.append("=" * 50)
    
    if "error" in alerts:
        output.append(f"❌ Error fetching alerts: {alerts['error']}")
        return "\n".join(output)
    
    alert_list = alerts.get("alerts", [])
    if not alert_list:
        output.append("✅ No active alerts")
    else:
        for alert in alert_list:
            severity = alert.get("severity", "INFO")
            message = alert.get("message", "Unknown alert")
            emoji = "🔴" if severity == "CRITICAL" else "🟡"
            output.append(f"{emoji} [{severity}] {message}")
    
    return "\n".join(output)


def format_health(health: Dict[str, Any]) -> str:
    """Format health status for display"""
    output = []
    output.append("\n💚 HEALTH STATUS")
    output.append("=" * 50)
    
    status = health.get("status", "unknown")
    if status == "healthy":
        output.append("✅ System is healthy")
    elif status == "error":
        output.append(f"❌ {health.get('message', 'Unknown error')}")
    else:
        output.append(f"⚠️ Status: {status}")
    
    return "\n".join(output)


def main():
    """Main monitoring loop"""
    print("🚀 Starting AIPROD Monitor...")
    print(f"API URL: {API_BASE_URL}")
    print(f"Refresh interval: {REFRESH_INTERVAL}s")
    print("\nPress Ctrl+C to exit\n")
    time.sleep(2)
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("=" * 50)
            print("AIPROD - Real-time Monitor")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)
            
            # Fetch data
            health = get_health()
            metrics = get_metrics()
            alerts = get_alerts()
            
            # Display
            print(format_health(health))
            print("\n" + format_metrics(metrics))
            print(format_alerts(alerts))
            
            print(f"\n\nRefreshing in {REFRESH_INTERVAL}s... (Ctrl+C to exit)")
            time.sleep(REFRESH_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()
