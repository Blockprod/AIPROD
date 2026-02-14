#!/usr/bin/env python3
"""
GPU Thermal Diagnostic Tool
Monitor GPU metrics before and after optimization
"""

import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from collections import deque

class GPUMonitor:
    """Monitor NVIDIA GPU thermal and performance metrics."""
    
    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.metrics_history = deque(maxlen=300)  # Keep last 10 minutes
        self.start_time = datetime.now()
        self._verify_nvidia_smi()
    
    def _verify_nvidia_smi(self):
        """Check if nvidia-smi is available."""
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                timeout=5
            )
        except FileNotFoundError:
            print("❌ ERROR: nvidia-smi not found in PATH")
            print("   Please install NVIDIA drivers with CUDA toolkit")
            sys.exit(1)
    
    def get_gpu_metrics(self) -> dict:
        """Get current GPU metrics."""
        try:
            # Query multiple metrics at once
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,utilization.gpu,power.draw,power.limit,"
                    "memory.used,memory.total,clocks.current.graphics,clocks.max.graphics",
                    "--format=csv,nounits,noheader"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            
            values = result.stdout.strip().split(", ")
            
            # Helper to safely convert values, handling [N/A]
            def safe_float(val, default=0.0):
                try:
                    val = val.strip()
                    if val.startswith('[') or val == 'N/A':
                        return default
                    return float(val)
                except (ValueError, AttributeError):
                    return default
            
            return {
                "timestamp": datetime.now(),
                "temperature_c": safe_float(values[0] if len(values) > 0 else "0"),
                "utilization_pct": safe_float(values[1] if len(values) > 1 else "0"),
                "power_draw_w": safe_float(values[2] if len(values) > 2 else "0", default=250),
                "power_limit_w": safe_float(values[3] if len(values) > 3 else "0", default=250),
                "memory_used_mb": safe_float(values[4] if len(values) > 4 else "0"),
                "memory_total_mb": safe_float(values[5] if len(values) > 5 else "8192", default=8192),
                "clock_current_mhz": safe_float(values[6] if len(values) > 6 else "0"),
                "clock_max_mhz": safe_float(values[7] if len(values) > 7 else "1800", default=1800),
            }
        except Exception as e:
            print(f"WARNING: Error reading GPU metrics: {e}")
            return None
    
    def print_header(self):
        """Print diagnostic header."""
        print("\n" + "="*90)
        print("GPU THERMAL DIAGNOSTIC MONITOR".center(90))
        print("="*90)
        print()
        
        # Get GPU info
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", 
                 "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            values = result.stdout.strip().split(", ")
            print(f"GPU: {values[0]}")
            print(f"Driver: {values[1]}")
            if len(values) > 2:
                print(f"Compute Capability: {values[2]}")
        except:
            pass
        
        print()
    
    def print_status(self, metrics: dict, baseline: dict = None):
        """Print current status with colors."""
        if not metrics:
            print("WARNING: Unable to read GPU metrics")
            return
        
        temp = metrics["temperature_c"]
        util = metrics["utilization_pct"]
        power = metrics["power_draw_w"]
        clock = metrics["clock_current_mhz"]
        mem_used = metrics["memory_used_mb"]
        mem_total = metrics["memory_total_mb"]
        
        # Status indicators based on thresholds
        def temp_status(t):
            if t >= 85:
                return "[CRIT]"  # Critical
            elif t >= 80:
                return "[WARN]"  # Warning
            elif t >= 75:
                return "[OK  ]"   # Caution
            else:
                return "[GOOD]"   # Healthy
        
        # Clear screen and print
        print(f"\r{datetime.now().strftime('%H:%M:%S')} | ", end="")
        print(f"Temp: {temp_status(temp)} {temp:5.1f}C | ", end="")
        print(f"Util: {util:5.1f}% | ", end="")
        print(f"Power: {power:6.1f}W | ", end="")
        print(f"Clock: {clock:4.0f}MHz | ", end="")
        print(f"Mem: {mem_used:6.0f}MB/{mem_total:6.0f}MB {mem_used/mem_total*100:5.1f}%", end="")
        
        # Show delta if baseline provided
        if baseline:
            temp_delta = temp - baseline["temperature_c"]
            power_delta = power - baseline["power_draw_w"]
            
            if temp_delta != 0:
                delta_str = f"{temp_delta:+.1f}C" if temp_delta else "- C"
                print(f" (Delta{delta_str})", end="")
        
        sys.stdout.flush()
        self.metrics_history.append(metrics)
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.metrics_history:
            print("\nERROR: No metrics recorded")
            return
        
        temps = [m["temperature_c"] for m in self.metrics_history]
        powers = [m["power_draw_w"] for m in self.metrics_history]
        clocks = [m["clock_current_mhz"] for m in self.metrics_history]
        
        print("\n\n" + "="*90)
        print("MONITORING SUMMARY".center(90))
        print("="*90 + "\n")
        
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"Duration: {duration:.1f} seconds ({len(self.metrics_history)} samples)")
        print()
        
        print("TEMPERATURE (C):")
        print(f"    Min: {min(temps):.1f}C")
        print(f"    Max: {max(temps):.1f}C")
        print(f"    Avg: {sum(temps)/len(temps):.1f}C")
        print()
        
        print("POWER DRAW (W):")
        print(f"    Min: {min(powers):.1f}W")
        print(f"    Max: {max(powers):.1f}W")
        print(f"    Avg: {sum(powers)/len(powers):.1f}W")
        print()
        
        print("GPU CLOCK (MHz):")
        print(f"    Min: {min(clocks):.0f}MHz")
        print(f"    Max: {max(clocks):.0f}MHz")
        print(f"    Avg: {sum(clocks)/len(clocks):.0f}MHz")
        print()
        
        # Health assessment
        avg_temp = sum(temps) / len(temps)
        print("HEALTH ASSESSMENT:")
        
        if avg_temp >= 85:
            print("    [CRIT] GPU running too hot")
            print("    Recommendations:")
            print("      - Reduce batch size")
            print("      - Reduce resolution")
            print("      - Check thermal paste")
            print("      - Improve case ventilation")
        elif avg_temp >= 80:
            print("    [WARN] GPU temperature elevated")
            print("    Recommendations:")
            print("      - Monitor closely during training")
            print("      - Consider reducing batch size")
            print("      - Check if thermal throttling occurs")
        elif avg_temp >= 75:
            print("    [OK  ] Normal but warm")
            print("    Status: Acceptable for training")
        else:
            print("    [GOOD] Optimal temperature")
            print("    Status: All good!")
        
        print()
    
    def run(self, duration: int = 300):
        """Run monitoring loop."""
        self.print_header()
        
        print(f"Monitoring for {duration} seconds (ctrl+c to stop)...\n")
        
        baseline = None
        try:
            elapsed = 0
            while elapsed < duration:
                metrics = self.get_gpu_metrics()
                if metrics:
                    if baseline is None:
                        baseline = metrics
                    
                    self.print_status(metrics, baseline)
                
                time.sleep(self.interval)
                elapsed += self.interval
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")
        finally:
            self.print_summary()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor GPU metrics during training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor for 5 minutes
  python scripts/gpu_thermal_monitor.py --duration 300
  
  # Monitor GPU during training run
  python scripts/gpu_thermal_monitor.py
        """
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Monitoring duration in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Sample interval in seconds (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(interval=args.interval)
    monitor.run(duration=args.duration)


if __name__ == "__main__":
    main()
