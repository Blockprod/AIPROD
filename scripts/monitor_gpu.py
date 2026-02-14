#!/usr/bin/env python3
"""
GPU Monitor - Real-time monitoring of PyTorch GPU usage
Run in a separate terminal while generating videos

Usage:
    python scripts/monitor_gpu.py
"""

import time
import torch
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️  nvidia-ml-py3 not installed. Install with: pip install nvidia-ml-py3")


def format_bytes(bytes_val):
    """Format bytes to MB/GB"""
    mb = bytes_val / 1e6
    if mb > 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.1f} MB"


def print_header():
    """Print column header"""
    print("\n" + "="*80)
    print(f"{'Time':<12} {'GPU Mem':<15} {'GPU Util':<12} {'Temp':<10} {'Status':<25}")
    print("="*80)


def main():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. GPU monitoring not possible.")
        return
    
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(0)
    print(f"\n✅ GPU: {device_name}")
    print(f"   Device ID: {torch.cuda.current_device()}")
    print(f"   Total Memory: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")
    
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print(f"   Driver Version: {pynvml.nvmlSystemGetDriverVersion()}")
        except Exception as e:
            print(f"   Error initializing PYNVML: {e}")
            PYNVML_AVAILABLE = False
    
    print_header()
    
    try:
        while True:
            time_str = datetime.now().strftime("%H:%M:%S")
            
            # PyTorch stats
            torch_allocated = torch.cuda.memory_allocated(device)
            torch_reserved = torch.cuda.memory_reserved(device)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            allocated_str = format_bytes(torch_allocated)
            reserved_str = format_bytes(torch_reserved)
            total_str = format_bytes(total_memory)
            util_pct = (torch_allocated / total_memory) * 100
            
            status = "Idle"
            if util_pct > 80:
                status = "🔴 HIGH LOAD"
            elif util_pct > 50:
                status = "🟠 MEDIUM LOAD"
            elif util_pct > 20:
                status = "🟡 SOME ACTIVITY"
            else:
                status = "🟢 LOW ACTIVITY"
            
            # PYNVML stats (if available)
            temp_str = "N/A"
            gpu_util_str = "N/A"
            
            if PYNVML_AVAILABLE:
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp_str = f"{temp}°C"
                    gpu_util_str = f"{gpu_util.gpu}%"
                except Exception:
                    pass
            
            mem_display = f"{allocated_str} / {total_str}"
            
            print(f"{time_str:<12} {mem_display:<15} {gpu_util_str:<12} {temp_str:<10} {status:<25}")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("Monitoring stopped by user")
        print("="*80 + "\n")
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


if __name__ == "__main__":
    main()
