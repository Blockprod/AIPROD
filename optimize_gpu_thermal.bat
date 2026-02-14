@echo off
REM AIPROD GPU Thermal Optimization Installer
REM Run as Administrator

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo   AIPROD GPU THERMAL OPTIMIZATION
echo ================================================================
echo.

REM Check if running as admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo.
    echo Please:
    echo   1. Right-click on Command Prompt or PowerShell
    echo   2. Select "Run as administrator"
    echo   3. Go to: C:\Users\averr\AIPROD
    echo   4. Run: optimize_gpu_thermal.bat
    echo.
    pause
    exit /b 1
)

echo Step 1: Verifying NVIDIA drivers...
nvidia-smi --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: NVIDIA drivers not found!
    echo Please install NVIDIA drivers from:
    echo https://www.nvidia.com/Download/driverDetails.aspx
    pause
    exit /b 1
)
echo [OK] NVIDIA drivers found

echo.
echo Step 2: Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH!
    exit /b 1
)
echo [OK] Python available

cd /d "C:\Users\averr\AIPROD"

echo.
echo Step 3: Applying GPU thermal configuration...
echo.

REM Run the PowerShell configuration script
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.\scripts\configure_gpu_thermal.ps1' -MaxClockMHz 1500"

if errorlevel 1 (
    echo.
    echo WARNING: GPU clock configuration may require manual setup.
    echo You can still train, but temperatures may be higher.
)

echo.
echo ================================================================
echo   OPTIMIZATION COMPLETE
echo ================================================================
echo.
echo Next steps:
echo.
echo   1. Start training with optimized settings:
echo      python packages/aiprod-core/src/training/train.py --start-phase 1
echo.
echo   2. In another terminal, monitor GPU temperature:
echo      python scripts/gpu_thermal_monitor.py --duration 600
echo.
echo   3. Watch for temperature range: 70-80 degrees C
echo.
echo For more info, see: docs/THERMAL_OPTIMIZATION_QUICK_START.md
echo.
pause
