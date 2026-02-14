@echo off
REM AIPROD Quick Activation Script for Windows
REM Usage: Run this batch file to activate venv and enter interactive Python shell

setlocal enabledelayedexpansion

echo.
echo =====================================
echo   AIPROD LOCAL DEVELOPMENT SHELL
echo =====================================
echo.

REM Manually set up environment without calling the standard activate script
SET "PATH=%CD%\.venv_311\Scripts;%SYSTEMROOT%\System32;%SYSTEMROOT%;%SYSTEMROOT%\System32\Wbem"
SET "VIRTUAL_ENV=%CD%\.venv_311"
SET "PROMPT=(.venv_311) $P$G"

REM Check GPU
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')" 2>nul

echo.
echo Useful commands in this shell:
echo   - python examples/quickstart.py --prompt "Your prompt"
echo   - python scripts/monitor_gpu.py (in separate terminal)
echo   - python -c "import torch; print(torch.cuda.memory_allocated() / 1e9, 'GB')"
echo   - exit (to deactivate venv)
echo.

cmd.exe /k
