@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\run_controlled_experiment.py" expA
pause
