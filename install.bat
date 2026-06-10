@echo off
REM Run this once after cloning the repo on Windows.
REM It installs all Python dependencies.

cd /d "%~dp0"
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Done. You can now double-click start.bat to launch the dashboard.
echo.
pause
