@echo off
REM Double-click this file on Windows to launch the dashboard.
REM The browser will open automatically at http://localhost:8501

cd /d "%~dp0"
streamlit run dashboard.py
pause
