#!/bin/bash
# Double-click this file on Mac to launch the dashboard.
# The browser will open automatically at http://localhost:8501

cd "$(dirname "$0")"
streamlit run dashboard.py
