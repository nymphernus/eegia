@echo off

cd /d "%~dp0"
call venv\Scripts\activate
cd app
streamlit run Home.py

@REM pause