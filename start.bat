@echo off

cd /d "%~dp0"

echo Starting MLflow UI
start "MLflow Server" cmd /c "call venv\Scripts\activate && mlflow ui"

echo Waiting for MLflow to start
timeout /t 5 /nobreak > nul
start http://127.0.0.1:5000

echo Starting Jupyter Notebook
call venv\Scripts\activate
jupyter notebook agent.ipynb

@REM pause