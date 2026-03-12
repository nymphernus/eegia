@echo off

cd /d "%~dp0"
call venv\Scripts\activate
jupyter notebook

@REM pause