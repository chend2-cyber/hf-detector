@echo off
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
start "" cmd /c "timeout /t 3 /nobreak >nul & start "" http://localhost:5000"
python code\app.py
