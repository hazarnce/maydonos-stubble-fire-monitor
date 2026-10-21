@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Proje Python ortami bulunamadi: .venv\Scripts\python.exe
  echo Once proje klasorunde python -m venv .venv ve pip install -r requirements.txt calistirin.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run legacy_streamlit/app.py
if errorlevel 1 pause
endlocal
