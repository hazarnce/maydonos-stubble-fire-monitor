@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Proje sanal ortami bulunamadi. Once requirements-web.txt dosyasini kurun.
  pause
  exit /b 1
)
start "MaydonOs Web Sunucusu" /min ".venv\Scripts\python.exe" -m uvicorn backend.web_api:app --host 127.0.0.1 --port 8000
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8000/"
endlocal
