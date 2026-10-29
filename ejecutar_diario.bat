@echo off
cd /d "%~dp0"

echo ===== Ejecucion %date% %time% ===== >> ejecucion.log
py analisis_diario.py >> ejecucion.log 2>&1
echo. >> ejecucion.log
