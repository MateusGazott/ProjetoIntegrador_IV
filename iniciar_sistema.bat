@echo off
chcp 65001 > nul
echo ======================================================================
echo    FinGuard Analytics - Sistema de Diagnóstico de Risco Financeiro
echo ======================================================================
echo.

set VENV_PYTHON=C:\Users\guibi\Downloads\Integrante4_Frontend_Streamlit\Integrante4_Frontend_Streamlit\.venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    set VENV_PYTHON=python
)

echo [1/2] Iniciando o Backend Preditivo na porta 5000...
start "FinGuard - Backend REST API" cmd /k ""%VENV_PYTHON%" 02_Backend_Integracao\backend_service.py 5000"

timeout /t 3 > nul

echo [2/2] Iniciando a Interface Web (Streamlit)...
start "FinGuard - Interface Web" cmd /k ""%VENV_PYTHON%" -m streamlit run 04_Frontend_Streamlit\app.py"

echo.
echo ======================================================================
echo    Sistema iniciado com sucesso!
echo    - Backend API: http://localhost:5000/api/diagnostico/saude
echo    - Frontend Web: http://localhost:8501
echo ======================================================================
echo Pressione qualquer tecla para sair desta janela...
pause > nul
