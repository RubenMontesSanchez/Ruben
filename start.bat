@echo off
echo ============================================
echo   3D Generator - Iniciando...
echo ============================================
echo.

echo Iniciando backend (Python FastAPI)...
start "Backend 3D" cmd /k "cd /d %~dp0backend && python main.py"

echo Esperando que el backend arranque...
timeout /t 3 /nobreak >nul

echo Iniciando frontend (Next.js)...
start "Frontend 3D" cmd /k "cd /d %~dp0frontend && npm run dev"

echo Esperando que el frontend arranque...
timeout /t 5 /nobreak >nul

echo Abriendo navegador...
start http://localhost:3000

echo.
echo App corriendo en http://localhost:3000
echo Backend API en   http://localhost:8000
echo.
echo Cierra las ventanas de terminal para detener la app.
