@echo off
REM Script to start frontend and backend services

echo =========================================
echo Starting AI Style Transfer Project
echo =========================================

cd /d %~dp0

REM Start backend service
echo.
echo Starting backend service...
start "Backend Service" cmd /k "cd src && python server.py"
timeout /t 3 /nobreak >nul
echo Backend service started

REM Start frontend service
echo.
echo Starting frontend service...
start "Frontend Service" cmd /k "cd frontend && npm run dev"
timeout /t 3 /nobreak >nul
echo Frontend service started

REM Wait for services to be ready and open browser
timeout /t 2 /nobreak >nul
echo.
echo Opening browser...
start http://localhost:3000

echo.
echo =========================================
echo All services are running!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo =========================================
echo.
echo Close the corresponding command windows to stop services
pause
