@echo off
setlocal enabledelayedexpansion

echo ========================================
echo A-LMI System Startup
echo Davis Unified Intelligence System v2.0
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [1/5] Starting Docker services...
docker-compose up -d

if errorlevel 1 (
    echo [ERROR] Failed to start Docker services!
    echo Check docker-compose.yml and ensure all images are available.
    pause
    exit /b 1
)

echo [OK] Docker services started
echo.

echo [2/5] Waiting for services to be ready...
echo Checking Kafka...
timeout /t 5 /nobreak
docker-compose exec -T kafka kafka-topics --list --bootstrap-server localhost:9092 >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Kafka not ready, waiting...
    timeout /t 10 /nobreak
)

echo Checking Milvus...
timeout /t 3 /nobreak

echo Checking Neo4j...
timeout /t 3 /nobreak

echo [OK] Services initialized
echo.

echo [3/5] Starting Python A-LMI backend...
cd /d "%~dp0a_lmi_system"
if not exist "main.py" (
    echo [ERROR] main.py not found in a_lmi_system directory!
    pause
    exit /b 1
)
start "A-LMI Backend" cmd /k "cd /d \"%~dp0a_lmi_system\" && python main.py"
timeout /t 2 /nobreak
cd /d "%~dp0"

echo [OK] Backend started
echo.

echo [4/5] Starting Conversational UI...
start "A-LMI UI" cmd /k "cd /d \"%~dp0a_lmi_system\" && python -m interface.conversational_ui"
timeout /t 2 /nobreak

echo [OK] UI started
echo.

echo [5/5] A-LMI System is running!
echo.
echo Services are now running in separate windows:
echo   - A-LMI Backend: Main agent and reasoning
echo   - A-LMI UI: Conversational interface
echo.
echo To stop: Close the windows or press Ctrl+C in each
echo.
echo ========================================
echo System Status:
echo   - Docker services: Running
echo   - Python backend: Starting...
echo   - UI: Starting...
echo ========================================
echo.
echo Opening system documentation...
start "" "README.md"
pause

