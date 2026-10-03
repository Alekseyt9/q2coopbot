@echo off
setlocal
cd /d "%~dp0"
set "GOTOOLCHAIN=auto"
set "GOCACHE=%CD%\workspace\build\go-cache"
set "Q2RUNS_GO=go"
where go >nul 2>nul
if errorlevel 1 set "Q2RUNS_GO=C:\Program Files\Go\bin\go.exe"
if not exist "workspace\build" mkdir "workspace\build"
"%Q2RUNS_GO%" build -o "workspace\build\q2runs.exe" ./cmd/q2runs
if errorlevel 1 goto failed
set "Q2RUNS_PORT=18791"
if not "%~1"=="" set "Q2RUNS_PORT=%~1"
echo Open http://127.0.0.1:%Q2RUNS_PORT%
"workspace\build\q2runs.exe" -listen "127.0.0.1:%Q2RUNS_PORT%"
if errorlevel 1 goto failed
exit /b 0
:failed
echo Run Explorer failed. See the error above.
pause
exit /b 1
