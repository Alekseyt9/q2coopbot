@echo off
setlocal
cd /d "%~dp0"
set "Q2RUNS_PORT=18791"
if not "%~1"=="" set "Q2RUNS_PORT=%~1"
powershell.exe -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing ('http://127.0.0.1:' + $env:Q2RUNS_PORT + '/') -TimeoutSec 2; if ($r.Content.Contains('<title>Quake II') -and $r.Content.Contains('RUN EXPLORER')) { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
    echo Run Explorer is already running: http://127.0.0.1:%Q2RUNS_PORT%
    echo This window can be closed. The server will keep running.
    pause
    exit /b 0
)
set "GOTOOLCHAIN=auto"
set "GOCACHE=%CD%\workspace\build\go-cache"
set "Q2RUNS_GO=go"
where go >nul 2>nul
if errorlevel 1 set "Q2RUNS_GO=C:\Program Files\Go\bin\go.exe"
if not exist "workspace\build" mkdir "workspace\build"
"%Q2RUNS_GO%" build -o "workspace\build\q2runs.exe" ./cmd/q2runs
if errorlevel 1 goto failed
echo Open http://127.0.0.1:%Q2RUNS_PORT%
"workspace\build\q2runs.exe" -listen "127.0.0.1:%Q2RUNS_PORT%"
if errorlevel 1 goto failed
echo Run Explorer stopped.
pause
exit /b 0
:failed
echo Run Explorer failed. See the error above.
pause
exit /b 1
