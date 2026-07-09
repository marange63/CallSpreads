@echo off
REM ============================================================
REM  Launch CallSpreads
REM  Double-click this file to start the Call Spread Finder
REM  web app. Leave this window open while using the app;
REM  close it (or press Ctrl+C) to stop the server.
REM ============================================================

REM Run from the folder this script lives in, regardless of
REM where it was launched from.
cd /d "%~dp0"

title Call Spread Finder

REM Prefer the Windows "py" launcher; fall back to python on PATH.
where py >nul 2>&1
if %errorlevel%==0 (
    py spx_call_spread_finder.py
) else (
    python spx_call_spread_finder.py
)

REM If the script exits or errors, keep the window open so any
REM message stays visible.
echo.
echo Server stopped. Press any key to close this window.
pause >nul
