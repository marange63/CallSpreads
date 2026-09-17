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

REM Activate the project's own conda environment. The dependencies live
REM only there. Do not fall back to the "py" launcher or to "python" on
REM PATH: those reach Python 3.14 and the Anaconda base environment
REM respectively, and neither is what this app runs on.
set "CONDA_BAT=C:\Users\wamfo\anaconda3\condabin\conda.bat"

if not exist "%CONDA_BAT%" (
    echo Could not find conda at "%CONDA_BAT%".
    goto :halt
)

call "%CONDA_BAT%" activate CallSpreads
if errorlevel 1 (
    echo Failed to activate the CallSpreads conda environment.
    echo.
    echo Create it from this folder with:
    echo     conda env create -f environment.yml
    goto :halt
)

python spx_call_spread_finder.py

:halt
REM If the script exits or errors, keep the window open so any
REM message stays visible.
echo.
echo Server stopped. Press any key to close this window.
pause >nul
