@echo off
REM Shim so `make <target>` works on Windows without GNU make.
REM Dispatches to the same CLI the Makefile calls.
setlocal
if "%PY%"=="" set PY=D:\Anaconda\envs\paper\python.exe
set PYTHONPATH=%~dp0src
set TARGET=%1
if "%TARGET%"=="" set TARGET=all

if /I "%TARGET%"=="test" (
  "%PY%" -m pytest -q "%~dp0tests"
  goto :eof
)
if /I "%TARGET%"=="env" (
  "%PY%" -m pip install -r "%~dp0requirements.txt"
  goto :eof
)
if /I "%TARGET%"=="all" (
  "%PY%" -m pytest -q "%~dp0tests" || goto :eof
  "%PY%" -m asdpipe.cli.main smoke || goto :eof
  "%PY%" -m asdpipe.cli.main validate || goto :eof
  "%PY%" -m asdpipe.cli.main checklist
  goto :eof
)
"%PY%" -m asdpipe.cli.main %TARGET%
