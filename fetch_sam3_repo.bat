@echo off
REM Clone/fetch the exact SAM3 source revision used by this exporter.
setlocal EnableExtensions

python "%~dp0python\fetch_sam3_repo.py" %*
