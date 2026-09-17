@echo off
setlocal
rem Use uv's base interpreter directly if the virtual-environment launcher fails.
set "SAM3_PYTHON="
for /f "tokens=1,* delims==" %%A in ('findstr /b "home = " "%~dp0.venv-export\pyvenv.cfg"') do set "SAM3_PYTHON=%%B"
for /f "tokens=*" %%A in ("%SAM3_PYTHON%") do set "SAM3_PYTHON=%%A"
set "SAM3_PYTHON=%SAM3_PYTHON:"=%\python.exe"
if not exist "%SAM3_PYTHON%" (
    echo Python 3.12 was not found at "%SAM3_PYTHON%".
    exit /b 1
)
if not exist "%~dp0.venv-export\Lib\site-packages\huggingface_hub\cli\hf.py" (
    echo The SAM3 export environment is missing. Run setup-windows.ps1 first.
    exit /b 1
)
"%SAM3_PYTHON%" -c "import runpy,site,sys; site.addsitedir(sys.argv.pop(1)); runpy.run_module('huggingface_hub.cli.hf',run_name='__main__')" "%~dp0.venv-export\Lib\site-packages" auth login %*
exit /b %errorlevel%
