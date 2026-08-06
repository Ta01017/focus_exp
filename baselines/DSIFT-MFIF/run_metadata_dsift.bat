@echo off
setlocal
if "%~2"=="" echo Usage: %~nx0 metadata.json output_dir [start_index] [max_samples] ^& exit /b 2
set "START=%~3"
if "%START%"=="" set "START=0"
set "MAX=%~4"
if "%MAX%"=="" set "MAX=-1"
python "%~dp0infer_metadata.py" --metadata "%~1" --output-dir "%~2" --start-index %START% --max-samples %MAX%
