@echo off
setlocal
set SCRIPT_DIR=%~dp0

python "%SCRIPT_DIR%openpose_research.py" ^
  --base-dir "%SCRIPT_DIR%" ^
  --views json_outputiew1 json_outputiew2 json_outputiew3 json_outputiew4 ^
  --p-mats P_matrix ^
  --out-dir analysis_output ^
  --fps 30 ^
  --person-index 0
pause
