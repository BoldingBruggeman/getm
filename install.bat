set REPO_DIR=%~dp0
set WORK_DIR=%TEMP%\build-pygetm-%RANDOM%

pip install -v %REPO_DIR%\extern\awex %REPO_DIR%\extern\pygotm %REPO_DIR%\python

cmake -S %REPO_DIR%\extern\pygsw -B %WORK_DIR% -DCMAKE_BUILD_TYPE=Release
cmake --build %WORK_DIR% --target install --config Release --parallel 4

rmdir /S /Q %WORK_DIR%
cmake -S %REPO_DIR%\extern\python-otps2 -B %WORK_DIR% -DCMAKE_BUILD_TYPE=Release
cmake --build %WORK_DIR% --target install --config Release --parallel 4

rmdir /S /Q %WORK_DIR%
