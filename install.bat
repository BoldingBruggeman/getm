set REPO_DIR=%~dp0
set WORK_DIR=%TEMP%\build-pygetm-%RANDOM%

set CMAKE_GENERATOR=Ninja
set FC=x86_64-w64-mingw32-gfortran
set CC=x86_64-w64-mingw32-gcc
set CFLAGS=-DMS_WIN64

pip install -v -e "%REPO_DIR%\python"
if errorlevel 1 exit /b 1

@SET PYGETM_DIR=
FOR /F %%I IN ('python -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('pygetm').origin))"') DO @SET "PYGETM_DIR=%%I"

cmake -S "%REPO_DIR%\extern\pygsw" -B "%WORK_DIR%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%WORK_DIR%" --target pygsw_wheel --config Release --parallel 4
if errorlevel 1 exit /b 1
xcopy /E /I /Y "%WORK_DIR%\pygsw" "%PYGETM_DIR%\pygsw"
if errorlevel 1 exit /b 1

rmdir /S /Q "%WORK_DIR%"
if errorlevel 1 exit /b 1
cmake -S "%REPO_DIR%\extern\python-otps2" -B "%WORK_DIR%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%WORK_DIR%" --target otps2_wheel --config Release --parallel 4
if errorlevel 1 exit /b 1
xcopy /E /I /Y "%WORK_DIR%\otps2" "%PYGETM_DIR%\otps2"
if errorlevel 1 exit /b 1

rmdir /S /Q "%WORK_DIR%"
