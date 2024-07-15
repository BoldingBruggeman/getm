set REPO_DIR=%~dp0
set WORK_DIR=%TEMP%\build-pygetm-%RANDOM%

@SET PYFABM_DIR=
FOR /F %%I IN ('python -c "import pyfabm,os;print(os.path.dirname(pyfabm.__file__))"') DO @SET "PYFABM_DIR=%%I"

pip install -v "%REPO_DIR%\python"
if errorlevel 1 exit /b 1

cmake -S "%REPO_DIR%\extern\pygsw" -B "%WORK_DIR%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%WORK_DIR%" --target pygsw_wheel --config Release --parallel 4
if errorlevel 1 exit /b 1
xcopy /E /I /Y "%WORK_DIR%\pygsw" "%PYFABM_DIR%\..\pygetm\pygsw"
if errorlevel 1 exit /b 1

rmdir /S /Q "%WORK_DIR%"
if errorlevel 1 exit /b 1
cmake -S "%REPO_DIR%\extern\python-otps2" -B "%WORK_DIR%" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%WORK_DIR%" --target otps2_wheel --config Release --parallel 4
if errorlevel 1 exit /b 1
xcopy /E /I /Y "%WORK_DIR%\otps2" "%PYFABM_DIR%\..\pygetm\otps2"
if errorlevel 1 exit /b 1

rmdir /S /Q "%WORK_DIR%"
