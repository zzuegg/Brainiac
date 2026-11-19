@echo off
rem Brainiac Studio launcher: updates the checkout to the newest main, then starts the studio.
rem
rem   studio.bat                      update, build and start, then open http://127.0.0.1:7860/
rem   studio.bat --data D:\Data       anything after studio.bat goes to the studio (see --help)
rem
rem Where it runs: if this file sits in a Brainiac checkout, that checkout. Anywhere else it
rem uses (and on the first run clones) the Brainiac-main folder next to this file.
rem
rem Local work is never touched. The update is skipped, with a warning, when the checkout has
rem uncommitted changes, is on another branch, or has commits that are not on GitHub's main.
rem
rem Settings (set them before calling, or edit the defaults below):
rem   BRAINIAC_HEAP        the Java heap, such as 24g (default: half of this PC's memory)
rem   BRAINIAC_CHECKOUT    the checkout to use instead of the one described above
rem   BRAINIAC_BRANCH      the branch to follow (default main)
rem   BRAINIAC_NO_BROWSER  set to 1 to not open the browser
setlocal EnableExtensions
title Brainiac Studio

rem The update can rewrite this very file, and cmd reads a running batch file as it goes, so
rem the work is done by a copy in %TEMP%. Call and exit share one line: nothing after it is read.
if "%BRAINIAC_LAUNCHER_COPY%"=="1" goto :copied
set "BRAINIAC_LAUNCHER_COPY=1"
set "BRAINIAC_LAUNCHER_HOME=%~dp0"
copy /y "%~f0" "%TEMP%\brainiac-studio-launcher.bat" >nul
call "%TEMP%\brainiac-studio-launcher.bat" %* & exit /b
:copied

if not defined BRAINIAC_REPO_URL set "BRAINIAC_REPO_URL=https://github.com/zzuegg/Brainiac.git"
if not defined BRAINIAC_BRANCH set "BRAINIAC_BRANCH=main"
set "PORT=7860"

set "HERE=%BRAINIAC_LAUNCHER_HOME%"
if not defined HERE set "HERE=%~dp0"
set "HERE=%HERE:~0,-1%"

where git >nul 2>nul
if errorlevel 1 (
    echo git was not found. Install Git for Windows and try again.
    goto :fail
)

rem ---- find the checkout --------------------------------------------------------------
if defined BRAINIAC_CHECKOUT (
    set "REPO=%BRAINIAC_CHECKOUT%"
    goto :have_repo
)
if exist "%HERE%\settings.gradle.kts" if exist "%HERE%\brainiac-studio\" (
    set "REPO=%HERE%"
    goto :have_repo
)
set "REPO=%HERE%\Brainiac-main"
rem models kept next to this file are found too
if not defined BRAINIAC_MODELS if exist "%HERE%\models\" set "BRAINIAC_MODELS=%HERE%\models"

:have_repo
if exist "%REPO%\.git" goto :update
if exist "%REPO%\" (
    echo "%REPO%" exists but is not a git checkout. Move it away or set BRAINIAC_CHECKOUT.
    goto :fail
)
echo Cloning Brainiac into "%REPO%" ...
git clone --branch "%BRAINIAC_BRANCH%" "%BRAINIAC_REPO_URL%" "%REPO%"
if errorlevel 1 goto :fail

rem ---- update ---------------------------------------------------------------------------
:update
cd /d "%REPO%"
if errorlevel 1 goto :fail

set "CUR="
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set "CUR=%%b"
if /i not "%CUR%"=="%BRAINIAC_BRANCH%" (
    echo.
    echo WARNING: "%REPO%" is on "%CUR%", not %BRAINIAC_BRANCH%. Not updating; starting it as it is.
    goto :skipped
)

set "DIRTY="
for /f "delims=" %%l in ('git status --porcelain --untracked-files^=no') do set "DIRTY=1"
if defined DIRTY (
    echo.
    echo WARNING: "%REPO%" has uncommitted changes. Not updating, so nothing is lost:
    git status --short --untracked-files=no
    echo Commit or stash them to get updates again. Starting the current version.
    goto :skipped
)

echo Fetching the newest %BRAINIAC_BRANCH% ...
git fetch origin "%BRAINIAC_BRANCH%"
if errorlevel 1 (
    echo.
    echo WARNING: could not reach GitHub. Starting the current version.
    goto :skipped
)
rem --ff-only never rewrites anything: with local commits that are not on GitHub it just fails
git merge --ff-only FETCH_HEAD
if errorlevel 1 (
    echo.
    echo WARNING: local %BRAINIAC_BRANCH% has commits that are not on GitHub. Not updating.
    goto :skipped
)
goto :run

:skipped
timeout /t 5 >nul

rem ---- run ------------------------------------------------------------------------------
:run
echo.
for /f "delims=" %%v in ('git log -1 "--format=%%h %%s (%%cr)"') do echo Version: %%v

netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
    echo Something is already listening on port %PORT%, probably a studio that is still running.
    echo Close it first to start the new version. Opening the running one.
    start "" "http://127.0.0.1:%PORT%/"
    goto :fail
)

if defined BRAINIAC_HEAP (
    set "HEAP=-Xmx%BRAINIAC_HEAP%"
) else (
    set "HEAP=-XX:MaxRAMPercentage=50"
)
rem NVIDIA's driver thread for OpenGL "threaded optimizations" crashed JVMs at exit on Linux
rem once Vulkan was loaded; turning it off costs nothing for the studio
set "__GL_THREADED_OPTIMIZATIONS=0"

if not "%BRAINIAC_NO_BROWSER%"=="1" (
    start "" /b powershell -NoProfile -ExecutionPolicy Bypass -Command "for ($i = 0; $i -lt 900; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://127.0.0.1:%PORT%/' | Out-Null; Start-Process 'http://127.0.0.1:%PORT%/'; break } catch { Start-Sleep -Seconds 1 } }"
)

echo Building and starting Brainiac Studio (heap %HEAP%) ...
if "%~1"=="" (
    call gradlew.bat --console=plain :brainiac-studio:run "-PstudioJvmArgs=%HEAP%"
) else (
    call gradlew.bat --console=plain :brainiac-studio:run "-PstudioJvmArgs=%HEAP%" "--args=%*"
)
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
pause
exit /b 1
