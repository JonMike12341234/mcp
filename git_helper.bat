@echo off
setlocal enabledelayedexpansion
REM Universal MCP Server - Git Helper Script

echo ================================================================
echo                Universal MCP Server Git Helper
echo ================================================================
echo.

REM Check if .git exists
if not exist ".git" (
    echo ERROR: This directory is not a git repository.
    echo Run 'git init' first or clone your repo.
    pause
    exit /b 1
)

REM Get current remote
set "REMOTE_URL="
for /f "delims=" %%r in ('git remote get-url origin 2^>nul') do set "REMOTE_URL=%%r"

REM Prompt for remote if not set
if "!REMOTE_URL!"=="" (
    set /p "REMOTE_URL=Enter your git remote URL (e.g. https://github.com/username/repo.git): "
    if defined REMOTE_URL (
        git remote add origin "!REMOTE_URL!"
    ) else (
        echo No remote URL entered. Exiting.
        pause
        exit /b 1
    )
) else (
    echo Current remote: !REMOTE_URL!
)

REM Ask for branch
set "BRANCH=main"
set "BRANCH_CHOICE=Y"
set /p "BRANCH_CHOICE=Push to '!BRANCH!' branch? (Y/n): "
if /I "!BRANCH_CHOICE!"=="n" (
    set "USER_BRANCH="
    set /p "USER_BRANCH=Enter branch name: "
    if defined USER_BRANCH (
        set "BRANCH=!USER_BRANCH!"
    ) else (
        echo No branch name entered. Using default 'main'.
        set "BRANCH=main"
    )
)

REM Stage all changes
echo.
echo Staging all changes...
git add -A

REM Commit
set "COMMIT_MSG="
set /p "COMMIT_MSG=Enter commit message: "
if not defined COMMIT_MSG (
    echo No commit message entered. Aborting commit.
    pause
    exit /b 1
)
git commit -m "!COMMIT_MSG!"

REM Pull latest changes
echo.
echo Pulling latest changes from origin/!BRANCH!...
git pull origin !BRANCH!
if errorlevel 1 (
    echo.
    echo Error pulling changes. Please resolve conflicts or issues manually.
    pause
    exit /b 1
)

REM Push
echo.
echo Pushing to origin/!BRANCH!...
git push origin !BRANCH!
if errorlevel 1 (
    echo.
    echo Push failed. If this is a new branch, attempting to set upstream...
    git push --set-upstream origin !BRANCH!
    if errorlevel 1 (
        echo.
        echo Failed to push and set upstream. Please check your connection, permissions, or branch status.
        pause
        exit /b 1
    )
)

echo.
echo ================================================================
echo                Git operation completed!
echo ================================================================
echo.
echo Useful commands:
echo   - git status
echo   - git log --oneline
echo   - git branch
echo   - git checkout ^<branch^>
echo   - git pull origin ^<branch^>
echo   - git push origin ^<branch^>
echo.
pause