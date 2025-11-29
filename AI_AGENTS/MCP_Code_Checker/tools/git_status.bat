@echo off
echo Showing all uncommitted changes with status...
echo.
git status --porcelain
echo.
echo Status codes:
echo   M  = modified (staged)
echo    M = modified (not staged)
echo   A  = added (staged)
echo   ?? = untracked (new file)
echo   D  = deleted (staged)
echo    D = deleted (not staged)
echo   R  = renamed
echo   C  = copied
pause
