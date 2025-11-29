@echo off
echo 🔧 FIXING SEQUENTIAL THINKING MCP TIMEOUT
echo ==================================================

set "target_file=C:\Users\Admin\AppData\Roaming\Qoder\SharedClientCache\mcp.json"
set "backup_file=C:\Users\Admin\AppData\Roaming\Qoder\SharedClientCache\mcp_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.json"
set "fixed_file=c:\Users\Admin\Documents\EA_SCALPER_XAUUSD\mcp_config_fixed_sequential_thinking.json"

echo 📋 Creating backup...
copy "%target_file%" "%backup_file%" >nul
if %errorlevel% neq 0 (
    echo ❌ Failed to create backup
    pause
    exit /b 1
)
echo ✅ Backup created: %backup_file%

echo 🔧 Applying fix...
copy "%fixed_file%" "%target_file%" >nul
if %errorlevel% neq 0 (
    echo ❌ Failed to apply fix
    pause
    exit /b 1
)

echo ✅ Configuration fixed successfully!
echo.
echo 📋 CHANGES MADE:
echo    - Removed '-y' parameter from sequential_thinking args
echo    - This eliminates the startup delay causing timeout
echo.
echo 🚀 NEXT STEPS:
echo    1. Restart Qoder IDE completely
echo    2. Sequential thinking should start without timeout
echo    3. Test by using sequential thinking features
echo.
echo 💡 EXPLANATION:
echo    The '-y' parameter was causing npm to take extra time
echo    during package resolution, triggering the timeout.
echo    Removing it uses the global installation directly.
echo.
echo ==================================================
echo 🎯 SEQUENTIAL THINKING MCP FIX COMPLETE!
pause