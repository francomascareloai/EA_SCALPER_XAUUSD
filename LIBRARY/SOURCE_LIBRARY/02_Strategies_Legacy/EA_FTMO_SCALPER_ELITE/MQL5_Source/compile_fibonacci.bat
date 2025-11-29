@echo off
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set EA_NAME=%1
if "%EA_NAME%"=="" set EA_NAME=EA_Fibonacci_Elite_v1.mq5
set LOGFILE=compilation_log_fibonacci_%TIMESTAMP%.txt

echo ========================================== > %LOGFILE%
echo EA Fibonacci Elite Compilation Log >> %LOGFILE%
echo Date: %date% %time% >> %LOGFILE%
echo Compiling: %EA_NAME% >> %LOGFILE%
echo ========================================== >> %LOGFILE%
echo. >> %LOGFILE%

echo Starting compilation of %EA_NAME%... >> %LOGFILE%

REM Verificar se arquivo MetaEditor existe
if exist "C:\Program Files\RoboForex MT5 Terminal\metaeditor64.exe" (
    echo Using RoboForex MetaEditor... >> %LOGFILE%
    "C:\Program Files\RoboForex MT5 Terminal\metaeditor64.exe" /compile:"%EA_NAME%" /log >> %LOGFILE% 2>&1
) else if exist "C:\Program Files\FTMO MetaTrader 5\MetaEditor64.exe" (
    echo Using FTMO MetaEditor... >> %LOGFILE%
    "C:\Program Files\FTMO MetaTrader 5\MetaEditor64.exe" /compile:"%EA_NAME%" /log >> %LOGFILE% 2>&1
) else if exist "C:\Program Files\MetaTrader 5\MetaEditor64.exe" (
    echo Using Standard MetaEditor... >> %LOGFILE%
    "C:\Program Files\MetaTrader 5\MetaEditor64.exe" /compile:"%EA_NAME%" /log >> %LOGFILE% 2>&1
) else if exist "C:\Program Files (x86)\MetaTrader 5\MetaEditor64.exe" (
    echo Using MetaEditor x86... >> %LOGFILE%
    "C:\Program Files (x86)\MetaTrader 5\MetaEditor64.exe" /compile:"%EA_NAME%" /log >> %LOGFILE% 2>&1
) else (
    echo ERROR: MetaEditor64.exe not found in standard locations >> %LOGFILE%
    echo Please check MetaTrader 5 installation >> %LOGFILE%
)

echo. >> %LOGFILE%
echo Compilation finished at %time% >> %LOGFILE%

REM Verificar se arquivo .ex5 foi criado
set EX5_NAME=%EA_NAME:.mq5=.ex5%
if exist "%EX5_NAME%" (
    echo SUCCESS: %EX5_NAME% compiled successfully! >> %LOGFILE%
    echo File size: >> %LOGFILE%
    dir "%EX5_NAME%" >> %LOGFILE%
    echo. >> %LOGFILE%
    echo ========================================== >> %LOGFILE%
    echo ✅ COMPILATION SUCCESSFUL! ✅ >> %LOGFILE%
    echo EA Fibonacci Elite está pronto para uso! >> %LOGFILE%
    echo ========================================== >> %LOGFILE%
) else (
    echo ERROR: Compilation failed - %EX5_NAME% file not found >> %LOGFILE%
    echo Possible causes: >> %LOGFILE%
    echo - Syntax errors in MQL5 code >> %LOGFILE%
    echo - Missing include files >> %LOGFILE%
    echo - MetaEditor path incorrect >> %LOGFILE%
)

echo. >> %LOGFILE%
echo ========================================== >> %LOGFILE%
echo Log saved as: %LOGFILE% >> %LOGFILE%
echo ========================================== >> %LOGFILE%

type %LOGFILE%
echo.
echo Pressione qualquer tecla para continuar...
pause > nul