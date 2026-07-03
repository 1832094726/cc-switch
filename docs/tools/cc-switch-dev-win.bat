@echo off
REM === cc-switch dev launcher for Windows ===

set ROOT_DIR=D:\syn\jd\hechengjun\cc-switch

REM --- Environment self-check ---
where node >nul 2>&1
if errorlevel 1 (
    echo [cc-switch-dev] ERROR: node not found in PATH
    echo [cc-switch-dev] install Node.js 18+ from https://nodejs.org
    exit /b 1
)
for /f "tokens=1 delims=." %%v in ('node -p "process.versions.node"') do set NODE_MAJOR=%%v
if %NODE_MAJOR% LSS 18 (
    echo [cc-switch-dev] ERROR: node v%NODE_MAJOR% is too old, need v18+
    exit /b 1
)
echo [cc-switch-dev] node: v%NODE_MAJOR%.x

where pnpm >nul 2>&1
if errorlevel 1 (
    where corepack >nul 2>&1
    if errorlevel 1 (
        set PNPM_CMD=npx pnpm@9
    ) else (
        set PNPM_CMD=corepack pnpm
    )
) else (
    set PNPM_CMD=pnpm
)
echo [cc-switch-dev] pnpm: %PNPM_CMD%

REM --- Dependency integrity check ---
if not exist "%ROOT_DIR%\node_modules\.bin\tauri.cmd" (
    echo [cc-switch-dev] node_modules missing, installing...
    cd /d "%ROOT_DIR%"
    call %PNPM_CMD% install
    if errorlevel 1 (
        echo [cc-switch-dev] ERROR: install failed
        exit /b 1
    )
)

REM Kill any process using the old dev ports (1420/1421 for Vite, 15721 for proxy)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":1420 " ^| findstr "LISTENING"') do (
    echo Killing PID %%a on port 1420
    taskkill /F /PID %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":1421 " ^| findstr "LISTENING"') do (
    echo Killing PID %%a on port 1421
    taskkill /F /PID %%a 2>nul
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":15721 " ^| findstr "LISTENING"') do (
    echo Killing PID %%a on port 15721
    taskkill /F /PID %%a 2>nul
)

call "D:\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set CARGO_TARGET_DIR=D:\cc-switch-cargo-target
cd /d D:\syn\jd\hechengjun\cc-switch

REM --- Build artifact cleanup ---
REM When the target dir exceeds a threshold, sweep orphaned artifacts
REM while keeping incremental compilation cache intact.
REM Default threshold is 8GB, override via CC_SWITCH_DEV_TARGET_MAX_GB.
if not defined CC_SWITCH_DEV_TARGET_MAX_GB set CC_SWITCH_DEV_TARGET_MAX_GB=8
where cargo-sweep >nul 2>&1
if not errorlevel 1 (
    if exist "%CARGO_TARGET_DIR%\debug" (
        echo [cc-switch-dev] checking target dir size...
        powershell -NoProfile -Command ^
            "$td='%CARGO_TARGET_DIR%'; $max=[int64]'%CC_SWITCH_DEV_TARGET_MAX_GB%'*1GB;" ^
            "$cur=(Get-ChildItem -Path $td -Recurse -File -ErrorAction SilentlyContinue ^| Measure-Object -Property Length -Sum).Sum;" ^
            "if ($cur -le $max) { exit 100 } else { exit 0 }"
        if not errorlevel 100 (
            echo [cc-switch-dev] target exceeds %CC_SWITCH_DEV_TARGET_MAX_GB%GB, cleaning orphaned artifacts...
            cargo-sweep sweep --maxsize "%CC_SWITCH_DEV_TARGET_MAX_GB%GB" "D:\syn\jd\hechengjun\cc-switch\src-tauri"
        ) else (
            echo [cc-switch-dev] target within threshold, skipping cleanup
        )
    )
) else (
    echo [cc-switch-dev] cargo-sweep not found, skipping cleanup
    echo [cc-switch-dev] install with: cargo install cargo-sweep
)
.\node_modules\.bin\tauri dev --no-watch --config "{\"identifier\":\"com.ccswitch.desktop.dev\",\"build\":{\"beforeDevCommand\":\"npm run dev:renderer\"}}"