@echo off
chcp 65001 > nul
title 乡镇服务反馈平台后端服务器

echo ========================================
echo   乡镇服务反馈平台 - 后端服务器启动
echo ========================================
echo.

REM 检查Python
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python，请先安装Python 3.9+
    pause
    exit /b 1
)

echo ✅ Python 已安装
echo.

REM 检查虚拟环境
if not exist "venv\Scripts\activate.bat" (
    echo ⚠ 正在创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo ✅ 虚拟环境创建成功
) else (
    echo ✅ 虚拟环境已存在
)
echo.

REM 激活虚拟环境
echo 🔄 正在激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 激活虚拟环境失败
    pause
    exit /b 1
)
echo ✅ 虚拟环境已激活
echo.

REM 安装依赖
echo 📦 正在检查并安装依赖包...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ❌ 安装依赖失败
    pause
    exit /b 1
)
echo ✅ 依赖包安装完成
echo.

REM 启动服务器
echo 🚀 正在启动后端服务器...
echo.
echo ========================================
python run.py

pause