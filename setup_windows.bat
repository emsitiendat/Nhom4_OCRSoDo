@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title OCR SoDo - Setup Release

set "REQ=requirements-release.txt"
set "VENV_PY=.venv\Scripts\python.exe"
set "PIP_VERSION=26.2.1"
set "PYTHON_EXE="
set "PYTHON_MODE="
set "PYTHON_URL=https://www.python.org/downloads/release/python-31011/"

echo ==========================================
echo      OCR SO DO / SO HONG - SETUP
echo ==========================================
echo.

if not exist "%REQ%" (
    echo [LOI] Khong tim thay %REQ%
    echo File nay phai nam cung thu muc voi setup_windows.bat.
    echo.
    pause
    exit /b 1
)

if not exist "src" (
    echo [LOI] Khong tim thay thu muc src.
    echo Hay dat setup_windows.bat trong thu muc goc cua project.
    echo.
    pause
    exit /b 1
)

rem ============================================================
rem 1. Neu .venv da ton tai, chi chap nhan Python 3.10
rem ============================================================
if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,10) and sys.maxsize > 2**32 else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [LOI] .venv hien tai khong dung Python 3.10.
        echo Hay xoa thu muc .venv roi chay lai setup_windows.bat.
        echo.
        "%VENV_PY%" --version
        pause
        exit /b 1
    )

    echo [1/5] Da tim thay .venv dung Python 3.10 - giu nguyen.
    "%VENV_PY%" --version
    goto install_packages
)

rem ============================================================
rem 2. Tim Python 3.10
rem    Uu tien py launcher, PATH, sau do cac vi tri Windows pho bien.
rem ============================================================
echo [1/5] Dang tim Python 3.10...

where py >nul 2>&1
if not errorlevel 1 (
    py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,10) and sys.maxsize > 2**32 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_MODE=launcher"
        goto python_found
    )
)

for /f "delims=" %%P in ('where python 2^>nul') do (
    if not defined PYTHON_EXE call :try_python "%%P"
)

if defined PYTHON_EXE goto python_found

where python3.10 >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('where python3.10 2^>nul') do (
        if not defined PYTHON_EXE call :try_python "%%P"
    )
)

if defined PYTHON_EXE goto python_found

if exist "%LocalAppData%\Programs\Python\Python310\python.exe" (
    call :try_python "%LocalAppData%\Programs\Python\Python310\python.exe"
)
if defined PYTHON_EXE goto python_found

if exist "%ProgramFiles%\Python310\python.exe" (
    call :try_python "%ProgramFiles%\Python310\python.exe"
)
if defined PYTHON_EXE goto python_found

echo.
echo [LOI] Khong tim thay Python 3.10 x64 co the su dung.
echo.
echo Project da duoc kiem thu voi Python 3.10.11 64-bit.
echo Trang tai chinh thuc:
echo %PYTHON_URL%
echo.
choice /C YN /N /M "Mo trang tai Python 3.10.11 chinh thuc? [Y/N]: "
if errorlevel 2 (
    echo.
    echo Da huy mo trinh duyet.
    echo Hay cai Python 3.10.x 64-bit roi chay lai setup_windows.bat.
    echo.
    pause
    exit /b 1
)

echo.
echo Dang mo Python.org...
start "" "%PYTHON_URL%"
echo.
echo Hay chon "Windows installer (64-bit)".
echo Sau khi cai Python xong, chay lai setup_windows.bat.
echo.
pause
exit /b 1

:python_found
echo [OK] Tim thay Python 3.10:
if "%PYTHON_MODE%"=="launcher" (
    py -3.10 --version
) else (
    "%PYTHON_EXE%" --version
)

echo.
echo [2/5] Tao moi truong .venv...
if "%PYTHON_MODE%"=="launcher" (
    py -3.10 -m venv .venv
) else (
    "%PYTHON_EXE%" -m venv .venv
)

if errorlevel 1 (
    echo.
    echo [LOI] Khong tao duoc .venv.
    pause
    exit /b 1
)

if not exist "%VENV_PY%" (
    echo.
    echo [LOI] .venv khong duoc tao day du.
    pause
    exit /b 1
)

:install_packages
echo.
echo [3/5] Cai pip %PIP_VERSION%...
"%VENV_PY%" -m pip install --upgrade pip==%PIP_VERSION%
if errorlevel 1 (
    echo.
    echo [LOI] Khong cai duoc pip %PIP_VERSION%.
    pause
    exit /b 1
)

echo.
echo [4/5] Cai dependency tu %REQ%...
"%VENV_PY%" -m pip install -r "%REQ%"
if errorlevel 1 (
    echo.
    echo [LOI] Cai dependency that bai.
    echo Kiem tra Internet va file %REQ%.
    pause
    exit /b 1
)

echo.
echo [5/5] Kiem tra dependency...
"%VENV_PY%" -m pip check
if errorlevel 1 (
    echo.
    echo [LOI] pip check phat hien dependency bi loi.
    pause
    exit /b 1
)

echo.
echo Kiem tra OpenCV / Paddle / PaddleOCR / Streamlit...
"%VENV_PY%" -c "import cv2, numpy, sklearn, skimage, pandas, streamlit; print('[OK] Scientific stack'); print('OpenCV', cv2.__version__, '| SIFT', hasattr(cv2,'SIFT_create'), '| ORB', hasattr(cv2,'ORB_create'))"
if errorlevel 1 goto import_failed

"%VENV_PY%" -c "import paddle; print('[OK] PaddlePaddle', paddle.__version__)"
if errorlevel 1 goto paddle_failed

"%VENV_PY%" -c "from paddleocr import PaddleOCR; print('[OK] PaddleOCR import')"
if errorlevel 1 goto paddle_failed

echo.
echo ==========================================
echo              SETUP HOAN TAT
echo ==========================================
echo Python project:
"%VENV_PY%" --version
echo Pip:
"%VENV_PY%" -m pip --version
echo.

if exist "Tesseract\tesseract.exe" (
    "Tesseract\tesseract.exe" --version >nul 2>&1
    if errorlevel 1 (
        echo [LUU Y] Tim thay Tesseract local nhung khong chay duoc.
        echo          Kiem tra Windows Security/Application Control.
    ) else (
        echo [OK] Tesseract local chay duoc: Tesseract\tesseract.exe
    )

    if exist "Tesseract\tessdata\vie.traineddata" (
        echo [OK] Tim thay vie.traineddata.
    ) else (
        echo [LUU Y] Thieu Tesseract\tessdata\vie.traineddata.
        echo          OCR Tuan 3 voi tieng Viet co the loi.
    )
) else (
    where tesseract >nul 2>&1
    if errorlevel 1 (
        echo [LUU Y] Chua tim thay Tesseract.
        echo          Chi anh huong Tuan 3 - Tesseract baseline.
    ) else (
        tesseract --version >nul 2>&1
        if errorlevel 1 (
            echo [LUU Y] Tesseract co trong PATH nhung khong chay duoc.
        ) else (
            echo [OK] Tesseract trong PATH chay duoc.
        )
    )
)

echo.
choice /C YN /N /M "Chay check_system.py ngay bay gio? [Y/N]: "
if errorlevel 2 goto setup_done

echo.
if exist "check_system.py" (
    echo Dang chay check_system.py...
    echo.
    "%VENV_PY%" "check_system.py"
) else (
    echo [LUU Y] Khong tim thay check_system.py trong thu muc project.
)

:setup_done
echo.
echo Tiep theo:
echo   - Neu SYSTEM READY: chay run_project.bat
echo   - Neu co FAIL/WARN: xu ly theo thong bao truoc khi chay project
echo.
pause
endlocal
exit /b 0

:import_failed
echo.
echo [LOI] Khong import duoc scientific stack.
echo Hay chup man hinh loi va gui cho nhom.
echo.
pause
exit /b 1

:paddle_failed
echo.
echo [LOI] Khong nap duoc Paddle/PaddleOCR.
echo Neu thay dong "Application Control policy has blocked this file"
echo thi day la chinh sach bao mat Windows chan DLL, khong phai loi code.
echo Khong nen tat toan bo Windows Defender.
echo.
pause
exit /b 1

:try_python
set "CANDIDATE=%~1"
"%CANDIDATE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,10) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=%CANDIDATE%"
exit /b 0
