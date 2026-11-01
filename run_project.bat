@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ============================================================
rem Unicode / Vietnamese console support
rem ============================================================
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

rem Paddle / Streamlit
set "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

title OCR SoDo - Run Project

set "PY=.venv\Scripts\python.exe"
set "RUNNER=run_with_heartbeat.py"

rem ============================================================
rem KIEM TRA MOI TRUONG
rem ============================================================
if not exist "%PY%" (
    echo.
    echo [LOI] Chua co moi truong .venv.
    echo Hay chay setup_windows.bat truoc.
    echo.
    pause
    exit /b 1
)

if not exist "%RUNNER%" (
    echo.
    echo [LOI] Khong tim thay %RUNNER%.
    echo File nay phai nam cung thu muc voi run_project.bat.
    echo.
    pause
    exit /b 1
)

rem ============================================================
rem KIEM TRA SOURCE
rem ============================================================
set "MISSING=0"

call :check_file "src\tuan1_dataset_tool_auto.py"
call :check_file "src\tuan1_dataset_tool_semi_auto.py"
call :check_file "src\tuan2_preprocess.py"
call :check_file "src\tuan3_baseline.py"
call :check_file "src\tuan4_pipeline.py"
call :check_file "src\tuan5_evaluate.py"
call :check_file "src\tuan7_app.py"

if "%MISSING%"=="1" (
    echo.
    echo [LOI] Bo source release khong day du.
    echo Kiem tra cac file [THIEU] o tren.
    echo.
    pause
    exit /b 1
)


rem ============================================================
rem MENU CHINH
rem ============================================================
:menu
cls

echo ============================================================
echo                 OCR SO DO / SO HONG
echo ============================================================
echo.
echo 1. Tuan 1 - Dataset / che PII / synthetic tamper
echo 2. Tuan 2 - Preprocess / tach ROI
echo 3. Tuan 3 - HSV / FFT / Tesseract baseline
echo 4. Tuan 4 - Proposed pipeline
echo 5. Tuan 5 - Danh gia thuc nghiem
echo 7. Tuan 7 - Streamlit demo
echo 0. Thoat
echo.

set /p "choice=Nhap lua chon: "

if "%choice%"=="1" goto week1_menu
if "%choice%"=="2" goto week2
if "%choice%"=="3" goto week3
if "%choice%"=="4" goto week4
if "%choice%"=="5" goto week5
if "%choice%"=="7" goto week7
if "%choice%"=="0" goto exit_program

echo.
echo [LOI] Lua chon khong hop le.
echo.
pause
goto menu


rem ============================================================
rem TUAN 1 MENU
rem ============================================================
:week1_menu
cls

echo ============================================================
echo                         TUAN 1
echo ============================================================
echo.
echo 1. AUTO      - Ban chinh, co the chon so anh
echo 2. SEMI-AUTO - Ho tro thu cong
echo 0. Quay lai
echo.

set /p "w1choice=Nhap lua chon: "

if "%w1choice%"=="1" goto week1_auto
if "%w1choice%"=="2" goto week1_semi
if "%w1choice%"=="0" goto menu

echo.
echo [LOI] Lua chon khong hop le.
echo.
pause
goto week1_menu


rem ============================================================
rem TUAN 1 AUTO
rem ============================================================
:week1_auto

if not exist "dataset\raw" (
    echo.
    echo [LOI] Khong tim thay:
    echo       dataset\raw
    echo.
    pause
    goto menu
)

rem ------------------------------------------------------------
rem Dem anh raw bang Python trong .venv.
rem Tranh loi dem sai do PowerShell / escaping Batch.
rem ------------------------------------------------------------
set "RAW_COUNT=0"
set "RAW_COUNT_FILE=%TEMP%\ocr_sodo_raw_count_%RANDOM%.txt"

"%PY%" -c "from pathlib import Path; exts={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}; p=Path(r'dataset/raw'); print(sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts))" > "%RAW_COUNT_FILE%" 2>nul

if errorlevel 1 (
    echo.
    echo [LOI] Khong dem duoc anh trong dataset\raw.
    echo Kiem tra .venv va quyen truy cap thu muc.
    echo.

    if exist "%RAW_COUNT_FILE%" (
        del /q "%RAW_COUNT_FILE%" >nul 2>&1
    )

    pause
    goto menu
)

set /p RAW_COUNT=<"%RAW_COUNT_FILE%"
del /q "%RAW_COUNT_FILE%" >nul 2>&1

if not defined RAW_COUNT (
    set "RAW_COUNT=0"
)

if "%RAW_COUNT%"=="0" (
    echo.
    echo [LOI] dataset\raw ton tai nhung khong co anh hop le.
    echo.
    echo Dinh dang ho tro:
    echo JPG JPEG PNG WEBP BMP TIF TIFF
    echo.
    pause
    goto menu
)

echo.
echo ============================================================
echo TUAN 1 AUTO
echo ============================================================
echo.
echo Tim thay %RAW_COUNT% anh raw.
echo.
echo - Enter        : xu ly tat ca anh
echo - Nhap 5 / 10 : smoke-test nhanh
echo.
echo Luu y:
echo Chay subset chi phu hop de test code.
echo Neu can tai tao dung dataset trong bao cao, hay chay full.
echo.

set "WEEK1_MAX_FILES="
set "W1COUNT="

set /p "W1COUNT=Nhap so anh muon xu ly [1-%RAW_COUNT%], Enter = tat ca: "

if "%W1COUNT%"=="" (
    goto week1_auto_run
)

rem ------------------------------------------------------------
rem Kiem tra input co phai so nguyen khong
rem ------------------------------------------------------------
echo %W1COUNT%| findstr /R "^[0-9][0-9]*$" >nul

if errorlevel 1 (
    echo.
    echo [LOI] Vui long nhap mot so nguyen.
    echo.
    pause
    goto week1_auto
)

set /a W1NUM=%W1COUNT%

if %W1NUM% LSS 1 (
    echo.
    echo [LOI] So anh phai tu 1 tro len.
    echo.
    pause
    goto week1_auto
)

if %W1NUM% GTR %RAW_COUNT% (
    echo.
    echo [LUU Y] So anh lon hon tong dataset.
    echo He thong se xu ly %RAW_COUNT% anh.
    set "W1NUM=%RAW_COUNT%"
)

set "WEEK1_MAX_FILES=%W1NUM%"


:week1_auto_run

echo.
if defined WEEK1_MAX_FILES (
    echo [INFO] Xu ly %WEEK1_MAX_FILES%/%RAW_COUNT% anh.
) else (
    echo [INFO] Xu ly FULL %RAW_COUNT% anh.
)

echo.
echo [INFO] PaddleOCR CPU co the mat nhieu thoi gian.
echo [INFO] Neu khong co log moi, heartbeat se thong bao
echo        chuong trinh van dang xu ly.
echo.

call :run_task "src\tuan1_dataset_tool_auto.py" "Tuan 1 AUTO"

set "WEEK1_MAX_FILES="

goto endrun


rem ============================================================
rem TUAN 1 SEMI AUTO
rem ============================================================
:week1_semi

echo.
echo [INFO] Dang mo Tuan 1 SEMI-AUTO.
echo.

call :run_task "src\tuan1_dataset_tool_semi_auto.py" "Tuan 1 SEMI-AUTO"

goto endrun


rem ============================================================
rem TUAN 2
rem ============================================================
:week2

call :check_week1_data

if errorlevel 1 (
    goto menu
)

echo.
echo ============================================================
echo TUAN 2 - PREPROCESS
echo ============================================================
echo.
echo [INFO] Preprocess + tach ROI text / diagram / seal.
echo.
echo [INFO] Neu mot buoc xu ly lau, heartbeat se xac nhan
echo        chuong trinh van dang chay.
echo.

call :run_task "src\tuan2_preprocess.py" "Tuan 2 Preprocess"

goto endrun


rem ============================================================
rem TUAN 3
rem ============================================================
:week3

call :check_processed

if errorlevel 1 (
    goto menu
)

echo.
echo ============================================================
echo TUAN 3 - BASELINE
echo ============================================================
echo.

rem ------------------------------------------------------------
rem Kiem tra Tesseract portable
rem ------------------------------------------------------------
if exist "Tesseract\tesseract.exe" (

    "Tesseract\tesseract.exe" --version >nul 2>&1

    if errorlevel 1 (
        echo [LUU Y] Tim thay Tesseract local nhung khong chay duoc.
        echo Co the bi Windows Security/Application Control chan.
    ) else (
        echo [OK] Tesseract local:
        echo      Tesseract\tesseract.exe
    )

) else (

    where tesseract >nul 2>&1

    if errorlevel 1 (
        echo [LUU Y] Khong tim thay Tesseract.
        echo Phan OCR baseline Tuan 3 co the khong chay.
    ) else (
        echo [OK] Tim thay Tesseract trong PATH.
    )
)

if exist "Tesseract\tessdata\vie.traineddata" (
    echo [OK] Tim thay vie.traineddata.
) else (
    echo [LUU Y] Khong thay vie.traineddata.
)

echo.

call :run_task "src\tuan3_baseline.py" "Tuan 3 Baseline"

goto endrun


rem ============================================================
rem TUAN 4
rem ============================================================
:week4

call :check_processed

if errorlevel 1 (
    goto menu
)

echo.
echo ============================================================
echo TUAN 4 - PROPOSED PIPELINE
echo ============================================================
echo.
echo [INFO] Geometry:
echo        Skeletonization + Hough
echo.
echo [INFO] OCR:
echo        Morphology + PaddleOCR
echo.
echo [INFO] Seal:
echo        HSV + LBP
echo.
echo [INFO] Lan dau PaddleOCR khoi tao model co the lau.
echo [INFO] Khong dong cua so neu heartbeat van xuat hien.
echo.

call :run_task "src\tuan4_pipeline.py" "Tuan 4 Proposed Pipeline"

goto endrun


rem ============================================================
rem TUAN 5
rem ============================================================
:week5

call :check_processed

if errorlevel 1 (
    goto menu
)

call :warn_subset

echo.
echo ============================================================
echo TUAN 5 - DANH GIA THUC NGHIEM
echo ============================================================
echo.
echo [INFO] Bao gom:
echo        - Tamper benchmark
echo        - OCR controlled benchmark
echo        - Geometry controlled benchmark
echo.
echo [INFO] PaddleOCR va benchmark co the mat thoi gian.
echo [INFO] Heartbeat se bao neu chuong trinh van dang chay.
echo.

call :run_task "src\tuan5_evaluate.py" "Tuan 5 Evaluation"

goto endrun


rem ============================================================
rem TUAN 7
rem ============================================================
:week7

call :check_processed

if errorlevel 1 (
    goto menu
)

echo.
echo ============================================================
echo TUAN 7 - STREAMLIT
echo ============================================================
echo.
echo Dang khoi dong giao dien...
echo.
echo Streamlit se:
echo - khong hoi email
echo - chay local
echo - tu mo browser
echo.


rem ------------------------------------------------------------
rem Tim port local dang trong
rem ------------------------------------------------------------
set "PORT="

for /f %%P in ('powershell -NoProfile -Command "$l=New-Object System.Net.Sockets.TcpListener([Net.IPAddress]::Loopback,0); $l.Start(); $p=$l.LocalEndpoint.Port; $l.Stop(); Write-Output $p"') do (
    set "PORT=%%P"
)

if not defined PORT (
    set "PORT=8501"
)

set "APP_URL=http://127.0.0.1:%PORT%"

echo Dia chi giao dien:
echo %APP_URL%
echo.

rem ------------------------------------------------------------
rem Streamlit:
rem headless = bo prompt email
rem telemetry = off
rem ------------------------------------------------------------
start "OCR SoDo - Streamlit" "%ComSpec%" /k ""%PY%" -m streamlit run "src\tuan7_app.py" --server.headless true --browser.gatherUsageStats false --server.address 127.0.0.1 --server.port %PORT%"

echo Dang cho server Streamlit san sang...

rem ------------------------------------------------------------
rem Cho toi da ~30 giay de server khoi dong
rem ------------------------------------------------------------
powershell -NoProfile -Command "$u='%APP_URL%'; for($i=0;$i -lt 30;$i++){try{Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 1 | Out-Null; exit 0}catch{Start-Sleep -Seconds 1}}; exit 1" >nul 2>&1

if errorlevel 1 (

    echo.
    echo [LUU Y] Streamlit chua phan hoi sau khi cho.
    echo.
    echo Kiem tra cua so:
    echo OCR SoDo - Streamlit
    echo.
    echo Hoac mo thu cong:
    echo %APP_URL%
    echo.

) else (

    echo.
    echo [OK] Streamlit da san sang.
    echo Dang mo trinh duyet...
    echo.
)

start "" "%APP_URL%"

echo Neu browser khong tu mo, truy cap:
echo %APP_URL%
echo.

pause

goto menu


rem ============================================================
rem CHAY PYTHON + HEARTBEAT
rem ============================================================
:run_task

set "FILE=%~1"
set "LABEL=%~2"

echo.
echo ------------------------------------------------------------
echo Dang chay:
echo %LABEL%
echo ------------------------------------------------------------
echo.

"%PY%" -u "%RUNNER%" "%FILE%" --label "%LABEL%" --heartbeat 12

exit /b %errorlevel%


rem ============================================================
rem KIEM TRA DATA TUAN 1
rem ============================================================
:check_week1_data

if exist "dataset\real" if exist "dataset\tampered" (
    exit /b 0
)

echo.
echo [LOI] Chua co:
echo       dataset\real
echo       dataset\tampered
echo.
echo Hay chay Tuan 1 truoc.
echo.

pause

exit /b 1


rem ============================================================
rem KIEM TRA DATA TUAN 2
rem ============================================================
:check_processed

if exist "dataset\processed_real\text" if exist "dataset\processed_real\diagram" if exist "dataset\processed_real\seal" (
    exit /b 0
)

echo.
echo [LOI] Chua co du lieu preprocess can thiet.
echo.
echo Hay chay Tuan 2 truoc khi chay:
echo Tuan 3 / Tuan 4 / Tuan 5 / Tuan 7
echo.

pause

exit /b 1


rem ============================================================
rem CANH BAO NEU TUAN 1 CHI CHAY SUBSET
rem ============================================================
:warn_subset

if not exist "dataset\week1_run_info.txt" (
    exit /b 0
)

set "RAW_TOTAL="
set "SELECTED="

for /f "tokens=1,2 delims==" %%A in (dataset\week1_run_info.txt) do (

    if /I "%%A"=="raw_total" (
        set "RAW_TOTAL=%%B"
    )

    if /I "%%A"=="selected" (
        set "SELECTED=%%B"
    )
)

if defined RAW_TOTAL if defined SELECTED if not "%RAW_TOTAL%"=="%SELECTED%" (

    echo.
    echo ============================================================
    echo [LUU Y]
    echo Tuan 1 truoc do chi chay subset:
    echo %SELECTED%/%RAW_TOTAL% anh
    echo.
    echo Metric Tuan 5 co the khac ket qua full dataset trong bao cao.
    echo ============================================================
    echo.
)

exit /b 0


rem ============================================================
rem KIEM TRA FILE
rem ============================================================
:check_file

if not exist "%~1" (
    echo [THIEU] %~1
    set "MISSING=1"
)

exit /b 0


rem ============================================================
rem KET THUC MOT TUAN
rem ============================================================
:endrun

echo.
echo ============================================================
echo Chuong trinh da ket thuc.
echo Nhan phim bat ky de quay lai menu.
echo ============================================================
echo.

pause

goto menu


rem ============================================================
rem THOAT
rem ============================================================
:exit_program

endlocal
exit /b 0