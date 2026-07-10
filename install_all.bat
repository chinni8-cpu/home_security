@echo off
echo ========================================
echo Installing CCTV Monitoring System
echo ========================================

echo.
echo Step 1: Installing Flask and Database packages...
pip install flask flask-sqlalchemy pymysql
echo.

echo Step 2: Installing OpenCV and Image processing...
pip install opencv-python numpy pillow
echo.

echo Step 3: Installing Face Recognition...
echo Trying dlib-bin...
pip install dlib-bin
if %errorlevel% neq 0 (
    echo dlib-bin failed, trying mediapipe...
    pip install mediapipe
)
echo.

echo Step 4: Installing additional packages...
pip install werkzeug python-dotenv
echo.

echo Step 5: Creating project directories...
mkdir uploads 2>nul
mkdir uploads\family_members 2>nul
mkdir static 2>nul
mkdir static\css 2>nul
mkdir static\js 2>nul
mkdir templates 2>nul
echo.

echo ========================================
echo Installation Complete!
echo To run: python app.py
echo ========================================
pause