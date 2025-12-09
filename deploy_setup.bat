@echo off
echo ========================================
echo Plant Disease App - Deployment Setup
echo ========================================
echo.

echo Step 1: Checking Git installation...
git --version
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed!
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)
echo Git is installed!
echo.

echo Step 2: Initializing Git repository...
git init
echo.

echo Step 3: Adding all files...
git add .
echo.

echo Step 4: Creating initial commit...
git commit -m "Initial commit - Plant Disease Predictor App"
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Create a new repository on GitHub (https://github.com/new)
echo 2. Run these commands (replace YOUR-USERNAME and YOUR-REPO):
echo.
echo    git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 3. Go to https://share.streamlit.io
echo 4. Sign in with GitHub
echo 5. Click "New app" and select your repository
echo 6. Set main file to: app1.py
echo 7. Click Deploy!
echo.
echo Your app will be live in 5-10 minutes!
echo ========================================
pause
