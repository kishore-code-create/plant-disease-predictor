@echo off
echo ========================================
echo Git LFS Setup (Optional)
echo ========================================
echo.
echo Your model file is 41MB - under GitHub's limit!
echo You DON'T need Git LFS for this project.
echo.
echo However, if you want to use Git LFS anyway:
echo.
echo 1. Install Git LFS from: https://git-lfs.github.com/
echo 2. Run: git lfs install
echo 3. Run: git lfs track "*.h5"
echo 4. Run: git add .gitattributes
echo 5. Run: git add plant_disease_model_15_class.h5
echo 6. Run: git commit -m "Add model with LFS"
echo.
pause
