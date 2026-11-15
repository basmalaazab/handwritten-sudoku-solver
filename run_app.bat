@echo off
echo ========================================================
echo Starting Handwritten Sudoku Reader and Solver Web App...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
