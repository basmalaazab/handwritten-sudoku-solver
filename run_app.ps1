Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting Handwritten Sudoku Reader and Solver Web App..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

python -m streamlit run app.py
