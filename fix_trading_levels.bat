@echo off
echo Fix Trading Levels and Rank Stocks
echo =================================
echo.
echo This script will:
echo 1. Find the most recent results file
echo 2. Fix any issues with prices (replace $0 prices)
echo 3. Fix any issues with trading levels (ideal entry, target price, stop loss)
echo 4. Recalculate risk-reward ratios
echo 5. Rank stocks based on a composite score
echo.
echo Results will be RANKED with #1 being the best stock to buy!
echo.

REM Show progress bar for the process
python progress_bar.py --title "Fixing Trading Levels and Ranking Stocks" --steps 50 --duration 5

python "%~dp0fix_trading_levels.py"

echo.
echo =============================================
echo PROCESS COMPLETE! Results are ready to view.
echo =============================================
echo.
echo The results are saved with "_fixed_ranked" suffix
echo The top-ranked stocks are your best candidates for the Perfect Setup pattern.
echo.
pause
