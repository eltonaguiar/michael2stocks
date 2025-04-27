@echo off
echo Fix Trading Levels and Rank Stocks
echo =================================
echo.
echo This script will:
echo 1. Fix negative or zero prices
echo 2. Fix negative or zero trading levels (ideal entry, target price, stop loss)
echo 3. Recalculate risk-reward ratios
echo 4. Rank stocks based on a composite score
echo.
echo Results will be RANKED with #1 being the best stock to buy!
echo.

python "%~dp0fix_trading_levels.py"

echo.
echo Process complete!
pause
