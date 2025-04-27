@echo off
echo Perfect Setup Complete Analysis
echo ==============================
echo.
echo FILTERS APPLIED:
echo 1. Monthly RSI under 25 with RSI period 14
echo 2. Perfect setup 4 steps:
echo    - Down-trending channel with clear support and resistance
echo    - Psychological bottom (W-pattern, triple bottom, reverse H^&S)
echo    - Price jumps to resistance, gets rejected, and sells off
echo    - Buy on support of trading channel (bottom is in)
echo 3. 1 million shares traded daily volume average
echo.
echo Results will be RANKED with #1 being the best stock to buy!
echo.
echo This script will:
echo 1. Run the custom screen with perfect setup parameters
echo 2. Fix any negative or zero prices
echo 3. Fix any negative or zero trading levels
echo 4. Recalculate risk-reward ratios
echo 5. Rank stocks based on a composite score
echo.

python "%~dp0perfect_setup_complete.py"

echo.
echo Process complete!
pause
