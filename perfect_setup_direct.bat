@echo off
echo Perfect Setup Direct Analysis
echo ===========================
echo.
echo FILTERS APPLIED:
echo 1. Monthly RSI under 25 with RSI period 14
echo 2. Perfect setup 4 steps
echo 3. 1 million shares traded daily volume average
echo.
echo IMPORTANT: All liquidity filters are DISABLED:
echo - Market cap minimum: 0 (disabled)
echo - Price minimum: 0 (disabled)
echo - RS minimum: 0 (disabled)
echo - Revenue growth minimum: 0 (disabled)
echo.
echo Running scan with all correct parameters...
echo.

python custom_screen.py --output perfect_setup_direct.csv --max-stocks 1000 --channels --rsi --rsi-period 14 --monthly-rsi-max 25.0 --volume-min 1000000 --market-cap-min 0 --price-min 0 --rs-min 0 --revenue-growth-min 0 --rs-bypass 0

echo.
echo Fixing trading levels and ranking stocks...
echo.

python fix_trading_levels.py

echo.
echo Process complete!
pause
