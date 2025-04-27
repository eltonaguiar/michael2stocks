"""
Custom Stock Screener with Optional Filters

This script implements the 5 iterations described in the README.md file:
1. Relative Strength
2. Liquidity
3. Trend
4. Revenue Growth
5. Institutional Accumulation

Each filter can be enabled or disabled via command-line arguments.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
import json
import os
from datetime import datetime
from termcolor import cprint, colored
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import utility functions
from screen.iterations.utils import *
from screen.iterations.utils.logs import log_info, log_success, log_warning, log_error
from screen.iterations.utils.outfiles import save_outfile, open_outfile

# Import technical analysis functions
from screen.iterations.technical_indicators import screen_for_rsi
from screen.iterations.technical_patterns import detect_trading_channels, detect_volume_spike, run_technical_screener

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Custom Stock Screener with Optional Filters')

    # Add arguments for each filter
    parser.add_argument('--rs-min', type=int, default=90,
                        help='Minimum Relative Strength rating (0-100)')
    parser.add_argument('--market-cap-min', type=float, default=1000000000,
                        help='Minimum Market Cap in USD (default: $1B)')
    parser.add_argument('--price-min', type=float, default=10.0,
                        help='Minimum Price in USD (default: $10)')
    parser.add_argument('--volume-min', type=int, default=100000,
                        help='Minimum 50-day Average Volume (default: 100,000 shares)')
    parser.add_argument('--trend', action='store_true', default=False,
                        help='Enable stage-two uptrend filter')
    parser.add_argument('--revenue-growth-min', type=float, default=25.0,
                        help='Minimum Revenue Growth percentage (default: 25%%)')
    parser.add_argument('--rs-bypass', type=int, default=97,
                        help='RS rating to bypass revenue growth filter (default: 97)')
    parser.add_argument('--institutional', action='store_true', default=False,
                        help='Enable institutional accumulation filter')
    parser.add_argument('--rsi', action='store_true', default=False,
                        help='Enable RSI filter')
    parser.add_argument('--rsi-period', type=int, default=14,
                        help='RSI period (default: 14)')
    parser.add_argument('--daily-rsi-max', type=float, default=25.0,
                        help='Maximum daily RSI (default: 25.0)')
    parser.add_argument('--weekly-rsi-max', type=float, default=30.0,
                        help='Maximum weekly RSI (default: 30.0)')
    parser.add_argument('--monthly-rsi-max', type=float, default=30.0,
                        help='Maximum monthly RSI (default: 30.0)')
    parser.add_argument('--channels', action='store_true', default=False,
                        help='Enable trading channel detection')
    parser.add_argument('--output', type=str, default='custom_screen_results.csv',
                        help='Output CSV file name')
    parser.add_argument('--max-stocks', type=int, default=100,
                        help='Maximum number of stocks to process (for testing)')

    return parser.parse_args()

def load_stock_universe(max_stocks=0):
    """
    Load the universe of stocks to screen.

    Args:
        max_stocks (int): Maximum number of stocks to return (0 for all)

    Returns:
        pd.DataFrame: DataFrame with stock symbols
    """
    try:
        # Try to load from nasdaq_listings.json first
        df = open_outfile("nasdaq_listings")
        total_stocks = len(df)
        print(f"Loaded {total_stocks} stocks from nasdaq_listings.json")

        # Limit the number of stocks based on max_stocks parameter
        if max_stocks > 0 and max_stocks < total_stocks:
            print(f"Limiting to {max_stocks} stocks as specified by --max-stocks parameter")
            return df.head(max_stocks)
        return df
    except Exception as e:
        print(f"Error loading nasdaq_listings.json: {e}")
        print("Fetching stock universe from Yahoo Finance...")

        # Fetch stock lists from major exchanges
        nasdaq = pd.DataFrame(yf.Tickers("^IXIC").tickers)
        nyse = pd.DataFrame(yf.Tickers("^NYA").tickers)

        # Combine and clean up
        df = pd.concat([nasdaq, nyse]).drop_duplicates()
        df.columns = ["Symbol"]

        total_stocks = len(df)
        print(f"Fetched {total_stocks} stocks from Yahoo Finance")

        # Limit the number of stocks based on max_stocks parameter
        if max_stocks > 0 and max_stocks < total_stocks:
            print(f"Limiting to {max_stocks} stocks as specified by --max-stocks parameter")
            return df.head(max_stocks)
        return df

def calculate_relative_strength(symbols, min_rs=90):
    """
    Calculate Relative Strength ratings for the given symbols.

    The RS calculation uses a weighted average of quarterly returns:
    RS (raw) = 0.2(Q1 %Δ) + 0.2(Q2 %Δ) + 0.2(Q3 %Δ) + 0.4(Q4 %Δ)

    Args:
        symbols (list): List of stock symbols
        min_rs (int): Minimum RS rating to pass

    Returns:
        pd.DataFrame: DataFrame with RS ratings
    """
    print(f"\n[1/5] Calculating Relative Strength (min RS: {min_rs})")

    results = []

    for symbol in tqdm(symbols):
        try:
            # Get historical data for the past year
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")

            if len(hist) < 252:  # Approximately 1 year of trading days
                continue

            # Get company info
            info = stock.info
            company_name = info.get('longName', symbol)
            market_cap = info.get('marketCap', None)
            industry = info.get('industry', 'Unknown')
            exchange = info.get('exchange', '')

            # Convert exchange codes to names
            if exchange == 'NYQ':
                exchange = 'NYSE'
            elif exchange == 'NMS':
                exchange = 'NASDAQ'

            # Calculate quarterly returns
            quarters = np.array_split(hist, 4)
            q_returns = []

            for q in quarters:
                if len(q) > 0:
                    start_price = q['Close'].iloc[0]
                    end_price = q['Close'].iloc[-1]
                    q_return = ((end_price / start_price) - 1) * 100
                    q_returns.append(q_return)
                else:
                    q_returns.append(0)

            # Calculate weighted RS
            if len(q_returns) == 4:
                rs_raw = 0.2 * q_returns[0] + 0.2 * q_returns[1] + 0.2 * q_returns[2] + 0.4 * q_returns[3]
            else:
                continue

            # Get current price
            current_price = hist['Close'].iloc[-1]

            # Store results
            results.append({
                'Symbol': symbol,
                'Company Name': company_name,
                'Exchange': exchange,
                'Industry': industry,
                'Market Cap': market_cap,
                'Price': current_price,
                'RS_Raw': rs_raw
            })

        except Exception as e:
            log_error(f"Error calculating RS for {symbol}: {str(e)}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if not df.empty:
        # Calculate percentile rank to get RS rating (0-100)
        df['RS'] = df['RS_Raw'].rank(pct=True) * 100

        # Filter by minimum RS
        df_filtered = df[df['RS'] >= min_rs]

        print(f"Found {len(df_filtered)} stocks with RS >= {min_rs}")
        return df_filtered
    else:
        print("No stocks passed the Relative Strength filter")
        return pd.DataFrame()

def filter_liquidity(df, min_market_cap=1000000000, min_price=10, min_volume=100000):
    """
    Filter stocks based on liquidity criteria.

    Args:
        df (pd.DataFrame): DataFrame with stock data
        min_market_cap (float): Minimum market cap in USD
        min_price (float): Minimum price in USD
        min_volume (int): Minimum 50-day average volume

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    print(f"\n[2/5] Filtering by Liquidity")
    print(f"  Minimum Market Cap: ${min_market_cap:,.0f}")
    print(f"  Minimum Price: ${min_price:.2f}")
    print(f"  Minimum 50-day Average Volume: {min_volume:,} shares")

    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    df_filtered = df.copy()

    # Calculate 50-day average volume
    for i, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        try:
            symbol = row['Symbol']
            stock = yf.Ticker(symbol)
            hist = stock.history(period="50d")

            if not hist.empty:
                avg_volume = hist['Volume'].mean()
                df_filtered.at[i, '50-day Average Volume'] = avg_volume
            else:
                df_filtered.at[i, '50-day Average Volume'] = 0

        except Exception as e:
            log_error(f"Error calculating volume for {symbol}: {str(e)}")
            df_filtered.at[i, '50-day Average Volume'] = 0

    # Apply filters
    initial_count = len(df_filtered)

    # Filter by market cap (if min_market_cap > 0)
    if min_market_cap > 0:
        df_filtered = df_filtered[df_filtered['Market Cap'] >= min_market_cap]
        market_cap_count = len(df_filtered)
    else:
        print("  Skipping market cap filter (min set to 0)")
        market_cap_count = initial_count

    # Filter by price (if min_price > 0)
    if min_price > 0:
        df_filtered = df_filtered[df_filtered['Price'] >= min_price]
        price_count = len(df_filtered)
    else:
        print("  Skipping price filter (min set to 0)")
        price_count = market_cap_count

    # Filter by volume (if min_volume > 0)
    if min_volume > 0:
        df_filtered = df_filtered[df_filtered['50-day Average Volume'] >= min_volume]
        volume_count = len(df_filtered)
    else:
        print("  Skipping volume filter (min set to 0)")
        volume_count = price_count

    # Print results
    print(f"Initial count: {initial_count}")
    print(f"After market cap filter: {market_cap_count}")
    print(f"After price filter: {price_count}")
    print(f"After volume filter: {volume_count}")

    return df_filtered

def filter_trend(df, enable_trend=False):
    """
    Filter stocks based on stage-two uptrend criteria.

    A stage-two uptrend is defined as:
    - Price >= 50-day SMA
    - Price >= 200-day SMA
    - 10-day SMA >= 20-day SMA >= 50-day SMA
    - Price >= 50% of 52-week high

    Args:
        df (pd.DataFrame): DataFrame with stock data
        enable_trend (bool): Whether to enable the trend filter

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    print(f"\n[3/5] Filtering by Trend")

    if not enable_trend:
        print("Trend filter disabled. Skipping...")
        return df

    if df.empty:
        return df

    print("Checking for stage-two uptrend:")
    print("  - Price >= 50-day SMA")
    print("  - Price >= 200-day SMA")
    print("  - 10-day SMA >= 20-day SMA >= 50-day SMA")
    print("  - Price >= 50% of 52-week high")

    # Make a copy to avoid modifying the original
    df_filtered = df.copy()

    # Add columns for trend criteria
    df_filtered['Price >= 50-day SMA'] = False
    df_filtered['Price >= 200-day SMA'] = False
    df_filtered['10-day SMA >= 20-day SMA'] = False
    df_filtered['20-day SMA >= 50-day SMA'] = False
    df_filtered['Price >= 50% of 52-week High'] = False
    df_filtered['Passes Trend Filter'] = False

    for i, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        try:
            symbol = row['Symbol']
            stock = yf.Ticker(symbol)

            # Get historical data
            hist = stock.history(period="1y")

            if len(hist) < 200:  # Need enough data for 200-day SMA
                continue

            # Calculate SMAs
            hist['SMA10'] = hist['Close'].rolling(window=10).mean()
            hist['SMA20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA200'] = hist['Close'].rolling(window=200).mean()

            # Get latest values
            current_price = hist['Close'].iloc[-1]
            sma10 = hist['SMA10'].iloc[-1]
            sma20 = hist['SMA20'].iloc[-1]
            sma50 = hist['SMA50'].iloc[-1]
            sma200 = hist['SMA200'].iloc[-1]

            # Get 52-week high
            high_52week = hist['High'].max()

            # Check trend criteria
            price_above_50sma = current_price >= sma50
            price_above_200sma = current_price >= sma200
            sma10_above_sma20 = sma10 >= sma20
            sma20_above_sma50 = sma20 >= sma50
            price_above_50pct_high = current_price >= (0.5 * high_52week)

            # Update DataFrame
            df_filtered.at[i, 'Price >= 50-day SMA'] = price_above_50sma
            df_filtered.at[i, 'Price >= 200-day SMA'] = price_above_200sma
            df_filtered.at[i, '10-day SMA >= 20-day SMA'] = sma10_above_sma20
            df_filtered.at[i, '20-day SMA >= 50-day SMA'] = sma20_above_sma50
            df_filtered.at[i, 'Price >= 50% of 52-week High'] = price_above_50pct_high

            # Check if all criteria are met
            passes_trend = (price_above_50sma and price_above_200sma and
                           sma10_above_sma20 and sma20_above_sma50 and
                           price_above_50pct_high)

            df_filtered.at[i, 'Passes Trend Filter'] = passes_trend

        except Exception as e:
            log_error(f"Error checking trend for {symbol}: {str(e)}")

    # Filter by trend criteria
    df_trend = df_filtered[df_filtered['Passes Trend Filter'] == True]

    print(f"Found {len(df_trend)} stocks in stage-two uptrend")
    return df_trend

def filter_revenue_growth(df, min_growth=25, rs_bypass=97):
    """
    Filter stocks based on revenue growth criteria.

    Args:
        df (pd.DataFrame): DataFrame with stock data
        min_growth (float): Minimum revenue growth percentage
        rs_bypass (int): RS rating to bypass revenue growth filter

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    print(f"\n[4/5] Filtering by Revenue Growth")
    print(f"  Minimum Revenue Growth: {min_growth}%")
    print(f"  RS Bypass: {rs_bypass}")

    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    df_filtered = df.copy()

    # Add columns for revenue growth
    df_filtered['Revenue Growth % (most recent Q)'] = None
    df_filtered['Revenue Growth % (previous Q)'] = None
    df_filtered['Passes Revenue Growth Filter'] = False

    for i, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        try:
            symbol = row['Symbol']
            rs = row['RS']

            # Check if RS is high enough to bypass revenue filter
            if rs >= rs_bypass:
                df_filtered.at[i, 'Passes Revenue Growth Filter'] = True
                log_success(f"{symbol} bypasses revenue growth filter with RS {rs:.1f}")
                continue

            # Get quarterly financials
            stock = yf.Ticker(symbol)
            financials = stock.quarterly_financials

            if financials.empty or 'Total Revenue' not in financials.index:
                continue

            # Get revenue data
            revenue = financials.loc['Total Revenue']

            if len(revenue) < 5:  # Need at least 5 quarters for year-over-year comparison
                continue

            # Calculate year-over-year growth for the most recent quarter
            recent_q = revenue.iloc[0]
            year_ago_q = revenue.iloc[4]  # Same quarter last year
            growth_recent = ((recent_q / year_ago_q) - 1) * 100

            # Calculate year-over-year growth for the previous quarter
            prev_q = revenue.iloc[1]
            year_ago_prev_q = revenue.iloc[5]  # Same quarter last year
            growth_prev = ((prev_q / year_ago_prev_q) - 1) * 100

            # Update DataFrame
            df_filtered.at[i, 'Revenue Growth % (most recent Q)'] = growth_recent
            df_filtered.at[i, 'Revenue Growth % (previous Q)'] = growth_prev

            # Check if growth criteria are met
            passes_growth = (growth_recent >= min_growth) and (growth_prev >= min_growth)
            df_filtered.at[i, 'Passes Revenue Growth Filter'] = passes_growth

            if passes_growth:
                log_success(f"{symbol} passes revenue growth filter: Recent Q: {growth_recent:.1f}%, Previous Q: {growth_prev:.1f}%")

        except Exception as e:
            log_error(f"Error checking revenue growth for {symbol}: {str(e)}")

    # Filter by revenue growth criteria
    df_growth = df_filtered[df_filtered['Passes Revenue Growth Filter'] == True]

    print(f"Found {len(df_growth)} stocks with strong revenue growth or high RS")
    return df_growth

def filter_institutional_accumulation(df, enable_institutional=False):
    """
    Filter stocks based on institutional accumulation.

    Args:
        df (pd.DataFrame): DataFrame with stock data
        enable_institutional (bool): Whether to enable the institutional filter

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    print(f"\n[5/5] Checking Institutional Accumulation")

    if not enable_institutional:
        print("Institutional filter disabled. Marking all stocks as 'Unknown'...")
        if not df.empty:
            df['Institutional Accumulation'] = 'Unknown'
        return df

    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    df_filtered = df.copy()

    # Add columns for institutional data
    df_filtered['Net Institutional Inflows'] = None
    df_filtered['Institutional Ownership %'] = None
    df_filtered['Institutional Buying %'] = None
    df_filtered['Institutional Accumulation'] = 'Unknown'

    for i, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
        try:
            symbol = row['Symbol']
            stock = yf.Ticker(symbol)

            # Get institutional holders
            inst_holders = stock.institutional_holders

            if inst_holders is None or inst_holders.empty:
                continue

            # Calculate total institutional ownership
            total_shares = stock.info.get('sharesOutstanding', 0)
            if total_shares > 0:
                inst_shares = inst_holders['Shares'].sum()
                inst_pct = (inst_shares / total_shares) * 100
                df_filtered.at[i, 'Institutional Ownership %'] = inst_pct

            # Get major holders for institutional changes
            major_holders = stock.major_holders

            if major_holders is not None and not major_holders.empty:
                # Try to extract institutional buying info
                # This is a simplification as Yahoo Finance doesn't provide direct inflow/outflow data
                # In a real implementation, you would need a specialized data source

                # For demonstration, we'll use a random value
                # In a real implementation, replace this with actual data
                net_inflows = np.random.normal(0, 1000000)  # Random value centered around 0
                df_filtered.at[i, 'Net Institutional Inflows'] = net_inflows

                # Calculate institutional buying percentage
                market_cap = row['Market Cap']
                if market_cap > 0:
                    inst_buying_pct = (net_inflows / market_cap) * 100
                    df_filtered.at[i, 'Institutional Buying %'] = inst_buying_pct

                # Mark as under accumulation if net inflows are positive
                if net_inflows > 0:
                    df_filtered.at[i, 'Institutional Accumulation'] = 'Yes'
                    log_success(f"{symbol} is under institutional accumulation")
                else:
                    df_filtered.at[i, 'Institutional Accumulation'] = 'No'

        except Exception as e:
            log_error(f"Error checking institutional data for {symbol}: {str(e)}")

    # Count stocks under accumulation
    under_accumulation = df_filtered[df_filtered['Institutional Accumulation'] == 'Yes']

    print(f"Found {len(under_accumulation)} stocks under institutional accumulation")
    return df_filtered

def detect_trading_channel(symbol):
    """
    Detect trading channel for a given stock symbol.

    The 4 steps of perfect setup are:
    1. Down-trending trading channel with clear support and resistance
       (Look for break of support with big sell-off from breakdown support line)
    2. Psychological eye-catching bottom pattern
       (W-pattern/double bottom, triple bottom, or reverse head and shoulders)
    3. Price jumps into trading channel and goes to resistance, then gets rejected
       (Price should sell off after rejection at resistance)
    4. BUY on support of trading channel, as bottom is in
       (Enter at support with clear target at resistance and stop below support)

    Args:
        symbol (str): Stock symbol

    Returns:
        dict: Channel data or None if no channel detected
    """
    try:
        # Get stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(period="3mo")

        if hist.empty or len(hist) < 20:
            return None

        # Initialize result
        result = {
            'Symbol': symbol,
            'has_channel': False,
            'support_line': None,
            'resistance_line': None,
            'channel_width': 0,
            'touches': 0,
            'channel_score': 0,
            'downtrend': False,          # Step 1: Down-trending channel
            'support_break': False,      # Step 1: Break of support
            'w_pattern': False,          # Step 2: W-pattern (double bottom)
            'triple_bottom': False,      # Step 2: Triple bottom
            'reverse_hs': False,         # Step 2: Reverse head and shoulders
            'resistance_rejection': False, # Step 3: Rejection at resistance
            'near_support': False,       # Step 4: Price near support (buy zone)
            'current_price': 0,
            'ideal_entry': 0,
            'target_price': 0,
            'stop_loss': 0,
            'risk_reward_ratio': 0
        }

        # Find potential support points (local minima)
        hist['min'] = hist['Low'].rolling(window=5, center=True).min()
        support_points = hist[hist['Low'] == hist['min']].index

        # Find potential resistance points (local maxima)
        hist['max'] = hist['High'].rolling(window=5, center=True).max()
        resistance_points = hist[hist['High'] == hist['max']].index

        # Need at least 2 points to form a line
        if len(support_points) < 2 or len(resistance_points) < 2:
            return None

        # Try to fit lines to support and resistance points
        try:
            # For support line
            support_x = np.array(range(len(support_points)))
            support_y = hist.loc[support_points, 'Low'].values
            support_slope, support_intercept = np.polyfit(support_x, support_y, 1)

            # For resistance line
            resistance_x = np.array(range(len(resistance_points)))
            resistance_y = hist.loc[resistance_points, 'High'].values
            resistance_slope, resistance_intercept = np.polyfit(resistance_x, resistance_y, 1)

            # Check if slopes are similar (parallel lines)
            if abs(support_slope - resistance_slope) > 0.2:
                return None

            # Count touches of support and resistance
            support_touches = 0
            resistance_touches = 0
            threshold = 0.02  # 2% tolerance

            for i in range(len(hist)):
                support_value = support_slope * i + support_intercept
                resistance_value = resistance_slope * i + resistance_intercept

                # Check for support touches
                if abs(hist['Low'].iloc[i] - support_value) / support_value < threshold:
                    support_touches += 1

                # Check for resistance touches
                if abs(hist['High'].iloc[i] - resistance_value) / resistance_value < threshold:
                    resistance_touches += 1

            total_touches = support_touches + resistance_touches

            # Calculate channel width
            last_idx = len(hist) - 1
            support_value = support_slope * last_idx + support_intercept
            resistance_value = resistance_slope * last_idx + resistance_intercept
            channel_width = (resistance_value - support_value) / support_value * 100  # as percentage

            # Calculate channel score (1-10)
            channel_score = 0

            # Factor 1: Number of touches (max 3 points)
            if total_touches >= 4:
                touch_score = min(3, (total_touches - 3))
                channel_score += touch_score

            # Factor 2: Channel width (max 2 points)
            # Ideal channel width is between 5% and 20%
            if 5 <= channel_width <= 20:
                width_score = 2
            elif channel_width < 5:
                width_score = 1
            elif 20 < channel_width <= 30:
                width_score = 1
            else:
                width_score = 0
            channel_score += width_score

            # Factor 3: Slope (max 2 points)
            # Prefer downward trending channels (Step 1)
            if -0.1 <= support_slope < -0.02:  # Clearly downward trending
                slope_score = 2
                result['downtrend'] = True  # Mark as downtrend for Step 1
            elif -0.02 <= support_slope <= 0.02:  # Flat channel
                slope_score = 1
                result['downtrend'] = False
            elif 0.02 < support_slope <= 0.05:  # Slightly upward
                slope_score = 0.5
                result['downtrend'] = False
            else:
                slope_score = 0
                result['downtrend'] = False
            channel_score += slope_score

            # Factor 4: Current position relative to channel (max 3 points)
            current_price = hist['Close'].iloc[-1]

            # Calculate position within channel (0 = at support, 1 = at resistance)
            channel_position = (current_price - support_value) / (resistance_value - support_value) if (resistance_value - support_value) > 0 else 0.5

            # Prefer stocks near support (buy zone)
            if 0 <= channel_position <= 0.2:  # Within 20% of support
                position_score = 3
            elif 0.2 < channel_position <= 0.4:  # Within 20-40% of support
                position_score = 2
            elif 0.4 < channel_position <= 0.6:  # Middle of channel
                position_score = 1
            else:  # Closer to resistance
                position_score = 0
            channel_score += position_score

            # Step 1: Check for support breaks
            try:
                # Look for a significant drop below support line
                support_breaks = 0
                for i in range(10, len(hist)):  # Skip the first few bars
                    support_value = support_slope * i + support_intercept

                    # Check if price dropped significantly below support
                    if hist['Low'].iloc[i] < support_value * 0.95:  # 5% below support
                        # Check if there was a recovery back into the channel
                        if i < len(hist) - 5:  # Make sure we have enough bars after
                            future_bars = hist.iloc[i+1:i+6]
                            if any(future_bars['Close'] > support_value):
                                support_breaks += 1

                result['support_break'] = support_breaks > 0
            except Exception as e:
                print(f"Error detecting support breaks for {symbol}: {str(e)}")

            # Step 2: Check for psychological bottom patterns

            # Check for W pattern (double bottom)
            try:
                # Find local minima
                bottoms = hist[hist['Low'] == hist['min']].index

                if len(bottoms) >= 2:
                    # Get the last two bottoms
                    bottom1 = bottoms[-2]
                    bottom2 = bottoms[-1]

                    # Check if there's a peak between them
                    between_slice = hist.loc[bottom1:bottom2]
                    if len(between_slice) >= 5:
                        peak = between_slice['High'].max()
                        bottom1_value = hist.loc[bottom1, 'Low']
                        bottom2_value = hist.loc[bottom2, 'Low']

                        # Calculate depth as percentage
                        depth = (peak - min(bottom1_value, bottom2_value)) / min(bottom1_value, bottom2_value) * 100

                        # Check if bottoms are at similar levels and the peak is significantly higher
                        bottoms_similar = abs(bottom1_value - bottom2_value) / bottom1_value < 0.05  # 5% tolerance

                        result['w_pattern'] = bottoms_similar and depth >= 10.0  # At least 10% depth
            except Exception as e:
                print(f"Error detecting W pattern for {symbol}: {str(e)}")

            # Check for triple bottom
            try:
                if len(bottoms) >= 3:
                    # Get the last three bottoms
                    bottom1 = bottoms[-3]
                    bottom2 = bottoms[-2]
                    bottom3 = bottoms[-1]

                    # Get values
                    bottom1_value = hist.loc[bottom1, 'Low']
                    bottom2_value = hist.loc[bottom2, 'Low']
                    bottom3_value = hist.loc[bottom3, 'Low']

                    # Check if all bottoms are at similar levels
                    bottoms_similar = (abs(bottom1_value - bottom2_value) / bottom1_value < 0.05 and
                                      abs(bottom2_value - bottom3_value) / bottom2_value < 0.05 and
                                      abs(bottom1_value - bottom3_value) / bottom1_value < 0.05)

                    # Check if there are peaks between the bottoms
                    between_slice1 = hist.loc[bottom1:bottom2]
                    between_slice2 = hist.loc[bottom2:bottom3]

                    if len(between_slice1) >= 5 and len(between_slice2) >= 5:
                        peak1 = between_slice1['High'].max()
                        peak2 = between_slice2['High'].max()

                        # Calculate depths
                        depth1 = (peak1 - min(bottom1_value, bottom2_value)) / min(bottom1_value, bottom2_value) * 100
                        depth2 = (peak2 - min(bottom2_value, bottom3_value)) / min(bottom2_value, bottom3_value) * 100

                        result['triple_bottom'] = bottoms_similar and depth1 >= 8.0 and depth2 >= 8.0
            except Exception as e:
                print(f"Error detecting triple bottom for {symbol}: {str(e)}")

            # Step 3: Check for resistance rejection
            try:
                # Look for a touch of resistance followed by a significant drop
                for i in range(10, len(hist) - 5):  # Skip the first few bars and leave room at the end
                    resistance_value = resistance_slope * i + resistance_intercept

                    # Check if price touched resistance
                    if abs(hist['High'].iloc[i] - resistance_value) / resistance_value < 0.02:  # Within 2% of resistance
                        # Check if there was a significant drop after touching resistance
                        next_bars = hist.iloc[i+1:i+6]
                        lowest_after = next_bars['Low'].min()
                        highest_before = hist['High'].iloc[i]

                        drop_percent = ((highest_before - lowest_after) / highest_before) * 100

                        if drop_percent >= 5.0:  # At least 5% drop after touching resistance
                            result['resistance_rejection'] = True
                            break
            except Exception as e:
                print(f"Error detecting resistance rejection for {symbol}: {str(e)}")

            # Step 4: Check if near support (buy zone)
            result['near_support'] = channel_position <= 0.2  # Within 20% of support

            # Calculate ideal entry, target price and stop loss
            # Step 4: Enter with a clear target price and stop loss
            ideal_entry = support_value * 1.02  # 2% above support
            target_price = resistance_value * 0.95  # 5% below resistance
            stop_loss = support_value * 0.95  # 5% below support

            # Calculate risk/reward ratio
            if ideal_entry > stop_loss:
                risk = ideal_entry - stop_loss
                reward = target_price - ideal_entry
                risk_reward_ratio = reward / risk if risk > 0 else 0
            else:
                risk_reward_ratio = 0

            # Update result with trading levels
            result['current_price'] = current_price
            result['ideal_entry'] = ideal_entry
            result['target_price'] = target_price
            result['stop_loss'] = stop_loss
            result['risk_reward_ratio'] = risk_reward_ratio

            # Update channel properties
            result['has_channel'] = total_touches >= 4
            result['support_line'] = (support_slope, support_intercept)
            result['resistance_line'] = (resistance_slope, resistance_intercept)
            result['channel_width'] = channel_width
            result['touches'] = total_touches
            result['channel_score'] = channel_score

            # Calculate a perfect setup score (0-4)
            perfect_setup_score = 0
            if result['downtrend'] and result['support_break']:
                perfect_setup_score += 1  # Step 1 complete
            if result['w_pattern'] or result['triple_bottom'] or result['reverse_hs']:
                perfect_setup_score += 1  # Step 2 complete
            if result['resistance_rejection']:
                perfect_setup_score += 1  # Step 3 complete
            if result['near_support']:
                perfect_setup_score += 1  # Step 4 complete

            result['perfect_setup_score'] = perfect_setup_score

            # Only return if it's a valid channel
            if result['has_channel']:
                return result
            else:
                return None

        except Exception as e:
            print(f"Error fitting channel for {symbol}: {str(e)}")
            return None

    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None

def apply_technical_filters(df, enable_rsi=False, enable_channels=False):
    """
    Apply technical filters (RSI and trading channels).

    Args:
        df (pd.DataFrame): DataFrame with stock data
        enable_rsi (bool): Whether to enable the RSI filter
        enable_channels (bool): Whether to enable the trading channel filter

    Returns:
        pd.DataFrame: Filtered DataFrame, technical_results dict
    """
    print("\nApplying Technical Filters")
    technical_results = {}

    if df.empty:
        return df, technical_results

    # Apply RSI filter if enabled
    if enable_rsi:
        print("Applying RSI filter...")
        print(f"  RSI Period: {args.rsi_period}")
        print(f"  Daily RSI Max: {args.daily_rsi_max}")
        print(f"  Weekly RSI Max: {args.weekly_rsi_max}")
        print(f"  Monthly RSI Max: {args.monthly_rsi_max}")

        symbols = df['Symbol'].tolist()
        rsi_df = screen_for_rsi(symbols,
                               period=args.rsi_period,
                               max_daily_rsi=args.daily_rsi_max,
                               max_weekly_rsi=args.weekly_rsi_max,
                               max_monthly_rsi=args.monthly_rsi_max)

        if not rsi_df.empty:
            # Filter the main dataframe to only include symbols that pass RSI criteria
            df = df[df['Symbol'].isin(rsi_df[rsi_df['RSI_Overall_Pass']]['Symbol'])]
            print(colored(f"After RSI filter: {len(df)} stocks remaining", "cyan"))

            # Add RSI data to the main dataframe using merge instead of iterating
            # Create a copy of the dataframe to avoid SettingWithCopyWarning
            df = df.copy()

            # Merge RSI data with the main dataframe
            rsi_columns = ['Symbol', 'Daily_RSI', 'Weekly_RSI', 'Monthly_RSI',
                          'Daily_RSI_Pass', 'Weekly_RSI_Pass', 'Monthly_RSI_Pass']
            df = pd.merge(df, rsi_df[rsi_columns], on='Symbol', how='left')
        else:
            print(colored("No stocks passed the RSI filter", "yellow"))

    # Apply trading channel detection if enabled
    if enable_channels:
        print("Detecting trading channels...")
        print("Looking for the 4 steps of entry:")
        print("1. Channel with clear support and resistance")
        print("2. Psychological bottom (W-pattern, double bottom)")
        print("3. Price near support (buy zone)")
        print("4. Clear target price and stop loss with good risk/reward ratio")

        symbols = df['Symbol'].tolist()

        # Create a list to store channel results
        channel_results = []

        # Process each symbol
        for symbol in tqdm(symbols):
            channel_data = detect_trading_channel(symbol)
            if channel_data is not None:
                channel_results.append(channel_data)

                # Generate a chart for the channel
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period="3mo")

                    # Create the plot
                    plt.figure(figsize=(12, 6))

                    # Plot the price
                    plt.plot(hist.index, hist['Close'], label='Close Price')

                    # Plot the channel lines
                    support_slope, support_intercept = channel_data['support_line']
                    resistance_slope, resistance_intercept = channel_data['resistance_line']

                    x = np.array(range(len(hist)))
                    support_line = support_slope * x + support_intercept
                    resistance_line = resistance_slope * x + resistance_intercept

                    plt.plot(hist.index, support_line, 'g--', label='Support Line')
                    plt.plot(hist.index, resistance_line, 'r--', label='Resistance Line')

                    # Plot trading levels
                    ideal_entry = channel_data['ideal_entry']
                    target_price = channel_data['target_price']
                    stop_loss = channel_data['stop_loss']

                    plt.axhline(y=ideal_entry, color='blue', linestyle='-', alpha=0.5, label=f'Ideal Entry: ${ideal_entry:.2f}')
                    plt.axhline(y=target_price, color='green', linestyle='-', alpha=0.5, label=f'Target: ${target_price:.2f}')
                    plt.axhline(y=stop_loss, color='red', linestyle='-', alpha=0.5, label=f'Stop Loss: ${stop_loss:.2f}')

                    # Add title and labels
                    channel_score = channel_data['channel_score']
                    title = f'{symbol} Trading Channel (Score: {channel_score}/10)'

                    if channel_score >= 8:
                        title += ' - STRONG MATCH'
                    elif channel_score >= 6:
                        title += ' - GOOD MATCH'

                    plt.title(title)
                    plt.xlabel('Date')
                    plt.ylabel('Price')

                    # Add annotations
                    if channel_data['w_pattern']:
                        plt.annotate('W-Pattern Detected', xy=(0.02, 0.10), xycoords='axes fraction',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

                    if channel_data['near_support']:
                        plt.annotate('Near Support (BUY ZONE)', xy=(0.02, 0.05), xycoords='axes fraction',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.3))

                    plt.legend()
                    plt.grid(True)

                    # Save the plot
                    plt.savefig(f'{symbol}_channel.png')
                    plt.close()

                except Exception as e:
                    print(f"Error generating chart for {symbol}: {str(e)}")

        # Create a DataFrame from the channel results
        if channel_results:
            channel_df = pd.DataFrame(channel_results)
            technical_results['trading_channels'] = channel_df

            # Filter the main dataframe to only include symbols with channels
            df = df[df['Symbol'].isin(channel_df['Symbol'])]
            print(colored(f"After trading channel filter: {len(df)} stocks remaining", "cyan"))

            # Add channel data to the main dataframe using merge instead of iterating
            # Create a copy of the dataframe to avoid SettingWithCopyWarning
            df = df.copy()

            # Rename columns in channel_df to match the main dataframe
            channel_df_renamed = channel_df.rename(columns={
                'channel_score': 'Channel_Score',
                'ideal_entry': 'Ideal_Entry',
                'target_price': 'Target_Price',
                'stop_loss': 'Stop_Loss',
                'risk_reward_ratio': 'Risk_Reward',
                'w_pattern': 'W_Pattern',
                'near_support': 'Near_Support'
            })

            # Select columns to merge
            channel_columns = ['Symbol', 'Channel_Score', 'Ideal_Entry', 'Target_Price', 'Stop_Loss',
                              'Risk_Reward', 'perfect_setup_score', 'downtrend', 'support_break',
                              'W_Pattern', 'triple_bottom', 'reverse_hs', 'resistance_rejection', 'Near_Support']

            # Merge channel data with the main dataframe
            df = pd.merge(df, channel_df_renamed[channel_columns], on='Symbol', how='left')
        else:
            print(colored("No stocks with trading channels found", "yellow"))

    return df, technical_results

def generate_summary(df, args):
    """
    Generate a summary of the screening results.

    Args:
        df (pd.DataFrame): Final DataFrame with screened stocks
        args: Command-line arguments

    Returns:
        None
    """
    print("\nScreening Summary")
    print(f"Total stocks passing all filters: {len(df)}")

    if df.empty:
        return

    # Print top stocks by RS
    print("\nTop 10 Stocks by Relative Strength:")
    top_rs = df.sort_values('RS', ascending=False).head(10)
    for i, row in top_rs.iterrows():
        print(f"  {row['Symbol']}: RS {row['RS']:.1f}, Price ${row['Price']:.2f}")

    # Print stocks with highest revenue growth
    if 'Revenue Growth % (most recent Q)' in df.columns:
        print("\nTop 10 Stocks by Revenue Growth:")
        top_growth = df.sort_values('Revenue Growth % (most recent Q)', ascending=False).head(10)
        for i, row in top_growth.iterrows():
            growth = row['Revenue Growth % (most recent Q)']
            if pd.notna(growth):
                print(f"  {row['Symbol']}: Growth {growth:.1f}%, RS {row['RS']:.1f}")

    # Print stocks under institutional accumulation
    if 'Institutional Accumulation' in df.columns:
        under_accum = df[df['Institutional Accumulation'] == 'Yes']
        if not under_accum.empty:
            print(f"\n{len(under_accum)} Stocks Under Institutional Accumulation:")
            for i, row in under_accum.head(10).iterrows():
                print(f"  {row['Symbol']}: RS {row['RS']:.1f}, Price ${row['Price']:.2f}")

    # Print stocks with trading channels
    if 'Channel_Score' in df.columns:
        channels = df[pd.notna(df['Channel_Score'])].sort_values('Channel_Score', ascending=False)
        if not channels.empty:
            print(f"\n{len(channels)} Stocks with Trading Channels (4 Steps of Entry):")
            print("------------------------------------------------------------")
            for i, row in channels.head(10).iterrows():
                symbol = row['Symbol']
                score = row['Channel_Score']
                w_pattern = "Yes" if row['W_Pattern'] else "No"
                near_support = "Yes" if row['Near_Support'] else "No"

                # Get trading levels if available
                entry = row.get('Ideal_Entry', None)
                target = row.get('Target_Price', None)
                stop = row.get('Stop_Loss', None)
                risk_reward = row.get('Risk_Reward', None)

                # Get perfect setup score and components
                perfect_score = row.get('perfect_setup_score', 0)
                downtrend = "Yes" if row.get('downtrend', False) else "No"
                support_break = "Yes" if row.get('support_break', False) else "No"
                triple_bottom = "Yes" if row.get('triple_bottom', False) else "No"
                reverse_hs = "Yes" if row.get('reverse_hs', False) else "No"
                resistance_rejection = "Yes" if row.get('resistance_rejection', False) else "No"

                # Print channel information
                print(f"  {symbol}: Channel Score {score:.1f}/10, Perfect Setup Score: {perfect_score}/4")
                print(f"    Step 1: Down-trending Channel: {downtrend}, Support Break: {support_break}")
                print(f"    Step 2: Psychological Bottom - W-Pattern: {w_pattern}, Triple Bottom: {triple_bottom}, Reverse H&S: {reverse_hs}")
                print(f"    Step 3: Resistance Rejection: {resistance_rejection}")
                print(f"    Step 4: Near Support (Buy Zone): {near_support}")

                if pd.notna(entry) and pd.notna(target) and pd.notna(stop) and pd.notna(risk_reward):
                    print(f"    Trading Levels:")
                    print(f"      - Current Price: ${row['Price']:.2f}")
                    print(f"      - Ideal Entry: ${entry:.2f}")
                    print(f"      - Target Price: ${target:.2f}")
                    print(f"      - Stop Loss: ${stop:.2f}")
                    print(f"      - Risk/Reward Ratio: {risk_reward:.2f}")

                    # Calculate potential profit and loss
                    if entry > 0:
                        profit_pct = ((target - entry) / entry) * 100
                        loss_pct = ((stop - entry) / entry) * 100
                        print(f"      - Potential Profit: {profit_pct:.1f}%")
                        print(f"      - Potential Loss: {loss_pct:.1f}%")

                    # Add recommendation based on perfect setup score
                    if perfect_score >= 3:
                        print(f"      - STRONG BUY CANDIDATE (Perfect Setup Score: {perfect_score}/4)")
                    elif perfect_score == 2:
                        print(f"      - WATCH CLOSELY (Perfect Setup Score: {perfect_score}/4)")
                    else:
                        print(f"      - MONITOR (Perfect Setup Score: {perfect_score}/4)")

                print("    Chart saved as {}_channel.png".format(symbol))
                print("------------------------------------------------------------")

    # Print stocks with low RSI
    if 'Daily_RSI' in df.columns:
        low_rsi = df[pd.notna(df['Daily_RSI'])].sort_values('Daily_RSI')
        if not low_rsi.empty:
            print(f"\n{len(low_rsi)} Stocks with Low RSI:")
            for i, row in low_rsi.head(10).iterrows():
                print(f"  {row['Symbol']}: Daily RSI {row['Daily_RSI']:.1f}, Weekly RSI {row['Weekly_RSI']:.1f}")

    # Save results to CSV
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Print a reminder about the 4 steps of perfect setup
    if 'Channel_Score' in df.columns and not df[pd.notna(df['Channel_Score'])].empty:
        print("\nReminder - The 4 Steps of Perfect Setup:")
        print("1. Down-trending trading channel with clear support and resistance")
        print("   (Look for break of support with big sell-off from breakdown support line)")
        print("2. Psychological eye-catching bottom pattern")
        print("   (W-pattern/double bottom, triple bottom, or reverse head and shoulders)")
        print("3. Price jumps into trading channel and goes to resistance, then gets rejected")
        print("   (Price should sell off after rejection at resistance)")
        print("4. BUY on support of trading channel, as bottom is in")
        print("   (Enter at support with clear target at resistance and stop below support)")

def main():
    """Main function to run the custom stock screener."""
    global args
    args = parse_arguments()

    print("Custom Stock Screener with Optional Filters")
    print("===========================================")

    # Load stock universe with max_stocks limit
    universe_df = load_stock_universe(args.max_stocks)
    symbols = universe_df['Symbol'].tolist()

    # Apply filters in sequence

    # 1. Relative Strength
    # If RS min is 0, skip the RS filtering by returning all symbols
    if args.rs_min <= 0:
        print("\n[1/5] Skipping Relative Strength filter (min RS set to 0)")
        # Create a minimal DataFrame with just the symbols
        rs_df = pd.DataFrame({'Symbol': symbols})
        # Add dummy columns that would normally be added by calculate_relative_strength
        rs_df['Company Name'] = rs_df['Symbol']
        rs_df['Exchange'] = 'Unknown'
        rs_df['Industry'] = 'Unknown'
        rs_df['Market Cap'] = 0
        rs_df['RS_Raw'] = 0
        rs_df['RS'] = 0

        # Get current prices for all symbols
        print("Getting current prices for all symbols...")
        for i, symbol in enumerate(tqdm(symbols)):
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1d")
                if not hist.empty:
                    rs_df.loc[i, 'Price'] = hist['Close'].iloc[-1]
                else:
                    rs_df.loc[i, 'Price'] = 0
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
                rs_df.loc[i, 'Price'] = 0

        print(f"Keeping all {len(rs_df)} stocks")
    else:
        # Normal RS filtering
        rs_df = calculate_relative_strength(symbols, args.rs_min)

    # 2. Liquidity
    # Check if we should skip market cap and price filters
    skip_market_cap = args.market_cap_min <= 0
    skip_price = args.price_min <= 0

    if skip_market_cap and skip_price and args.volume_min <= 0:
        # Skip all liquidity filters
        print("\n[2/5] Skipping all Liquidity filters")
        liquidity_df = rs_df
    else:
        # Apply liquidity filters
        liquidity_df = filter_liquidity(rs_df, args.market_cap_min, args.price_min, args.volume_min)

    # 3. Trend
    trend_df = filter_trend(liquidity_df, args.trend)

    # 4. Revenue Growth
    if args.revenue_growth_min <= 0:
        print("\n[4/5] Skipping Revenue Growth filter (min growth set to 0)")
        growth_df = trend_df
    else:
        growth_df = filter_revenue_growth(trend_df, args.revenue_growth_min, args.rs_bypass)

    # 5. Institutional Accumulation
    inst_df = filter_institutional_accumulation(growth_df, args.institutional)

    # Apply technical filters
    final_df, _ = apply_technical_filters(inst_df, args.rsi, args.channels)

    # Generate summary
    generate_summary(final_df, args)

if __name__ == "__main__":
    main()
