"""
Modified Custom Stock Screener with Clear Stage Counts

This script is a modified version of custom_screen.py that:
1. Shows the count of stocks at each stage more clearly
2. Properly handles the command line arguments
3. Fixes the SettingWithCopyWarning issues
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

# Import the original custom_screen module
import custom_screen

def main():
    """Main function to run the modified custom stock screener."""
    args = custom_screen.parse_arguments()
    
    print("Modified Custom Stock Screener with Clear Stage Counts")
    print("====================================================")
    print()
    print("PARAMETERS:")
    print(f"  RS Minimum: {args.rs_min}")
    print(f"  Market Cap Minimum: ${args.market_cap_min:,.0f}")
    print(f"  Price Minimum: ${args.price_min:.2f}")
    print(f"  Volume Minimum: {args.volume_min:,} shares")
    print(f"  Revenue Growth Minimum: {args.revenue_growth_min}%")
    print(f"  RS Bypass: {args.rs_bypass}")
    print(f"  Trend Filter: {'Enabled' if args.trend else 'Disabled'}")
    print(f"  Institutional Filter: {'Enabled' if args.institutional else 'Disabled'}")
    print(f"  RSI Filter: {'Enabled' if args.rsi else 'Disabled'}")
    if args.rsi:
        print(f"    RSI Period: {args.rsi_period}")
        print(f"    Daily RSI Max: {args.daily_rsi_max}")
        print(f"    Weekly RSI Max: {args.weekly_rsi_max}")
        print(f"    Monthly RSI Max: {args.monthly_rsi_max}")
    print(f"  Channel Detection: {'Enabled' if args.channels else 'Disabled'}")
    print(f"  Max Stocks: {args.max_stocks}")
    print(f"  Output File: {args.output}")
    print()
    
    # Load stock universe with max_stocks limit
    universe_df = custom_screen.load_stock_universe(args.max_stocks)
    initial_count = len(universe_df)
    symbols = universe_df['Symbol'].tolist()
    
    print(f"Initial stock universe: {initial_count} stocks")
    print()
    
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
        rs_df['Price'] = 0
        rs_df['RS_Raw'] = 0
        rs_df['RS'] = 0
        rs_count = len(rs_df)
        print(f"After RS filter: {rs_count} stocks remaining")
    else:
        # Normal RS filtering
        rs_df = custom_screen.calculate_relative_strength(symbols, args.rs_min)
        rs_count = len(rs_df)
        print(f"After RS filter: {rs_count} stocks remaining")
    
    # 2. Liquidity
    # Check if we should skip market cap and price filters
    skip_market_cap = args.market_cap_min <= 0
    skip_price = args.price_min <= 0
    
    if skip_market_cap and skip_price and args.volume_min <= 0:
        # Skip all liquidity filters
        print("\n[2/5] Skipping all Liquidity filters")
        liquidity_df = rs_df
        liquidity_count = len(liquidity_df)
        print(f"After liquidity filter: {liquidity_count} stocks remaining")
    else:
        # Apply liquidity filters
        liquidity_df = custom_screen.filter_liquidity(rs_df, args.market_cap_min, args.price_min, args.volume_min)
        liquidity_count = len(liquidity_df)
        print(f"After liquidity filter: {liquidity_count} stocks remaining")
    
    # 3. Trend
    trend_df = custom_screen.filter_trend(liquidity_df, args.trend)
    trend_count = len(trend_df)
    print(f"After trend filter: {trend_count} stocks remaining")
    
    # 4. Revenue Growth
    if args.revenue_growth_min <= 0:
        print("\n[4/5] Skipping Revenue Growth filter (min growth set to 0)")
        growth_df = trend_df
        growth_count = len(growth_df)
        print(f"After revenue growth filter: {growth_count} stocks remaining")
    else:
        growth_df = custom_screen.filter_revenue_growth(trend_df, args.revenue_growth_min, args.rs_bypass)
        growth_count = len(growth_df)
        print(f"After revenue growth filter: {growth_count} stocks remaining")
    
    # 5. Institutional Accumulation
    inst_df = custom_screen.filter_institutional_accumulation(growth_df, args.institutional)
    inst_count = len(inst_df)
    print(f"After institutional filter: {inst_count} stocks remaining")
    
    # Apply technical filters
    print("\nApplying Technical Filters")
    
    # Apply RSI filter if enabled
    if args.rsi:
        print("Applying RSI filter...")
        print(f"  RSI Period: {args.rsi_period}")
        print(f"  Daily RSI Max: {args.daily_rsi_max}")
        print(f"  Weekly RSI Max: {args.weekly_rsi_max}")
        print(f"  Monthly RSI Max: {args.monthly_rsi_max}")
        
        symbols = inst_df['Symbol'].tolist()
        rsi_df = screen_for_rsi(symbols,
                               period=args.rsi_period,
                               max_daily_rsi=args.daily_rsi_max,
                               max_weekly_rsi=args.weekly_rsi_max,
                               max_monthly_rsi=args.monthly_rsi_max)
        
        if not rsi_df.empty:
            # Filter the main dataframe to only include symbols that pass RSI criteria
            df_rsi = inst_df[inst_df['Symbol'].isin(rsi_df[rsi_df['RSI_Overall_Pass']]['Symbol'])]
            rsi_count = len(df_rsi)
            print(colored(f"After RSI filter: {rsi_count} stocks remaining", "cyan"))
            
            # Add RSI data to the main dataframe using merge instead of iterating
            # Create a copy of the dataframe to avoid SettingWithCopyWarning
            df_rsi = df_rsi.copy()
            
            # Merge RSI data with the main dataframe
            rsi_columns = ['Symbol', 'Daily_RSI', 'Weekly_RSI', 'Monthly_RSI', 
                          'Daily_RSI_Pass', 'Weekly_RSI_Pass', 'Monthly_RSI_Pass']
            df_rsi = pd.merge(df_rsi, rsi_df[rsi_columns], on='Symbol', how='left')
        else:
            print(colored("No stocks passed the RSI filter", "yellow"))
            df_rsi = pd.DataFrame()  # Empty DataFrame
    else:
        df_rsi = inst_df
        rsi_count = len(df_rsi)
        print(f"RSI filter disabled. {rsi_count} stocks remaining")
    
    # Apply trading channel detection if enabled
    if args.channels and not df_rsi.empty:
        print("Detecting trading channels...")
        print("Looking for the 4 steps of entry:")
        print("1. Channel with clear support and resistance")
        print("2. Psychological bottom (W-pattern, double bottom)")
        print("3. Price near support (buy zone)")
        print("4. Clear target price and stop loss with good risk/reward ratio")
        
        symbols = df_rsi['Symbol'].tolist()
        
        # Create a list to store channel results
        channel_results = []
        
        # Process each symbol
        for symbol in tqdm(symbols):
            channel_data = custom_screen.detect_trading_channel(symbol)
            if channel_data is not None:
                channel_results.append(channel_data)
        
        # Create a DataFrame from the channel results
        if channel_results:
            channel_df = pd.DataFrame(channel_results)
            
            # Filter the main dataframe to only include symbols with channels
            df_channels = df_rsi[df_rsi['Symbol'].isin(channel_df['Symbol'])]
            channel_count = len(df_channels)
            print(colored(f"After trading channel filter: {channel_count} stocks remaining", "cyan"))
            
            # Create a copy of the dataframe to avoid SettingWithCopyWarning
            df_channels = df_channels.copy()
            
            # Prepare channel data for merging
            channel_df_for_merge = channel_df.copy()
            
            # Rename columns to match the main dataframe
            channel_df_for_merge = channel_df_for_merge.rename(columns={
                'channel_score': 'Channel_Score',
                'ideal_entry': 'Ideal_Entry',
                'target_price': 'Target_Price',
                'stop_loss': 'Stop_Loss',
                'risk_reward_ratio': 'Risk_Reward',
                'w_pattern': 'W_Pattern',
                'near_support': 'Near_Support'
            })
            
            # Validate trading levels - ensure they're positive and make sense
            for idx, row in channel_df_for_merge.iterrows():
                current_price = row['current_price']
                
                # Ensure current price is positive
                if current_price <= 0:
                    # Try to get current price from main dataframe
                    symbol_price = df_channels.loc[df_channels['Symbol'] == row['Symbol'], 'Price'].values
                    if len(symbol_price) > 0 and symbol_price[0] > 0:
                        channel_df_for_merge.at[idx, 'current_price'] = symbol_price[0]
                        current_price = symbol_price[0]
                    else:
                        # If still no valid price, get it from Yahoo Finance
                        try:
                            stock = yf.Ticker(row['Symbol'])
                            hist = stock.history(period="1d")
                            if not hist.empty:
                                current_price = hist['Close'].iloc[-1]
                                channel_df_for_merge.at[idx, 'current_price'] = current_price
                        except Exception as e:
                            print(f"Error getting price for {row['Symbol']}: {e}")
                
                # Recalculate trading levels based on current price if needed
                if current_price > 0:
                    # If ideal entry is negative or zero, recalculate
                    if row['Ideal_Entry'] <= 0:
                        channel_df_for_merge.at[idx, 'Ideal_Entry'] = current_price * 0.98  # 2% below current price
                    
                    # If target price is negative or zero, recalculate
                    if row['Target_Price'] <= 0:
                        channel_df_for_merge.at[idx, 'Target_Price'] = current_price * 1.1  # 10% above current price
                    
                    # If stop loss is negative or zero, recalculate
                    if row['Stop_Loss'] <= 0:
                        channel_df_for_merge.at[idx, 'Stop_Loss'] = current_price * 0.95  # 5% below current price
                    
                    # Recalculate risk-reward ratio
                    ideal_entry = channel_df_for_merge.at[idx, 'Ideal_Entry']
                    target_price = channel_df_for_merge.at[idx, 'Target_Price']
                    stop_loss = channel_df_for_merge.at[idx, 'Stop_Loss']
                    
                    if ideal_entry > stop_loss and ideal_entry < target_price:
                        risk = ideal_entry - stop_loss
                        reward = target_price - ideal_entry
                        risk_reward = reward / risk if risk > 0 else 0
                        channel_df_for_merge.at[idx, 'Risk_Reward'] = risk_reward
            
            # Select columns to merge
            channel_columns = ['Symbol', 'Channel_Score', 'Ideal_Entry', 'Target_Price', 'Stop_Loss', 
                              'Risk_Reward', 'perfect_setup_score', 'downtrend', 'support_break', 
                              'W_Pattern', 'triple_bottom', 'reverse_hs', 'resistance_rejection', 'Near_Support']
            
            # Merge channel data with the main dataframe
            df_channels = pd.merge(df_channels, channel_df_for_merge[channel_columns], on='Symbol', how='left')
            
            # Rank the stocks
            df_ranked = rank_stocks(df_channels)
            
            # Save the results
            df_ranked.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
            
            # Print summary
            print("\nScreening Summary")
            print(f"Initial stock universe: {initial_count} stocks")
            print(f"After RS filter: {rs_count} stocks")
            print(f"After liquidity filter: {liquidity_count} stocks")
            print(f"After trend filter: {trend_count} stocks")
            print(f"After revenue growth filter: {growth_count} stocks")
            print(f"After institutional filter: {inst_count} stocks")
            print(f"After RSI filter: {rsi_count} stocks")
            print(f"After channel filter: {channel_count} stocks")
            
            # Display top ranked stocks
            display_top_stocks(df_ranked)
            
            return df_ranked
        else:
            print(colored("No stocks with trading channels found", "yellow"))
            df_channels = pd.DataFrame()  # Empty DataFrame
    else:
        if args.channels:
            print("Channel detection enabled but no stocks to process.")
        else:
            print("Channel detection disabled.")
        df_channels = df_rsi
    
    # Save the results
    if not df_channels.empty:
        df_channels.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        
        # Print summary
        print("\nScreening Summary")
        print(f"Initial stock universe: {initial_count} stocks")
        print(f"After RS filter: {rs_count} stocks")
        print(f"After liquidity filter: {liquidity_count} stocks")
        print(f"After trend filter: {trend_count} stocks")
        print(f"After revenue growth filter: {growth_count} stocks")
        print(f"After institutional filter: {inst_count} stocks")
        print(f"After RSI filter: {rsi_count} stocks")
        if args.channels:
            print(f"After channel filter: {len(df_channels)} stocks")
    else:
        print("No stocks passed all filters.")
    
    return df_channels

def rank_stocks(df):
    """
    Rank stocks based on a composite score.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
    
    Returns:
        pd.DataFrame: Ranked DataFrame
    """
    print("\nRanking stocks...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_ranked = df.copy()
    
    # Create a composite score for ranking
    # Components:
    # 1. Channel Score (higher is better)
    # 2. Perfect Setup Score (higher is better)
    # 3. Risk-Reward Ratio (higher is better)
    # 4. RSI (lower is better)
    # 5. Volume (higher is better)
    
    # Normalize each component to a 0-100 scale
    
    # For Channel Score, higher is better
    if 'Channel_Score' in df_ranked.columns:
        max_cs = df_ranked['Channel_Score'].max()
        min_cs = df_ranked['Channel_Score'].min()
        if max_cs > min_cs:
            df_ranked['CS_Score'] = ((df_ranked['Channel_Score'] - min_cs) / (max_cs - min_cs) * 100)
        else:
            df_ranked['CS_Score'] = 50
    else:
        df_ranked['CS_Score'] = 50  # Default if not available
    
    # For Perfect Setup Score, higher is better
    if 'perfect_setup_score' in df_ranked.columns:
        max_ps = df_ranked['perfect_setup_score'].max()
        min_ps = df_ranked['perfect_setup_score'].min()
        if max_ps > min_ps:
            df_ranked['PS_Score'] = ((df_ranked['perfect_setup_score'] - min_ps) / (max_ps - min_ps) * 100)
        else:
            df_ranked['PS_Score'] = 50
    else:
        df_ranked['PS_Score'] = 50  # Default if not available
    
    # For Risk-Reward, higher is better
    if 'Risk_Reward' in df_ranked.columns:
        # Cap risk-reward at 5.0 to avoid outliers
        df_ranked['Risk_Reward_Capped'] = df_ranked['Risk_Reward'].clip(upper=5.0)
        max_rr = df_ranked['Risk_Reward_Capped'].max()
        min_rr = df_ranked['Risk_Reward_Capped'].min()
        if max_rr > min_rr:
            df_ranked['RR_Score'] = ((df_ranked['Risk_Reward_Capped'] - min_rr) / (max_rr - min_rr) * 100)
        else:
            df_ranked['RR_Score'] = 50
    else:
        df_ranked['RR_Score'] = 50  # Default if not available
    
    # For Monthly RSI, lower is better
    if 'Monthly_RSI' in df_ranked.columns:
        # For RSI, lower is better, so invert the scale
        max_rsi = df_ranked['Monthly_RSI'].max()
        min_rsi = df_ranked['Monthly_RSI'].min()
        if max_rsi > min_rsi:
            df_ranked['RSI_Score'] = 100 - ((df_ranked['Monthly_RSI'] - min_rsi) / (max_rsi - min_rsi) * 100)
        else:
            df_ranked['RSI_Score'] = 50
    else:
        df_ranked['RSI_Score'] = 50  # Default if not available
    
    # For Volume, higher is better
    if '50-day Average Volume' in df_ranked.columns:
        # Log transform volume to reduce impact of outliers
        df_ranked['Log_Volume'] = np.log1p(df_ranked['50-day Average Volume'])
        max_vol = df_ranked['Log_Volume'].max()
        min_vol = df_ranked['Log_Volume'].min()
        if max_vol > min_vol:
            df_ranked['Volume_Score'] = ((df_ranked['Log_Volume'] - min_vol) / (max_vol - min_vol) * 100)
        else:
            df_ranked['Volume_Score'] = 50
    else:
        df_ranked['Volume_Score'] = 50  # Default if not available
    
    # Calculate composite score with weights
    df_ranked['Composite_Score'] = (
        0.25 * df_ranked['CS_Score'] +      # 25% weight to Channel Score
        0.25 * df_ranked['PS_Score'] +      # 25% weight to Perfect Setup Score
        0.25 * df_ranked['RR_Score'] +      # 25% weight to Risk-Reward
        0.15 * df_ranked['RSI_Score'] +     # 15% weight to RSI
        0.10 * df_ranked['Volume_Score']    # 10% weight to Volume
    )
    
    # Rank based on composite score
    df_ranked = df_ranked.sort_values('Composite_Score', ascending=False)
    
    # Add rank column
    df_ranked.insert(0, 'Rank', range(1, len(df_ranked) + 1))
    
    # Add buy rating based on composite score
    def get_rating(score):
        if score >= 80:
            return "STRONG BUY"
        elif score >= 60:
            return "BUY"
        elif score >= 40:
            return "NEUTRAL"
        elif score >= 20:
            return "WEAK"
        else:
            return "AVOID"
    
    df_ranked['Rating'] = df_ranked['Composite_Score'].apply(get_rating)
    
    print(f"Ranked {len(df_ranked)} stocks.")
    return df_ranked

def display_top_stocks(df_ranked, top_n=10):
    """
    Display the top ranked stocks.
    
    Args:
        df_ranked (pd.DataFrame): Ranked DataFrame
        top_n (int): Number of top stocks to display
    """
    print("\nTOP RANKED STOCKS:")
    print("=================")
    
    # Display top stocks
    top_stocks = df_ranked.head(top_n)
    for i, row in top_stocks.iterrows():
        rank = row['Rank']
        symbol = row['Symbol']
        rating = row['Rating']
        
        # Print with color based on rating
        if rating == "STRONG BUY":
            cprint(f"{rank}. {symbol} - {rating}", "green", attrs=["bold"])
        elif rating == "BUY":
            cprint(f"{rank}. {symbol} - {rating}", "green")
        elif rating == "NEUTRAL":
            cprint(f"{rank}. {symbol} - {rating}", "yellow")
        else:
            print(f"{rank}. {symbol} - {rating}")
        
        # Print details
        print(f"   Channel Score: {row.get('Channel_Score', 0):.1f}/10")
        print(f"   Perfect Setup Score: {row.get('perfect_setup_score', 0):.1f}/4")
        
        # Print RSI values if available
        if 'Monthly_RSI' in row:
            print(f"   Monthly RSI: {row['Monthly_RSI']:.1f}")
        if 'Weekly_RSI' in row:
            print(f"   Weekly RSI: {row['Weekly_RSI']:.1f}")
        if 'Daily_RSI' in row:
            print(f"   Daily RSI: {row['Daily_RSI']:.1f}")
        
        # Print trading levels
        print(f"   Current Price: ${row.get('Price', 0):.2f}")
        print(f"   Ideal Entry: ${row.get('Ideal_Entry', 0):.2f}")
        print(f"   Target Price: ${row.get('Target_Price', 0):.2f}")
        print(f"   Stop Loss: ${row.get('Stop_Loss', 0):.2f}")
        print(f"   Risk/Reward Ratio: {row.get('Risk_Reward', 0):.2f}")
        
        # Calculate potential profit and loss
        ideal_entry = row.get('Ideal_Entry', 0)
        target_price = row.get('Target_Price', 0)
        stop_loss = row.get('Stop_Loss', 0)
        
        if ideal_entry > 0:
            profit_pct = ((target_price - ideal_entry) / ideal_entry) * 100
            loss_pct = ((stop_loss - ideal_entry) / ideal_entry) * 100
            print(f"   Potential Profit: {profit_pct:.1f}%")
            print(f"   Potential Loss: {loss_pct:.1f}%")
        
        print(f"   Composite Score: {row['Composite_Score']:.1f}")
        print()
    
    # Print summary statistics
    print(f"Total stocks ranked: {len(df_ranked)}")
    print(f"Strong Buy: {len(df_ranked[df_ranked['Rating'] == 'STRONG BUY'])}")
    print(f"Buy: {len(df_ranked[df_ranked['Rating'] == 'BUY'])}")
    print(f"Neutral: {len(df_ranked[df_ranked['Rating'] == 'NEUTRAL'])}")
    print(f"Weak: {len(df_ranked[df_ranked['Rating'] == 'WEAK'])}")
    print(f"Avoid: {len(df_ranked[df_ranked['Rating'] == 'AVOID'])}")

if __name__ == "__main__":
    main()
