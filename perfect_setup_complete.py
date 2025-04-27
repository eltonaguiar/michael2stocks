"""
Perfect Setup Complete Script

This script:
1. Runs the custom screen with perfect setup parameters
2. Fixes trading levels (prices, ideal entry, target, stop loss)
3. Calculates risk-reward ratios
4. Ranks stocks based on a composite score
5. Provides a clear summary of the best stocks to buy

Filters applied:
- Monthly RSI under 25 with RSI period 14
- Perfect setup 4 steps
- 1 million shares traded daily volume average
"""

import subprocess
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from termcolor import colored, cprint

def main():
    """Main function"""
    print("Perfect Setup Complete Analysis")
    print("==============================")
    print()
    print("FILTERS APPLIED:")
    print("1. Monthly RSI under 25 with RSI period 14")
    print("2. Perfect setup 4 steps:")
    print("   - Down-trending channel with clear support and resistance")
    print("   - Psychological bottom (W-pattern, triple bottom, reverse H&S)")
    print("   - Price jumps to resistance, gets rejected, and sells off")
    print("   - Buy on support of trading channel (bottom is in)")
    print("3. 1 million shares traded daily volume average")
    print()
    print("All other filters (RS, market cap, price) have been disabled")
    print("to maximize the number of stocks found.")
    print()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the custom_screen.py file
    custom_screen_path = os.path.join(script_dir, "custom_screen.py")
    
    # Check if the file exists
    if not os.path.exists(custom_screen_path):
        print(f"ERROR: {custom_screen_path} does not exist!")
        input("Press Enter to continue...")
        return
    
    # Output file
    output_file = "perfect_setup_results.csv"
    
    # Command to run with perfect setup parameters
    cmd = [
        sys.executable,  # Use the current Python interpreter
        custom_screen_path,
        "--rs-min", "0",                 # No RS minimum requirement
        "--market-cap-min", "0",         # No market cap minimum
        "--price-min", "0",              # No price minimum
        "--volume-min", "1000000",       # 1 million shares traded per day
        "--revenue-growth-min", "0",     # No revenue growth requirement
        "--rs-bypass", "0",              # No RS bypass
        "--output", output_file,
        "--max-stocks", "1000",          # Scan more stocks
        "--channels",                    # Enable channel detection
        "--rsi",                         # Enable RSI filter
        "--rsi-period", "14",            # RSI period 14
        "--monthly-rsi-max", "25.0"      # Monthly RSI under 25
    ]
    
    print("Running the scan...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        
        # Check if the output file was created
        output_path = os.path.join(script_dir, output_file)
        if os.path.exists(output_path):
            print(f"\nSuccess! Results saved to: {output_file}")
            
            # Now fix trading levels and rank the results
            df_fixed_ranked = process_results(output_path)
            
            if df_fixed_ranked is not None:
                # Save the fixed and ranked results
                ranked_output = output_path.replace('.csv', '_ranked.csv')
                df_fixed_ranked.to_csv(ranked_output, index=False)
                print(f"\nRanked results saved to: {os.path.basename(ranked_output)}")
                
                # Display top stocks
                display_top_stocks(df_fixed_ranked)
        else:
            print(f"\nWarning: Output file was not created.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running the command: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    input("Press Enter to continue...")

def process_results(csv_path):
    """
    Process the results by fixing trading levels and ranking stocks.
    
    Args:
        csv_path (str): Path to the CSV file with results
    
    Returns:
        pd.DataFrame: Processed DataFrame with fixed trading levels and rankings
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print("No stocks found that match the criteria.")
            return None
        
        print(f"Found {len(df)} stocks that match the criteria.")
        
        # Fix trading levels
        df_fixed = fix_trading_levels(df)
        
        # Rank stocks
        df_ranked = rank_stocks(df_fixed)
        
        return df_ranked
        
    except Exception as e:
        print(f"Error processing results: {e}")
        import traceback
        traceback.print_exc()
        return None

def fix_trading_levels(df):
    """
    Fix trading levels in the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
    
    Returns:
        pd.DataFrame: DataFrame with fixed trading levels
    """
    print("\nFixing trading levels...")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_fixed = df.copy()
    
    # Add columns if they don't exist
    if 'Price' not in df_fixed.columns:
        df_fixed['Price'] = 0.0
    
    if 'Ideal_Entry' not in df_fixed.columns:
        df_fixed['Ideal_Entry'] = 0.0
    
    if 'Target_Price' not in df_fixed.columns:
        df_fixed['Target_Price'] = 0.0
    
    if 'Stop_Loss' not in df_fixed.columns:
        df_fixed['Stop_Loss'] = 0.0
    
    if 'Risk_Reward' not in df_fixed.columns:
        df_fixed['Risk_Reward'] = 0.0
    
    # Process each stock
    for i, row in df_fixed.iterrows():
        symbol = row['Symbol']
        current_price = row['Price']
        
        # Ensure current price is positive
        if pd.isna(current_price) or current_price <= 0:
            try:
                # Get current price from Yahoo Finance
                stock = yf.Ticker(symbol)
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    df_fixed.loc[i, 'Price'] = current_price
                    print(f"Updated price for {symbol}: ${current_price:.2f}")
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")
                continue
        
        # Skip if we still don't have a valid price
        if pd.isna(current_price) or current_price <= 0:
            continue
        
        # Fix ideal entry
        ideal_entry = row['Ideal_Entry']
        if pd.isna(ideal_entry) or ideal_entry <= 0:
            ideal_entry = current_price * 0.98  # 2% below current price
            df_fixed.loc[i, 'Ideal_Entry'] = ideal_entry
            print(f"Fixed ideal entry for {symbol}: ${ideal_entry:.2f}")
        
        # Fix target price
        target_price = row['Target_Price']
        if pd.isna(target_price) or target_price <= 0:
            target_price = current_price * 1.1  # 10% above current price
            df_fixed.loc[i, 'Target_Price'] = target_price
            print(f"Fixed target price for {symbol}: ${target_price:.2f}")
        
        # Fix stop loss
        stop_loss = row['Stop_Loss']
        if pd.isna(stop_loss) or stop_loss <= 0:
            stop_loss = current_price * 0.95  # 5% below current price
            df_fixed.loc[i, 'Stop_Loss'] = stop_loss
            print(f"Fixed stop loss for {symbol}: ${stop_loss:.2f}")
        
        # Recalculate risk-reward ratio
        if ideal_entry > 0 and target_price > ideal_entry and stop_loss > 0 and stop_loss < ideal_entry:
            risk = ideal_entry - stop_loss
            reward = target_price - ideal_entry
            risk_reward = reward / risk
            df_fixed.loc[i, 'Risk_Reward'] = risk_reward
            print(f"Recalculated risk/reward for {symbol}: {risk_reward:.2f}")
    
    print(f"Fixed trading levels for {len(df_fixed)} stocks.")
    return df_fixed

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
