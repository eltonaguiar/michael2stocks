"""
Perfect Setup Detection with Ranking System

This script finds stocks that match the following criteria:
1. Monthly RSI under 25 with RSI period 14
2. Perfect setup 4 steps
3. 1 million shares traded daily volume average

Results are ranked based on a composite score.
"""

import subprocess
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def main():
    """Main function"""
    print("Perfect Setup Detection with Ranking System")
    print("=========================================")
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
            
            # Now rank the results
            rank_results(output_path)
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

def rank_results(csv_path):
    """
    Rank the results based on a composite score.
    
    Args:
        csv_path (str): Path to the CSV file with results
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print("No stocks found that match the criteria.")
            return
        
        print(f"Found {len(df)} stocks that match the criteria.")
        
        # Create a copy to avoid SettingWithCopyWarning
        ranked_df = df.copy()
        
        # Calculate risk-reward ratio if not already calculated
        if 'Risk_Reward_Ratio' not in ranked_df.columns or ranked_df['Risk_Reward_Ratio'].isnull().all():
            ranked_df['Risk_Reward_Ratio'] = 0.0
            for i in range(len(ranked_df)):
                current_price = ranked_df.loc[i, 'Price']
                target_price = ranked_df.loc[i, 'Target_Price']
                stop_loss = ranked_df.loc[i, 'Stop_Loss']
                
                if pd.notna(current_price) and pd.notna(target_price) and pd.notna(stop_loss) and current_price > 0 and stop_loss > 0:
                    potential_gain = target_price - current_price
                    potential_loss = current_price - stop_loss
                    
                    if potential_loss > 0:
                        risk_reward = potential_gain / potential_loss
                        ranked_df.loc[i, 'Risk_Reward_Ratio'] = risk_reward
        
        # Create a composite score for ranking
        # Components:
        # 1. Monthly RSI (lower is better)
        # 2. Risk-Reward Ratio (higher is better)
        # 3. Channel Score (higher is better)
        # 4. Volume (higher is better)
        
        # Normalize each component to a 0-100 scale
        if 'Monthly_RSI' in ranked_df.columns:
            # For RSI, lower is better, so invert the scale
            max_rsi = 25.0  # Our filter is already for RSI < 25
            min_rsi = ranked_df['Monthly_RSI'].min()
            if max_rsi > min_rsi:
                ranked_df['RSI_Score'] = 100 - ((ranked_df['Monthly_RSI'] - min_rsi) / (max_rsi - min_rsi) * 100)
            else:
                ranked_df['RSI_Score'] = 100
        else:
            ranked_df['RSI_Score'] = 50  # Default if not available
        
        # For Risk-Reward, higher is better
        if 'Risk_Reward_Ratio' in ranked_df.columns:
            max_rr = ranked_df['Risk_Reward_Ratio'].max()
            min_rr = ranked_df['Risk_Reward_Ratio'].min()
            if max_rr > min_rr:
                ranked_df['RR_Score'] = ((ranked_df['Risk_Reward_Ratio'] - min_rr) / (max_rr - min_rr) * 100)
            else:
                ranked_df['RR_Score'] = 50
        else:
            ranked_df['RR_Score'] = 50  # Default if not available
        
        # For Channel Score, higher is better
        if 'Channel_Score' in ranked_df.columns:
            max_cs = ranked_df['Channel_Score'].max()
            min_cs = ranked_df['Channel_Score'].min()
            if max_cs > min_cs:
                ranked_df['CS_Score'] = ((ranked_df['Channel_Score'] - min_cs) / (max_cs - min_cs) * 100)
            else:
                ranked_df['CS_Score'] = 50
        else:
            ranked_df['CS_Score'] = 50  # Default if not available
        
        # For Volume, higher is better
        if '50-day Average Volume' in ranked_df.columns:
            max_vol = ranked_df['50-day Average Volume'].max()
            min_vol = ranked_df['50-day Average Volume'].min()
            if max_vol > min_vol:
                ranked_df['Volume_Score'] = ((ranked_df['50-day Average Volume'] - min_vol) / (max_vol - min_vol) * 100)
            else:
                ranked_df['Volume_Score'] = 50
        else:
            ranked_df['Volume_Score'] = 50  # Default if not available
        
        # Calculate composite score with weights
        ranked_df['Composite_Score'] = (
            0.30 * ranked_df['RSI_Score'] +      # 30% weight to RSI
            0.30 * ranked_df['RR_Score'] +       # 30% weight to Risk-Reward
            0.25 * ranked_df['CS_Score'] +       # 25% weight to Channel Score
            0.15 * ranked_df['Volume_Score']     # 15% weight to Volume
        )
        
        # Rank based on composite score
        ranked_df = ranked_df.sort_values('Composite_Score', ascending=False)
        
        # Add rank column
        ranked_df.insert(0, 'Rank', range(1, len(ranked_df) + 1))
        
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
        
        ranked_df['Rating'] = ranked_df['Composite_Score'].apply(get_rating)
        
        # Save the ranked results
        ranked_output = csv_path.replace('.csv', '_ranked.csv')
        ranked_df.to_csv(ranked_output, index=False)
        
        print(f"\nRanked results saved to: {os.path.basename(ranked_output)}")
        print("\nTOP 10 RANKED STOCKS:")
        print("=====================")
        
        # Display top 10 stocks
        top_10 = ranked_df.head(10)
        for i, row in top_10.iterrows():
            print(f"{row['Rank']}. {row['Symbol']} - {row['Rating']}")
            print(f"   Monthly RSI: {row['Monthly_RSI']:.1f}")
            print(f"   Risk-Reward: {row['Risk_Reward_Ratio']:.2f}")
            print(f"   Channel Score: {row['Channel_Score']:.1f}")
            print(f"   Composite Score: {row['Composite_Score']:.1f}")
            print()
        
        print(f"Total stocks found: {len(ranked_df)}")
        print(f"Strong Buy: {len(ranked_df[ranked_df['Rating'] == 'STRONG BUY'])}")
        print(f"Buy: {len(ranked_df[ranked_df['Rating'] == 'BUY'])}")
        print(f"Neutral: {len(ranked_df[ranked_df['Rating'] == 'NEUTRAL'])}")
        print(f"Weak: {len(ranked_df[ranked_df['Rating'] == 'WEAK'])}")
        print(f"Avoid: {len(ranked_df[ranked_df['Rating'] == 'AVOID'])}")
        
    except Exception as e:
        print(f"Error ranking results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
