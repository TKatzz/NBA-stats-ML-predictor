"""
NBA Data Exploration
Explore the data you've fetched to understand it better
"""

import pandas as pd
import matplotlib.pyplot as plt

def explore_nba_data(csv_file):
    """
    Load and explore NBA game log data
    """
    df = pd.read_csv(csv_file)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
    print("="*70)
    print("NBA DATA EXPLORATION")
    print("="*70)
    
    # Basic info
    print(f"\n1. BASIC INFO")
    print(f"   Total games: {len(df)}")
    print(f"   Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    print(f"   Seasons: {df['SEASON'].unique().tolist() if 'SEASON' in df.columns else 'N/A'}")
    
    # Available columns
    print(f"\n2. AVAILABLE COLUMNS ({len(df.columns)} total):")
    print("   ", ", ".join(df.columns.tolist()))
    
    # Key statistics
    print(f"\n3. AVERAGE STATS PER GAME:")
    key_stats = ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    available_stats = [col for col in key_stats if col in df.columns]
    print(df[available_stats].mean().round(2).to_string())
    
    # Distribution insights
    print(f"\n4. POINTS DISTRIBUTION:")
    print(f"   Min: {df['PTS'].min()}")
    print(f"   25th percentile: {df['PTS'].quantile(0.25)}")
    print(f"   Median: {df['PTS'].median()}")
    print(f"   75th percentile: {df['PTS'].quantile(0.75)}")
    print(f"   Max: {df['PTS'].max()}")
    print(f"   Std Dev: {df['PTS'].std():.2f}")
    
    # Home vs Away
    if 'MATCHUP' in df.columns:
        df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')
        print(f"\n5. HOME vs AWAY PERFORMANCE:")
        print(f"   Home games: {df['IS_HOME'].sum()}")
        print(f"   Away games: {(~df['IS_HOME']).sum()}")
        print(f"\n   Average PTS at home: {df[df['IS_HOME']]['PTS'].mean():.2f}")
        print(f"   Average PTS away: {df[~df['IS_HOME']]['PTS'].mean():.2f}")
    
    # Missing values
    print(f"\n6. MISSING VALUES:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0].to_string())
    else:
        print("   No missing values ✓")
    
    # Recent form
    print(f"\n7. RECENT FORM (last 10 games):")
    recent = df.tail(10)
    print(recent[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST']].to_string(index=False))
    
    print("\n" + "="*70)
    print("Exploration complete! This data is ready for ML modeling.")
    print("="*70)
    
    return df


def create_quick_visualizations(df, player_name):
    """
    Create some quick visualizations
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{player_name} - Performance Analysis', fontsize=16)
    
    # Points over time
    axes[0, 0].plot(df['GAME_DATE'], df['PTS'], marker='o', linestyle='-', alpha=0.6)
    axes[0, 0].set_title('Points Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Points')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Points distribution
    axes[0, 1].hist(df['PTS'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Points Distribution')
    axes[0, 1].set_xlabel('Points')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df['PTS'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # Minutes vs Points
    axes[1, 0].scatter(df['MIN'], df['PTS'], alpha=0.6)
    axes[1, 0].set_title('Minutes vs Points')
    axes[1, 0].set_xlabel('Minutes Played')
    axes[1, 0].set_ylabel('Points')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rolling average
    df_sorted = df.sort_values('GAME_DATE')
    rolling_avg = df_sorted['PTS'].rolling(window=5).mean()
    axes[1, 1].plot(df_sorted['GAME_DATE'], df_sorted['PTS'], alpha=0.3, label='Actual')
    axes[1, 1].plot(df_sorted['GAME_DATE'], rolling_avg, color='red', linewidth=2, label='5-game avg')
    axes[1, 1].set_title('Points with Rolling Average')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Points')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{player_name.replace(" ", "_")}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{player_name.replace(' ', '_')}_analysis.png'")
    
    return fig


# Example usage
if __name__ == "__main__":
    import sys
    
    # You can pass a CSV file as argument, or it will look for common names
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Try to find a CSV file in current directory
        import os
        csv_files = [f for f in os.listdir('.') if f.endswith('_data.csv')]
        
        if csv_files:
            csv_file = csv_files[0]
            print(f"Found: {csv_file}\n")
        else:
            print("No CSV file found. Please run quick_start_fetch.py first!")
            sys.exit(1)
    
    # Explore the data
    df = explore_nba_data(csv_file)
    
    # Create visualizations
    player_name = csv_file.replace('_data.csv', '').replace('_', ' ')
    create_quick_visualizations(df, player_name)
