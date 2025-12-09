# Complete Pipeline for Moving Averages Analysis on S&P 500 Historical Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Dataframe presentation configuration
desired_width = 320                                                 # shows columns with X or fewer characters
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

# Define the short and long window lengths
var_short_window = 30
var_long_window = 200

# Define datapoints in range
n_days = 1000     # 250 ~ 1 year   # max = 3,500


class MovingAveragesPipeline:
    # Complete pipeline for moving averages analysis

    def __init__(self, filepath):
        """Initialize with data filepath"""
        self.filepath = filepath
        self.df = None
        self.results = {}

    def load_and_process_data(self):
        """Load and preprocess the S&P 500 data"""
        print("\nSTEP 1: LOADING AND PROCESSING DATA\n")

        # Load data
        self.df = pd.read_csv(self.filepath)
        self.df = self.df.tail(n_days)              # Constrain to most recent n records
        print(f"\nOriginal dataset shape: {self.df.shape}")
        print(f"\nSample data:\n{self.df.head()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")

        # Convert date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Sort by date
        self.df = self.df.sort_values('Date').reset_index(drop=True)

        # Handle missing values in Adj Close
        if self.df['Adj Close'].isnull().sum() > 0:
            print(f"\nHandling {self.df['Adj Close'].isnull().sum()} missing values...")
            self.df['Adj Close'].fillna(method='ffill', inplace=True)

        # Create a clean working copy - new column = Adj_Close
        self.df['Adj_Close'] = self.df['Adj Close'].copy()

        print(f"\nProcessed dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"\nBasic statistics of Adj Close:\n{self.df['Adj_Close'].describe()}")

    def calculate_sma(self, window):
        """Calculate Simple Moving Average"""
        return self.df['Adj_Close'].rolling(window=window).mean()

    def calculate_ema(self, window):
        """Calculate Exponential Moving Average"""
        return self.df['Adj_Close'].ewm(span=window, adjust=False).mean()

    def calculate_wma(self, window):
        """Calculate Weighted Moving Average"""
        weights = np.arange(1, window + 1)

        def weighted_average(values):
            if len(values) < window:
                return np.nan
            return np.dot(values[-window:], weights) / weights.sum()

        return self.df['Adj_Close'].rolling(window=window).apply(weighted_average, raw=True)

    def calculate_moving_averages(self, short_window=20, long_window=100):
        """Calculate all moving averages for both window sizes"""
        print("\nSTEP 2: CALCULATING MOVING AVERAGES")
        print(f"\nShort window: {short_window} days")
        print(f"Long window: {long_window} days")

        # Short window calculations
        self.df[f'SMA_{short_window}'] = self.calculate_sma(short_window)
        self.df[f'EMA_{short_window}'] = self.calculate_ema(short_window)
        self.df[f'WMA_{short_window}'] = self.calculate_wma(short_window)

        # Long window calculations
        self.df[f'SMA_{long_window}'] = self.calculate_sma(long_window)
        self.df[f'EMA_{long_window}'] = self.calculate_ema(long_window)
        self.df[f'WMA_{long_window}'] = self.calculate_wma(long_window)

        print("\nMoving averages calculated successfully")
        print(f"\nSample of results (last 5 rows):")
        cols_to_show = ['Date', 'Adj_Close', f'SMA_{short_window}',
                        f'EMA_{short_window}', f'WMA_{short_window}']
        print(self.df[cols_to_show].tail())
        # print(self.df)

    def calculate_metrics(self, ma_column):
        """Calculate accuracy metrics for a moving average"""
        # Remove NaN values for comparison
        valid_idx = ~(self.df['Adj_Close'].isna() | self.df[ma_column].isna())
        actual = self.df.loc[valid_idx, 'Adj_Close']
        predicted = self.df.loc[valid_idx, ma_column]

        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # Smoothness (variance of first differences)
        smoothness = np.var(np.diff(predicted))

        return {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Smoothness': smoothness
        }

    def generate_metrics(self, short_window=20, long_window=100):
        """Generate metrics for all moving averages"""
        print("\nSTEP 3: CALCULATING ACCURACY AND SMOOTHNESS METRICS")

        ma_types = ['SMA', 'EMA', 'WMA']
        windows = [short_window, long_window]

        metrics_data = []

        for ma_type in ma_types:
            for window in windows:
                col_name = f'{ma_type}_{window}'
                metrics = self.calculate_metrics(col_name)
                metrics['MA_Type'] = ma_type
                metrics['Window'] = window
                metrics_data.append(metrics)

        self.metrics_df = pd.DataFrame(metrics_data)

        print("\nMetrics Summary - per MA type, per Window")
        print(self.metrics_df.to_string(index=False))

        # Store for interpretation
        self.results['metrics'] = self.metrics_df

    def plot_original_data(self):
        """Plot original time series data"""
        print("\n" + "=" * 80)
        print("STEP 4: GENERATING VISUALIZATIONS")
        print("=" * 80)

        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Date'], self.df['Adj_Close'], linewidth=1, alpha=0.8)
        plt.title('S&P 500 Adjusted Close Price - Original Data', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Adjusted Close Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("\n✓ Original data plot generated")

    def plot_comparison_all_mas(self, short_window=20):
        """Plot original data with all three moving averages"""
        plt.figure(figsize=(14, 7))

        # Plot original data
        plt.plot(self.df['Date'], self.df['Adj_Close'],
                 label='Actual Price', linewidth=1.5, alpha=0.7, color='black')

        # Plot all moving averages
        plt.plot(self.df['Date'], self.df[f'SMA_{short_window}'],
                 label=f'SMA ({short_window})', linewidth=2, alpha=0.8)
        plt.plot(self.df['Date'], self.df[f'EMA_{short_window}'],
                 label=f'EMA ({short_window})', linewidth=2, alpha=0.8)
        plt.plot(self.df['Date'], self.df[f'WMA_{short_window}'],
                 label=f'WMA ({short_window})', linewidth=2, alpha=0.8)

        plt.title(f'S&P 500: Actual Price vs Moving Averages (Window={short_window})',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("\n✓ Comparison plot (All MAs) generated")

    def plot_sma_comparison(self, short_window=20, long_window=100):
        """Plot SMA with different window sizes"""
        plt.figure(figsize=(14, 7))

        plt.plot(self.df['Date'], self.df['Adj_Close'],
                 label='Actual Price', linewidth=1, alpha=0.6, color='gray')
        plt.plot(self.df['Date'], self.df[f'SMA_{short_window}'],
                 label=f'SMA ({short_window} days)', linewidth=2, alpha=0.9)
        plt.plot(self.df['Date'], self.df[f'SMA_{long_window}'],
                 label=f'SMA ({long_window} days)', linewidth=2, alpha=0.9)

        plt.title('Simple Moving Average: Short vs Long Window Comparison',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("\n✓ SMA comparison plot generated")

    def plot_ema_comparison(self, short_window=20, long_window=100):
        """Plot EMA with different window sizes"""
        plt.figure(figsize=(14, 7))

        plt.plot(self.df['Date'], self.df['Adj_Close'],
                 label='Actual Price', linewidth=1, alpha=0.6, color='gray')
        plt.plot(self.df['Date'], self.df[f'EMA_{short_window}'],
                 label=f'EMA ({short_window} days)', linewidth=2, alpha=0.9)
        plt.plot(self.df['Date'], self.df[f'EMA_{long_window}'],
                 label=f'EMA ({long_window} days)', linewidth=2, alpha=0.9)

        plt.title('Exponential Moving Average: Short vs Long Window Comparison',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("\n✓ EMA comparison plot generated")

    def plot_wma_comparison(self, short_window=20, long_window=100):
        """Plot WMA with different window sizes"""
        plt.figure(figsize=(14, 7))

        plt.plot(self.df['Date'], self.df['Adj_Close'],
                 label='Actual Price', linewidth=1, alpha=0.6, color='gray')
        plt.plot(self.df['Date'], self.df[f'WMA_{short_window}'],
                 label=f'WMA ({short_window} days)', linewidth=2, alpha=0.9)
        plt.plot(self.df['Date'], self.df[f'WMA_{long_window}'],
                 label=f'WMA ({long_window} days)', linewidth=2, alpha=0.9)

        plt.title('Weighted Moving Average: Short vs Long Window Comparison',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("\n✓ WMA comparison plot generated")

    def plot_metrics_comparison(self):
        """Plot metrics comparison"""
        metrics_df = self.metrics_df

        # MAE Comparison
        # plt.figure(figsize=(14, 7))
        pivot_mae = metrics_df.pivot(index='MA_Type', columns='Window', values='MAE')
        pivot_mae.plot(kind='bar', width=0.8)
        plt.title('Mean Absolute Error (MAE) Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Moving Average Type', fontsize=12)
        plt.ylabel('MAE ($)', fontsize=12)
        plt.legend(title='Window Size', fontsize=10)
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        print("\n✓ MAE comparison plot generated")

        # MAPE Comparison
        # plt.figure(figsize=(12, 6))
        pivot_mape = metrics_df.pivot(index='MA_Type', columns='Window', values='MAPE')
        pivot_mape.plot(kind='bar', width=0.8)
        plt.title('Mean Absolute Percentage Error (MAPE) Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Moving Average Type', fontsize=12)
        plt.ylabel('MAPE (%)', fontsize=12)
        plt.legend(title='Window Size', fontsize=10)
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        print("\n✓ MAPE comparison plot generated")

        # Smoothness Comparison
        # plt.figure(figsize=(12, 6))
        pivot_smooth = metrics_df.pivot(index='MA_Type', columns='Window', values='Smoothness')
        pivot_smooth.plot(kind='bar', width=0.8)
        plt.title('Smoothness Comparison (Lower is Smoother)', fontsize=16, fontweight='bold')
        plt.xlabel('Moving Average Type', fontsize=12)
        plt.ylabel('Smoothness (Variance)', fontsize=12)
        plt.legend(title='Window Size', fontsize=10)
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        print("\n✓ Smoothness comparison plot generated")

    def interpret_results(self):
        """Provide interpretation and conclusions"""
        print("\n" + "=" * 80)
        print("STEP 5: INTERPRETATION AND CONCLUSIONS")
        print("=" * 80)

        metrics_df = self.metrics_df

        print("\n1. ACCURACY ANALYSIS (MAE & MAPE):")
        print("-" * 80)

        # Find best performing MA for short window
        short_metrics = metrics_df[metrics_df['Window'] == metrics_df['Window'].min()]
        best_short = short_metrics.loc[short_metrics['MAE'].idxmin()]

        print(f"\n   Short Window ({best_short['Window']:.0f} days):")
        print(f"   • Best performer: {best_short['MA_Type']} with MAE = ${best_short['MAE']:.2f}")
        print(f"   • This represents {best_short['MAPE']:.2f}% average error")

        # Find best performing MA for long window
        long_metrics = metrics_df[metrics_df['Window'] == metrics_df['Window'].max()]
        best_long = long_metrics.loc[long_metrics['MAE'].idxmin()]

        print(f"\n   Long Window ({best_long['Window']:.0f} days):")
        print(f"   • Best performer: {best_long['MA_Type']} with MAE = ${best_long['MAE']:.2f}")
        print(f"   • This represents {best_long['MAPE']:.2f}% average error")

        print("\n2. SMOOTHNESS ANALYSIS:")
        print("-" * 80)

        # Smoothness comparison
        smoothest_short = short_metrics.loc[short_metrics['Smoothness'].idxmin()]
        smoothest_long = long_metrics.loc[long_metrics['Smoothness'].idxmin()]

        print(f"\n   • Smoothest short window MA: {smoothest_short['MA_Type']} "
              f"(Variance: {smoothest_short['Smoothness']:.2f})")
        print(f"   • Smoothest long window MA: {smoothest_long['MA_Type']} "
              f"(Variance: {smoothest_long['Smoothness']:.2f})")
        print(f"   • Long windows produce {(smoothest_short['Smoothness'] / smoothest_long['Smoothness']):.1f}x "
              f"smoother curves than short windows")

        print("\n3. MOVING AVERAGE TYPE COMPARISON:")
        print("-" * 80)

        print("\n   SMA (Simple Moving Average):")
        print("   • Gives equal weight to all data points in the window")
        print("   • Slowest to react to price changes")
        print("   • Most stable and smoothest of the three methods")

        print("\n   EMA (Exponential Moving Average):")
        print("   • Gives more weight to recent prices")
        print("   • Faster reaction to price changes than SMA")
        print("   • More responsive to trends while maintaining smoothness")

        print("\n   WMA (Weighted Moving Average):")
        print("   • Linearly weights recent prices higher")
        print("   • Reactivity between SMA and EMA")
        print("   • Good balance of responsiveness and stability")

        print("\n4. WINDOW SIZE EFFECTS:")
        print("-" * 80)

        # Calculate average error increase for longer windows
        avg_mae_increase = ((long_metrics['MAE'].mean() / short_metrics['MAE'].mean() - 1) * 100)

        print(f"\n   Short Window ({short_metrics['Window'].iloc[0]:.0f} days):")
        print("   • More responsive to recent price movements")
        print("   • Higher volatility in the moving average")
        print(f"   • Average MAE: ${short_metrics['MAE'].mean():.2f}")

        print(f"\n   Long Window ({long_metrics['Window'].iloc[0]:.0f} days):")
        print("   • Captures longer-term trends")
        print("   • Much smoother, less reactive to short-term fluctuations")
        print(f"   • Average MAE: ${long_metrics['MAE'].mean():.2f}")
        print(f"   • {abs(avg_mae_increase):.1f}% {'higher' if avg_mae_increase > 0 else 'lower'} "
              f"error due to lag effect")

        print("\n5. KEY CONCLUSIONS:")
        print("-" * 80)

        print("\n   a) Trade-off Between Accuracy and Smoothness:")
        print("      • Shorter windows track prices more closely (lower MAE)")
        print("      • Longer windows are smoother but lag behind actual prices")

        print("\n   b) Optimal Moving Average Selection:")
        print(f"      • For trend following: Use {best_long['MA_Type']} with {best_long['Window']:.0f}-day window")
        print(f"      • For trading signals: Use {best_short['MA_Type']} with {best_short['Window']:.0f}-day window")
        print("      • For robust analysis: Combine both short and long windows")

        print("\n   c) Practical Applications:")
        print("      • EMA is often preferred for trading due to its responsiveness")
        print("      • SMA is better for identifying major trend reversals")
        print("      • WMA provides a middle ground for balanced analysis")
        print("      • Crossover strategies (short MA crossing long MA) signal potential trends")

        print("\n   d) Model Performance:")
        best_overall = metrics_df.loc[metrics_df['MAE'].idxmin()]
        print(f"      • Best overall model: {best_overall['MA_Type']} with {best_overall['Window']:.0f}-day window")
        print(f"      • Achieved MAE of ${best_overall['MAE']:.2f} ({best_overall['MAPE']:.2f}% MAPE)")
        print(f"      • Smoothness variance: {best_overall['Smoothness']:.2f}")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

    def run_complete_pipeline(self, short_window=20, long_window=100):
        """Execute the complete pipeline"""
        print("MOVING AVERAGES ANALYSIS PIPELINE - S&P 500")

        # Step 1: Load and process data
        self.load_and_process_data()

        # Step 2: Calculate moving averages
        self.calculate_moving_averages(short_window, long_window)

        # Step 3: Calculate metrics
        self.generate_metrics(short_window, long_window)

        # Step 4: Generate visualizations
        self.plot_original_data()
        self.plot_comparison_all_mas(short_window)
        self.plot_sma_comparison(short_window, long_window)
        self.plot_ema_comparison(short_window, long_window)
        self.plot_wma_comparison(short_window, long_window)
        self.plot_metrics_comparison()

        # Step 5: Interpret results
        self.interpret_results()

        return self.df, self.metrics_df


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    # IMPORTANT: Update this path to your actual file location
    filepath = 'sp500_index.csv'  # Update with your file path

    pipeline = MovingAveragesPipeline(filepath)

    # Run complete pipeline with custom window sizes
    df_results, metrics = pipeline.run_complete_pipeline(
        short_window=var_short_window,  # Short-term: 20 days (~1 month)
        long_window=var_long_window  # Long-term: 100 days (~5 months)
    )

    print("\n✓ Pipeline completed successfully!")
    print("\nTo rerun with different window sizes:")
    print("pipeline.run_complete_pipeline(short_window=30, long_window=200)")