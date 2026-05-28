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
var_short_window = 30           # 30 is the normal value
var_long_window = 200           # 200 is the normal value

# Define datapoints in range
n_days = 1000     # note: 250 ~ 1 year, and the max = 3,500


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

        # Check for duplicate dates
        n_duplicates = self.df['Date'].duplicated().sum()
        if n_duplicates > 0:
            print(f"\nWARNING: {n_duplicates} duplicate date(s) found - removing...")
            self.df = self.df.drop_duplicates(subset='Date', keep='last')
        else:
            print(f"\nDuplicate dates:               None found")

        # Check for zero or negative prices
        n_anomalies = (self.df['Adj Close'] <= 0).sum()
        if n_anomalies > 0:
            print(f"WARNING: {n_anomalies} zero or negative price value(s) detected")
        else:
            print(f"Price anomalies (zero/negative): None found")

        # Check for large time gaps in the series (>5 days indicates missing data beyond normal weekends and market holidays)
        date_diffs = self.df['Date'].diff().dt.days.dropna()
        large_gaps = date_diffs[date_diffs > 5]
        if len(large_gaps) > 0:
            print(f"\nLarge time gaps (>5 days) detected: {len(large_gaps)}")
            for idx, gap in large_gaps.items():
                print(f"  Gap of {int(gap)} days ending at {self.df.loc[idx, 'Date'].date()}")
        else:
            print(f"Time series gaps (>5 days):      None detected")

        # Handle missing values in Adj Close
        if self.df['Adj Close'].isnull().sum() > 0:
            print(f"\nHandling {self.df['Adj Close'].isnull().sum()} missing values...")
            # self.df['Adj Close'].fillna(method='ffill', inplace=True)
            self.df['Adj Close'] = self.df['Adj Close'].ffill()

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

    def calculate_returns_and_volatility(self):
        """Calculate daily percentage returns and rolling volatility metrics."""
        print("\nSTEP 2b: CALCULATING RETURNS AND VOLATILITY METRICS")

        # --- Daily percentage returns ---
        self.df['Daily_Return'] = self.df['Adj_Close'].pct_change() * 100

        # --- Rolling 30-day annualised volatility ---
        # Annualisation uses sqrt(252): the approximate number of US trading
        # days per year. This expresses volatility on the same scale as
        # annual return figures commonly quoted in financial analysis.
        self.df['Rolling_Volatility_30'] = (
                self.df['Daily_Return'].rolling(window=var_short_window).std() * np.sqrt(252)
        )

        # --- Summary statistics ---
        returns_clean = self.df['Daily_Return'].dropna()
        ann_vol = returns_clean.std() * np.sqrt(252)

        print(f"\nDaily Returns Summary:")
        print(f"  Mean daily return:           {returns_clean.mean():.4f}%")
        print(f"  Std dev of daily return:     {returns_clean.std():.4f}%")
        print(f"  Annualised volatility:       {ann_vol:.2f}%")
        print(f"  Largest single-day loss:     {returns_clean.min():.4f}%  "
              f"({self.df.loc[returns_clean.idxmin(), 'Date'].date()})")
        print(f"  Largest single-day gain:     {returns_clean.max():.4f}%  "
              f"({self.df.loc[returns_clean.idxmax(), 'Date'].date()})")
        print(f"  Positive return days:        "
              f"{(returns_clean > 0).sum()} "
              f"({(returns_clean > 0).sum() / len(returns_clean) * 100:.1f}%)")
        print(f"  Negative return days:        "
              f"{(returns_clean < 0).sum()} "
              f"({(returns_clean < 0).sum() / len(returns_clean) * 100:.1f}%)")

        return self.df['Daily_Return'], self.df['Rolling_Volatility_30']



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
        metrics_df_reordered = self.metrics_df[['MA_Type', 'Window', 'MAE', 'MAPE', 'RMSE', 'Smoothness']].round(2)
        print(metrics_df_reordered)
        # print(self.metrics_df.to_string(index=False))

        # Store for interpretation
        self.results['metrics'] = self.metrics_df

    def plot_original_data(self):
        """Plot original time series data"""
        print("\nSTEP 4: GENERATING VISUALIZATIONS")

        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Date'], self.df['Adj_Close'], linewidth=1, alpha=0.8)
        plt.title('S&P 500 Adjusted Close Price - Original Data', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Adjusted Close Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"ma_data_{n_days}.png", dpi=150)
        plt.show()

    def plot_comparison_all_mas(self, short_window=20, data_points=None, filename_suffix=None):
        """Plot original data with all three moving averages.

        Parameters
        ----------
        short_window : int
            Window size for the MAs to display.
        data_points : int or None
            Number of most recent data points to include in the chart.
            If None, all loaded data points are used.
        filename_suffix : str or None
            Suffix appended to the saved filename (e.g. '1000' or '250').
            Defaults to the string value of data_points.
        """

        # Determine subset of data to plot
        if data_points is not None:
            plot_df = self.df.tail(data_points).copy()
        else:
            data_points = len(self.df)
            plot_df = self.df.copy()

        if filename_suffix is None:
            filename_suffix = str(data_points)

        plt.figure(figsize=(14, 7))

        plt.plot(plot_df['Date'], plot_df['Adj_Close'],
                 label='Actual Price', linewidth=1.5, alpha=0.7, color='black')
        plt.plot(plot_df['Date'], plot_df[f'SMA_{short_window}'],
                 label=f'SMA ({short_window})', linewidth=2, alpha=0.8)
        plt.plot(plot_df['Date'], plot_df[f'EMA_{short_window}'],
                 label=f'EMA ({short_window})', linewidth=2, alpha=0.8)
        plt.plot(plot_df['Date'], plot_df[f'WMA_{short_window}'],
                 label=f'WMA ({short_window})', linewidth=2, alpha=0.8)

        plt.title(
            f'S&P 500: Actual Price vs Moving Averages '
            f'(Window = {short_window} days, Last {data_points} Data Points)',
            fontsize=16, fontweight='bold'
        )
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'ma_smooth_{filename_suffix}.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved: ma_smooth_{filename_suffix}.png")


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
        plt.savefig(f"ma_sma_{n_days}.png", dpi=150)
        plt.show()

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
        plt.savefig(f"ma_ema_{n_days}.png", dpi=150)
        plt.show()

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
        plt.savefig(f"ma_wma_{n_days}.png", dpi=150)
        plt.show()

    def plot_returns_and_volatility(self):
        """Plot daily returns and rolling volatility analysis."""

        returns = self.df['Daily_Return'].dropna()

        fig, axes = plt.subplots(3, 1, figsize=(14, 14))
        fig.suptitle('S&P 500: Returns and Volatility Analysis',
                     fontsize=16, fontweight='bold')

        # --- Panel 1: Daily returns bar chart ---
        bar_colours = ['steelblue' if r >= 0 else 'tomato' for r in returns]
        axes[0].bar(self.df['Date'].iloc[1:], returns,
                    color=bar_colours, alpha=0.75, linewidth=0)
        axes[0].axhline(0, color='black', linewidth=0.8)
        axes[0].axhline(returns.mean(), color='darkblue', linewidth=1.5,
                        linestyle='--', label=f'Mean: {returns.mean():.3f}%')
        axes[0].set_title('Daily Returns (%)', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Date', fontsize=11)
        axes[0].set_ylabel('Daily Return (%)', fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # --- Panel 2: Returns distribution ---
        axes[1].hist(returns, bins=60, color='steelblue', alpha=0.7,
                     edgecolor='white', linewidth=0.3)
        axes[1].axvline(returns.mean(), color='darkblue', linewidth=2,
                        linestyle='--',
                        label=f'Mean: {returns.mean():.3f}%')
        axes[1].axvline(returns.mean() - returns.std(), color='orange',
                        linewidth=1.5, linestyle=':',
                        label=f'Mean \u2212 1 SD: {returns.mean() - returns.std():.3f}%')
        axes[1].axvline(returns.mean() + returns.std(), color='orange',
                        linewidth=1.5, linestyle=':',
                        label=f'Mean + 1 SD: {returns.mean() + returns.std():.3f}%')
        axes[1].set_title('Daily Returns Distribution', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Daily Return (%)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # --- Panel 3: Rolling 30-day annualised volatility ---
        axes[2].plot(self.df['Date'], self.df['Rolling_Volatility_30'],
                     color='darkorange', linewidth=1.5,
                     label=f'Rolling {var_short_window}-Day Annualised Volatility')
        axes[2].fill_between(self.df['Date'], self.df['Rolling_Volatility_30'],
                             alpha=0.2, color='darkorange')
        axes[2].set_title(
            f'Rolling {var_short_window}-Day Annualised Volatility (%)',
            fontsize=13, fontweight='bold'
        )
        axes[2].set_xlabel('Date', fontsize=11)
        axes[2].set_ylabel('Annualised Volatility (%)', fontsize=11)
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'ma_returns_volatility_{n_days}.png', dpi=150, bbox_inches='tight')
        plt.show()

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
        plt.savefig(f"ma_mae_{n_days}.png", dpi=150)
        plt.show()

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
        plt.savefig(f"ma_mape_{n_days}.png", dpi=150)
        plt.show()

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
        plt.savefig(f"ma_smoothness_{n_days}.png", dpi=150)
        plt.show()

    def plot_crossover_signals(self, short_window=30, long_window=200):
        """Plot Golden Cross and Death Cross events for SMA short vs long window.

        Parameters
        ----------
        short_window : int
            Short-term SMA window (typically 30 days).
        long_window : int
            Long-term SMA window (typically 200 days).
        """
        print(f"\n  Generating crossover signals chart "
              f"(SMA {short_window} vs SMA {long_window})...")

        short_col = f'SMA_{short_window}'
        long_col = f'SMA_{long_window}'

        # Check columns are present (require calculate_moving_averages to have run)
        for col in [short_col, long_col]:
            if col not in self.df.columns:
                print(f"  WARNING: Column '{col}' not found. "
                      f"Ensure calculate_moving_averages() was called with "
                      f"short_window={short_window} and long_window={long_window}.")
                return

        # --- Identify crossover points ---
        # Signal = +1 where short MA is above long MA, -1 where below.
        # A crossover occurs where the sign changes.
        signal = np.sign(self.df[short_col] - self.df[long_col])

        # diff() == +2 means signal moved from -1 to +1: Golden Cross (bullish)
        # diff() == -2 means signal moved from +1 to -1: Death Cross (bearish)
        crossover = signal.diff().fillna(0)
        golden_cross_idx = self.df.index[crossover == 2]
        death_cross_idx = self.df.index[crossover == -2]

        print(f"  Golden Crosses (bullish) detected: {len(golden_cross_idx)}")
        for idx in golden_cross_idx:
            print(f"    {self.df.loc[idx, 'Date'].date()}  |  "
                  f"Price: ${self.df.loc[idx, 'Adj_Close']:.2f}  |  "
                  f"SMA{short_window}: ${self.df.loc[idx, short_col]:.2f}")

        print(f"  Death Crosses  (bearish) detected:  {len(death_cross_idx)}")
        for idx in death_cross_idx:
            print(f"    {self.df.loc[idx, 'Date'].date()}  |  "
                  f"Price: ${self.df.loc[idx, 'Adj_Close']:.2f}  |  "
                  f"SMA{short_window}: ${self.df.loc[idx, short_col]:.2f}")

        # --- MA Spread series ---
        ma_spread = self.df[short_col] - self.df[long_col]

        # --- Plot ---
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 10),
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True
        )
        fig.suptitle(
            f'S&P 500: Golden Cross & Death Cross Analysis '
            f'(SMA {short_window}-day vs SMA {long_window}-day)',
            fontsize=16, fontweight='bold'
        )

        # --- Panel 1: Price + SMAs + crossover events ---
        ax1.plot(self.df['Date'], self.df['Adj_Close'],
                 label='Adj Close Price', linewidth=1, alpha=0.45, color='gray')
        ax1.plot(self.df['Date'], self.df[short_col],
                 label=f'SMA {short_window}-day (short)', linewidth=2, color='steelblue')
        ax1.plot(self.df['Date'], self.df[long_col],
                 label=f'SMA {long_window}-day (long)', linewidth=2, color='darkorange')

        # Annotate Golden Crosses
        gc_label_added = False
        for idx in golden_cross_idx:
            label = 'Golden Cross (Bullish \u2191)' if not gc_label_added else None
            ax1.axvline(x=self.df.loc[idx, 'Date'], color='green',
                        linewidth=1.2, linestyle='--', alpha=0.75)
            ax1.scatter(self.df.loc[idx, 'Date'],
                        self.df.loc[idx, short_col],
                        color='green', s=130, zorder=5,
                        marker='^', label=label)
            gc_label_added = True

        # Annotate Death Crosses
        dc_label_added = False
        for idx in death_cross_idx:
            label = 'Death Cross (Bearish \u2193)' if not dc_label_added else None
            ax1.axvline(x=self.df.loc[idx, 'Date'], color='red',
                        linewidth=1.2, linestyle='--', alpha=0.75)
            ax1.scatter(self.df.loc[idx, 'Date'],
                        self.df.loc[idx, short_col],
                        color='red', s=130, zorder=5,
                        marker='v', label=label)
            dc_label_added = True

        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: MA Spread ---
        spread_colours = ['steelblue' if v >= 0 else 'tomato' for v in ma_spread]
        ax2.bar(self.df['Date'], ma_spread,
                color=spread_colours, alpha=0.75, linewidth=0,
                label=f'SMA Spread (SMA{short_window} \u2212 SMA{long_window})')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Spread ($)', fontsize=11)
        ax2.set_title(
            'MA Spread  —  Blue: Bullish (short MA above long MA)  |  '
            'Red: Bearish (short MA below long MA)',
            fontsize=10
        )
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'ma_crossover_{n_days}.png', dpi=150, bbox_inches='tight')
        plt.show()

    def interpret_results(self):
        """Provide interpretation and conclusions"""
        print("\nSTEP 5: INTERPRETATION AND CONCLUSIONS")

        metrics_df = self.metrics_df

        print("\n1. ACCURACY ANALYSIS (MAE & MAPE):")

        # Find best performing MA for short window
        short_metrics = metrics_df[metrics_df['Window'] == metrics_df['Window'].min()]
        best_short = short_metrics.loc[short_metrics['MAE'].idxmin()]

        print(f"\nShort Window ({best_short['Window']:.0f} days):")
        print(f"• Best performer: {best_short['MA_Type']} with MAE = ${best_short['MAE']:.2f}")
        print(f"• This represents {best_short['MAPE']:.2f}% average error")

        # Find best performing MA for long window
        long_metrics = metrics_df[metrics_df['Window'] == metrics_df['Window'].max()]
        best_long = long_metrics.loc[long_metrics['MAE'].idxmin()]

        print(f"\nLong Window ({best_long['Window']:.0f} days):")
        print(f"• Best performer: {best_long['MA_Type']} with MAE = ${best_long['MAE']:.2f}")
        print(f"• This represents {best_long['MAPE']:.2f}% average error")

        print("\n2. SMOOTHNESS ANALYSIS:")

        # Smoothness comparison
        smoothest_short = short_metrics.loc[short_metrics['Smoothness'].idxmin()]
        smoothest_long = long_metrics.loc[long_metrics['Smoothness'].idxmin()]

        print(f"\n• Smoothest short window MA: {smoothest_short['MA_Type']} "
              f"(Variance: {smoothest_short['Smoothness']:.2f})")
        print(f"• Smoothest long window MA: {smoothest_long['MA_Type']} "
              f"(Variance: {smoothest_long['Smoothness']:.2f})")
        print(f"• Long windows produce {(smoothest_short['Smoothness'] / smoothest_long['Smoothness']):.1f} x "
              f"smoother curves than short windows")

        print("\n3. MOVING AVERAGE TYPE COMPARISON:")

        print("\nSMA (Simple Moving Average):")
        print("• Gives equal weight to all data points in the window")
        print("• Slowest to react to price changes")
        print("• Most stable and smoothest of the three methods")

        print("\nEMA (Exponential Moving Average):")
        print("• Gives more weight to recent prices")
        print("• Faster reaction to price changes than SMA")
        print("• More responsive to trends while maintaining smoothness")

        print("\nWMA (Weighted Moving Average):")
        print("• Linearly weights recent prices higher")
        print("• Reactivity between SMA and EMA")
        print("• Good balance of responsiveness and stability")

        print("\n4. WINDOW SIZE EFFECTS:")

        # Calculate average error increase for longer windows
        avg_mae_increase = ((long_metrics['MAE'].mean() / short_metrics['MAE'].mean() - 1) * 100)

        print(f"\nShort Window ({short_metrics['Window'].iloc[0]:.0f} days):")
        print("• More responsive to recent price movements")
        print("• Higher volatility in the moving average")
        print(f"• Average MAE: ${short_metrics['MAE'].mean():.2f}")

        print(f"\nLong Window ({long_metrics['Window'].iloc[0]:.0f} days):")
        print("• Captures longer-term trends")
        print("• Much smoother, less reactive to short-term fluctuations")
        print(f"• Average MAE: ${long_metrics['MAE'].mean():.2f}")
        print(f"• {abs(avg_mae_increase):.1f}% {'higher' if avg_mae_increase > 0 else 'lower'} "
              f"error due to lag effect")

        print("\n5. KEY CONCLUSIONS:")

        print("\na) Trade-off Between Accuracy and Smoothness:")
        print("• Shorter windows track prices more closely (lower MAE)")
        print("• Longer windows are smoother but lag behind actual prices")

        print("\nb) Optimal Moving Average Selection:")
        print(f"• For trend following: Use {best_long['MA_Type']} with {best_long['Window']:.0f}-day window")
        print(f"• For trading signals: Use {best_short['MA_Type']} with {best_short['Window']:.0f}-day window")
        print("• For robust analysis: Combine both short and long windows")

        print("\nc) Practical Applications:")
        print("• EMA is often preferred for trading due to its responsiveness")
        print("• SMA is better for identifying major trend reversals")
        print("• WMA provides a middle ground for balanced analysis")
        print("• Crossover strategies (short MA crossing long MA) signal potential trends")

        print("\nd) Model Performance:")
        best_overall = metrics_df.loc[metrics_df['MAE'].idxmin()]
        print(f"• Best overall model: {best_overall['MA_Type']} with {best_overall['Window']:.0f}-day window")
        print(f"• Achieved MAE of ${best_overall['MAE']:.2f} ({best_overall['MAPE']:.2f}% MAPE)")
        print(f"• Smoothness variance: {best_overall['Smoothness']:.2f}")

        print("\nANALYSIS COMPLETE")

    def run_complete_pipeline(self, short_window=20, long_window=100):
        """Execute the complete pipeline"""
        print("MOVING AVERAGES ANALYSIS PIPELINE - S&P 500")

        # Step 1: Load and process data
        self.load_and_process_data()

        # Step 2: Calculate moving averages
        self.calculate_moving_averages(short_window, long_window)

        # Step 2b: Calculate returns and volatility
        self.calculate_returns_and_volatility()

        # Step 3: Calculate metrics
        self.generate_metrics(short_window, long_window)

        # Step 4: Generate visualizations
        self.plot_original_data()

        # Full dataset comparison chart (e.g. ma_smooth_1000.png)
        self.plot_comparison_all_mas(short_window,
                                     data_points=n_days,
                                     filename_suffix=str(n_days))

        # 250-day subset comparison chart (ma_smooth_250.png)
        self.plot_comparison_all_mas(short_window,
                                     data_points=250,
                                     filename_suffix='250')


        self.plot_sma_comparison(short_window, long_window)
        self.plot_ema_comparison(short_window, long_window)
        self.plot_wma_comparison(short_window, long_window)
        self.plot_metrics_comparison()
        self.plot_returns_and_volatility()
        self.plot_crossover_signals(short_window, long_window)

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

    print(f"\nPipeline completed successfully, with short window = {var_short_window} and long window = {var_long_window}")
    # print("\nTo rerun with different window sizes:")
    # print("pipeline.run_complete_pipeline(short_window=30, long_window=200)")