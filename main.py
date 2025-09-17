import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import inspect
from typing import Dict, List, Any, Optional
warnings.filterwarnings('ignore')

# Page configuration with dark theme
st.set_page_config(
    page_title="Technical Indicators Performance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark theme CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #4a4a5a;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def download_data(ticker, start_date, end_date):
    """Download and cache stock data"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

@st.cache_data
def calculate_returns(data, periods):
    """Vectorized returns calculation"""
    returns = pd.DataFrame(index=data.index)
    close_prices = data['Close'].values
    
    for period in periods:
        returns[f'returns_{period}d'] = (pd.Series(close_prices).pct_change(period) * 100).values
    
    return returns

def get_all_pandas_ta_indicators():
    """Dynamically get all available pandas_ta indicators and their categories"""
    
    # Get all indicators from pandas_ta
    all_indicators = data.ta.indicators(as_list=True) if hasattr(ta, 'indicators') else []
    
    # Manual categorization of known indicators
    categories = {
        'Momentum': ['rsi', 'stoch', 'stochrsi', 'willr', 'uo', 'mom', 'cmo', 'cci', 
                    'ppo', 'apo', 'bop', 'trix', 'mfi', 'cfo', 'cg', 'cti', 'er', 
                    'fisher', 'inertia', 'kst', 'psl', 'pvo', 'qqe', 'rvi', 'roc',
                    'slope', 'smi', 'squeeze', 'squeeze_pro', 'stc', 'tsi', 'kdj',
                    'kst', 'macd', 'dm', 'pgo', 'pretty_good_oscillator'],
        'Trend': ['adx', 'aroon', 'chop', 'cksp', 'decay', 'dpo', 'increasing',
                 'decreasing', 'long_run', 'short_run', 'psar', 'qstick',
                 'supertrend', 'tsignals', 'ttm_trend', 'vhf', 'vortex', 
                 'xsignals', 'amat'],
        'Volatility': ['atr', 'natr', 'true_range', 'aberration', 'accbands',
                      'bbands', 'donchian', 'hwc', 'kc', 'massi', 'pdist', 'rvi',
                      'thermo', 'ui'],
        'Volume': ['ad', 'adosc', 'aobv', 'cmf', 'efi', 'eom', 'kvo', 'mfi',
                  'nvi', 'obv', 'pvi', 'pvol', 'pvr', 'pvt', 'vp', 'vwap',
                  'vwma', 'wb'],
        'Moving Averages': ['alma', 'dema', 'ema', 'fwma', 'hilo', 'hma', 'hwma',
                           'jma', 'kama', 'linreg', 'mama', 'mcgd', 'midpoint',
                           'midprice', 'ohlc4', 'pwma', 'rma', 'sinwma', 'sma',
                           'ssf', 'swma', 't3', 'tema', 'trima', 'vidya', 'vwap',
                           'vwma', 'wcp', 'wma', 'zlma'],
        'Statistics': ['entropy', 'kurtosis', 'mad', 'median', 'quantile', 'skew',
                      'stdev', 'tos_stdevall', 'variance', 'zscore'],
        'Patterns': ['cdl_pattern', 'cdl_doji', 'cdl_inside', 'cdl_z'],
        'Transform': ['hl2', 'hlc3', 'ohlc4', 'log_return', 'percent_return',
                     'cum_log_return', 'cum_percent_return'],
        'Utility': ['above', 'above_value', 'below', 'below_value', 'cross',
                   'cross_value'],
        'Performance': ['cagr', 'calmar', 'downside_deviation', 'jensens_alpha',
                       'log_max_drawdown', 'max_drawdown', 'optimal_f',
                       'pure_profit_score', 'sharpe', 'sortino', 'ulcer_index'],
        'Other': ['ao', 'apo', 'bias', 'brar', 'ebsw', 'efi', 'ha', 'ichimoku',
                 'pivots', 'pivot_camarilla', 'pivot_demark', 'pivot_fibonacci',
                 'pivot_woodie']
    }
    
    return categories

class IndicatorCalculator:
    """Efficient indicator calculator with automatic parameter detection"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.ohlcv = {
            'open': data['Open'],
            'high': data['High'],
            'low': data['Low'],
            'close': data['Close'],
            'volume': data['Volume'] if 'Volume' in data.columns else None
        }
        
    def get_indicator_params(self, indicator_name: str) -> Dict[str, Any]:
        """Get the parameters for a specific indicator"""
        
        # Common parameter mappings
        param_map = {
            # Single period indicators
            'single_period': ['rsi', 'atr', 'natr', 'cci', 'willr', 'cmo', 'roc',
                            'mom', 'dpo', 'cti', 'er', 'cg', 'cfo', 'kama',
                            'sma', 'ema', 'wma', 'tema', 'dema', 'trima', 'hma',
                            'fwma', 'hwma', 'rma', 'sinwma', 'swma', 't3', 'zlma',
                            'midpoint', 'entropy', 'kurtosis', 'mad', 'median',
                            'skew', 'stdev', 'variance', 'zscore', 'linreg',
                            'ui', 'chop', 'vhf', 'qstick', 'jma', 'alma', 'vidya',
                            'mcgd', 'pwma'],
            
            # Fast/Slow period indicators
            'fast_slow': ['macd', 'ppo', 'apo', 'tsi', 'kvo', 'smi', 'stc', 
                         'adosc', 'pvo', 'copp', 'massi'],
            
            # Special indicators
            'stoch': ['stoch', 'stochrsi', 'kdj'],
            'bbands': ['bbands', 'kc'],
            'adx': ['adx', 'dmi'],
            'aroon': ['aroon'],
            'supertrend': ['supertrend'],
            'donchian': ['donchian'],
            'ichimoku': ['ichimoku'],
            'vortex': ['vortex'],
            'psar': ['psar'],
            'squeeze': ['squeeze', 'squeeze_pro'],
            
            # No period indicators
            'no_period': ['vwap', 'obv', 'ad', 'pvt', 'ha', 'hl2', 'hlc3', 
                         'ohlc4', 'true_range', 'bop', 'ao', 'wcp', 'pvol',
                         'pvr', 'nvi', 'pvi']
        }
        
        # Reverse mapping
        for param_type, indicators in param_map.items():
            if indicator_name.lower() in indicators:
                return {'type': param_type}
        
        return {'type': 'single_period'}  # Default
    
    def calculate_indicator(self, indicator_name: str, periods: List[int]) -> pd.DataFrame:
        """Calculate a single indicator for multiple periods"""
        
        results = pd.DataFrame(index=self.data.index)
        indicator_func = getattr(ta, indicator_name.lower(), None)
        
        if not indicator_func:
            return results
        
        param_info = self.get_indicator_params(indicator_name)
        
        try:
            if param_info['type'] == 'no_period':
                # Indicators that don't take period parameter
                result = indicator_func(self.ohlcv['high'], self.ohlcv['low'], 
                                       self.ohlcv['close'], self.ohlcv['volume'])
                if isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        results[f'{indicator_name}_{col}'] = result[col]
                elif isinstance(result, pd.Series):
                    results[indicator_name] = result
                    
            elif param_info['type'] == 'single_period':
                # Single period indicators
                for period in periods:
                    try:
                        # Try different parameter combinations
                        if indicator_name.lower() in ['sma', 'ema', 'wma', 'rma', 'tema', 
                                                      'dema', 'hma', 'zlma', 't3', 'kama',
                                                      'fwma', 'hwma', 'sinwma', 'swma',
                                                      'trima', 'midpoint', 'linreg', 'jma',
                                                      'alma', 'vidya', 'mcgd', 'pwma']:
                            result = indicator_func(self.ohlcv['close'], length=period)
                        elif indicator_name.lower() in ['rsi', 'cmo', 'roc', 'mom', 
                                                        'dpo', 'cti', 'er', 'cg', 'cfo']:
                            result = indicator_func(self.ohlcv['close'], length=period)
                        elif indicator_name.lower() in ['atr', 'natr']:
                            result = indicator_func(self.ohlcv['high'], self.ohlcv['low'], 
                                                  self.ohlcv['close'], length=period)
                        elif indicator_name.lower() in ['cci', 'willr']:
                            result = indicator_func(self.ohlcv['high'], self.ohlcv['low'], 
                                                  self.ohlcv['close'], length=period)
                        elif indicator_name.lower() in ['mfi']:
                            result = indicator_func(self.ohlcv['high'], self.ohlcv['low'], 
                                                  self.ohlcv['close'], self.ohlcv['volume'], 
                                                  length=period)
                        elif indicator_name.lower() in ['midprice']:
                            result = indicator_func(self.ohlcv['high'], self.ohlcv['low'], 
                                                  length=period)
                        elif indicator_name.lower() in ['entropy', 'kurtosis', 'skew', 
                                                        'stdev', 'variance', 'zscore']:
                            result = indicator_func(self.ohlcv['close'], length=period)
                        else:
                            # Try with close only
                            result = indicator_func(self.ohlcv['close'], length=period)
                        
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                results[f'{indicator_name}_{period}_{col}'] = result[col]
                        elif isinstance(result, pd.Series):
                            results[f'{indicator_name}_{period}'] = result
                    except:
                        continue
                        
            elif param_info['type'] == 'fast_slow':
                # Fast/Slow indicators
                for period in periods:
                    fast = max(period // 2, 2)
                    slow = period
                    try:
                        if indicator_name.lower() == 'macd':
                            result = self.data.ta.macd(fast=fast, slow=slow)
                        elif indicator_name.lower() in ['ppo', 'apo']:
                            result = indicator_func(self.ohlcv['close'], fast=fast, slow=slow)
                        elif indicator_name.lower() in ['tsi', 'smi']:
                            result = indicator_func(self.ohlcv['close'], fast=fast, slow=slow)
                        elif indicator_name.lower() in ['kvo', 'adosc']:
                            result = indicator_func(self.ohlcv['high'], self.ohlcv['low'],
                                                  self.ohlcv['close'], self.ohlcv['volume'],
                                                  fast=fast, slow=slow)
                        else:
                            continue
                            
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                results[f'{indicator_name}_{period}_{col}'] = result[col]
                        elif isinstance(result, pd.Series):
                            results[f'{indicator_name}_{period}'] = result
                    except:
                        continue
                        
            elif param_info['type'] == 'stoch':
                # Stochastic indicators
                for period in periods:
                    try:
                        result = self.data.ta.stoch(k=period)
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                results[f'{indicator_name}_{period}_{col}'] = result[col]
                    except:
                        continue
                        
            elif param_info['type'] == 'bbands':
                # Bollinger Bands type
                for period in periods:
                    try:
                        result = self.data.ta.bbands(length=period)
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                results[f'{indicator_name}_{period}_{col}'] = result[col]
                    except:
                        continue
                        
            elif param_info['type'] == 'adx':
                # ADX type
                for period in periods:
                    try:
                        result = self.data.ta.adx(length=period)
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                results[f'{indicator_name}_{period}_{col}'] = result[col]
                    except:
                        continue
                        
            elif param_info['type'] == 'aroon':
                # Aroon
                for period in periods:
                    try:
                        result = self.data.ta.aroon(length=period)
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                results[f'{indicator_name}_{period}_{col}'] = result[col]
                    except:
                        continue
                        
            elif param_info['type'] == 'supertrend':
                # Supertrend
                for period in periods:
                    try:
                        result = self.data.ta.supertrend(length=period)
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                results[f'{indicator_name}_{period}_{col}'] = result[col]
                    except:
                        continue
                        
            elif param_info['type'] == 'donchian':
                # Donchian
                for period in periods:
                    try:
                        result = self.data.ta.donchian(lower_length=period, upper_length=period)
                        if isinstance(result, pd.DataFrame):
                            for col in result.columns:
                                results[f'{indicator_name}_{period}_{col}'] = result[col]
                    except:
                        continue
                        
        except Exception as e:
            pass
        
        return results

def calculate_all_indicators_vectorized(data: pd.DataFrame, 
                                       selected_indicators: List[str], 
                                       periods: List[int]) -> pd.DataFrame:
    """Calculate all indicators using vectorized operations"""
    
    # Initialize calculator
    calc = IndicatorCalculator(data)
    
    # Use pandas_ta strategy for bulk calculation (most efficient)
    try:
        # Create a custom strategy
        my_strategy = ta.Strategy(
            name="Custom Strategy",
            description="Performance Analysis Strategy",
            ta=[]
        )
        
        # Add indicators to strategy
        for indicator_name in selected_indicators:
            param_info = calc.get_indicator_params(indicator_name)
            
            if param_info['type'] == 'no_period':
                my_strategy.ta.append({indicator_name.lower(): {}})
            elif param_info['type'] == 'single_period':
                for period in periods:
                    my_strategy.ta.append({indicator_name.lower(): {"length": period}})
            elif param_info['type'] == 'fast_slow':
                for period in periods:
                    my_strategy.ta.append({
                        indicator_name.lower(): {
                            "fast": max(period // 2, 2),
                            "slow": period
                        }
                    })
            elif indicator_name.lower() == 'stoch':
                for period in periods:
                    my_strategy.ta.append({"stoch": {"k": period}})
            elif indicator_name.lower() == 'bbands':
                for period in periods:
                    my_strategy.ta.append({"bbands": {"length": period}})
            elif indicator_name.lower() == 'adx':
                for period in periods:
                    my_strategy.ta.append({"adx": {"length": period}})
        
        # Run strategy
        data.ta.strategy(my_strategy)
        
    except:
        # Fallback to individual calculation
        all_results = []
        
        for indicator_name in selected_indicators:
            result = calc.calculate_indicator(indicator_name, periods)
            if not result.empty:
                all_results.append(result)
        
        if all_results:
            data = pd.concat([data] + all_results, axis=1)
    
    # Add all available candlestick patterns
    try:
        cdl_patterns = data.ta.cdl_pattern(name="all")
        if cdl_patterns is not None and not cdl_patterns.empty:
            data = pd.concat([data, cdl_patterns], axis=1)
    except:
        pass
    
    # Return only indicator columns
    indicator_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    return data[indicator_cols]

@st.cache_data
def get_available_indicators():
    """Get list of all available indicators in pandas_ta"""
    
    # Get all methods from pandas_ta
    all_methods = []
    for name, obj in inspect.getmembers(ta):
        if callable(obj) and not name.startswith('_'):
            all_methods.append(name.upper())
    
    # Common indicators that should be available
    common_indicators = [
        'RSI', 'MACD', 'BBANDS', 'SMA', 'EMA', 'ATR', 'ADX', 'STOCH',
        'CCI', 'WILLR', 'MFI', 'OBV', 'ROC', 'MOM', 'PPO', 'APO',
        'AROON', 'CMO', 'DPO', 'TRIX', 'ULTOSC', 'TSI', 'KST', 'VORTEX',
        'SUPERTREND', 'PSAR', 'KAMA', 'TEMA', 'DEMA', 'HMA', 'WMA',
        'VWAP', 'VWMA', 'DONCHIAN', 'KC', 'ENTROPY', 'KURTOSIS', 'ZSCORE'
    ]
    
    # Combine and deduplicate
    all_indicators = list(set(all_methods + common_indicators))
    all_indicators.sort()
    
    return all_indicators

@st.cache_data  
def analyze_performance_by_quantiles(data, indicators, returns, quantiles, indicator_col):
    """Analyze performance by quantiles with vectorization"""
    # Drop NaN values
    valid_idx = ~(indicators[indicator_col].isna() | returns.isna().any(axis=1))
    clean_indicators = indicators.loc[valid_idx, indicator_col]
    clean_returns = returns[valid_idx]
    
    if len(clean_indicators) < quantiles:
        return None
    
    # Check if it's a pattern indicator
    if 'CDL' in indicator_col:
        # For pattern indicators, group by signal value
        unique_vals = clean_indicators.unique()
        results = []
        
        for val in unique_vals:
            mask = clean_indicators == val
            if mask.sum() > 0:
                signal_type = 'Bearish' if val < 0 else 'Bullish' if val > 0 else 'Neutral'
                stats = {
                    'Signal': signal_type,
                    'Value': val,
                    'Count': mask.sum(),
                    'Percentage': mask.sum() / len(clean_indicators) * 100
                }
                
                for ret_col in clean_returns.columns:
                    stats[f'{ret_col}_mean'] = clean_returns.loc[mask, ret_col].mean()
                    stats[f'{ret_col}_std'] = clean_returns.loc[mask, ret_col].std()
                    stats[f'{ret_col}_win_rate'] = (clean_returns.loc[mask, ret_col] > 0).mean() * 100
                
                results.append(stats)
        
        return pd.DataFrame(results)
    
    # For regular indicators, use quantiles
    try:
        quantile_labels = pd.qcut(clean_indicators, q=quantiles, duplicates='drop')
    except:
        return None
    
    # Calculate statistics for each quantile
    results = []
    for quantile in quantile_labels.cat.categories:
        mask = quantile_labels == quantile
        quantile_returns = clean_returns[mask]
        
        stats = {
            'Quantile': str(quantile),
            'Count': mask.sum(),
            'Indicator_Mean': clean_indicators[mask].mean(),
            'Indicator_Std': clean_indicators[mask].std()
        }
        
        for ret_col in clean_returns.columns:
            stats[f'{ret_col}_mean'] = quantile_returns[ret_col].mean()
            stats[f'{ret_col}_std'] = quantile_returns[ret_col].std()
            stats[f'{ret_col}_sharpe'] = stats[f'{ret_col}_mean'] / stats[f'{ret_col}_std'] if stats[f'{ret_col}_std'] != 0 else 0
            stats[f'{ret_col}_win_rate'] = (quantile_returns[ret_col] > 0).mean() * 100
        
        results.append(stats)
    
    return pd.DataFrame(results)

def create_performance_plots(data, indicators, returns, indicator_col, return_period, quantiles):
    """Create interactive performance plots using Plotly"""
    
    # Check if this is a pattern indicator
    is_pattern = 'CDL' in indicator_col
    
    # Filter valid data
    valid_idx = ~(indicators[indicator_col].isna() | returns[f'returns_{return_period}d'].isna())
    clean_indicators = indicators.loc[valid_idx, indicator_col]
    clean_returns = returns.loc[valid_idx, f'returns_{return_period}d']
    
    if len(clean_indicators) < 10:
        st.warning(f"Not enough data points for {indicator_col}")
        return None
    
    if is_pattern:
        # Special handling for pattern indicators
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{indicator_col} Signal Distribution',
                f'Average {return_period}-Day Returns by Signal',
                f'Signal Frequency Over Time',
                f'Returns Distribution by Signal Type'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # Signal distribution and returns analysis
        signal_counts = clean_indicators.value_counts().sort_index()
        
        if not signal_counts.empty:
            signal_labels = ['Bearish' if x < 0 else 'Bullish' if x > 0 else 'Neutral' 
                            for x in signal_counts.index]
            
            fig.add_trace(
                go.Bar(x=signal_labels, y=signal_counts.values, 
                       marker_color=['red' if x < 0 else 'green' if x > 0 else 'gray' 
                                    for x in signal_counts.index]),
                row=1, col=1
            )
            
            # Average returns by signal
            avg_returns = []
            for signal_val in signal_counts.index:
                mask = clean_indicators == signal_val
                avg_ret = clean_returns[mask].mean() if mask.sum() > 0 else 0
                avg_returns.append(avg_ret)
            
            fig.add_trace(
                go.Bar(x=signal_labels, y=avg_returns,
                       marker_color=['red' if x < 0 else 'green' if x > 0 else 'gray' 
                                    for x in signal_counts.index]),
                row=1, col=2
            )
    
    else:
        # Regular indicators
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'{indicator_col} Distribution',
                f'Average {return_period}-Day Returns by Quantile',
                f'Rolling Correlation (126 days)',
                f'{indicator_col} vs {return_period}-Day Returns'
            ),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=clean_indicators, nbinsx=50, name='Distribution',
                         marker_color='#3366CC'),
            row=1, col=1
        )
        
        # Mean line
        fig.add_vline(x=clean_indicators.mean(), line_dash="dash", 
                      line_color="red", row=1, col=1)
        
        # Quantile analysis
        perf_analysis = analyze_performance_by_quantiles(
            data, indicators, returns, quantiles, indicator_col
        )
        
        if perf_analysis is not None and not perf_analysis.empty:
            if 'Quantile' in perf_analysis.columns:
                fig.add_trace(
                    go.Bar(x=perf_analysis['Quantile'], 
                           y=perf_analysis[f'returns_{return_period}d_mean'],
                           name='Avg Returns',
                           marker_color='#DC3912'),
                    row=1, col=2
                )
        
        # Rolling correlation
        window = min(126, len(clean_indicators) // 4)
        if len(clean_indicators) > window and window > 10:
            rolling_corr = clean_indicators.rolling(window).corr(clean_returns)
            fig.add_trace(
                go.Scatter(x=rolling_corr.index, y=rolling_corr.values,
                          mode='lines', name='Rolling Correlation',
                          line=dict(color='#FF9900')),
                row=2, col=1
            )
            
            overall_corr = clean_indicators.corr(clean_returns)
            fig.add_hline(y=overall_corr, line_dash="dash", 
                          line_color="red", row=2, col=1)
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=clean_indicators, y=clean_returns,
                      mode='markers', name='Data Points',
                      marker=dict(color='#109618', size=3, opacity=0.5)),
            row=2, col=2
        )
        
        # Trend line
        if len(clean_indicators) > 1:
            try:
                z = np.polyfit(clean_indicators, clean_returns, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(clean_indicators.min(), clean_indicators.max(), 100)
                fig.add_trace(
                    go.Scatter(x=x_trend, y=p(x_trend),
                              mode='lines', name='Trend Line',
                              line=dict(color='red', width=2)),
                    row=2, col=2
                )
            except:
                pass
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=True,
        title_text=f"Performance Analysis: {indicator_col}",
        title_font_size=20
    )
    
    return fig

def main():
    st.title("🎯 Technical Indicators Performance Analyzer")
    st.markdown("### All pandas_ta Indicators - Vectorized Edition")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ticker selection
        ticker = st.text_input("Stock Ticker", value="SPY", 
                              help="Enter a valid stock ticker symbol")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                      value=datetime.now() - timedelta(days=5*365))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Return periods
        st.subheader("📈 Return Periods")
        return_periods = st.multiselect(
            "Select return periods (days)",
            options=[1, 2, 3, 5, 10, 20, 30, 60],
            default=[1, 5, 10, 20]
        )
        
        # Quantiles
        quantiles = st.slider("Number of Quantiles", 
                            min_value=3, max_value=20, value=10,
                            help="Number of bins to divide indicator values")
        
        # Indicator selection
        st.subheader("📊 Indicator Selection")
        
        # Get all available indicators
        all_indicators = get_available_indicators()
        categories = get_all_pandas_ta_indicators()
        
        # Selection method
        selection_method = st.radio(
            "Selection Method",
            ["Quick Sets", "By Category", "Custom Selection", "All Indicators"],
            horizontal=False
        )
        
        selected_indicators = []
        
        if selection_method == "Quick Sets":
            quick_set = st.selectbox(
                "Choose a predefined set",
                ["Popular", "Momentum", "Trend", "Volatility", "Volume", "All Moving Averages"]
            )
            
            quick_sets = {
                "Popular": ['RSI', 'MACD', 'BBANDS', 'ATR', 'ADX', 'STOCH', 'CCI', 'MFI', 'OBV'],
                "Momentum": ['RSI', 'STOCH', 'CCI', 'WILLR', 'MOM', 'ROC', 'TSI', 'CMO'],
                "Trend": ['ADX', 'AROON', 'PSAR', 'SUPERTREND', 'VORTEX', 'DPO'],
                "Volatility": ['ATR', 'NATR', 'BBANDS', 'DONCHIAN', 'KC'],
                "Volume": ['OBV', 'MFI', 'AD', 'CMF', 'VWAP', 'VWMA'],
                "All Moving Averages": ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'HMA', 'KAMA', 'T3']
            }
            
            selected_indicators = quick_sets[quick_set]
            st.info(f"Selected {len(selected_indicators)} {quick_set} indicators")
            
        elif selection_method == "By Category":
            selected_categories = st.multiselect(
                "Select Categories",
                options=list(categories.keys()),
                default=["Momentum", "Volatility"]
            )
            
            for category in selected_categories:
                # Get indicators from category that are available
                cat_indicators = [ind.upper() for ind in categories[category] 
                                 if ind.upper() in all_indicators]
                selected_indicators.extend(cat_indicators)
            
            selected_indicators = list(set(selected_indicators))
            st.info(f"Selected {len(selected_indicators)} indicators from {len(selected_categories)} categories")
            
        elif selection_method == "Custom Selection":
            selected_indicators = st.multiselect(
                "Select individual indicators",
                options=all_indicators,
                default=['RSI', 'MACD', 'BBANDS', 'ATR']
            )
            
        else:  # All Indicators
            selected_indicators = all_indicators
            st.warning(f"⚠️ Calculating ALL {len(selected_indicators)} indicators - this may take time!")
        
        # Include patterns
        include_patterns = st.checkbox("Include Candlestick Patterns", value=False)
        
        # Period configuration
        st.subheader("⏱️ Period Configuration")
        
        period_method = st.radio(
            "Period Selection",
            ["Range", "Specific Values"],
            horizontal=True
        )
        
        if period_method == "Range":
            col1, col2, col3 = st.columns(3)
            with col1:
                min_period = st.number_input("Min", value=5, min_value=2, max_value=200)
            with col2:
                max_period = st.number_input("Max", value=50, min_value=2, max_value=200)
            with col3:
                step_period = st.number_input("Step", value=5, min_value=1, max_value=50)
            
            periods = list(range(min_period, max_period + 1, step_period))
        else:
            periods_str = st.text_input(
                "Enter periods (comma-separated)",
                value="5, 10, 14, 20, 30, 50",
                help="Example: 5, 10, 14, 20, 30, 50"
            )
            try:
                periods = [int(p.strip()) for p in periods_str.split(',')]
            except:
                periods = [5, 10, 14, 20, 30, 50]
        
        st.info(f"Periods to calculate: {periods}")
        
        # Analysis button
        analyze_button = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    
    # Main content area
    if analyze_button:
        # Download data
        with st.spinner("📥 Downloading data..."):
            data = download_data(ticker, start_date, end_date)
        
        if data is not None:
            # Calculate returns
            with st.spinner("📊 Calculating returns..."):
                returns = calculate_returns(data, return_periods)
            
            # Calculate indicators
            with st.spinner(f"⚡ Calculating {len(selected_indicators)} indicators with vectorization..."):
                indicators = calculate_all_indicators_vectorized(data, selected_indicators, periods)
            
            # Get available indicators
            available_indicators = [col for col in indicators.columns if not indicators[col].isna().all()]
            
            if not available_indicators:
                st.error("No indicators were calculated successfully. Please check your selection.")
                return
            
            # Display results
            st.success(f"✅ Successfully calculated {len(available_indicators)} indicator configurations!")
            
            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Analysis", "📈 Performance", "🏆 Top Performers", 
                "🔥 Correlation", "📋 Summary"
            ])
            
            with tab1:
                st.header("Indicator Performance Analysis")
                
                # Filter indicators
                pattern_indicators = [col for col in available_indicators if 'CDL' in col]
                regular_indicators = [col for col in available_indicators if 'CDL' not in col]
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if regular_indicators:
                        selected_indicator = st.selectbox(
                            "Select indicator for analysis",
                            options=regular_indicators + pattern_indicators,
                            format_func=lambda x: x.replace('_', ' ')
                        )
                    else:
                        st.warning("No indicators available")
                        selected_indicator = None
                
                with col2:
                    if selected_indicator:
                        selected_return = st.selectbox(
                            "Return period",
                            options=return_periods,
                            format_func=lambda x: f"{x} days"
                        )
                
                if selected_indicator:
                    # Create plots
                    fig = create_performance_plots(
                        data, indicators, returns,
                        selected_indicator, selected_return, quantiles
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics
                    with st.expander("📊 Detailed Metrics"):
                        perf_analysis = analyze_performance_by_quantiles(
                            data, indicators, returns, quantiles, selected_indicator
                        )
                        if perf_analysis is not None:
                            st.dataframe(perf_analysis, use_container_width=True)
            
            with tab2:
                st.header("Performance Metrics Dashboard")
                
                # Calculate metrics for all indicators
                with st.spinner("Calculating performance metrics..."):
                    performance_summary = []
                    
                    # Limit analysis to prevent timeout
                    indicators_to_analyze = available_indicators[:100]
                    
                    for ind_col in indicators_to_analyze:
                        for ret_period in return_periods[:2]:  # Top 2 return periods
                            perf = analyze_performance_by_quantiles(
                                data, indicators, returns, quantiles, ind_col
                            )
                            
                            if perf is not None and not perf.empty:
                                if 'Signal' in perf.columns:
                                    # Pattern indicator
                                    for _, row in perf.iterrows():
                                        performance_summary.append({
                                            'Indicator': ind_col,
                                            'Type': 'Pattern',
                                            'Signal': row['Signal'],
                                            'Return_Period': f'{ret_period}d',
                                            'Mean_Return': row[f'returns_{ret_period}d_mean'],
                                            'Win_Rate': row[f'returns_{ret_period}d_win_rate'],
                                            'Count': row['Count']
                                        })
                                else:
                                    # Regular indicator
                                    top_q = perf.iloc[-1]
                                    bottom_q = perf.iloc[0]
                                    
                                    performance_summary.append({
                                        'Indicator': ind_col,
                                        'Type': 'Regular',
                                        'Return_Period': f'{ret_period}d',
                                        'Top_Q_Return': top_q[f'returns_{ret_period}d_mean'],
                                        'Bottom_Q_Return': bottom_q[f'returns_{ret_period}d_mean'],
                                        'Spread': top_q[f'returns_{ret_period}d_mean'] - 
                                                bottom_q[f'returns_{ret_period}d_mean'],
                                        'Top_Win_Rate': top_q[f'returns_{ret_period}d_win_rate']
                                    })
                
                if performance_summary:
                    perf_df = pd.DataFrame(performance_summary)
                    
                    # Regular indicators
                    regular_perf = perf_df[perf_df['Type'] == 'Regular']
                    if not regular_perf.empty:
                        st.subheader("📊 Top Regular Indicators by Spread")
                        top_regular = regular_perf.nlargest(20, 'Spread')
                        
                        fig_spread = px.bar(
                            top_regular,
                            x='Indicator',
                            y='Spread',
                            color='Return_Period',
                            title="Top 20 Indicators by Return Spread",
                            template="plotly_dark"
                        )
                        fig_spread.update_xaxis(tickangle=45)
                        st.plotly_chart(fig_spread, use_container_width=True)
                    
                    # Pattern indicators
                    pattern_perf = perf_df[perf_df['Type'] == 'Pattern']
                    if not pattern_perf.empty:
                        st.subheader("🕯️ Pattern Recognition Performance")
                        
                        # Best bullish patterns
                        bullish = pattern_perf[pattern_perf['Signal'] == 'Bullish'].nlargest(10, 'Mean_Return')
                        if not bullish.empty:
                            st.write("**Top Bullish Patterns**")
                            st.dataframe(
                                bullish[['Indicator', 'Return_Period', 'Mean_Return', 'Win_Rate', 'Count']],
                                use_container_width=True
                            )
            
            with tab3:
                st.header("🏆 Top Performing Configurations")
                
                # Calculate correlations
                with st.spinner("Finding best configurations..."):
                    best_configs = []
                    
                    for ind_col in available_indicators[:100]:
                        for ret_period in return_periods:
                            valid_idx = ~(indicators[ind_col].isna() | 
                                        returns[f'returns_{ret_period}d'].isna())
                            
                            if valid_idx.sum() > 100:
                                corr = indicators.loc[valid_idx, ind_col].corr(
                                    returns.loc[valid_idx, f'returns_{ret_period}d']
                                )
                                
                                best_configs.append({
                                    'Indicator': ind_col,
                                    'Return_Period': f'{ret_period}d',
                                    'Correlation': corr,
                                    'Abs_Correlation': abs(corr),
                                    'Sample_Size': valid_idx.sum()
                                })
                
                if best_configs:
                    best_df = pd.DataFrame(best_configs)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Strongest Positive Correlations")
                        positive = best_df.nlargest(15, 'Correlation')
                        st.dataframe(
                            positive[['Indicator', 'Return_Period', 'Correlation', 'Sample_Size']].style.format({
                                'Correlation': '{:.4f}'
                            }).background_gradient(cmap='Greens', subset=['Correlation']),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.subheader("📉 Strongest Negative Correlations")
                        negative = best_df.nsmallest(15, 'Correlation')
                        st.dataframe(
                            negative[['Indicator', 'Return_Period', 'Correlation', 'Sample_Size']].style.format({
                                'Correlation': '{:.4f}'
                            }).background_gradient(cmap='Reds_r', subset=['Correlation']),
                            use_container_width=True
                        )
            
            with tab4:
                st.header("🔥 Correlation Analysis")
                
                # Select indicators for correlation matrix
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    matrix_indicators = st.multiselect(
                        "Select indicators for correlation matrix",
                        options=available_indicators[:50],
                        default=available_indicators[:min(15, len(available_indicators))]
                    )
                
                with col2:
                    correlation_method = st.radio(
                        "Method",
                        ["Pearson", "Spearman"],
                        help="Pearson for linear, Spearman for monotonic relationships"
                    )
                
                if matrix_indicators and len(matrix_indicators) > 1:
                    # Calculate correlation matrix
                    if correlation_method == "Pearson":
                        corr_matrix = indicators[matrix_indicators].corr()
                    else:
                        corr_matrix = indicators[matrix_indicators].corr(method='spearman')
                    
                    # Create heatmap
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text}',
                        textfont={"size": 8},
                        colorbar=dict(title="Correlation")
                    ))
                    
                    fig_corr.update_layout(
                        template="plotly_dark",
                        title=f"{correlation_method} Correlation Matrix",
                        height=max(400, len(matrix_indicators) * 25),
                        xaxis={'side': 'bottom'},
                        yaxis={'side': 'left'}
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Correlation statistics
                    with st.expander("📊 Correlation Statistics"):
                        corr_flat = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Correlation", f"{np.mean(corr_flat):.3f}")
                        with col2:
                            st.metric("Max Correlation", f"{np.max(corr_flat):.3f}")
                        with col3:
                            st.metric("Min Correlation", f"{np.min(corr_flat):.3f}")
                        with col4:
                            st.metric("Std Correlation", f"{np.std(corr_flat):.3f}")
            
            with tab5:
                st.header("📋 Analysis Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("📊 Data Overview")
                    st.metric("Total Trading Days", f"{len(data):,}")
                    st.metric("Date Range", f"{(end_date - start_date).days} days")
                    st.metric("Start Price", f"${data['Close'].iloc[0]:.2f}")
                    st.metric("End Price", f"${data['Close'].iloc[-1]:.2f}")
                    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    st.metric("Total Return", f"{total_return:.2f}%")
                
                with col2:
                    st.subheader("📈 Indicators Calculated")
                    st.metric("Total Configurations", f"{len(available_indicators):,}")
                    st.metric("Unique Indicators", f"{len(selected_indicators):,}")
                    st.metric("Periods Analyzed", f"{len(periods)}")
                    
                    pattern_count = sum(1 for col in available_indicators if 'CDL' in col)
                    st.metric("Pattern Indicators", f"{pattern_count:,}")
                    st.metric("Regular Indicators", f"{len(available_indicators) - pattern_count:,}")
                
                with col3:
                    st.subheader("📊 Return Periods")
                    for period in return_periods:
                        ret_col = f'returns_{period}d'
                        if ret_col in returns.columns:
                            mean_ret = returns[ret_col].mean()
                            std_ret = returns[ret_col].std()
                            sharpe = mean_ret / std_ret if std_ret != 0 else 0
                            st.write(f"**{period}-day**")
                            st.write(f"μ: {mean_ret:.2f}%, σ: {std_ret:.2f}%, SR: {sharpe:.3f}")
                
                # Indicator category breakdown
                st.subheader("📊 Indicators by Category")
                
                category_counts = {}
                for col in available_indicators:
                    # Simple categorization based on indicator name patterns
                    if 'SMA' in col or 'EMA' in col or 'WMA' in col or 'TEMA' in col or 'DEMA' in col:
                        category = 'Moving Averages'
                    elif 'RSI' in col or 'STOCH' in col or 'CCI' in col or 'MOM' in col or 'ROC' in col:
                        category = 'Momentum'
                    elif 'BB' in col or 'ATR' in col or 'NATR' in col or 'DC' in col:
                        category = 'Volatility'
                    elif 'ADX' in col or 'AROON' in col or 'PSAR' in col or 'SUPERTREND' in col:
                        category = 'Trend'
                    elif 'OBV' in col or 'MFI' in col or 'AD' in col or 'VWAP' in col:
                        category = 'Volume'
                    elif 'CDL' in col:
                        category = 'Patterns'
                    else:
                        category = 'Other'
                    
                    category_counts[category] = category_counts.get(category, 0) + 1
                
                if category_counts:
                    cat_df = pd.DataFrame(list(category_counts.items()), 
                                         columns=['Category', 'Count'])
                    cat_df = cat_df.sort_values('Count', ascending=False)
                    
                    fig_cat = px.pie(
                        cat_df,
                        values='Count',
                        names='Category',
                        title="Indicator Distribution by Category",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)

if __name__ == "__main__":
    main()
