import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
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

def get_indicator_categories():
    """Return all indicators organized by category"""
    return {
        'Momentum Indicators': [
            'RSI', 'STOCH_K', 'STOCH_D', 'STOCHF_K', 'STOCHF_D', 'STOCHRSI_K', 'STOCHRSI_D',
            'WILLR', 'ULTOSC', 'MOMENTUM', 'CMO', 'CCI', 'DX', 'PPO', 'APO', 
            'BOP', 'TRIX', 'MFI', 'ADOSC'
        ],
        'Trend Indicators': [
            'ADX', 'ADXR', 'AROON_UP', 'AROON_DOWN', 'AROONOSC', 
            'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM',
            'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'MACDEXT', 'MACDFIX',
            'HT_TRENDLINE', 'HT_TRENDMODE'
        ],
        'Volatility Indicators': [
            'ATR', 'NATR', 'TRANGE', 
            'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER',
            'STDDEV', 'VAR', 'AVGDEV'
        ],
        'Moving Averages': [
            'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'FAMA',
            'T3', 'MA', 'MIDPOINT', 'MIDPRICE', 'HT_DCPERIOD', 'HT_DCPHASE'
        ],
        'Volume Indicators': [
            'AD', 'ADOSC', 'OBV', 'MFI'
        ],
        'Cycle Indicators': [
            'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_INPHASE', 'HT_PHASOR_QUADRATURE',
            'HT_SINE_SINE', 'HT_SINE_LEADSINE', 'HT_TRENDMODE'
        ],
        'Price Transform': [
            'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'
        ],
        'Statistical Functions': [
            'LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE',
            'TSF', 'BETA', 'CORREL', 'STDDEV', 'VAR', 'AVGDEV'
        ],
        'Pattern Recognition': [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS',
            'CDLINNECK', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLKICKINGBYLENGTH',
            'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU',
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
            'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
            'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
            'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
            'CDLXSIDEGAP3METHODS'
        ],
        'Other Indicators': [
            'ROC', 'ROCP', 'ROCR', 'ROCR100', 'BETA', 'CORREL', 'MIN', 'MAX',
            'MININDEX', 'MAXINDEX', 'MINMAX', 'MINMAXINDEX', 'SUM', 
            'SAR', 'SAREXT', 'HT_TRENDLINE'
        ]
    }

def calculate_all_indicators(data, indicator_params):
    """Calculate all technical indicators using talib with vectorization"""
    indicators = pd.DataFrame(index=data.index)
    open_prices = data['Open'].values
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    volume = data['Volume'].values
    
    # Define all indicator functions with their parameters
    indicator_functions = {
        # Momentum Indicators
        'RSI': lambda p: talib.RSI(close, timeperiod=p),
        'STOCH_K': lambda p: talib.STOCH(high, low, close, fastk_period=p)[0],
        'STOCH_D': lambda p: talib.STOCH(high, low, close, fastk_period=p)[1],
        'STOCHF_K': lambda p: talib.STOCHF(high, low, close, fastk_period=p)[0],
        'STOCHF_D': lambda p: talib.STOCHF(high, low, close, fastk_period=p)[1],
        'STOCHRSI_K': lambda p: talib.STOCHRSI(close, timeperiod=p)[0],
        'STOCHRSI_D': lambda p: talib.STOCHRSI(close, timeperiod=p)[1],
        'WILLR': lambda p: talib.WILLR(high, low, close, timeperiod=p),
        'ULTOSC': lambda p: talib.ULTOSC(high, low, close, timeperiod1=p//3, timeperiod2=p//2, timeperiod3=p),
        'MOMENTUM': lambda p: talib.MOM(close, timeperiod=p),
        'CMO': lambda p: talib.CMO(close, timeperiod=p),
        'CCI': lambda p: talib.CCI(high, low, close, timeperiod=p),
        'DX': lambda p: talib.DX(high, low, close, timeperiod=p),
        'PPO': lambda p: talib.PPO(close, fastperiod=max(p//2, 2), slowperiod=p),
        'APO': lambda p: talib.APO(close, fastperiod=max(p//2, 2), slowperiod=p),
        'BOP': lambda p: talib.BOP(open_prices, high, low, close),
        'TRIX': lambda p: talib.TRIX(close, timeperiod=p),
        'MFI': lambda p: talib.MFI(high, low, close, volume, timeperiod=p),
        'ADOSC': lambda p: talib.ADOSC(high, low, close, volume, fastperiod=max(p//3, 2), slowperiod=p),
        
        # Trend Indicators
        'ADX': lambda p: talib.ADX(high, low, close, timeperiod=p),
        'ADXR': lambda p: talib.ADXR(high, low, close, timeperiod=p),
        'AROON_UP': lambda p: talib.AROON(high, low, timeperiod=p)[0],
        'AROON_DOWN': lambda p: talib.AROON(high, low, timeperiod=p)[1],
        'AROONOSC': lambda p: talib.AROONOSC(high, low, timeperiod=p),
        'PLUS_DI': lambda p: talib.PLUS_DI(high, low, close, timeperiod=p),
        'MINUS_DI': lambda p: talib.MINUS_DI(high, low, close, timeperiod=p),
        'PLUS_DM': lambda p: talib.PLUS_DM(high, low, timeperiod=p),
        'MINUS_DM': lambda p: talib.MINUS_DM(high, low, timeperiod=p),
        'MACD': lambda p: talib.MACD(close, fastperiod=max(p//2, 2), slowperiod=p, signalperiod=9)[0],
        'MACD_SIGNAL': lambda p: talib.MACD(close, fastperiod=max(p//2, 2), slowperiod=p, signalperiod=9)[1],
        'MACD_HIST': lambda p: talib.MACD(close, fastperiod=max(p//2, 2), slowperiod=p, signalperiod=9)[2],
        'MACDEXT': lambda p: talib.MACDEXT(close, fastperiod=max(p//2, 2), slowperiod=p, signalperiod=9)[0],
        'MACDFIX': lambda p: talib.MACDFIX(close, signalperiod=p)[0],
        
        # Volatility Indicators
        'ATR': lambda p: talib.ATR(high, low, close, timeperiod=p),
        'NATR': lambda p: talib.NATR(high, low, close, timeperiod=p),
        'TRANGE': lambda p: talib.TRANGE(high, low, close),
        'BBANDS_UPPER': lambda p: talib.BBANDS(close, timeperiod=p, nbdevup=2, nbdevdn=2)[0],
        'BBANDS_MIDDLE': lambda p: talib.BBANDS(close, timeperiod=p, nbdevup=2, nbdevdn=2)[1],
        'BBANDS_LOWER': lambda p: talib.BBANDS(close, timeperiod=p, nbdevup=2, nbdevdn=2)[2],
        'STDDEV': lambda p: talib.STDDEV(close, timeperiod=p, nbdev=1),
        'VAR': lambda p: talib.VAR(close, timeperiod=p, nbdev=1),
        
        # Moving Averages
        'SMA': lambda p: talib.SMA(close, timeperiod=p),
        'EMA': lambda p: talib.EMA(close, timeperiod=p),
        'WMA': lambda p: talib.WMA(close, timeperiod=p),
        'DEMA': lambda p: talib.DEMA(close, timeperiod=p),
        'TEMA': lambda p: talib.TEMA(close, timeperiod=p),
        'TRIMA': lambda p: talib.TRIMA(close, timeperiod=p),
        'KAMA': lambda p: talib.KAMA(close, timeperiod=p),
        'T3': lambda p: talib.T3(close, timeperiod=p, vfactor=0),
        'MA': lambda p: talib.MA(close, timeperiod=p),
        'MIDPOINT': lambda p: talib.MIDPOINT(close, timeperiod=p),
        'MIDPRICE': lambda p: talib.MIDPRICE(high, low, timeperiod=p),
        
        # Volume Indicators
        'AD': lambda p: talib.AD(high, low, close, volume),
        'OBV': lambda p: talib.OBV(close, volume),
        
        # Price Transform
        'AVGPRICE': lambda p: talib.AVGPRICE(open_prices, high, low, close),
        'MEDPRICE': lambda p: talib.MEDPRICE(high, low),
        'TYPPRICE': lambda p: talib.TYPPRICE(high, low, close),
        'WCLPRICE': lambda p: talib.WCLPRICE(high, low, close),
        
        # Statistical Functions
        'LINEARREG': lambda p: talib.LINEARREG(close, timeperiod=p),
        'LINEARREG_ANGLE': lambda p: talib.LINEARREG_ANGLE(close, timeperiod=p),
        'LINEARREG_INTERCEPT': lambda p: talib.LINEARREG_INTERCEPT(close, timeperiod=p),
        'LINEARREG_SLOPE': lambda p: talib.LINEARREG_SLOPE(close, timeperiod=p),
        'TSF': lambda p: talib.TSF(close, timeperiod=p),
        'BETA': lambda p: talib.BETA(high, low, timeperiod=p),
        'CORREL': lambda p: talib.CORREL(high, low, timeperiod=p),
        
        # Other Indicators
        'ROC': lambda p: talib.ROC(close, timeperiod=p),
        'ROCP': lambda p: talib.ROCP(close, timeperiod=p),
        'ROCR': lambda p: talib.ROCR(close, timeperiod=p),
        'ROCR100': lambda p: talib.ROCR100(close, timeperiod=p),
        'MIN': lambda p: talib.MIN(close, timeperiod=p),
        'MAX': lambda p: talib.MAX(close, timeperiod=p),
        'MININDEX': lambda p: talib.MININDEX(close, timeperiod=p),
        'MAXINDEX': lambda p: talib.MAXINDEX(close, timeperiod=p),
        'SUM': lambda p: talib.SUM(close, timeperiod=p),
        'SAR': lambda p: talib.SAR(high, low, acceleration=0.02, maximum=0.2),
        'SAREXT': lambda p: talib.SAREXT(high, low),
        'HT_TRENDLINE': lambda p: talib.HT_TRENDLINE(close),
        'HT_TRENDMODE': lambda p: talib.HT_TRENDMODE(close),
        'HT_DCPERIOD': lambda p: talib.HT_DCPERIOD(close),
        'HT_DCPHASE': lambda p: talib.HT_DCPHASE(close),
        'HT_PHASOR_INPHASE': lambda p: talib.HT_PHASOR(close)[0],
        'HT_PHASOR_QUADRATURE': lambda p: talib.HT_PHASOR(close)[1],
        'HT_SINE_SINE': lambda p: talib.HT_SINE(close)[0],
        'HT_SINE_LEADSINE': lambda p: talib.HT_SINE(close)[1],
    }
    
    # Calculate indicators for specified parameters
    for indicator_name, periods in indicator_params.items():
        if indicator_name in indicator_functions:
            for period in periods:
                try:
                    result = indicator_functions[indicator_name](period)
                    if result is not None:
                        indicators[f'{indicator_name}_{period}'] = result
                except:
                    continue
    # Add Candlestick Pattern Recognition (these don't use period parameter)
    pattern_functions = {
        'CDL2CROWS': talib.CDL2CROWS,
        'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
        'CDL3INSIDE': talib.CDL3INSIDE,
        'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
        'CDL3OUTSIDE': talib.CDL3OUTSIDE,
        'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH,
        'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
        'CDLABANDONEDBABY': lambda: talib.CDLABANDONEDBABY(open_prices, high, low, close, penetration=0),
        'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK,
        'CDLBELTHOLD': talib.CDLBELTHOLD,
        'CDLBREAKAWAY': talib.CDLBREAKAWAY,
        'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU,
        'CDLCONCEALBABYSWALL': talib.CDLCONCEALBABYSWALL,
        'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK,
        'CDLDARKCLOUDCOVER': lambda: talib.CDLDARKCLOUDCOVER(open_prices, high, low, close, penetration=0),
        'CDLDOJI': talib.CDLDOJI,
        'CDLDOJISTAR': talib.CDLDOJISTAR,
        'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
        'CDLENGULFING': talib.CDLENGULFING,
        'CDLEVENINGDOJISTAR': lambda: talib.CDLEVENINGDOJISTAR(open_prices, high, low, close, penetration=0),
        'CDLEVENINGSTAR': lambda: talib.CDLEVENINGSTAR(open_prices, high, low, close, penetration=0),
        'CDLGAPSIDESIDEWHITE': talib.CDLGAPSIDESIDEWHITE,
        'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
        'CDLHAMMER': talib.CDLHAMMER,
        'CDLHANGINGMAN': talib.CDLHANGINGMAN,
        'CDLHARAMI': talib.CDLHARAMI,
        'CDLHARAMICROSS': talib.CDLHARAMICROSS,
        'CDLHIGHWAVE': talib.CDLHIGHWAVE,
        'CDLHIKKAKE': talib.CDLHIKKAKE,
        'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
        'CDLHOMINGPIGEON': talib.CDLHOMINGPIGEON,
        'CDLIDENTICAL3CROWS': talib.CDLIDENTICAL3CROWS,
        'CDLINNECK': talib.CDLINNECK,
        'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
        'CDLKICKING': talib.CDLKICKING,
        'CDLKICKINGBYLENGTH': talib.CDLKICKINGBYLENGTH,
        'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM,
        'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI,
        'CDLLONGLINE': talib.CDLLONGLINE,
        'CDLMARUBOZU': talib.CDLMARUBOZU,
        'CDLMATCHINGLOW': talib.CDLMATCHINGLOW,
        'CDLMATHOLD': lambda: talib.CDLMATHOLD(open_prices, high, low, close, penetration=0),
        'CDLMORNINGDOJISTAR': lambda: talib.CDLMORNINGDOJISTAR(open_prices, high, low, close, penetration=0),
        'CDLMORNINGSTAR': lambda: talib.CDLMORNINGSTAR(open_prices, high, low, close, penetration=0),
        'CDLONNECK': talib.CDLONNECK,
        'CDLPIERCING': talib.CDLPIERCING,
        'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN,
        'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS,
        'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES,
        'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
        'CDLSHORTLINE': talib.CDLSHORTLINE,
        'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
        'CDLSTALLEDPATTERN': talib.CDLSTALLEDPATTERN,
        'CDLSTICKSANDWICH': talib.CDLSTICKSANDWICH,
        'CDLTAKURI': talib.CDLTAKURI,
        'CDLTASUKIGAP': talib.CDLTASUKIGAP,
        'CDLTHRUSTING': talib.CDLTHRUSTING,
        'CDLTRISTAR': talib.CDLTRISTAR,
        'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER,
        'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS,
        'CDLXSIDEGAP3METHODS': talib.CDLXSIDEGAP3METHODS
    }
    
    # Calculate pattern recognition indicators (they don't use periods)
    for pattern_name in indicator_params.keys():
        if pattern_name.startswith('CDL') and pattern_name in pattern_functions:
            try:
                func = pattern_functions[pattern_name]
                if callable(func):
                    if pattern_name in ['CDLABANDONEDBABY', 'CDLDARKCLOUDCOVER', 'CDLEVENINGDOJISTAR', 
                                       'CDLEVENINGSTAR', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR']:
                        result = func()
                    else:
                        result = func(open_prices, high, low, close)
                    
                    if result is not None:
                        indicators[pattern_name] = result
            except:
                continue
    
    return indicators

@st.cache_data
def analyze_performance_by_quantiles(data, indicators, returns, quantiles, indicator_col):
    """Analyze performance by quantiles with vectorization"""
    # Drop NaN values
    valid_idx = ~(indicators[indicator_col].isna() | returns.isna().any(axis=1))
    clean_indicators = indicators.loc[valid_idx, indicator_col]
    clean_returns = returns[valid_idx]
    
    if len(clean_indicators) < quantiles:
        return None
    
    # Create quantile bins
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
        
        # 1. Signal distribution
        signal_counts = clean_indicators.value_counts().sort_index()
        signal_labels = ['Bearish' if x < 0 else 'Bullish' if x > 0 else 'Neutral' 
                        for x in signal_counts.index]
        
        fig.add_trace(
            go.Bar(x=signal_labels, y=signal_counts.values, 
                   marker_color=['red' if x < 0 else 'green' if x > 0 else 'gray' 
                                for x in signal_counts.index]),
            row=1, col=1
        )
        
        # 2. Average returns by signal
        avg_returns = []
        for signal_val in [-100, 0, 100]:
            if signal_val in signal_counts.index:
                mask = clean_indicators == signal_val
                avg_ret = clean_returns[mask].mean() if mask.sum() > 0 else 0
                avg_returns.append(avg_ret)
            else:
                avg_returns.append(0)
        
        fig.add_trace(
            go.Bar(x=['Bearish', 'Neutral', 'Bullish'], y=avg_returns,
                   marker_color=['red', 'gray', 'green']),
            row=1, col=2
        )
        
        # 3. Signal frequency over time (rolling)
        window = 252  # 1 year
        bullish_freq = (clean_indicators > 0).rolling(window).mean() * 100
        bearish_freq = (clean_indicators < 0).rolling(window).mean() * 100
        
        fig.add_trace(
            go.Scatter(x=bullish_freq.index, y=bullish_freq.values,
                      mode='lines', name='Bullish %', line=dict(color='green')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=bearish_freq.index, y=bearish_freq.values,
                      mode='lines', name='Bearish %', line=dict(color='red')),
            row=2, col=1
        )
        
        # 4. Returns distribution by signal
        returns_by_signal = []
        colors = []
        labels = []
        
        for signal_val, label, color in [(-100, 'Bearish', 'red'), 
                                         (0, 'Neutral', 'gray'), 
                                         (100, 'Bullish', 'green')]:
            mask = clean_indicators == signal_val
            if mask.sum() > 0:
                returns_by_signal.append(clean_returns[mask])
                labels.append(f"{label} (n={mask.sum()})")
                colors.append(color)
        
        for i, (ret_data, label, color) in enumerate(zip(returns_by_signal, labels, colors)):
            fig.add_trace(
                go.Box(y=ret_data, name=label, marker_color=color),
                row=2, col=2
            )
    
    else:
        # Original handling for regular indicators
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
        
        # 1. Histogram of indicator values
        fig.add_trace(
            go.Histogram(x=clean_indicators, nbinsx=50, name='Distribution',
                         marker_color='#3366CC'),
            row=1, col=1
        )
        
        # Add mean line
        fig.add_vline(x=clean_indicators.mean(), line_dash="dash", 
                      line_color="red", row=1, col=1)
        
        # 2. Average returns by quantile
        perf_analysis = analyze_performance_by_quantiles(
            data, indicators, returns, quantiles, indicator_col
        )
        
        if perf_analysis is not None:
            fig.add_trace(
                go.Bar(x=perf_analysis['Quantile'], 
                       y=perf_analysis[f'returns_{return_period}d_mean'],
                       name='Avg Returns',
                       marker_color='#DC3912'),
                row=1, col=2
            )
        
        # 3. Rolling correlation
        window = 126
        if len(clean_indicators) > window:
            rolling_corr = clean_indicators.rolling(window).corr(clean_returns)
            fig.add_trace(
                go.Scatter(x=rolling_corr.index, y=rolling_corr.values,
                          mode='lines', name='Rolling Correlation',
                          line=dict(color='#FF9900')),
                row=2, col=1
            )
            
            # Add overall correlation line
            overall_corr = clean_indicators.corr(clean_returns)
            fig.add_hline(y=overall_corr, line_dash="dash", 
                          line_color="red", row=2, col=1)
        
        # 4. Scatter plot with trend line
        fig.add_trace(
            go.Scatter(x=clean_indicators, y=clean_returns,
                      mode='markers', name='Data Points',
                      marker=dict(color='#109618', size=3, opacity=0.5)),
            row=2, col=2
        )
        
        # Add trend line
        z = np.polyfit(clean_indicators, clean_returns, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(clean_indicators.min(), clean_indicators.max(), 100)
        fig.add_trace(
            go.Scatter(x=x_trend, y=p(x_trend),
                      mode='lines', name='Trend Line',
                      line=dict(color='red', width=2)),
            row=2, col=2
        )
    
    # Update layout for dark theme
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=True,
        title_text=f"Performance Analysis: {indicator_col}",
        title_font_size=20
    )
    
    # Update axes labels based on indicator type
    if is_pattern:
        fig.update_xaxes(title_text="Signal Type", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Signal Type", row=1, col=2)
        fig.update_yaxes(title_text=f"Avg Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Frequency (%)", row=2, col=1)
        fig.update_xaxes(title_text="Signal Type", row=2, col=2)
        fig.update_yaxes(title_text=f"Returns (%)", row=2, col=2)
    else:
        fig.update_xaxes(title_text=indicator_col, row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Quantile", row=1, col=2)
        fig.update_yaxes(title_text=f"Avg Return (%)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Correlation", row=2, col=1)
        fig.update_xaxes(title_text=indicator_col, row=2, col=2)
        fig.update_yaxes(title_text=f"Returns (%)", row=2, col=2)
    
    return fig

def main():
    st.title("🎯 Technical Indicators Performance Analyzer")
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
        
        # Get indicator categories
        indicator_categories = get_indicator_categories()
        
        # Selection method
        selection_method = st.radio(
            "Selection Method",
            ["By Category", "Individual Indicators", "All Indicators"],
            horizontal=True
        )
        
        selected_indicators = []
        
        if selection_method == "By Category":
            selected_categories = st.multiselect(
                "Select Categories",
                options=list(indicator_categories.keys()),
                default=["Momentum Indicators", "Volatility Indicators"]
            )
            
            # Get all indicators from selected categories
            for category in selected_categories:
                selected_indicators.extend(indicator_categories[category])
            
            # Show selected indicators as info
            with st.expander("Selected Indicators", expanded=False):
                for category in selected_categories:
                    st.write(f"**{category}:**")
                    st.write(", ".join(indicator_categories[category]))
        
        elif selection_method == "Individual Indicators":
            # Flatten all indicators
            all_indicators = []
            for indicators_list in indicator_categories.values():
                all_indicators.extend(indicators_list)
            all_indicators = sorted(list(set(all_indicators)))
            
            selected_indicators = st.multiselect(
                "Select individual indicators",
                options=all_indicators,
                default=['RSI', 'CCI', 'MOMENTUM', 'ATR', 'MACD', 'BBANDS_UPPER']
            )
        
        else:  # All Indicators
            # Get all unique indicators
            for indicators_list in indicator_categories.values():
                selected_indicators.extend(indicators_list)
            selected_indicators = list(set(selected_indicators))
            st.info(f"Selected all {len(selected_indicators)} indicators")
        
        # Remove duplicates and filter out pattern recognition for performance
        selected_indicators = list(set(selected_indicators))
        
        # Optional: Exclude pattern recognition indicators for performance
        exclude_patterns = st.checkbox("Exclude Pattern Recognition (CDL*) indicators for performance", value=True)
        if exclude_patterns:
            selected_indicators = [ind for ind in selected_indicators if not ind.startswith('CDL')]
        
        # Period range for indicators
        st.subheader("⏱️ Indicator Periods")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_period = st.number_input("Min Period", value=5, min_value=2, max_value=200)
        with col2:
            max_period = st.number_input("Max Period", value=50, min_value=2, max_value=200)
        with col3:
            step_period = st.number_input("Step", value=5, min_value=1, max_value=50)
        
        # Analysis button
        analyze_button = st.button("🚀 Run Analysis", use_container_width=True)
    
    # Main content area
    if analyze_button:
        with st.spinner("Downloading data..."):
            data = download_data(ticker, start_date, end_date)
        
        if data is not None:
            # Calculate returns
            with st.spinner("Calculating returns..."):
                returns = calculate_returns(data, return_periods)
            
            # Prepare indicator parameters
            # Separate pattern recognition indicators from others
            pattern_indicators = [ind for ind in selected_indicators if ind.startswith('CDL')]
            regular_indicators = [ind for ind in selected_indicators if not ind.startswith('CDL')]
            
            indicator_params = {}
            
            # Regular indicators use period ranges
            for ind in regular_indicators:
                indicator_params[ind] = list(range(min_period, max_period + 1, step_period))
            
            # Pattern indicators don't use periods (we'll just pass them to be calculated once)
            for ind in pattern_indicators:
                indicator_params[ind] = [0]  # Dummy value, won't be used
            
            # Calculate indicators
            with st.spinner("Calculating indicators..."):
                indicators = calculate_all_indicators(data, indicator_params)
            
            # Display tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Indicator Analysis", 
                                                    "📈 Performance Metrics", 
                                                    "🎯 Best Performers",
                                                    "📉 Correlation Matrix",
                                                    "📋 Summary Statistics"])
            
            with tab1:
                st.header("Indicator Performance Analysis")
                
                # Select specific indicator and period for detailed analysis
                available_indicators = [col for col in indicators.columns if not indicators[col].isna().all()]
                
                if available_indicators:
                    # Separate pattern and regular indicators for better UI
                    pattern_indicators_list = [col for col in available_indicators if 'CDL' in col]
                    regular_indicators_list = [col for col in available_indicators if 'CDL' not in col]
                    
                    indicator_type = st.radio("Indicator Type", ["Regular", "Pattern Recognition"], horizontal=True)
                    
                    if indicator_type == "Regular" and regular_indicators_list:
                        selected_indicator = st.selectbox(
                            "Select indicator for detailed analysis",
                            options=regular_indicators_list
                        )
                    elif indicator_type == "Pattern Recognition" and pattern_indicators_list:
                        selected_indicator = st.selectbox(
                            "Select pattern for detailed analysis",
                            options=pattern_indicators_list,
                            format_func=lambda x: x.replace('CDL', 'Pattern: ')
                        )
                    else:
                        st.warning(f"No {indicator_type} indicators available.")
                        selected_indicator = None
                    
                    if selected_indicator:
                        selected_return = st.selectbox(
                            "Select return period for analysis",
                            options=return_periods,
                            format_func=lambda x: f"{x} days"
                        )
                        
                        # Create performance plots
                        fig = create_performance_plots(
                            data, indicators, returns, 
                            selected_indicator, selected_return, quantiles
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No indicators calculated. Please check your parameters.")
            
            with tab2:
                st.header("Performance Metrics Summary")
                
                # Calculate performance for all indicators
                performance_summary = []
                
                for ind_col in available_indicators[:20]:  # Limit to first 20 for display
                    for ret_period in return_periods[:3]:  # Top 3 return periods
                        perf = analyze_performance_by_quantiles(
                            data, indicators, returns, quantiles, ind_col
                        )
                        
                        if perf is not None:
                            # Get top and bottom quantile performance
                            top_quantile = perf.iloc[-1]
                            bottom_quantile = perf.iloc[0]
                            
                            performance_summary.append({
                                'Indicator': ind_col,
                                'Return_Period': f'{ret_period}d',
                                'Top_Quantile_Return': top_quantile[f'returns_{ret_period}d_mean'],
                                'Bottom_Quantile_Return': bottom_quantile[f'returns_{ret_period}d_mean'],
                                'Spread': top_quantile[f'returns_{ret_period}d_mean'] - 
                                         bottom_quantile[f'returns_{ret_period}d_mean'],
                                'Top_Sharpe': top_quantile[f'returns_{ret_period}d_sharpe'],
                                'Top_Win_Rate': top_quantile[f'returns_{ret_period}d_win_rate']
                            })
                
                if performance_summary:
                    perf_df = pd.DataFrame(performance_summary)
                    perf_df = perf_df.sort_values('Spread', ascending=False)
                    
                    # Display metrics
                    st.dataframe(
                        perf_df.style.format({
                            'Top_Quantile_Return': '{:.2f}%',
                            'Bottom_Quantile_Return': '{:.2f}%',
                            'Spread': '{:.2f}%',
                            'Top_Sharpe': '{:.3f}',
                            'Top_Win_Rate': '{:.1f}%'
                        }).background_gradient(cmap='RdYlGn', subset=['Spread', 'Top_Sharpe']),
                        use_container_width=True
                    )
                    
                    # Plot spread comparison
                    fig_spread = px.bar(
                        perf_df.head(15), 
                        x='Indicator', 
                        y='Spread',
                        color='Return_Period',
                        title="Top 15 Indicators by Return Spread",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_spread, use_container_width=True)
            
            with tab3:
                st.header("Best Performing Indicator Configurations")
                
                # Separate pattern and regular indicators
                pattern_cols = [col for col in available_indicators if col.startswith('CDL')]
                regular_cols = [col for col in available_indicators if not col.startswith('CDL')]
                
                # Find best indicator/period combinations for regular indicators
                best_configs = []
                
                for ind_col in regular_cols:
                    for ret_period in return_periods:
                        # Calculate correlation
                        valid_idx = ~(indicators[ind_col].isna() | 
                                    returns[f'returns_{ret_period}d'].isna())
                        
                        if valid_idx.sum() > 100:  # Minimum sample size
                            corr = indicators.loc[valid_idx, ind_col].corr(
                                returns.loc[valid_idx, f'returns_{ret_period}d']
                            )
                            
                            best_configs.append({
                                'Indicator': ind_col,
                                'Return_Period': f'{ret_period}d',
                                'Correlation': corr,
                                'Abs_Correlation': abs(corr)
                            })
                
                # Pattern recognition performance
                pattern_performance = []
                if pattern_cols:
                    for pattern in pattern_cols:
                        for ret_period in return_periods:
                            # Get bullish and bearish signals
                            bullish = indicators[pattern] > 0
                            bearish = indicators[pattern] < 0
                            
                            if bullish.sum() > 5:  # Minimum signals
                                bullish_returns = returns.loc[bullish, f'returns_{ret_period}d'].mean()
                                pattern_performance.append({
                                    'Pattern': pattern.replace('CDL', ''),
                                    'Signal': 'Bullish',
                                    'Return_Period': f'{ret_period}d',
                                    'Avg_Return': bullish_returns,
                                    'Count': bullish.sum()
                                })
                            
                            if bearish.sum() > 5:
                                bearish_returns = returns.loc[bearish, f'returns_{ret_period}d'].mean()
                                pattern_performance.append({
                                    'Pattern': pattern.replace('CDL', ''),
                                    'Signal': 'Bearish',
                                    'Return_Period': f'{ret_period}d',
                                    'Avg_Return': bearish_returns,
                                    'Count': bearish.sum()
                                })
                
                # Display results
                if best_configs:
                    best_df = pd.DataFrame(best_configs)
                    best_df = best_df.sort_values('Abs_Correlation', ascending=False).head(20)
                    
                    st.subheader("🎯 Technical Indicators Performance")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Strongest Positive Correlations**")
                        positive = best_df[best_df['Correlation'] > 0].head(10)
                        st.dataframe(
                            positive.style.format({
                                'Correlation': '{:.4f}',
                                'Abs_Correlation': '{:.4f}'
                            }).background_gradient(cmap='Greens', subset=['Correlation']),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.write("**Strongest Negative Correlations**")
                        negative = best_df[best_df['Correlation'] < 0].head(10)
                        st.dataframe(
                            negative.style.format({
                                'Correlation': '{:.4f}',
                                'Abs_Correlation': '{:.4f}'
                            }).background_gradient(cmap='Reds_r', subset=['Correlation']),
                            use_container_width=True
                        )
                
                if pattern_performance:
                    st.subheader("🕯️ Candlestick Pattern Performance")
                    
                    pattern_df = pd.DataFrame(pattern_performance)
                    pattern_df = pattern_df.sort_values('Avg_Return', ascending=False)
                    
                    # Best bullish patterns
                    bullish_patterns = pattern_df[pattern_df['Signal'] == 'Bullish'].head(10)
                    if not bullish_patterns.empty:
                        st.write("**Best Bullish Patterns**")
                        st.dataframe(
                            bullish_patterns.style.format({
                                'Avg_Return': '{:.3f}%',
                                'Count': '{:,.0f}'
                            }).background_gradient(cmap='Greens', subset=['Avg_Return']),
                            use_container_width=True
                        )
                    
                    # Best bearish patterns
                    bearish_patterns = pattern_df[pattern_df['Signal'] == 'Bearish'].head(10)
                    if not bearish_patterns.empty:
                        st.write("**Best Bearish Patterns**")
                        st.dataframe(
                            bearish_patterns.style.format({
                                'Avg_Return': '{:.3f}%',
                                'Count': '{:,.0f}'
                            }).background_gradient(cmap='Reds_r', subset=['Avg_Return']),
                            use_container_width=True
                        )
            
            with tab4:
                st.header("Correlation Matrix")
                
                # Select subset of indicators for correlation matrix
                matrix_indicators = st.multiselect(
                    "Select indicators for correlation matrix",
                    options=available_indicators[:30],
                    default=available_indicators[:10]
                )
                
                if matrix_indicators:
                    # Calculate correlation matrix
                    corr_matrix = indicators[matrix_indicators].corr()
                    
                    # Create heatmap
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 8}
                    ))
                    
                    fig_corr.update_layout(
                        template="plotly_dark",
                        title="Indicator Correlation Matrix",
                        height=600
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            with tab5:
                st.header("Summary Statistics")
                
                # Categorize indicators
                pattern_cols = [col for col in available_indicators if col.startswith('CDL')]
                regular_cols = [col for col in available_indicators if not col.startswith('CDL')]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Indicators Calculated")
                    st.metric("Total Indicators", len(available_indicators))
                    st.metric("Regular Indicators", len(regular_cols))
                    st.metric("Pattern Recognition", len(pattern_cols))
                    
                    # Show indicators by category
                    indicator_categories = get_indicator_categories()
                    category_counts = {}
                    for category, ind_list in indicator_categories.items():
                        count = sum(1 for col in available_indicators 
                                  for ind in ind_list if col.startswith(ind))
                        if count > 0:
                            category_counts[category] = count
                    
                    if category_counts:
                        st.subheader("📈 By Category")
                        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"• {cat}: **{count}**")
                
                with col2:
                    st.subheader("🎯 Performance Overview")
                    
                    # Calculate overall statistics
                    correlations = []
                    for ind_col in regular_cols[:100]:  # Limit for performance
                        for ret_period in return_periods:
                            valid_idx = ~(indicators[ind_col].isna() | 
                                        returns[f'returns_{ret_period}d'].isna())
                            if valid_idx.sum() > 100:
                                corr = indicators.loc[valid_idx, ind_col].corr(
                                    returns.loc[valid_idx, f'returns_{ret_period}d']
                                )
                                correlations.append(abs(corr))
                    
                    if correlations:
                        st.metric("Avg Absolute Correlation", f"{np.mean(correlations):.4f}")
                        st.metric("Max Absolute Correlation", f"{np.max(correlations):.4f}")
                        st.metric("Indicators > 0.1 Correlation", 
                                f"{sum(c > 0.1 for c in correlations)}/{len(correlations)}")
                    
                    # Pattern recognition statistics
                    if pattern_cols:
                        st.subheader("🕯️ Pattern Recognition")
                        pattern_signals = []
                        for pattern in pattern_cols:
                            signals = (indicators[pattern] != 0).sum()
                            if signals > 0:
                                pattern_signals.append({
                                    'Pattern': pattern.replace('CDL', ''),
                                    'Signals': signals,
                                    'Frequency': f"{signals/len(indicators)*100:.2f}%"
                                })
                        
                        if pattern_signals:
                            pattern_df = pd.DataFrame(pattern_signals).sort_values('Signals', ascending=False).head(10)
                            st.dataframe(pattern_df, use_container_width=True, hide_index=True)
            
            # Display data info
            with st.expander("📊 Data Information"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Days", len(data))
                with col2:
                    st.metric("Start Price", f"${data['Close'].iloc[0]:.2f}")
                with col3:
                    st.metric("End Price", f"${data['Close'].iloc[-1]:.2f}")
                with col4:
                    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    st.metric("Total Return", f"{total_return:.2f}%")
                
                st.dataframe(data.tail(10), use_container_width=True)

if __name__ == "__main__":
    main()
