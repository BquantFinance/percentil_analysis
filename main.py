import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ===================== PAGE CONFIGURATION =====================
st.set_page_config(
    page_title="Complete Quantitative Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CSS STYLING =====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #151932 100%);
        font-family: 'Inter', sans-serif;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 1rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .trading-rule {
        background: rgba(30, 34, 56, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
    }
    
    .momentum-badge {
        background: rgba(255, 152, 0, 0.2);
        color: #FF9800;
        border: 1px solid #FF9800;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
    }
    
    .mean-reversion-badge {
        background: rgba(33, 150, 243, 0.2);
        color: #2196F3;
        border: 1px solid #2196F3;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== TECHNICAL INDICATORS CLASS =====================
class TechnicalIndicators:
    """Complete TALib indicators manager (200+)"""
    
    INDICATOR_CONFIG = {
        # ============ OVERLAP STUDIES ============
        'BBANDS': ('BBANDS', {'timeperiod': 'p', 'nbdevup': 2, 'nbdevdn': 2, 'matype': 0}),
        'DEMA': ('DEMA', {'timeperiod': 'p'}),
        'EMA': ('EMA', {'timeperiod': 'p'}),
        'HT_TRENDLINE': ('HT_TRENDLINE', {}),
        'KAMA': ('KAMA', {'timeperiod': 'p'}),
        'MA': ('MA', {'timeperiod': 'p', 'matype': 0}),
        'MAMA': ('MAMA', {'fastlimit': 0.5, 'slowlimit': 0.05}),
        'MAVP': ('MAVP', {'minperiod': 2, 'maxperiod': 30, 'matype': 0}),
        'MIDPOINT': ('MIDPOINT', {'timeperiod': 'p'}),
        'MIDPRICE': ('MIDPRICE', {'timeperiod': 'p'}),
        'SAR': ('SAR', {'acceleration': 0.02, 'maximum': 0.2}),
        'SAREXT': ('SAREXT', {'startvalue': 0, 'offsetonreverse': 0, 'accelerationinitlong': 0.02,
                              'accelerationlong': 0.02, 'accelerationmaxlong': 0.20,
                              'accelerationinitshort': 0.02, 'accelerationshort': 0.02, 
                              'accelerationmaxshort': 0.20}),
        'SMA': ('SMA', {'timeperiod': 'p'}),
        'T3': ('T3', {'timeperiod': 'p', 'vfactor': 0}),
        'TEMA': ('TEMA', {'timeperiod': 'p'}),
        'TRIMA': ('TRIMA', {'timeperiod': 'p'}),
        'WMA': ('WMA', {'timeperiod': 'p'}),
        
        # ============ MOMENTUM INDICATORS ============
        'ADX': ('ADX', {'timeperiod': 'p'}),
        'ADXR': ('ADXR', {'timeperiod': 'p'}),
        'APO': ('APO', {'fastperiod': 'max(p//2, 2)', 'slowperiod': 'p', 'matype': 0}),
        'AROON': ('AROON', {'timeperiod': 'p'}),
        'AROONOSC': ('AROONOSC', {'timeperiod': 'p'}),
        'BOP': ('BOP', {}),
        'CCI': ('CCI', {'timeperiod': 'p'}),
        'CMO': ('CMO', {'timeperiod': 'p'}),
        'DX': ('DX', {'timeperiod': 'p'}),
        'MACD': ('MACD', {'fastperiod': 'max(p//2, 2)', 'slowperiod': 'p', 'signalperiod': 9}),
        'MACDEXT': ('MACDEXT', {'fastperiod': 'max(p//2, 2)', 'fastmatype': 0, 
                                'slowperiod': 'p', 'slowmatype': 0, 
                                'signalperiod': 9, 'signalmatype': 0}),
        'MACDFIX': ('MACDFIX', {'signalperiod': 9}),
        'MFI': ('MFI', {'timeperiod': 'p'}),
        'MINUS_DI': ('MINUS_DI', {'timeperiod': 'p'}),
        'MINUS_DM': ('MINUS_DM', {'timeperiod': 'p'}),
        'MOM': ('MOM', {'timeperiod': 'p'}),
        'PLUS_DI': ('PLUS_DI', {'timeperiod': 'p'}),
        'PLUS_DM': ('PLUS_DM', {'timeperiod': 'p'}),
        'PPO': ('PPO', {'fastperiod': 'max(p//2, 2)', 'slowperiod': 'p', 'matype': 0}),
        'ROC': ('ROC', {'timeperiod': 'p'}),
        'ROCP': ('ROCP', {'timeperiod': 'p'}),
        'ROCR': ('ROCR', {'timeperiod': 'p'}),
        'ROCR100': ('ROCR100', {'timeperiod': 'p'}),
        'RSI': ('RSI', {'timeperiod': 'p'}),
        'STOCH': ('STOCH', {'fastk_period': 'p', 'slowk_period': 3, 
                           'slowk_matype': 0, 'slowd_period': 3, 'slowd_matype': 0}),
        'STOCHF': ('STOCHF', {'fastk_period': 'p', 'fastd_period': 3, 'fastd_matype': 0}),
        'STOCHRSI': ('STOCHRSI', {'timeperiod': 'p', 'fastk_period': 5, 
                                  'fastd_period': 3, 'fastd_matype': 0}),
        'TRIX': ('TRIX', {'timeperiod': 'p'}),
        'ULTOSC': ('ULTOSC', {'timeperiod1': 'max(p//3, 2)', 
                              'timeperiod2': 'max(p//2, 3)', 'timeperiod3': 'p'}),
        'WILLR': ('WILLR', {'timeperiod': 'p'}),
        
        # ============ VOLUME INDICATORS ============
        'AD': ('AD', {}),
        'ADOSC': ('ADOSC', {'fastperiod': 'max(p//3, 2)', 'slowperiod': 'p'}),
        'OBV': ('OBV', {}),
        
        # ============ VOLATILITY INDICATORS ============
        'ATR': ('ATR', {'timeperiod': 'p'}),
        'NATR': ('NATR', {'timeperiod': 'p'}),
        'TRANGE': ('TRANGE', {}),
        
        # ============ CYCLE INDICATORS ============
        'HT_DCPERIOD': ('HT_DCPERIOD', {}),
        'HT_DCPHASE': ('HT_DCPHASE', {}),
        'HT_PHASOR': ('HT_PHASOR', {}),
        'HT_SINE': ('HT_SINE', {}),
        'HT_TRENDMODE': ('HT_TRENDMODE', {}),
        
        # ============ STATISTIC FUNCTIONS ============
        'BETA': ('BETA', {'timeperiod': 'p'}),
        'CORREL': ('CORREL', {'timeperiod': 'p'}),
        'LINEARREG': ('LINEARREG', {'timeperiod': 'p'}),
        'LINEARREG_ANGLE': ('LINEARREG_ANGLE', {'timeperiod': 'p'}),
        'LINEARREG_INTERCEPT': ('LINEARREG_INTERCEPT', {'timeperiod': 'p'}),
        'LINEARREG_SLOPE': ('LINEARREG_SLOPE', {'timeperiod': 'p'}),
        'STDDEV': ('STDDEV', {'timeperiod': 'p', 'nbdev': 1}),
        'TSF': ('TSF', {'timeperiod': 'p'}),
        'VAR': ('VAR', {'timeperiod': 'p', 'nbdev': 1}),
        
        # ============ MATH TRANSFORM ============
        'ACOS': ('ACOS', {}),
        'ASIN': ('ASIN', {}),
        'ATAN': ('ATAN', {}),
        'CEIL': ('CEIL', {}),
        'COS': ('COS', {}),
        'COSH': ('COSH', {}),
        'EXP': ('EXP', {}),
        'FLOOR': ('FLOOR', {}),
        'LN': ('LN', {}),
        'LOG10': ('LOG10', {}),
        'SIN': ('SIN', {}),
        'SINH': ('SINH', {}),
        'SQRT': ('SQRT', {}),
        'TAN': ('TAN', {}),
        'TANH': ('TANH', {}),
        
        # ============ MATH OPERATORS ============
        'ADD': ('ADD', {}),
        'DIV': ('DIV', {}),
        'MAX': ('MAX', {'timeperiod': 'p'}),
        'MAXINDEX': ('MAXINDEX', {'timeperiod': 'p'}),
        'MIN': ('MIN', {'timeperiod': 'p'}),
        'MININDEX': ('MININDEX', {'timeperiod': 'p'}),
        'MINMAX': ('MINMAX', {'timeperiod': 'p'}),
        'MINMAXINDEX': ('MINMAXINDEX', {'timeperiod': 'p'}),
        'MULT': ('MULT', {}),
        'SUB': ('SUB', {}),
        'SUM': ('SUM', {'timeperiod': 'p'}),
        
        # ============ PRICE TRANSFORM ============
        'AVGPRICE': ('AVGPRICE', {}),
        'MEDPRICE': ('MEDPRICE', {}),
        'TYPPRICE': ('TYPPRICE', {}),
        'WCLPRICE': ('WCLPRICE', {}),
    }
    
    CANDLE_PATTERNS = [
        'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 'CDL3OUTSIDE',
        'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
        'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
        'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR',
        'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
        'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
        'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
        'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER',
        'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI',
        'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 
        'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING',
        'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES',
        'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLSTALLEDPATTERN',
        'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP', 'CDLTHRUSTING',
        'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
    ]
    
    CATEGORIES = {
        "📈 Overlaps (17)": ['BBANDS', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 
                            'MAMA', 'MAVP', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT',
                            'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'],
        "💫 Momentum (30)": ['ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP', 
                            'CCI', 'CMO', 'DX', 'MACD', 'MACDEXT', 'MACDFIX', 
                            'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 
                            'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 
                            'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX', 
                            'ULTOSC', 'WILLR'],
        "📊 Volume (3)": ['AD', 'ADOSC', 'OBV'],
        "📉 Volatility (3)": ['ATR', 'NATR', 'TRANGE'],
        "🎯 Cycles (5)": ['HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE'],
        "📐 Statistics (9)": ['BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 
                             'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR'],
        "🔢 Math Transform (15)": ['ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH', 
                                   'EXP', 'FLOOR', 'LN', 'LOG10', 'SIN', 'SINH', 
                                   'SQRT', 'TAN', 'TANH'],
        "➕ Math Operators (11)": ['ADD', 'DIV', 'MAX', 'MAXINDEX', 'MIN', 'MININDEX',
                                  'MINMAX', 'MINMAXINDEX', 'MULT', 'SUB', 'SUM'],
        "💹 Price Transform (4)": ['AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'],
        "🕯️ Pattern Recognition (61)": CANDLE_PATTERNS
    }
    
    @classmethod
    def _get_indicator_inputs(cls, func_name):
        """Detect input requirements for each indicator"""
        if func_name.startswith('CDL'):
            return 'ohlc'
        elif func_name in ['AD', 'ADOSC']:
            return 'hlcv'
        elif func_name in ['OBV']:
            return 'cv'
        elif func_name in ['AVGPRICE']:
            return 'ohlc'
        elif func_name in ['MEDPRICE', 'MIDPRICE']:
            return 'hl'
        elif func_name in ['TYPPRICE', 'WCLPRICE']:
            return 'hlc'
        elif func_name in ['ATR', 'NATR', 'ADX', 'ADXR', 'CCI', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM']:
            return 'hlc'
        elif func_name == 'MFI':
            return 'hlcv'
        elif func_name in ['BOP']:
            return 'ohlc'
        elif func_name in ['SAR', 'SAREXT']:
            return 'hl'
        elif func_name in ['TRANGE']:
            return 'hlc'
        elif func_name in ['STOCH', 'STOCHF', 'WILLR']:
            return 'hlc'
        elif func_name in ['BETA', 'CORREL']:
            return 'hl'
        elif func_name in ['AROON']:
            return 'hl'
        elif func_name in ['ADD', 'DIV', 'MULT', 'SUB']:
            return 'cc'
        elif func_name in ['MAX', 'MAXINDEX', 'MIN', 'MININDEX', 'MINMAX', 'MINMAXINDEX', 'SUM']:
            return 'c'
        else:
            return 'c'
    
    @classmethod
    def calculate_indicator(cls, indicator_name, high, low, close, volume, open_prices, period):
        """Calculate any TALib indicator with error handling"""
        try:
            # Check data sufficiency
            if period > 0 and len(close) < period * 2:
                return None
            
            # Candle patterns
            if indicator_name.startswith('CDL'):
                func = getattr(talib, indicator_name)
                result = func(open_prices, high, low, close)
                if result is not None and np.any(result != 0):
                    return result
                return None
            
            # Regular indicators
            if indicator_name not in cls.INDICATOR_CONFIG:
                if hasattr(talib, indicator_name):
                    func = getattr(talib, indicator_name)
                    return func(close)
                return None
            
            func_name, params = cls.INDICATOR_CONFIG[indicator_name]
            func = getattr(talib, func_name)
            
            data_type = cls._get_indicator_inputs(func_name)
            
            # Prepare arguments
            if data_type == 'ohlc':
                args = [open_prices, high, low, close]
            elif data_type == 'hlcv':
                args = [high, low, close, volume]
            elif data_type == 'hlc':
                args = [high, low, close]
            elif data_type == 'hl':
                args = [high, low]
            elif data_type == 'cv':
                args = [close, volume]
            elif data_type == 'cc':
                args = [close, close]
            else:
                args = [close]
            
            # Process parameters
            kwargs = {}
            for key, value in params.items():
                if isinstance(value, str):
                    if value == 'p':
                        kwargs[key] = period
                    elif 'p' in value:
                        kwargs[key] = eval(value, {'p': period, 'max': max})
                else:
                    kwargs[key] = value
            
            result = func(*args, **kwargs)
            
            if isinstance(result, tuple):
                result = result[0]
            
            # Validate result
            if result is not None and not np.all(np.isnan(result)):
                if np.std(result[~np.isnan(result)]) > 1e-10:
                    return result
            
            return None
            
        except Exception:
            return None
    
    @classmethod
    def needs_period(cls, indicator_name):
        """Check if indicator needs period parameter"""
        no_period = [
            'HT_TRENDLINE', 'BOP', 'MACDFIX', 'AD', 'OBV', 'TRANGE', 
            'SAR', 'SAREXT', 'MAMA', 'MAVP',
            'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE',
            'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
            'ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH', 
            'EXP', 'FLOOR', 'LN', 'LOG10', 'SIN', 'SINH', 
            'SQRT', 'TAN', 'TANH',
            'ADD', 'DIV', 'MULT', 'SUB'
        ] + cls.CANDLE_PATTERNS
        
        return indicator_name not in no_period
    
    @classmethod
    def get_all_indicators(cls):
        """Get list of all available indicators"""
        return list(cls.INDICATOR_CONFIG.keys()) + cls.CANDLE_PATTERNS
    
    @classmethod
    def get_total_count(cls):
        """Get total count of indicators"""
        return len(cls.INDICATOR_CONFIG) + len(cls.CANDLE_PATTERNS)

# ===================== CALCULATION FUNCTIONS =====================
@st.cache_data
def download_data(ticker: str, period: str, start_date: str = None, end_date: str = None, multi_level_index=False) -> Optional[pd.DataFrame]:
    """Download historical data using yfinance period or date range"""
    try:
        if period != "custom":
            # Use yfinance period
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        else:
            # Use custom date range
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty:
            st.error(f"❌ No data found for {ticker}")
            return None
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error downloading data: {str(e)}")
        return None

@st.cache_data
def calculate_all_indicators(ticker: str, period: str, start_date: str, end_date: str,
                             quantiles: int, return_days: int, 
                             periods_to_test: List[int],
                             selected_categories: List[str]) -> Tuple:
    """Calculate all selected indicators"""
    
    data = download_data(ticker, period, start_date, end_date)
    if data is None:
        return None, None, None, None
    
    # Calculate returns for multiple periods
    for i in range(1, return_days + 1):
        data[f'returns_{i}_days'] = data['Close'].pct_change(i) * 100
    
    # Prepare data arrays
    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    close = data['Close'].values.astype(np.float64)
    volume = data['Volume'].values.astype(np.float64) if 'Volume' in data.columns else np.zeros_like(close)
    open_prices = data['Open'].values.astype(np.float64)
    
    indicators = pd.DataFrame(index=data.index)
    
    # Get indicators to calculate based on selected categories
    indicators_to_calc = []
    for category, indicator_list in TechnicalIndicators.CATEGORIES.items():
        if category in selected_categories or "ALL" in selected_categories:
            indicators_to_calc.extend(indicator_list)
    
    # Remove duplicates
    indicators_to_calc = list(set(indicators_to_calc))
    
    # Count total calculations
    total_calculations = sum(
        len(periods_to_test) if TechnicalIndicators.needs_period(ind) else 1 
        for ind in indicators_to_calc
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    calculation_counter = 0
    successful = 0
    
    # Calculate indicators
    for indicator_name in indicators_to_calc:
        if TechnicalIndicators.needs_period(indicator_name):
            for period in periods_to_test:
                calculation_counter += 1
                status_text.text(f"📊 Calculating {indicator_name}_{period}... ({calculation_counter}/{total_calculations})")
                
                result = TechnicalIndicators.calculate_indicator(
                    indicator_name, high, low, close, volume, open_prices, period
                )
                
                if result is not None and not np.all(np.isnan(result)):
                    indicators[f'{indicator_name}_{period}'] = result
                    successful += 1
                
                progress_bar.progress(calculation_counter / total_calculations)
        else:
            calculation_counter += 1
            status_text.text(f"📊 Calculating {indicator_name}... ({calculation_counter}/{total_calculations})")
            
            result = TechnicalIndicators.calculate_indicator(
                indicator_name, high, low, close, volume, open_prices, 0
            )
            
            if result is not None and not np.all(np.isnan(result)):
                indicators[indicator_name] = result
                successful += 1
            
            progress_bar.progress(calculation_counter / total_calculations)
    
    progress_bar.empty()
    status_text.empty()
    
    # Drop empty columns
    indicators = indicators.dropna(axis=1, how='all')
    
    # Calculate percentile analysis
    returns_data = {}
    
    for indicator_col in indicators.columns:
        try:
            temp_df = pd.DataFrame({'indicator': indicators[indicator_col]})
            
            for i in range(1, return_days + 1):
                ret_col = f'returns_{i}_days'
                if ret_col in data.columns:
                    temp_df[ret_col] = data[ret_col]
            
            temp_df = temp_df.dropna()
            
            if len(temp_df) >= quantiles:
                temp_df['quantile'] = pd.qcut(temp_df['indicator'], q=quantiles, duplicates='drop')
                
                returns_data[indicator_col] = pd.DataFrame()
                for i in range(1, return_days + 1):
                    ret_col = f'returns_{i}_days'
                    if ret_col in temp_df.columns:
                        grouped = temp_df.groupby('quantile')[ret_col].agg(['mean', 'std', 'count'])
                        returns_data[indicator_col][f'returns_{i}_days_mean'] = grouped['mean']
                        returns_data[indicator_col][f'returns_{i}_days_std'] = grouped['std']
                        returns_data[indicator_col][f'returns_{i}_days_count'] = grouped['count']
        except:
            continue
    
    st.success(f"✅ Calculated {successful} configurations out of {total_calculations} attempted")
    
    # Summary statistics
    summary = {
        'total_attempted': total_calculations,
        'successful': successful,
        'indicators_count': len(indicators.columns),
        'data_points': len(data),
        'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    }
    
    return returns_data, indicators, data, summary

def analyze_for_rules(indicator_values, returns, quantiles=10):
    """Analyze indicator for trading rules"""
    try:
        temp_df = pd.DataFrame({
            'indicator': indicator_values,
            'returns': returns
        }).dropna()
        
        if len(temp_df) < max(20, quantiles):
            return None
        
        # Try creating quantiles
        try:
            temp_df['percentile'] = pd.qcut(temp_df['indicator'], q=quantiles, labels=False, duplicates='drop')
        except:
            try:
                temp_df['percentile'] = pd.qcut(temp_df['indicator'], q=5, labels=False, duplicates='drop')
            except:
                return None
        
        percentile_returns = temp_df.groupby('percentile')['returns'].agg(['mean', 'std', 'count'])
        
        if len(percentile_returns) < 3:
            return None
        
        metrics = {}
        
        top_return = percentile_returns['mean'].iloc[-1]
        bottom_return = percentile_returns['mean'].iloc[0]
        metrics['spread'] = top_return - bottom_return
        metrics['top_return'] = top_return
        metrics['bottom_return'] = bottom_return
        
        if len(percentile_returns) >= 3:
            correlation, p_value = spearmanr(range(len(percentile_returns)), percentile_returns['mean'].values)
            metrics['direction'] = correlation
            metrics['p_value'] = p_value
        else:
            metrics['direction'] = 1.0 if top_return > bottom_return else -1.0
            metrics['p_value'] = 0.5
        
        metrics['sharpe'] = abs(metrics['spread']) / (percentile_returns['std'].mean() + 1e-8)
        metrics['best_long_percentile'] = percentile_returns['mean'].idxmax() + 1
        metrics['best_short_percentile'] = percentile_returns['mean'].idxmin() + 1
        metrics['min_samples'] = percentile_returns['count'].min()
        metrics['total_samples'] = percentile_returns['count'].sum()
        
        return metrics
        
    except Exception:
        return None

def generate_trading_rules(indicators, data, return_days=5, min_spread=1.0, max_p_value=0.15, top_n=30):
    """Generate trading rules from all indicators"""
    
    returns = data['Close'].pct_change(return_days).shift(-return_days) * 100
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(indicators.columns)
    
    for idx, indicator_col in enumerate(indicators.columns):
        status_text.text(f"🎯 Analyzing {indicator_col}... ({idx+1}/{total})")
        
        indicator_values = indicators[indicator_col].values
        metrics = analyze_for_rules(indicator_values, returns.values, quantiles=10)
        
        if metrics and metrics['min_samples'] >= 5:
            metrics['indicator_name'] = indicator_col
            all_results.append(metrics)
        
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_results:
        return []
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df['score'] = (
        abs(results_df['spread']) * 0.5 +
        results_df['sharpe'] * 10 +
        (1 / (results_df['p_value'] + 0.001)) * 0.1
    )
    
    # Filter quality signals
    quality = results_df[
        (abs(results_df['spread']) >= min_spread) & 
        (results_df['p_value'] <= max_p_value)
    ]
    
    # If not enough, relax criteria
    if len(quality) < 10:
        quality = results_df[abs(results_df['spread']) >= 0.5]
    
    quality = quality.nlargest(min(top_n, len(quality)), 'score')
    
    # Generate rules
    rules = []
    for _, row in quality.iterrows():
        if row['direction'] > 0.3:
            strategy = "MOMENTUM"
            entry = f"When {row['indicator_name']} is HIGH (top 20%)"
            action = "STRONG BUY"
            exit_signal = f"When {row['indicator_name']} is LOW (bottom 20%)"
            exit_action = "STRONG SELL"
        elif row['direction'] < -0.3:
            strategy = "MEAN REVERSION"
            entry = f"When {row['indicator_name']} is LOW (bottom 20%)"
            action = "STRONG BUY"
            exit_signal = f"When {row['indicator_name']} is HIGH (top 20%)"
            exit_action = "STRONG SELL"
        else:
            strategy = "SELECTIVE"
            entry = f"When {row['indicator_name']} is at Percentile {int(row['best_long_percentile'])}"
            action = "BUY"
            exit_signal = f"When {row['indicator_name']} is at Percentile {int(row['best_short_percentile'])}"
            exit_action = "AVOID"
        
        rules.append({
            'rank': len(rules) + 1,
            'indicator': row['indicator_name'],
            'strategy': strategy,
            'entry': entry,
            'action': action,
            'exit': exit_signal,
            'exit_action': exit_action,
            'spread': row['spread'],
            'top_return': row['top_return'],
            'bottom_return': row['bottom_return'],
            'confidence': (1 - row['p_value']) * 100,
            'sharpe': row['sharpe']
        })
    
    return rules

def create_percentile_plot(indicators, returns_data, data, indicator_name, return_days):
    """Create percentile analysis plot with KDE"""
    
    if indicator_name not in indicators.columns or indicator_name not in returns_data:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'<b>Distribution</b>',
            f'<b>Returns by Percentile</b>',
            f'<b>Rolling Correlation</b>',
            f'<b>Scatter Plot</b>'
        ),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 1. Histogram with KDE
    hist_data = indicators[indicator_name].dropna()
    
    fig.add_trace(
        go.Histogram(
            x=hist_data,
            nbinsx=50,
            marker=dict(color='rgba(102, 126, 234, 0.6)'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add KDE
    if len(hist_data) > 1:
        try:
            from scipy import stats as scipy_stats
            kde = scipy_stats.gaussian_kde(hist_data)
            x_range = np.linspace(hist_data.min(), hist_data.max(), 200)
            kde_values = kde(x_range)
            
            hist_counts, _ = np.histogram(hist_data, bins=50)
            kde_scale = max(hist_counts) / max(kde_values) * 0.8
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_values * kde_scale,
                    mode='lines',
                    line=dict(color='#FFD93D', width=3),
                    showlegend=False
                ),
                row=1, col=1
            )
        except:
            pass
    
    mean_val = hist_data.mean()
    fig.add_vline(x=mean_val, line=dict(color='red', width=2),
                  row=1, col=1, annotation_text=f'μ={mean_val:.2f}')
    
    # 2. Returns by percentile
    returns_col = f'returns_{return_days}_days_mean'
    if returns_col in returns_data[indicator_name].columns:
        returns_values = returns_data[indicator_name][returns_col]
        x_labels = [f'P{i+1}' for i in range(len(returns_values))]
        
        colors = ['red' if val < 0 else 'green' for val in returns_values]
        
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=returns_values,
                marker=dict(color=colors),
                text=[f'{val:.2f}%' for val in returns_values],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Rolling correlation
    if f'returns_{return_days}_days' in data.columns:
        common_idx = data.index.intersection(indicators[indicator_name].index)
        if len(common_idx) > 126:
            aligned_returns = data.loc[common_idx, f'returns_{return_days}_days']
            aligned_indicator = indicators.loc[common_idx, indicator_name]
            
            rolling_corr = aligned_returns.rolling(126).corr(aligned_indicator).dropna()
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line=dict(color='gray', width=1), row=2, col=1)
    
    # 4. Scatter plot
    if f'returns_{return_days}_days' in data.columns:
        common_idx = data.index.intersection(indicators[indicator_name].index)
        if len(common_idx) > 0:
            x_data = indicators.loc[common_idx, indicator_name]
            y_data = data.loc[common_idx, f'returns_{return_days}_days']
            
            mask = ~(x_data.isna() | y_data.isna())
            if mask.sum() > 1:
                x_clean = x_data[mask]
                y_clean = y_data[mask]
                
                fig.add_trace(
                    go.Scattergl(
                        x=x_clean,
                        y=y_clean,
                        mode='markers',
                        marker=dict(size=3, color=y_clean, colorscale='RdYlGn', opacity=0.5),
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                # Regression line
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=p(x_trend),
                        mode='lines',
                        line=dict(color='yellow', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=2, col=2
                )
    
    fig.update_layout(
        template="plotly_dark",
        height=800,
        title=f"<b>{indicator_name} Analysis</b>",
        showlegend=False
    )
    
    return fig

# ===================== MAIN APPLICATION =====================
def main():
    st.markdown("""
        <h1 style='text-align: center;'>
            🔬 Complete Quantitative Analyzer
        </h1>
        <p style='text-align: center; color: #8892B0; font-size: 1.1rem;'>
            Analyze {total} TALib Indicators with Custom Period Ranges
        </p>
    """.format(total=TechnicalIndicators.get_total_count()), unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.returns_data = None
        st.session_state.indicators = None
        st.session_state.data = None
        st.session_state.trading_rules = None
        st.session_state.summary = None
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Data Download Settings
        with st.expander("📈 **DATA DOWNLOAD**", expanded=True):
            ticker = st.text_input("Symbol", value="SPY", help="Enter stock ticker symbol")
            
            period_option = st.selectbox(
                "Download Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max", "custom"],
                index=6,  # Default to 2y
                help="Select time period for data download"
            )
            
            if period_option == "custom":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
                with col2:
                    end_date = st.date_input("End Date", value=datetime.now())
            else:
                start_date = None
                end_date = None
        
        # Period Range Settings
        with st.expander("🎯 **PERIOD RANGE**", expanded=True):
            st.markdown("Define exact period range to test")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                min_period = st.number_input("Min Period", value=5, min_value=2, max_value=500)
            with col2:
                max_period = st.number_input("Max Period", value=50, min_value=5, max_value=500)
            with col3:
                step_period = st.number_input("Step", value=5, min_value=1, max_value=50)
            
            # Generate periods list
            periods_to_test = list(range(min_period, max_period + 1, step_period))
            st.info(f"📊 Will test {len(periods_to_test)} periods: {periods_to_test[:5]}{'...' if len(periods_to_test) > 5 else ''}")
        
        # Analysis Settings
        with st.expander("📊 **ANALYSIS SETTINGS**", expanded=True):
            return_days = st.select_slider(
                "Forward Return Days",
                options=[1, 2, 3, 5, 7, 10, 14, 20, 30],
                value=5
            )
            
            quantiles = st.slider(
                "Number of Percentiles",
                min_value=5,
                max_value=20,
                value=10,
                step=5
            )
            
            min_spread = st.slider(
                "Min Spread for Rules (%)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5
            )
            
            max_p_value = st.slider(
                "Max P-Value",
                min_value=0.01,
                max_value=0.30,
                value=0.15,
                step=0.01
            )
        
        # Indicator Selection
        with st.expander("📐 **INDICATOR SELECTION**", expanded=True):
            select_mode = st.radio(
                "Selection Mode",
                ["Quick Presets", "Categories", "All Indicators"]
            )
            
            if select_mode == "Quick Presets":
                preset = st.selectbox(
                    "Choose Preset",
                    ["Essential (30)", "Momentum Focus (50)", "Complete (100)", "Everything (200+)"]
                )
                
                if preset == "Essential (30)":
                    selected_categories = ["📈 Overlaps (17)", "💫 Momentum (30)"][:1]
                elif preset == "Momentum Focus (50)":
                    selected_categories = ["💫 Momentum (30)", "📉 Volatility (3)", "📊 Volume (3)"]
                elif preset == "Complete (100)":
                    selected_categories = list(TechnicalIndicators.CATEGORIES.keys())[:6]
                else:
                    selected_categories = ["ALL"]
            
            elif select_mode == "Categories":
                selected_categories = st.multiselect(
                    "Select Categories",
                    list(TechnicalIndicators.CATEGORIES.keys()),
                    default=["💫 Momentum (30)", "📈 Overlaps (17)"]
                )
            else:
                selected_categories = ["ALL"]
            
            # Count indicators
            if "ALL" in selected_categories:
                indicator_count = TechnicalIndicators.get_total_count()
            else:
                indicator_count = sum(
                    len(TechnicalIndicators.CATEGORIES[cat]) 
                    for cat in selected_categories 
                    if cat in TechnicalIndicators.CATEGORIES
                )
            
            st.success(f"📊 {indicator_count} indicators × {len(periods_to_test)} periods = {indicator_count * len(periods_to_test)} calculations")
        
        st.markdown("---")
        
        # Run Analysis Button
        analyze_button = st.button(
            "🚀 **ANALYZE**",
            use_container_width=True,
            type="primary"
        )
    
    # Main Analysis
    if analyze_button:
        with st.spinner('🔬 Running comprehensive analysis...'):
            # Calculate all indicators
            returns_data, indicators, data, summary = calculate_all_indicators(
                ticker,
                period_option,
                start_date.strftime('%Y-%m-%d') if start_date else None,
                end_date.strftime('%Y-%m-%d') if end_date else None,
                quantiles,
                return_days,
                periods_to_test,
                selected_categories
            )
            
            if returns_data and indicators is not None and data is not None:
                st.session_state.analysis_done = True
                st.session_state.returns_data = returns_data
                st.session_state.indicators = indicators
                st.session_state.data = data
                st.session_state.summary = summary
                
                # Generate trading rules
                with st.spinner('🎯 Generating trading rules...'):
                    st.session_state.trading_rules = generate_trading_rules(
                        indicators, data, return_days, min_spread, max_p_value, top_n=30
                    )
    
    # Display Results
    if st.session_state.analysis_done:
        returns_data = st.session_state.returns_data
        indicators = st.session_state.indicators
        data = st.session_state.data
        trading_rules = st.session_state.trading_rules
        summary = st.session_state.summary
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Indicators", summary['indicators_count'])
        with col2:
            st.metric("📈 Success Rate", f"{(summary['successful']/summary['total_attempted']*100):.1f}%")
        with col3:
            st.metric("📋 Trading Rules", len(trading_rules) if trading_rules else 0)
        with col4:
            st.metric("📅 Data Points", summary['data_points'])
        
        st.info(f"📆 Date Range: {summary['date_range']}")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Individual Analysis",
            "📋 Trading Rules",
            "🏆 Top Performers",
            "💾 Export"
        ])
        
        with tab1:
            st.markdown("### 📈 Individual Indicator Analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_indicator = st.selectbox(
                    "Select Indicator",
                    sorted(indicators.columns),
                    help="Choose any calculated indicator"
                )
            with col2:
                return_period = st.selectbox(
                    "Return Period",
                    list(range(1, return_days + 1)),
                    index=min(4, return_days - 1) if return_days >= 5 else 0
                )
            
            if selected_indicator:
                # Check for trading rule
                has_rule = any(r['indicator'] == selected_indicator for r in (trading_rules or []))
                if has_rule:
                    st.success(f"✅ Trading rule exists for {selected_indicator}")
                
                # Create plot
                fig = create_percentile_plot(
                    indicators, returns_data, data,
                    selected_indicator, return_period
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 📋 Trading Rules")
            
            if trading_rules:
                # Filter options
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    strategy_filter = st.selectbox(
                        "Filter by Strategy",
                        ["All"] + list(set(r['strategy'] for r in trading_rules))
                    )
                with filter_col2:
                    min_confidence = st.slider("Min Confidence %", 0, 100, 50)
                
                # Filter rules
                filtered_rules = [
                    r for r in trading_rules 
                    if (strategy_filter == "All" or r['strategy'] == strategy_filter)
                    and r['confidence'] >= min_confidence
                ]
                
                st.info(f"Showing {len(filtered_rules)} rules")
                
                # Display rules
                for rule in filtered_rules[:20]:
                    badge_class = {
                        'MOMENTUM': 'momentum-badge',
                        'MEAN REVERSION': 'mean-reversion-badge'
                    }.get(rule['strategy'], 'momentum-badge')
                    
                    badge_html = f'<span class="{badge_class}">{"🟠" if rule["strategy"] == "MOMENTUM" else "🔵"} {rule["strategy"]}</span>'
                    
                    st.markdown(f"""
                    <div class="trading-rule">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <h4>#{rule['rank']}: {rule['indicator']}</h4>
                            {badge_html}
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            <div>
                                <p><strong>Entry:</strong> {rule['entry']}</p>
                                <p>→ <strong>{rule['action']}</strong></p>
                                <p><strong>Exit:</strong> {rule['exit']}</p>
                                <p>→ <strong>{rule['exit_action']}</strong></p>
                            </div>
                            <div>
                                <p>📊 <strong>Spread:</strong> {rule['spread']:.2f}%</p>
                                <p>📈 <strong>Top Return:</strong> {rule['top_return']:.2f}%</p>
                                <p>📉 <strong>Bottom Return:</strong> {rule['bottom_return']:.2f}%</p>
                                <p>🎯 <strong>Confidence:</strong> {rule['confidence']:.1f}%</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No trading rules generated. Try adjusting parameters.")
        
        with tab3:
            st.markdown("### 🏆 Top Performing Indicators")
            
            # Create performance summary
            performance = []
            for ind_col in indicators.columns:
                if ind_col in returns_data:
                    ret_col = f'returns_{return_days}_days_mean'
                    if ret_col in returns_data[ind_col].columns:
                        values = returns_data[ind_col][ret_col]
                        if len(values) > 1:
                            spread = values.iloc[-1] - values.iloc[0]
                            performance.append({
                                'Indicator': ind_col,
                                'Spread': spread,
                                'Top Return': values.iloc[-1],
                                'Bottom Return': values.iloc[0],
                                'Has Rule': '✅' if any(r['indicator'] == ind_col for r in (trading_rules or [])) else '❌'
                            })
            
            if performance:
                perf_df = pd.DataFrame(performance)
                perf_df = perf_df.sort_values('Spread', ascending=False).head(30)
                
                st.dataframe(
                    perf_df.style.format({
                        'Spread': '{:.2f}%',
                        'Top Return': '{:.2f}%',
                        'Bottom Return': '{:.2f}%'
                    }).background_gradient(subset=['Spread'], cmap='RdYlGn'),
                    use_container_width=True,
                    height=600
                )
        
        with tab4:
            st.markdown("### 💾 Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if trading_rules:
                    rules_df = pd.DataFrame(trading_rules)
                    csv = rules_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Trading Rules (CSV)",
                        data=csv,
                        file_name=f"{ticker}_rules_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                if performance:
                    perf_csv = perf_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Performance (CSV)",
                        data=perf_csv,
                        file_name=f"{ticker}_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Prepare Indicator Data"):
                    indicators_csv = indicators.to_csv()
                    st.download_button(
                        "📥 Download All Indicators (CSV)",
                        data=indicators_csv,
                        file_name=f"{ticker}_indicators_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
