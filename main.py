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
    page_title="Quantitative Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== ELEGANT GRAY DARK MODE STYLING =====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: #0f0f0f;
    }
    
    .main-header {
        color: #f0f0f0;
        font-weight: 200;
        font-size: 3rem;
        text-align: center;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }
    
    .sub-header {
        text-align: center;
        color: #808080;
        font-size: 0.95rem;
        font-weight: 300;
        margin-bottom: 3rem;
        letter-spacing: 0.02em;
    }
    
    .config-section {
        background: linear-gradient(135deg, #1a1a1a 0%, #222222 100%);
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .section-title {
        color: #d0d0d0;
        font-weight: 400;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #333333;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border: 1px solid #404040;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="metric-container"] label {
        color: #a0a0a0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="metric-container"] > div {
        color: #f0f0f0 !important;
        font-weight: 500 !important;
        font-size: 1.4rem !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4a4a4a 0%, #606060 100%);
        color: #f0f0f0;
        border: 1px solid #666666;
        padding: 0.8rem 3rem;
        font-weight: 400;
        font-size: 0.95rem;
        border-radius: 8px;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #606060 0%, #707070 100%);
        border-color: #888888;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 0.3rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #808080;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
        font-weight: 400;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #333333 0%, #404040 100%);
        color: #f0f0f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .info-badge {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        color: #b0b0b0;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border: 1px solid #404040;
        font-size: 0.85rem;
        margin: 1rem 0;
        font-weight: 300;
    }
    
    .success-badge {
        background: linear-gradient(135deg, #1a2818 0%, #253023 100%);
        color: #90ee90;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border: 1px solid #4a5a48;
        font-size: 0.85rem;
        margin: 1rem 0;
        font-weight: 300;
    }
    
    .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {
        color: #d0d0d0 !important;
        font-size: 0.85rem !important;
        font-weight: 400 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #404040 !important;
        border-radius: 6px !important;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #606060 0%, #808080 100%);
    }
    
    .stProgress > div > div {
        background: #2a2a2a;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #404040, transparent);
        margin: 2.5rem 0;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 0.85rem !important;
    }
    
    /* Plotly chart background */
    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== TECHNICAL INDICATORS CLASS =====================
class TechnicalIndicators:
    """Complete TALib indicators manager (200+ indicators)"""
    
    @staticmethod
    def calculate_bbands(h, l, c, v, o, p):
        try:
            result = talib.BBANDS(c, timeperiod=p or 20, nbdevup=2, nbdevdn=2)
            return result[0]  # Return upper band
        except: return None
    
    @staticmethod
    def calculate_single_indicator(name, h, l, c, v, o, p):
        """Calculate a single indicator with proper error handling"""
        try:
            h = np.asarray(h, dtype=np.float64)
            l = np.asarray(l, dtype=np.float64)
            c = np.asarray(c, dtype=np.float64)
            v = np.asarray(v, dtype=np.float64)
            o = np.asarray(o, dtype=np.float64)
            
            # Overlap Studies
            if name == 'BBANDS':
                result = talib.BBANDS(c, timeperiod=p or 20)
                return result[0]
            elif name == 'DEMA': return talib.DEMA(c, timeperiod=p or 30)
            elif name == 'EMA': return talib.EMA(c, timeperiod=p or 30)
            elif name == 'HT_TRENDLINE': return talib.HT_TRENDLINE(c)
            elif name == 'KAMA': return talib.KAMA(c, timeperiod=p or 30)
            elif name == 'MA': return talib.MA(c, timeperiod=p or 30)
            elif name == 'MAMA':
                result = talib.MAMA(c, fastlimit=0.5, slowlimit=0.05)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'MIDPOINT': return talib.MIDPOINT(c, timeperiod=p or 14)
            elif name == 'MIDPRICE': return talib.MIDPRICE(h, l, timeperiod=p or 14)
            elif name == 'SAR': return talib.SAR(h, l)
            elif name == 'SAREXT': return talib.SAREXT(h, l)
            elif name == 'SMA': return talib.SMA(c, timeperiod=p or 30)
            elif name == 'T3': return talib.T3(c, timeperiod=p or 5)
            elif name == 'TEMA': return talib.TEMA(c, timeperiod=p or 30)
            elif name == 'TRIMA': return talib.TRIMA(c, timeperiod=p or 30)
            elif name == 'WMA': return talib.WMA(c, timeperiod=p or 30)
            
            # Momentum Indicators
            elif name == 'ADX': return talib.ADX(h, l, c, timeperiod=p or 14)
            elif name == 'ADXR': return talib.ADXR(h, l, c, timeperiod=p or 14)
            elif name == 'APO': return talib.APO(c, fastperiod=max(p//2, 2) if p else 12, slowperiod=p or 26)
            elif name == 'AROON':
                result = talib.AROON(h, l, timeperiod=p or 14)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'AROONOSC': return talib.AROONOSC(h, l, timeperiod=p or 14)
            elif name == 'BOP': return talib.BOP(o, h, l, c)
            elif name == 'CCI': return talib.CCI(h, l, c, timeperiod=p or 14)
            elif name == 'CMO': return talib.CMO(c, timeperiod=p or 14)
            elif name == 'DX': return talib.DX(h, l, c, timeperiod=p or 14)
            elif name == 'MACD':
                result = talib.MACD(c, fastperiod=max(p//2, 2) if p else 12, slowperiod=p or 26)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'MACDEXT':
                result = talib.MACDEXT(c, fastperiod=max(p//2, 2) if p else 12, slowperiod=p or 26)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'MACDFIX':
                result = talib.MACDFIX(c)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'MFI': return talib.MFI(h, l, c, v, timeperiod=p or 14)
            elif name == 'MINUS_DI': return talib.MINUS_DI(h, l, c, timeperiod=p or 14)
            elif name == 'MINUS_DM': return talib.MINUS_DM(h, l, timeperiod=p or 14)
            elif name == 'MOM': return talib.MOM(c, timeperiod=p or 10)
            elif name == 'PLUS_DI': return talib.PLUS_DI(h, l, c, timeperiod=p or 14)
            elif name == 'PLUS_DM': return talib.PLUS_DM(h, l, timeperiod=p or 14)
            elif name == 'PPO': return talib.PPO(c, fastperiod=max(p//2, 2) if p else 12, slowperiod=p or 26)
            elif name == 'ROC': return talib.ROC(c, timeperiod=p or 10)
            elif name == 'ROCP': return talib.ROCP(c, timeperiod=p or 10)
            elif name == 'ROCR': return talib.ROCR(c, timeperiod=p or 10)
            elif name == 'ROCR100': return talib.ROCR100(c, timeperiod=p or 10)
            elif name == 'RSI': return talib.RSI(c, timeperiod=p or 14)
            elif name == 'STOCH':
                result = talib.STOCH(h, l, c, fastk_period=p or 5)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'STOCHF':
                result = talib.STOCHF(h, l, c, fastk_period=p or 5)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'STOCHRSI':
                result = talib.STOCHRSI(c, timeperiod=p or 14)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'TRIX': return talib.TRIX(c, timeperiod=p or 30)
            elif name == 'ULTOSC': return talib.ULTOSC(h, l, c, timeperiod1=max(p//3, 2) if p else 7, timeperiod2=max(p//2, 3) if p else 14, timeperiod3=p or 28)
            elif name == 'WILLR': return talib.WILLR(h, l, c, timeperiod=p or 14)
            
            # Volume Indicators
            elif name == 'AD': return talib.AD(h, l, c, v)
            elif name == 'ADOSC': return talib.ADOSC(h, l, c, v, fastperiod=max(p//3, 2) if p else 3, slowperiod=p or 10)
            elif name == 'OBV': return talib.OBV(c, v)
            
            # Volatility
            elif name == 'ATR': return talib.ATR(h, l, c, timeperiod=p or 14)
            elif name == 'NATR': return talib.NATR(h, l, c, timeperiod=p or 14)
            elif name == 'TRANGE': return talib.TRANGE(h, l, c)
            
            # Cycle Indicators
            elif name == 'HT_DCPERIOD': return talib.HT_DCPERIOD(c)
            elif name == 'HT_DCPHASE': return talib.HT_DCPHASE(c)
            elif name == 'HT_PHASOR':
                result = talib.HT_PHASOR(c)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'HT_SINE':
                result = talib.HT_SINE(c)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'HT_TRENDMODE': return talib.HT_TRENDMODE(c)
            
            # Statistics
            elif name == 'BETA': return talib.BETA(h, l, timeperiod=p or 5)
            elif name == 'CORREL': return talib.CORREL(h, l, timeperiod=p or 30)
            elif name == 'LINEARREG': return talib.LINEARREG(c, timeperiod=p or 14)
            elif name == 'LINEARREG_ANGLE': return talib.LINEARREG_ANGLE(c, timeperiod=p or 14)
            elif name == 'LINEARREG_INTERCEPT': return talib.LINEARREG_INTERCEPT(c, timeperiod=p or 14)
            elif name == 'LINEARREG_SLOPE': return talib.LINEARREG_SLOPE(c, timeperiod=p or 14)
            elif name == 'STDDEV': return talib.STDDEV(c, timeperiod=p or 5)
            elif name == 'TSF': return talib.TSF(c, timeperiod=p or 14)
            elif name == 'VAR': return talib.VAR(c, timeperiod=p or 5)
            
            # Math Transform
            elif name == 'ACOS': return talib.ACOS(c)
            elif name == 'ASIN': return talib.ASIN(c)
            elif name == 'ATAN': return talib.ATAN(c)
            elif name == 'CEIL': return talib.CEIL(c)
            elif name == 'COS': return talib.COS(c)
            elif name == 'COSH': return talib.COSH(c)
            elif name == 'EXP': return talib.EXP(c)
            elif name == 'FLOOR': return talib.FLOOR(c)
            elif name == 'LN': return talib.LN(c)
            elif name == 'LOG10': return talib.LOG10(c)
            elif name == 'SIN': return talib.SIN(c)
            elif name == 'SINH': return talib.SINH(c)
            elif name == 'SQRT': return talib.SQRT(c)
            elif name == 'TAN': return talib.TAN(c)
            elif name == 'TANH': return talib.TANH(c)
            
            # Math Operators
            elif name == 'ADD': return talib.ADD(c, c)
            elif name == 'DIV': return talib.DIV(c, c)
            elif name == 'MAX': return talib.MAX(c, timeperiod=p or 30)
            elif name == 'MAXINDEX': return talib.MAXINDEX(c, timeperiod=p or 30)
            elif name == 'MIN': return talib.MIN(c, timeperiod=p or 30)
            elif name == 'MININDEX': return talib.MININDEX(c, timeperiod=p or 30)
            elif name == 'MINMAX':
                result = talib.MINMAX(c, timeperiod=p or 30)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'MINMAXINDEX':
                result = talib.MINMAXINDEX(c, timeperiod=p or 30)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'MULT': return talib.MULT(c, c)
            elif name == 'SUB': return talib.SUB(c, c)
            elif name == 'SUM': return talib.SUM(c, timeperiod=p or 30)
            
            # Price Transform
            elif name == 'AVGPRICE': return talib.AVGPRICE(o, h, l, c)
            elif name == 'MEDPRICE': return talib.MEDPRICE(h, l)
            elif name == 'TYPPRICE': return talib.TYPPRICE(h, l, c)
            elif name == 'WCLPRICE': return talib.WCLPRICE(h, l, c)
            
            # Candle patterns
            elif name.startswith('CDL'):
                if hasattr(talib, name):
                    func = getattr(talib, name)
                    return func(o, h, l, c)
            
            return None
        except:
            return None
    
    # All indicators list
    ALL_INDICATORS = [
        # Overlap Studies (17)
        'BBANDS', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA',
        'MAMA', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT',
        'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA',
        
        # Momentum (31)
        'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP',
        'CCI', 'CMO', 'DX', 'MACD', 'MACDEXT', 'MACDFIX',
        'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI',
        'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100',
        'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX',
        'ULTOSC', 'WILLR',
        
        # Volume (3)
        'AD', 'ADOSC', 'OBV',
        
        # Volatility (3)
        'ATR', 'NATR', 'TRANGE',
        
        # Cycles (5)
        'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE',
        
        # Statistics (9)
        'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE',
        'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR',
        
        # Math Transform (15)
        'ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH',
        'EXP', 'FLOOR', 'LN', 'LOG10', 'SIN', 'SINH',
        'SQRT', 'TAN', 'TANH',
        
        # Math Operators (11)
        'ADD', 'DIV', 'MAX', 'MAXINDEX', 'MIN', 'MININDEX',
        'MINMAX', 'MINMAXINDEX', 'MULT', 'SUB', 'SUM',
        
        # Price Transform (4)
        'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
    ]
    
    # Candle patterns (61)
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
        "Overlaps": ['BBANDS', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA',
                     'MAMA', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT',
                     'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'],
        "Momentum": ['ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP',
                     'CCI', 'CMO', 'DX', 'MACD', 'MACDEXT', 'MACDFIX',
                     'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI',
                     'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100',
                     'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX',
                     'ULTOSC', 'WILLR'],
        "Volume": ['AD', 'ADOSC', 'OBV'],
        "Volatility": ['ATR', 'NATR', 'TRANGE'],
        "Cycles": ['HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE'],
        "Statistics": ['BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE',
                       'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR'],
        "Math Transform": ['ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH',
                          'EXP', 'FLOOR', 'LN', 'LOG10', 'SIN', 'SINH',
                          'SQRT', 'TAN', 'TANH'],
        "Math Operators": ['ADD', 'DIV', 'MAX', 'MAXINDEX', 'MIN', 'MININDEX',
                          'MINMAX', 'MINMAXINDEX', 'MULT', 'SUB', 'SUM'],
        "Price Transform": ['AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'],
        "Patterns": CANDLE_PATTERNS
    }
    
    @classmethod
    def needs_period(cls, indicator_name):
        """Check if indicator needs period parameter"""
        no_period = [
            'HT_TRENDLINE', 'BOP', 'MACDFIX', 'AD', 'OBV', 'TRANGE',
            'SAR', 'SAREXT', 'MAMA',
            'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE',
            'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
            'ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH',
            'EXP', 'FLOOR', 'LN', 'LOG10', 'SIN', 'SINH',
            'SQRT', 'TAN', 'TANH',
            'ADD', 'DIV', 'MULT', 'SUB'
        ] + cls.CANDLE_PATTERNS
        
        return indicator_name not in no_period
    
    @classmethod
    def calculate_indicator(cls, indicator_name, high, low, close, volume, open_prices, period):
        """Calculate any indicator with error handling"""
        try:
            result = cls.calculate_single_indicator(indicator_name, high, low, close, volume, open_prices, period)
            
            if result is not None:
                # Check if result has valid values
                if not np.all(np.isnan(result)):
                    return result
            
            return None
        except:
            return None
    
    @classmethod
    def get_total_count(cls):
        return len(cls.ALL_INDICATORS) + len(cls.CANDLE_PATTERNS)

# ===================== CALCULATION FUNCTIONS =====================
@st.cache_data
def download_data(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Download historical data"""
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True, multi_level_index=False)
        
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
        
        return data
        
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

@st.cache_data
def calculate_all_indicators(ticker: str, period: str, quantiles: int, return_days: int, 
                             periods_to_test: List[int], selected_categories: List[str]) -> Tuple:
    """Calculate all selected indicators"""
    
    data = download_data(ticker, period)
    if data is None:
        return None, None, None, None
    
    # Calculate returns
    for i in range(1, return_days + 1):
        data[f'returns_{i}_days'] = data['Close'].pct_change(i) * 100
    
    # Prepare data
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    volume = data['Volume'].values if 'Volume' in data.columns else np.zeros_like(close)
    open_prices = data['Open'].values
    
    indicators = pd.DataFrame(index=data.index)
    
    # Get indicators to calculate
    indicators_to_calc = []
    
    if "ALL" in selected_categories:
        indicators_to_calc = TechnicalIndicators.ALL_INDICATORS + TechnicalIndicators.CANDLE_PATTERNS
    else:
        for category in selected_categories:
            if category in TechnicalIndicators.CATEGORIES:
                indicators_to_calc.extend(TechnicalIndicators.CATEGORIES[category])
    
    indicators_to_calc = list(set(indicators_to_calc))
    
    # Count calculations
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
                status_text.text(f"⏳ Calculating {indicator_name}_{period}...")
                
                result = TechnicalIndicators.calculate_indicator(
                    indicator_name, high, low, close, volume, open_prices, period
                )
                
                if result is not None:
                    indicators[f'{indicator_name}_{period}'] = result
                    successful += 1
                
                progress_bar.progress(calculation_counter / total_calculations)
        else:
            calculation_counter += 1
            status_text.text(f"⏳ Calculating {indicator_name}...")
            
            result = TechnicalIndicators.calculate_indicator(
                indicator_name, high, low, close, volume, open_prices, 0
            )
            
            if result is not None:
                indicators[indicator_name] = result
                successful += 1
            
            progress_bar.progress(calculation_counter / total_calculations)
    
    progress_bar.empty()
    status_text.empty()
    
    # Drop completely empty columns
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
            
            if len(temp_df) >= quantiles * 2:  # Need enough data for quantiles
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
    
    st.markdown(f"""
        <div class="success-badge">
            ✓ Successfully calculated {successful} out of {total_calculations} configurations
        </div>
    """, unsafe_allow_html=True)
    
    summary = {
        'total_attempted': total_calculations,
        'successful': successful,
        'indicators_count': len(indicators.columns),
        'data_points': len(data),
        'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
    }
    
    return returns_data, indicators, data, summary

def create_percentile_plot(indicators, returns_data, data, indicator_name, return_days):
    """Create enhanced analysis plots with beautiful dark theme aesthetics"""
    
    if indicator_name not in indicators.columns or indicator_name not in returns_data:
        return None
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '<b>Distribution & Statistics</b>', '<b>Returns by Percentile</b>',
            '<b>Rolling Correlation (126-day)</b>', '<b>Scatter Analysis with Density</b>',
            '<b>Box Plot by Quantile</b>', '<b>Cumulative Returns</b>'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "box"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.12,
        row_heights=[0.35, 0.35, 0.3]
    )
    
    # Color palette
    gradient_colors = ['#FF6B6B', '#FE8C68', '#FEAA68', '#FEC868', '#FFE66D', 
                       '#C7E66D', '#8FE66D', '#5FE668', '#4FC668', '#51CF66']
    
    # 1. ENHANCED DISTRIBUTION WITH KDE
    hist_data = indicators[indicator_name].dropna()
    
    if len(hist_data) > 0:
        # Calculate statistics
        mean_val = hist_data.mean()
        median_val = hist_data.median()
        std_val = hist_data.std()
        q25 = hist_data.quantile(0.25)
        q75 = hist_data.quantile(0.75)
        
        # Histogram with more bins
        fig.add_trace(
            go.Histogram(
                x=hist_data,
                nbinsx=100,
                marker=dict(
                    color='rgba(100, 150, 255, 0.3)',
                    line=dict(color='rgba(100, 150, 255, 0.5)', width=0.5)
                ),
                name='Distribution',
                showlegend=False,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        # Add KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(hist_data.values)
        x_range = np.linspace(hist_data.min(), hist_data.max(), 200)
        kde_values = kde(x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_values,
                mode='lines',
                line=dict(color='#FFE66D', width=2.5),
                name='KDE',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add vertical lines for statistics
        for val, color, name, dash in [
            (mean_val, '#51CF66', 'Mean', 'solid'),
            (median_val, '#FF6B6B', 'Median', 'dash'),
            (q25, '#808080', 'Q1', 'dot'),
            (q75, '#808080', 'Q3', 'dot')
        ]:
            fig.add_vline(
                x=val, 
                line=dict(color=color, width=1.5, dash=dash),
                annotation_text=f"{name}: {val:.2f}",
                annotation_position="top",
                annotation_font_color=color,
                annotation_font_size=10,
                row=1, col=1
            )
    
    # 2. ENHANCED RETURNS BY PERCENTILE
    returns_col = f'returns_{return_days}_days_mean'
    if returns_col in returns_data[indicator_name].columns:
        returns_values = returns_data[indicator_name][returns_col]
        x_labels = [f'P{i+1}' for i in range(len(returns_values))]
        
        # Use gradient colors based on value
        max_abs = max(abs(returns_values.max()), abs(returns_values.min()))
        normalized_values = [(val + max_abs) / (2 * max_abs) for val in returns_values]
        colors = [gradient_colors[int(norm * (len(gradient_colors) - 1))] for norm in normalized_values]
        
        # Add error bars if std available
        std_col = f'returns_{return_days}_days_std'
        error_y = None
        if std_col in returns_data[indicator_name].columns:
            error_y = dict(
                type='data',
                array=returns_data[indicator_name][std_col],
                visible=True,
                color='rgba(255, 255, 255, 0.3)',
                thickness=1.5,
                width=4
            )
        
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=returns_values,
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                ),
                text=[f'{val:.2f}%' for val in returns_values],
                textposition='outside',
                textfont=dict(size=10),
                error_y=error_y,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add trend line
        x_numeric = list(range(len(returns_values)))
        z = np.polyfit(x_numeric, returns_values, 1)
        p = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=p(x_numeric),
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. ENHANCED ROLLING CORRELATION
    if f'returns_{return_days}_days' in data.columns:
        common_idx = data.index.intersection(indicators[indicator_name].index)
        if len(common_idx) > 126:
            aligned_returns = data.loc[common_idx, f'returns_{return_days}_days']
            aligned_indicator = indicators.loc[common_idx, indicator_name]
            
            rolling_corr = aligned_returns.rolling(126).corr(aligned_indicator).dropna()
            
            # Create gradient fill
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    line=dict(color='#4FC668', width=0),
                    fill='tozeroy',
                    fillcolor='rgba(79, 198, 104, 0.2)',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add main line
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    line=dict(color='#4FC668', width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_hline(y=0, line=dict(color='rgba(255, 255, 255, 0.3)', width=1), row=2, col=1)
            
            # Add correlation bands
            for y, alpha in [(0.5, 0.1), (-0.5, 0.1), (0.75, 0.05), (-0.75, 0.05)]:
                fig.add_hline(
                    y=y,
                    line=dict(color=f'rgba(255, 255, 255, {alpha})', width=1, dash='dot'),
                    row=2, col=1
                )
    
    # 4. ENHANCED SCATTER PLOT WITH DENSITY
    if f'returns_{return_days}_days' in data.columns:
        common_idx = data.index.intersection(indicators[indicator_name].index)
        if len(common_idx) > 0:
            x_data = indicators.loc[common_idx, indicator_name]
            y_data = data.loc[common_idx, f'returns_{return_days}_days']
            
            mask = ~(x_data.isna() | y_data.isna())
            if mask.sum() > 1:
                x_clean = x_data[mask]
                y_clean = y_data[mask]
                
                # Calculate point density for coloring
                from scipy.stats import gaussian_kde
                try:
                    xy = np.vstack([x_clean, y_clean])
                    density = gaussian_kde(xy)(xy)
                except:
                    density = y_clean
                
                fig.add_trace(
                    go.Scattergl(
                        x=x_clean,
                        y=y_clean,
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=density,
                            colorscale=[
                                [0, '#0D1117'],
                                [0.25, '#1F2937'],
                                [0.5, '#6366F1'],
                                [0.75, '#A78BFA'],
                                [1, '#FDE68A']
                            ],
                            opacity=0.8,
                            line=dict(width=0),
                            showscale=True,
                            colorbar=dict(
                                title="Density",
                                titlefont=dict(size=10),
                                tickfont=dict(size=9),
                                len=0.5,
                                y=0.5,
                                yanchor='middle',
                                thickness=10
                            )
                        ),
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                # Add regression line
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=p(x_line),
                        mode='lines',
                        line=dict(color='#FF6B6B', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=2, col=2
                )
    
    # 5. BOX PLOT BY QUANTILE
    if f'returns_{return_days}_days' in data.columns:
        temp_df = pd.DataFrame({
            'indicator': indicators[indicator_name],
            'returns': data[f'returns_{return_days}_days']
        }).dropna()
        
        if len(temp_df) >= quantiles * 2:
            temp_df['quantile'] = pd.qcut(temp_df['indicator'], q=min(quantiles, 20), duplicates='drop')
            
            for i, (name, group) in enumerate(temp_df.groupby('quantile')):
                color_idx = int((i / (len(temp_df.groupby('quantile')) - 1)) * (len(gradient_colors) - 1))
                
                fig.add_trace(
                    go.Box(
                        y=group['returns'],
                        name=f'Q{i+1}',
                        marker=dict(
                            color=gradient_colors[color_idx],
                            opacity=0.7
                        ),
                        boxmean='sd',
                        showlegend=False
                    ),
                    row=3, col=1
                )
    
    # 6. CUMULATIVE RETURNS
    if f'returns_{return_days}_days' in data.columns:
        temp_df = pd.DataFrame({
            'indicator': indicators[indicator_name],
            'returns': data[f'returns_{return_days}_days'],
            'date': data.index
        }).dropna()
        
        if len(temp_df) >= quantiles:
            temp_df['quantile'] = pd.qcut(temp_df['indicator'], q=min(quantiles, 5), duplicates='drop')
            
            for i, (name, group) in enumerate(temp_df.groupby('quantile')):
                group = group.sort_values('date')
                cumulative_returns = (1 + group['returns'] / 100).cumprod() - 1
                
                color_idx = int((i / max(len(temp_df.groupby('quantile')) - 1, 1)) * (len(gradient_colors) - 1))
                
                fig.add_trace(
                    go.Scatter(
                        x=group['date'],
                        y=cumulative_returns * 100,
                        mode='lines',
                        line=dict(color=gradient_colors[color_idx], width=2),
                        name=f'Q{i+1}',
                        showlegend=True
                    ),
                    row=3, col=2
                )
    
    # UPDATE LAYOUT
    fig.update_layout(
        template="plotly_dark",
        height=1000,
        title={
            'text': f"<b>{indicator_name}</b> <span style='font-size:14px; color:#808080;'>| {return_days}-Day Return Analysis</span>",
            'font': {'size': 24, 'color': '#f0f0f0', 'family': 'Inter'},
            'x': 0.5,
            'xanchor': 'center'
        },
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        font=dict(color='#C9D1D9', family='Inter', size=11),
        margin=dict(t=100, b=100, l=80, r=80)
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor='#30363D',
        showgrid=True,
        zeroline=False,
        linecolor='#30363D',
        tickfont=dict(size=10)
    )
    fig.update_yaxes(
        gridcolor='#30363D',
        showgrid=True,
        zeroline=False,
        linecolor='#30363D',
        tickfont=dict(size=10)
    )
    
    # Update specific axes labels
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Percentile", row=1, col=2)
    fig.update_yaxes(title_text=f"Returns ({return_days}d) %", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    fig.update_xaxes(title_text="Indicator Value", row=2, col=2)
    fig.update_yaxes(title_text=f"Returns ({return_days}d) %", row=2, col=2)
    fig.update_xaxes(title_text="Quantile", row=3, col=1)
    fig.update_yaxes(title_text=f"Returns ({return_days}d) %", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=2)
    fig.update_yaxes(title_text="Cumulative Return %", row=3, col=2)
    
    return fig

# ===================== MAIN APPLICATION =====================
def main():
    # Header
    st.markdown("""
        <h1 class='main-header'>Quantitative Analysis Platform</h1>
        <p class='sub-header'>
            {total} TECHNICAL INDICATORS · MULTI-PERIOD TESTING · PERCENTILE ANALYSIS
        </p>
    """.format(total=TechnicalIndicators.get_total_count()), unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.returns_data = None
        st.session_state.indicators = None
        st.session_state.data = None
        st.session_state.summary = None
    
    # Main Configuration
    st.markdown("<div class='config-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Data Configuration</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([2, 1.5, 1, 1])
    
    with col1:
        ticker = st.text_input("TICKER", value="SPY", help="Stock symbol to analyze")
    
    with col2:
        period_option = st.selectbox(
            "PERIOD",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=4  # 2y default
        )
    
    with col3:
        return_days = st.number_input("RETURN DAYS", value=5, min_value=1, max_value=30)
    
    with col4:
        quantiles = st.number_input("PERCENTILES", value=10, min_value=5, max_value=20, step=5)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Period Range Configuration
    st.markdown("<div class='config-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Period Range Configuration</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        min_period = st.number_input("MIN", value=5, min_value=2, max_value=500)
    
    with col2:
        max_period = st.number_input("MAX", value=50, min_value=5, max_value=500)
    
    with col3:
        step_period = st.number_input("STEP", value=5, min_value=1, max_value=50)
    
    with col4:
        periods_to_test = list(range(min_period, max_period + 1, step_period))
        st.markdown(f"""
            <div class="info-badge">
                Testing periods: {', '.join(map(str, periods_to_test))}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Indicator Selection
    st.markdown("<div class='config-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Indicator Selection</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        select_mode = st.radio(
            "MODE",
            ["Presets", "Categories", "All"]
        )
    
    with col2:
        if select_mode == "Presets":
            preset = st.selectbox(
                "PRESET",
                ["Essential (30 indicators)", 
                 "Extended (60 indicators)", 
                 "Complete (100 indicators)", 
                 "Everything (158+ indicators)"]
            )
            
            if "Essential" in preset:
                selected_categories = ["Overlaps", "Momentum"][:1]
            elif "Extended" in preset:
                selected_categories = ["Momentum", "Volatility", "Volume", "Overlaps"]
            elif "Complete" in preset:
                selected_categories = list(TechnicalIndicators.CATEGORIES.keys())[:7]
            else:
                selected_categories = ["ALL"]
        
        elif select_mode == "Categories":
            selected_categories = st.multiselect(
                "SELECT CATEGORIES",
                list(TechnicalIndicators.CATEGORIES.keys()),
                default=["Momentum", "Overlaps"]
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
    
    # Calculate total
    total_with_periods = sum(
        len(periods_to_test) if TechnicalIndicators.needs_period(ind) else 1
        for cat in (["ALL"] if "ALL" in selected_categories else selected_categories)
        for ind in (TechnicalIndicators.ALL_INDICATORS + TechnicalIndicators.CANDLE_PATTERNS 
                   if cat == "ALL" 
                   else TechnicalIndicators.CATEGORIES.get(cat, []))
    )
    
    st.markdown(f"""
        <div class="info-badge">
            {indicator_count} indicators × {len(periods_to_test)} periods = {total_with_periods} calculations
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Analyze Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_button = st.button(
            "ANALYZE",
            use_container_width=True,
            type="primary"
        )
    
    # Analysis
    if analyze_button:
        with st.spinner('Processing indicators...'):
            returns_data, indicators, data, summary = calculate_all_indicators(
                ticker,
                period_option,
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
    
    # Results
    if st.session_state.analysis_done:
        returns_data = st.session_state.returns_data
        indicators = st.session_state.indicators
        data = st.session_state.data
        summary = st.session_state.summary
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("INDICATORS", summary['indicators_count'])
        with col2:
            st.metric("SUCCESS RATE", f"{(summary['successful']/summary['total_attempted']*100):.1f}%")
        with col3:
            st.metric("DATA POINTS", summary['data_points'])
        with col4:
            st.metric("DATE RANGE", summary['date_range'].split(' to ')[0])
        
        # Analysis Tabs
        tab1, tab2, tab3 = st.tabs(["📊 ANALYSIS", "🏆 PERFORMANCE", "💾 EXPORT"])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_indicator = st.selectbox(
                    "SELECT INDICATOR",
                    sorted(indicators.columns)
                )
            with col2:
                return_period = st.number_input(
                    "DAYS",
                    min_value=1,
                    max_value=return_days,
                    value=min(5, return_days)
                )
            
            if selected_indicator:
                fig = create_percentile_plot(
                    indicators, returns_data, data,
                    selected_indicator, return_period
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Performance analysis
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
                                'Spread (%)': spread,
                                'Top Percentile (%)': values.iloc[-1],
                                'Bottom Percentile (%)': values.iloc[0],
                                'Sharpe': spread / (returns_data[ind_col][f'returns_{return_days}_days_std'].mean() + 1e-8)
                            })
            
            if performance:
                perf_df = pd.DataFrame(performance)
                perf_df = perf_df.sort_values('Spread (%)', ascending=False).head(50)
                
                st.dataframe(
                    perf_df.style.format({
                        'Spread (%)': '{:.2f}',
                        'Top Percentile (%)': '{:.2f}',
                        'Bottom Percentile (%)': '{:.2f}',
                        'Sharpe': '{:.3f}'
                    }).background_gradient(
                        subset=['Spread (%)'],
                        cmap='RdYlGn',
                        vmin=-5,
                        vmax=5
                    ),
                    use_container_width=True,
                    height=600
                )
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("GENERATE PERFORMANCE CSV"):
                    if performance:
                        csv = perf_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Performance",
                            data=csv,
                            file_name=f"{ticker}_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if st.button("GENERATE INDICATORS CSV"):
                    csv = indicators.to_csv()
                    st.download_button(
                        "📥 Download Indicators",
                        data=csv,
                        file_name=f"{ticker}_indicators_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
