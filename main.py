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
from scipy.stats import gaussian_kde, pointbiserialr
from itertools import combinations, product
import re
from dataclasses import dataclass
import json

warnings.filterwarnings('ignore')

# ===================== PAGE CONFIGURATION =====================
st.set_page_config(
    page_title="Advanced Rule-Extraction Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== ELEGANT DARK MODE STYLING =====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700&display=swap');
    
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
    
    .optimization-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #0f3460;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease;
    }
    
    .optimization-card:hover {
        transform: translateY(-4px);
        border-color: #4a69bd;
    }
    
    .method-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0.5rem 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3a 100%);
        border: 1px solid #3a3a4a;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 600;
        color: #4ade80;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
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
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #764ba2;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.4);
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
    
    .warning-badge {
        background: linear-gradient(135deg, #2a1a18 0%, #302520 100%);
        color: #f59e0b;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        border: 1px solid #5a4a48;
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
    
    .progress-text {
        font-size: 1.1rem;
        color: #4ade80;
        font-weight: 500;
        text-align: center;
        margin: 1rem 0;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #404040, transparent);
        margin: 2.5rem 0;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #404040;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #505050;
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== TECHNICAL INDICATORS CLASS =====================
class TechnicalIndicators:
    """Complete TALib indicators manager (200+ indicators)"""
    
    @staticmethod
    def calculate_single_indicator(name, h, l, c, v, o, p):
        """Calculate an indicator with error handling"""
        try:
            h = np.asarray(h, dtype=np.float64)
            l = np.asarray(l, dtype=np.float64)
            c = np.asarray(c, dtype=np.float64)
            v = np.asarray(v, dtype=np.float64)
            o = np.asarray(o, dtype=np.float64)
            
            # Overlay Studies
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
            
            # Math Transforms
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
    
    ALL_INDICATORS = [
        'BBANDS', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA',
        'MAMA', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT',
        'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA',
        'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP',
        'CCI', 'CMO', 'DX', 'MACD', 'MACDEXT', 'MACDFIX',
        'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI',
        'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100',
        'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX',
        'ULTOSC', 'WILLR',
        'AD', 'ADOSC', 'OBV',
        'ATR', 'NATR', 'TRANGE',
        'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE',
        'BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE',
        'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR',
        'ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH',
        'EXP', 'FLOOR', 'LN', 'LOG10', 'SIN', 'SINH',
        'SQRT', 'TAN', 'TANH',
        'ADD', 'DIV', 'MAX', 'MAXINDEX', 'MIN', 'MININDEX',
        'MINMAX', 'MINMAXINDEX', 'MULT', 'SUB', 'SUM',
        'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
    ]
    
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
        "Superposición": ['BBANDS', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA',
                     'MAMA', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT',
                     'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'],
        "Momentum": ['ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP',
                     'CCI', 'CMO', 'DX', 'MACD', 'MACDEXT', 'MACDFIX',
                     'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI',
                     'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100',
                     'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX',
                     'ULTOSC', 'WILLR'],
        "Volumen": ['AD', 'ADOSC', 'OBV'],
        "Volatilidad": ['ATR', 'NATR', 'TRANGE'],
        "Ciclos": ['HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE'],
        "Estadísticas": ['BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE',
                       'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR'],
        "Transformación Matemática": ['ACOS', 'ASIN', 'ATAN', 'CEIL', 'COS', 'COSH',
                          'EXP', 'FLOOR', 'LN', 'LOG10', 'SIN', 'SINH',
                          'SQRT', 'TAN', 'TANH'],
        "Operadores Matemáticos": ['ADD', 'DIV', 'MAX', 'MAXINDEX', 'MIN', 'MININDEX',
                          'MINMAX', 'MINMAXINDEX', 'MULT', 'SUB', 'SUM'],
        "Transformación de Precio": ['AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE'],
        "Patrones de Velas": CANDLE_PATTERNS
    }
    
    @classmethod
    def needs_period(cls, indicator_name):
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
        try:
            result = cls.calculate_single_indicator(indicator_name, high, low, close, volume, open_prices, period)
            if result is not None:
                if not np.all(np.isnan(result)):
                    return result
            return None
        except:
            return None
    
    @classmethod
    def get_total_count(cls):
        return len(cls.ALL_INDICATORS) + len(cls.CANDLE_PATTERNS)

# ===================== HELPER FUNCTIONS =====================

def calculate_indicators_for_dataset(data: pd.DataFrame, periods_to_test: List[int], 
                                    selected_categories: List[str]) -> pd.DataFrame:
    """Calculate indicators WITHOUT look-ahead bias"""
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    volume = data['Volume'].values if 'Volume' in data.columns else np.zeros_like(close)
    open_prices = data['Open'].values
    
    indicators = pd.DataFrame(index=data.index)
    
    indicators_to_calc = []
    if "TODO" in selected_categories:
        indicators_to_calc = TechnicalIndicators.ALL_INDICATORS + TechnicalIndicators.CANDLE_PATTERNS
    else:
        for category in selected_categories:
            if category in TechnicalIndicators.CATEGORIES:
                indicators_to_calc.extend(TechnicalIndicators.CATEGORIES[category])
    
    indicators_to_calc = list(set(indicators_to_calc))
    
    for indicator_name in indicators_to_calc:
        if TechnicalIndicators.needs_period(indicator_name):
            for period in periods_to_test:
                result = TechnicalIndicators.calculate_indicator(
                    indicator_name, high, low, close, volume, open_prices, period
                )
                if result is not None:
                    indicators[f'{indicator_name}_{period}'] = result
        else:
            result = TechnicalIndicators.calculate_indicator(
                indicator_name, high, low, close, volume, open_prices, 0
            )
            if result is not None:
                indicators[indicator_name] = result
    
    indicators = indicators.dropna(axis=1, how='all')
    return indicators


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
def calculate_all_indicators(ticker: str, period: str, quantiles: int, min_return_days: int, 
                             max_return_days: int, periods_to_test: List[int], 
                             selected_categories: List[str]) -> Tuple:
    """Calculate all selected indicators - for visualization only"""
    
    data = download_data(ticker, period)
    if data is None:
        return None, None, None, None
    
    for i in range(min_return_days, max_return_days + 1):
        data[f'retornos_{i}_dias'] = data['Close'].pct_change(i).shift(-i) * 100
    
    indicators_to_calc = []
    if "TODO" in selected_categories:
        indicators_to_calc = TechnicalIndicators.ALL_INDICATORS + TechnicalIndicators.CANDLE_PATTERNS
    else:
        for category in selected_categories:
            if category in TechnicalIndicators.CATEGORIES:
                indicators_to_calc.extend(TechnicalIndicators.CATEGORIES[category])
    
    indicators_to_calc = list(set(indicators_to_calc))
    
    total_calculations = sum(
        len(periods_to_test) if TechnicalIndicators.needs_period(ind) else 1 
        for ind in indicators_to_calc
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    calculation_counter = 0
    successful = 0
    
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    volume = data['Volume'].values if 'Volume' in data.columns else np.zeros_like(close)
    open_prices = data['Open'].values
    
    indicators = pd.DataFrame(index=data.index)
    
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
    
    indicators = indicators.dropna(axis=1, how='all')
    
    returns_data = {}
    for indicator_col in indicators.columns:
        try:
            returns_data[indicator_col] = {}
            for i in range(min_return_days, max_return_days + 1):
                temp_df = pd.DataFrame({'indicator': indicators[indicator_col]})
                ret_col = f'retornos_{i}_dias'
                if ret_col in data.columns:
                    temp_df[ret_col] = data[ret_col]
                temp_df = temp_df.dropna()
                
                if len(temp_df) >= quantiles * 2:
                    temp_df['quantile'] = pd.qcut(temp_df['indicator'], q=quantiles, duplicates='drop')
                    grouped = temp_df.groupby('quantile')[ret_col].agg(['mean', 'std', 'count'])
                    returns_data[indicator_col][f'retornos_{i}_dias_mean'] = grouped['mean']
                    returns_data[indicator_col][f'retornos_{i}_dias_std'] = grouped['std']
                    returns_data[indicator_col][f'retornos_{i}_dias_count'] = grouped['count']
        except:
            continue
    
    st.markdown(f"""
        <div class="success-badge">
            ✓ Successfully calculated {successful} of {total_calculations} configurations
        </div>
    """, unsafe_allow_html=True)
    
    summary = {
        'total_attempted': total_calculations,
        'successful': successful,
        'indicators_count': len(indicators.columns),
        'data_points': len(data),
        'date_range': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
        'min_return_days': min_return_days,
        'max_return_days': max_return_days
    }
    
    return returns_data, indicators, data, summary


def create_percentile_plot(indicators, returns_data, data, indicator_name, return_days, quantiles=10):
    """Create enhanced analysis plots"""
    
    if indicator_name not in indicators.columns or indicator_name not in returns_data:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>Distribution & Statistics</b>', '<b>Returns by Percentile</b>',
            '<b>Rolling Correlation (126 days)</b>', '<b>Scatter Analysis</b>'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    gradient_colors = ['#FF6B6B', '#FE8C68', '#FEAA68', '#FEC868', '#FFE66D', 
                       '#C7E66D', '#8FE66D', '#5FE668', '#4FC668', '#51CF66']
    
    hist_data = indicators[indicator_name].dropna()
    
    if len(hist_data) > 0:
        q1 = hist_data.quantile(0.01)
        q99 = hist_data.quantile(0.99)
        filtered_data = hist_data[(hist_data >= q1) & (hist_data <= q99)]
        
        mean_val = filtered_data.mean()
        median_val = filtered_data.median()
        
        fig.add_trace(
            go.Histogram(
                x=filtered_data,
                nbinsx=100,
                marker=dict(
                    color='rgba(100, 150, 255, 0.4)',
                    line=dict(color='rgba(100, 150, 255, 0.6)', width=0.5)
                ),
                name='Distribution',
                showlegend=False,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        kde = gaussian_kde(filtered_data.values)
        x_range = np.linspace(filtered_data.min(), filtered_data.max(), 200)
        kde_values = kde(x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_values,
                mode='lines',
                line=dict(color='#FFE66D', width=3),
                name='KDE',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_vline(x=mean_val, line=dict(color='#51CF66', width=2, dash='solid'), row=1, col=1)
        fig.add_vline(x=median_val, line=dict(color='#FF6B6B', width=2, dash='dash'), row=1, col=1)
        
        fig.update_xaxes(range=[q1, q99], row=1, col=1)
    
    returns_col = f'retornos_{return_days}_dias_mean'
    if returns_col in returns_data[indicator_name]:
        returns_values = returns_data[indicator_name][returns_col]
        x_labels = [f'P{i+1}' for i in range(len(returns_values))]
        
        max_abs = max(abs(returns_values.max()), abs(returns_values.min())) if returns_values.max() != returns_values.min() else 1
        normalized_values = [(val + max_abs) / (2 * max_abs) for val in returns_values]
        colors = [gradient_colors[min(int(norm * (len(gradient_colors) - 1)), len(gradient_colors)-1)] for norm in normalized_values]
        
        std_col = f'retornos_{return_days}_dias_std'
        error_y = None
        if std_col in returns_data[indicator_name]:
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
                textfont=dict(size=10, color='white'),
                error_y=error_y,
                showlegend=False
            ),
            row=1, col=2
        )
    
    if f'retornos_{return_days}_dias' in data.columns:
        common_idx = data.index.intersection(indicators[indicator_name].index)
        if len(common_idx) > 126:
            aligned_returns = data.loc[common_idx, f'retornos_{return_days}_dias']
            aligned_indicator = indicators.loc[common_idx, indicator_name]
            
            rolling_corr = aligned_returns.rolling(126).corr(aligned_indicator).dropna()
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    line=dict(color='#FFFFFF', width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line=dict(color='rgba(255, 255, 255, 0.3)', width=1), row=2, col=1)
    
    if f'retornos_{return_days}_dias' in data.columns:
        common_idx = data.index.intersection(indicators[indicator_name].index)
        if len(common_idx) > 0:
            x_data = indicators.loc[common_idx, indicator_name]
            y_data = data.loc[common_idx, f'retornos_{return_days}_dias']
            
            mask = ~(x_data.isna() | y_data.isna())
            if mask.sum() > 1:
                x_clean = x_data[mask]
                y_clean = y_data[mask]
                
                x_q1, x_q99 = x_clean.quantile([0.01, 0.99])
                y_q1, y_q99 = y_clean.quantile([0.01, 0.99])
                
                scatter_mask = (x_clean >= x_q1) & (x_clean <= x_q99) & (y_clean >= y_q1) & (y_clean <= y_q99)
                x_filtered = x_clean[scatter_mask]
                y_filtered = y_clean[scatter_mask]
                
                fig.add_trace(
                    go.Scattergl(
                        x=x_filtered,
                        y=y_filtered,
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=y_filtered,
                            colorscale='RdYlGn',
                            opacity=0.6,
                            line=dict(width=0),
                            showscale=True
                        ),
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                fig.update_xaxes(range=[x_q1, x_q99], row=2, col=2)
                fig.update_yaxes(range=[y_q1, y_q99], row=2, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        height=800,
        title={
            'text': f"<b>{indicator_name}</b> | Returns Analysis at {return_days} Days",
            'font': {'size': 24, 'color': '#f0f0f0', 'family': 'Inter'},
            'x': 0.5,
            'xanchor': 'center'
        },
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        showlegend=False,
        font=dict(color='#C9D1D9', family='Inter', size=11),
        margin=dict(t=80, b=60, l=60, r=120)
    )
    
    fig.update_xaxes(gridcolor='#30363D', showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor='#30363D', showgrid=True, zeroline=False)
    
    return fig

# ===================== PARTICLE SWARM OPTIMIZATION =====================

@dataclass
class Particle:
    """Particle for PSO optimization"""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    current_fitness: float

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for trading rules - IMPROVED"""
    
    def __init__(self, n_particles: int = 30, n_iterations: int = 50,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        self.fitness_history = []
        self.valid_rules_found = 0
        self.total_evaluations = 0


class ImprovedParticleSwarmOptimizer(ParticleSwarmOptimizer):
    """Enhanced PSO with Multi-Swarm, Diversity Archive, and Adaptive Parameters"""
    
    def __init__(self, n_particles: int = 50, n_iterations: int = 80,
                 w_max: float = 0.9, w_min: float = 0.4, c1: float = 1.5, c2: float = 1.5,
                 n_swarms: int = 3, archive_size: int = 50):
        super().__init__(n_particles, n_iterations, w_max, c1, c2)
        self.w_max = w_max
        self.w_min = w_min
        self.n_swarms = n_swarms
        self.archive_size = archive_size
        self.diversity_archive = []  # Store (rule, fitness, indicators_used)
        self.swarms_history = []
    
    def optimize_rule_parameters(self, 
                                indicators_df: pd.DataFrame,
                                returns: pd.Series,
                                indicator_names: List[str],
                                progress_callback=None) -> Dict:
        """Multi-Swarm PSO with diversity preservation"""
        
        # Filter valid indicators
        valid_indicators = [ind for ind in indicator_names 
                          if ind in indicators_df.columns 
                          and len(indicators_df[ind].dropna()) >= 20]
        
        if len(valid_indicators) < 2:
            return {
                'rules': [],
                'fitness_scores': [],
                'history': [-999],
                'valid_rules_found': 0,
                'message': f"Not enough valid indicators (found {len(valid_indicators)})"
            }
        
        n_conditions = min(2, len(valid_indicators))
        n_dims = n_conditions * 2
        
        # Create multiple swarms for diversity
        particles_per_swarm = max(self.n_particles // self.n_swarms, 10)
        swarms = []
        
        for swarm_id in range(self.n_swarms):
            swarm_particles = []
            
            for i in range(particles_per_swarm):
                # Different initialization per swarm
                if swarm_id == 0:
                    # Swarm 1: Low percentiles (oversold)
                    position = np.random.uniform(0.0, 0.35, n_dims)
                elif swarm_id == 1:
                    # Swarm 2: Mid percentiles (neutral)
                    position = np.random.uniform(0.35, 0.65, n_dims)
                else:
                    # Swarm 3: High percentiles (overbought)
                    position = np.random.uniform(0.65, 1.0, n_dims)
                
                position = np.clip(position + np.random.randn(n_dims) * 0.03, 0, 1)
                velocity = np.random.randn(n_dims) * 0.05
                
                particle = Particle(
                    position=position,
                    velocity=velocity,
                    best_position=position.copy(),
                    best_fitness=-np.inf,
                    current_fitness=-np.inf
                )
                swarm_particles.append(particle)
            
            swarms.append({
                'particles': swarm_particles,
                'best_position': None,
                'best_fitness': -np.inf
            })
        
        # Track global best
        global_best_position = None
        global_best_fitness = -np.inf
        
        # Optimization loop
        for iteration in range(self.n_iterations):
            # Adaptive inertia weight (decreases over time)
            w = self.w_max - (self.w_max - self.w_min) * (iteration / self.n_iterations)
            
            valid_count = 0
            
            # Evaluate each swarm
            for swarm_id, swarm in enumerate(swarms):
                for particle in swarm['particles']:
                    # Multi-objective fitness
                    fitness = self._evaluate_particle_multiobjective(
                        particle.position, indicators_df, returns, valid_indicators
                    )
                    
                    self.total_evaluations += 1
                    if fitness > -500:
                        valid_count += 1
                        self.valid_rules_found += 1
                    
                    particle.current_fitness = fitness
                    
                    # Update personal best
                    if fitness > particle.best_fitness:
                        particle.best_fitness = fitness
                        particle.best_position = particle.position.copy()
                    
                    # Update swarm best
                    if fitness > swarm['best_fitness']:
                        swarm['best_fitness'] = fitness
                        swarm['best_position'] = particle.position.copy()
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = particle.position.copy()
                
                # Update velocities and positions for this swarm
                for particle in swarm['particles']:
                    r1, r2, r3 = np.random.rand(3)
                    
                    # Personal, swarm, and global influence
                    cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                    social_swarm = self.c2 * r2 * (swarm['best_position'] - particle.position)
                    social_global = 0.5 * r3 * (global_best_position - particle.position)
                    
                    particle.velocity = (w * particle.velocity + cognitive + 
                                       social_swarm + social_global)
                    
                    # Velocity clamping
                    v_max = 0.2 * (1.0 - 0.5 * iteration / self.n_iterations)  # Decrease over time
                    particle.velocity = np.clip(particle.velocity, -v_max, v_max)
                    
                    particle.position = particle.position + particle.velocity
                    particle.position = np.clip(particle.position, 0, 1)
            
            # Inter-swarm migration (every 15 iterations)
            if iteration % 15 == 0 and iteration > 0:
                self._migrate_best_particles(swarms)
            
            # Add to diversity archive
            if iteration % 10 == 0:
                self._update_diversity_archive(
                    swarms, indicators_df, valid_indicators
                )
            
            # Restart worst particles if stagnating
            if iteration > 20 and iteration % 20 == 0:
                self._restart_worst_particles(swarms, n_dims, iteration)
            
            self.fitness_history.append(global_best_fitness)
            
            if progress_callback:
                progress_callback(iteration + 1, self.n_iterations, 
                                global_best_fitness, valid_count)
        
        # Extract diverse rules from archive + top particles
        final_rules = self._extract_final_diverse_rules(
            swarms, indicators_df, valid_indicators
        )
        
        return final_rules
    
    def _migrate_best_particles(self, swarms):
        """Exchange best particles between swarms"""
        n_migrants = 2
        
        for i in range(len(swarms)):
            source_swarm = swarms[i]
            target_swarm = swarms[(i + 1) % len(swarms)]
            
            # Sort by fitness
            sorted_particles = sorted(source_swarm['particles'], 
                                    key=lambda p: p.best_fitness, reverse=True)
            
            # Copy best to target swarm's worst
            target_sorted = sorted(target_swarm['particles'],
                                 key=lambda p: p.best_fitness)
            
            for j in range(min(n_migrants, len(sorted_particles), len(target_sorted))):
                target_sorted[j].position = sorted_particles[j].position.copy()
                target_sorted[j].velocity = sorted_particles[j].velocity.copy() * 0.5
    
    def _restart_worst_particles(self, swarms, n_dims, iteration):
        """Restart worst performing particles to maintain diversity"""
        for swarm in swarms:
            particles = swarm['particles']
            sorted_particles = sorted(particles, key=lambda p: p.best_fitness)
            
            # Restart worst 30%
            n_restart = max(len(particles) // 3, 2)
            
            for i in range(n_restart):
                # Re-initialize with high diversity
                sorted_particles[i].position = np.random.rand(n_dims)
                sorted_particles[i].velocity = np.random.randn(n_dims) * 0.05
                sorted_particles[i].best_fitness = -np.inf
    
    def _update_diversity_archive(self, swarms, indicators_df, valid_indicators):
        """Maintain archive of diverse high-quality rules"""
        candidates = []
        
        for swarm in swarms:
            for particle in swarm['particles']:
                if particle.best_fitness > 0.1:  # Threshold for archive
                    rule = self._decode_position(
                        particle.best_position, indicators_df, valid_indicators
                    )
                    
                    # Extract indicators
                    indicators_used = set()
                    for ind in valid_indicators:
                        if ind in rule:
                            indicators_used.add(ind)
                    
                    candidates.append({
                        'rule': rule,
                        'fitness': particle.best_fitness,
                        'indicators': indicators_used,
                        'position': particle.best_position.copy()
                    })
        
        # Merge with existing archive
        all_candidates = self.diversity_archive + candidates
        
        # Sort by fitness
        all_candidates.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Select diverse subset
        selected = []
        for candidate in all_candidates:
            is_diverse = True
            
            for selected_item in selected:
                # Check indicator overlap
                overlap = len(candidate['indicators'] & selected_item['indicators'])
                max_indicators = max(len(candidate['indicators']), 
                                   len(selected_item['indicators']))
                
                if max_indicators > 0:
                    similarity = overlap / max_indicators
                    if similarity > 0.7:  # Too similar
                        is_diverse = False
                        break
            
            if is_diverse:
                selected.append(candidate)
            
            if len(selected) >= self.archive_size:
                break
        
        self.diversity_archive = selected
    
    def _extract_final_diverse_rules(self, swarms, indicators_df, valid_indicators):
        """Extract final diverse rule set from archive and top particles"""
        # Collect all candidates
        all_candidates = list(self.diversity_archive)
        
        # Add top particles from each swarm
        for swarm in swarms:
            sorted_particles = sorted(swarm['particles'], 
                                    key=lambda p: p.best_fitness, reverse=True)
            
            for particle in sorted_particles[:5]:
                if particle.best_fitness > 0.1:
                    rule = self._decode_position(
                        particle.best_position, indicators_df, valid_indicators
                    )
                    
                    indicators_used = set()
                    for ind in valid_indicators:
                        if ind in rule:
                            indicators_used.add(ind)
                    
                    all_candidates.append({
                        'rule': rule,
                        'fitness': particle.best_fitness,
                        'indicators': indicators_used
                    })
        
        # Remove duplicates and select diverse subset
        all_candidates.sort(key=lambda x: x['fitness'], reverse=True)
        
        final_rules = []
        final_fitness = []
        seen_indicators = []
        
        for candidate in all_candidates:
            is_diverse = True
            
            for seen_set in seen_indicators:
                if candidate['indicators'] == seen_set:
                    is_diverse = False
                    break
            
            if is_diverse or len(final_rules) < 5:  # Always keep top 5
                final_rules.append(candidate['rule'])
                final_fitness.append(candidate['fitness'])
                seen_indicators.append(candidate['indicators'])
            
            if len(final_rules) >= 30:  # Return more rules!
                break
        
        return {
            'rules': final_rules,
            'fitness_scores': final_fitness,
            'history': self.fitness_history,
            'valid_rules_found': self.valid_rules_found,
            'total_evaluations': self.total_evaluations,
            'archive_size': len(self.diversity_archive)
        }
    
    def _evaluate_particle_multiobjective(self, position: np.ndarray, 
                                         indicators_df: pd.DataFrame,
                                         returns: pd.Series, 
                                         indicator_names: List[str]) -> float:
        """Multi-objective fitness: Sharpe + Win Rate + Signal Count + Uniqueness"""
        try:
            rule_condition = self._decode_position(position, indicators_df, indicator_names)
            
            parser = RobustConditionParser()
            signals = parser.evaluate_condition(rule_condition, indicators_df)
            
            if signals.sum() < 3:
                return -500
            
            # Align indices
            common_idx = indicators_df.index.intersection(returns.index)
            if len(common_idx) < 3:
                return -450
            
            aligned_signals = signals[indicators_df.index.isin(common_idx)]
            aligned_returns = returns.loc[common_idx]
            
            signal_returns = aligned_returns[aligned_signals].dropna()
            
            if len(signal_returns) < 3:
                return -400
            
            # Calculate metrics
            mean_return = signal_returns.mean()
            if np.isnan(mean_return) or np.isinf(mean_return):
                return -350
            
            std_return = signal_returns.std()
            if std_return < 1e-8 or np.isnan(std_return):
                std_return = 1.0
            
            sharpe = mean_return / std_return
            win_rate = (signal_returns > 0).mean()
            signal_count = signals.sum()
            
            # Multi-objective fitness with balanced weights
            fitness_sharpe = sharpe * 0.35        # 35% - profitability
            fitness_winrate = win_rate * 0.25     # 25% - consistency
            fitness_signals = min(signal_count / 50, 1.0) * 0.20  # 20% - data sufficiency
            fitness_return = (mean_return / 10) * 0.20  # 20% - absolute returns
            
            # Penalties
            if signal_count < 10:
                fitness_signals -= 0.3
            if abs(sharpe) > 5:  # Extreme values
                fitness_sharpe -= 0.5
            
            total_fitness = fitness_sharpe + fitness_winrate + fitness_signals + fitness_return
            
            return max(min(total_fitness, 5), -5)
            
        except Exception as e:
            return -300
    
    def optimize_rule_parameters(self, 
                                indicators_df: pd.DataFrame,
                                returns: pd.Series,
                                indicator_names: List[str],
                                progress_callback=None) -> Dict:
        """Optimize rule parameters using PSO - RETURN TOP RULES"""
        
        # Filter valid indicators first
        valid_indicators = [ind for ind in indicator_names 
                          if ind in indicators_df.columns 
                          and len(indicators_df[ind].dropna()) >= 20]
        
        if len(valid_indicators) < 2:
            return {
                'rules': [],
                'fitness_scores': [],
                'history': [-999],
                'valid_rules_found': 0,
                'message': f"Not enough valid indicators (found {len(valid_indicators)})"
            }
        
        # Reduce dimensions for better convergence - use 1-2 indicators
        n_conditions = min(2, len(valid_indicators))
        n_dims = n_conditions * 2  # indicator index + threshold for each condition
        
        particles = []
        
        # Initialize particles with MORE diverse positions
        for i in range(self.n_particles):
            # Use FIVE different initialization strategies for more diversity
            strategy = i % 5
            
            if strategy == 0:
                # Low percentiles (bearish)
                position = np.random.uniform(0.0, 0.3, n_dims)
            elif strategy == 1:
                # Mid-low percentiles
                position = np.random.uniform(0.2, 0.5, n_dims)
            elif strategy == 2:
                # Mid percentiles (neutral)
                position = np.random.uniform(0.4, 0.6, n_dims)
            elif strategy == 3:
                # Mid-high percentiles
                position = np.random.uniform(0.5, 0.8, n_dims)
            else:
                # High percentiles (bullish)
                position = np.random.uniform(0.7, 1.0, n_dims)
            
            # Add small noise
            position = np.clip(position + np.random.randn(n_dims) * 0.05, 0, 1)
            velocity = np.random.randn(n_dims) * 0.05
            
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=-np.inf,
                current_fitness=-np.inf
            )
            particles.append(particle)
        
        # Optimization loop
        stagnation_counter = 0
        best_fitness_history = []
        
        for iteration in range(self.n_iterations):
            valid_count = 0
            
            for i, particle in enumerate(particles):
                fitness = self._evaluate_particle(
                    particle.position, indicators_df, returns, valid_indicators
                )
                
                self.total_evaluations += 1
                if fitness > -500:
                    valid_count += 1
                    self.valid_rules_found += 1
                
                particle.current_fitness = fitness
                
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            
            # Update velocities and positions
            for particle in particles:
                r1, r2 = np.random.rand(2)
                
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                
                # Adaptive inertia
                w_adaptive = self.w * (0.9 - 0.4 * iteration / self.n_iterations)
                
                particle.velocity = w_adaptive * particle.velocity + cognitive + social
                
                # Velocity clamping
                v_max = 0.2
                particle.velocity = np.clip(particle.velocity, -v_max, v_max)
                
                particle.position = particle.position + particle.velocity
                particle.position = np.clip(particle.position, 0, 1)
            
            # Add diversity if stagnating
            if stagnation_counter > 10 and iteration < self.n_iterations - 5:
                # Reinitialize worst particles
                worst_indices = np.argsort([p.best_fitness for p in particles])[:self.n_particles // 3]
                for idx in worst_indices:
                    particles[idx].position = np.random.rand(n_dims)
                    particles[idx].velocity = np.random.randn(n_dims) * 0.05
                stagnation_counter = 0
            
            self.fitness_history.append(self.global_best_fitness)
            best_fitness_history.append(self.global_best_fitness)
            
            if progress_callback:
                progress_callback(iteration + 1, self.n_iterations, 
                                self.global_best_fitness, valid_count)
        
        # Return TOP 10 DIVERSE rules instead of just 1
        particles.sort(key=lambda p: p.best_fitness, reverse=True)
        
        top_rules = []
        top_fitness = []
        seen_indicators = []  # Track indicators used, not exact rules
        
        for particle in particles[:min(50, len(particles))]:  # Check top 50
            # STRICTER: Only rules with positive fitness (actual edge)
            if particle.best_fitness > 0.1:  # Must have some positive fitness
                rule = self._decode_position(particle.best_position, indicators_df, valid_indicators)
                
                # Extract indicators from rule for diversity check
                indicators_in_rule = set()
                for ind in valid_indicators:
                    if ind in rule:
                        indicators_in_rule.add(ind)
                
                # Check if this indicator combination is new
                is_diverse = True
                for seen_set in seen_indicators:
                    if indicators_in_rule == seen_set:
                        is_diverse = False
                        break
                
                if is_diverse or len(top_rules) < 3:  # Always allow first 3
                    top_rules.append(rule)
                    top_fitness.append(particle.best_fitness)
                    seen_indicators.append(indicators_in_rule)
                    
                    if len(top_rules) >= 15:  # Collect more rules
                        break
        
        return {
            'rules': top_rules,
            'fitness_scores': top_fitness,
            'history': self.fitness_history,
            'valid_rules_found': self.valid_rules_found,
            'total_evaluations': self.total_evaluations,
            'final_valid_rate': f"{valid_count}/{self.n_particles}"
        }
    
    def _evaluate_particle(self, position: np.ndarray, indicators_df: pd.DataFrame,
                          returns: pd.Series, indicator_names: List[str]) -> float:
        """Evaluate fitness - ANTI-OVERFITTING VERSION"""
        try:
            rule_condition = self._decode_position(position, indicators_df, indicator_names)
            
            parser = RobustConditionParser()
            signals = parser.evaluate_condition(rule_condition, indicators_df)
            
            # Very lenient minimum
            if signals.sum() < 3:
                return -500
            
            # Align indices
            common_idx = indicators_df.index.intersection(returns.index)
            if len(common_idx) < 3:
                return -450
            
            aligned_signals = signals[indicators_df.index.isin(common_idx)]
            aligned_returns = returns.loc[common_idx]
            
            signal_returns = aligned_returns[aligned_signals].dropna()
            
            if len(signal_returns) < 3:
                return -400
            
            # Calculate metrics
            mean_return = signal_returns.mean()
            
            if np.isnan(mean_return) or np.isinf(mean_return):
                return -350
            
            std_return = signal_returns.std()
            if std_return < 1e-8 or np.isnan(std_return):
                std_return = 1.0
            
            sharpe = mean_return / std_return
            win_rate = (signal_returns > 0).mean()
            
            # Penalty for too few signals (overfitting indicator)
            signal_penalty = 0
            if signals.sum() < 10:
                signal_penalty = -0.5
            elif signals.sum() < 20:
                signal_penalty = -0.2
            
            # Penalty for extreme Sharpe (likely overfitting)
            extreme_penalty = 0
            if abs(sharpe) > 5:
                extreme_penalty = -1.0
            elif abs(sharpe) > 3:
                extreme_penalty = -0.5
            
            # Reward consistency and reasonable performance
            base_fitness = sharpe * 0.4 + win_rate * 0.3 + (mean_return / 10) * 0.3
            
            # Apply penalties
            fitness = base_fitness + signal_penalty + extreme_penalty
            
            # More moderate bounds
            return max(min(fitness, 5), -5)
            
        except Exception as e:
            return -300
    
    def _decode_position(self, position: np.ndarray, indicators_df: pd.DataFrame,
                        indicator_names: List[str]) -> str:
        """Decode particle position to trading rule - DETERMINISTIC"""
        n_conditions = len(position) // 2
        conditions = []
        
        valid_indicators = [ind for ind in indicator_names 
                          if ind in indicators_df.columns 
                          and len(indicators_df[ind].dropna()) >= 10]
        
        if not valid_indicators:
            return "RSI_14 > 70"
        
        # Use first dimension to determine number of conditions (deterministic)
        if len(position) >= 2:
            # If first position < 0.6, use 1 condition, else 2
            if position[0] < 0.6:
                n_conditions = 1
            else:
                n_conditions = min(2, len(position) // 2)
        else:
            n_conditions = 1
        
        for i in range(n_conditions):
            idx_offset = i * 2
            if idx_offset + 1 >= len(position):
                break
                
            # Select indicator deterministically
            ind_idx = int(position[idx_offset] * len(valid_indicators))
            ind_idx = min(ind_idx, len(valid_indicators) - 1)
            indicator = valid_indicators[ind_idx]
            
            # Select threshold using percentile
            threshold_pct = position[idx_offset + 1]
            
            # Snap to common percentiles for stability
            if threshold_pct < 0.25:
                threshold_pct = 0.2
            elif threshold_pct < 0.45:
                threshold_pct = 0.3
            elif threshold_pct < 0.55:
                threshold_pct = 0.5
            elif threshold_pct < 0.75:
                threshold_pct = 0.7
            else:
                threshold_pct = 0.8
            
            ind_data = indicators_df[indicator].dropna()
            
            if len(ind_data) < 10:
                continue  # Skip this indicator if insufficient data
            
            # Use percentile for threshold
            threshold = ind_data.quantile(threshold_pct)
            
            # CORRECTED: ENFORCE BOUNDS for bounded indicators (STOCHRSI, RSI, etc.)
            # This prevents impossible thresholds like STOCHRSI > 100
            indicator_upper = indicator.upper()
            if 'STOCHRSI' in indicator_upper or 'RSI' in indicator_upper or 'MFI' in indicator_upper:
                # Bounded 0-100
                threshold = np.clip(threshold, 0, 100)
            elif 'STOCH' in indicator_upper and 'STOCHRSI' not in indicator_upper:
                # Stochastic (not StochRSI) also 0-100
                threshold = np.clip(threshold, 0, 100)
            elif 'WILLR' in indicator_upper:
                # Williams %R: -100 to 0
                threshold = np.clip(threshold, -100, 0)
            elif 'ADX' in indicator_upper or 'AROON' in indicator_upper:
                # ADX and Aroon: 0-100
                threshold = np.clip(threshold, 0, 100)
            
            # Ensure threshold is valid and reasonable
            if np.isnan(threshold) or np.isinf(threshold):
                threshold = ind_data.median()
                if np.isnan(threshold) or np.isinf(threshold):
                    continue  # Skip this indicator
            
            # Skip if indicator has no variance (all same value)
            if ind_data.std() < 1e-10:
                continue
            
            # Skip if threshold is unreasonably small (likely bad indicator)
            if abs(threshold) < 1e-6 and ind_data.std() < 0.01:
                continue
            
            # Choose operator based on threshold position
            operator = '>' if threshold_pct > 0.5 else '<'
            
            # Format threshold properly with reasonable precision
            if abs(threshold) < 0.01:
                threshold_str = f"{threshold:.6f}"
            elif abs(threshold) < 1:
                threshold_str = f"{threshold:.4f}"
            elif abs(threshold) < 100:
                threshold_str = f"{threshold:.2f}"
            else:
                threshold_str = f"{threshold:.1f}"
            
            conditions.append(f"{indicator} {operator} {threshold_str}")
        
        if not conditions:
            # Fallback
            ind = valid_indicators[0]
            threshold = indicators_df[ind].median()
            return f"{ind} > {threshold:.6f}"
        
        # Return simple condition for better stability
        if len(conditions) == 1:
            return conditions[0]
        else:
            return f"({conditions[0]}) AND ({conditions[1]})"


# ===================== GENETIC ALGORITHM =====================

@dataclass
class Chromosome:
    """Chromosome for genetic algorithm"""
    genes: List[str]
    fitness: float = -np.inf
    
class GeneticAlgorithm:
    """Genetic Algorithm for trading rule evolution - IMPROVED"""
    
    def __init__(self, population_size: int = 50, n_generations: int = 30,
                 mutation_rate: float = 0.15, crossover_rate: float = 0.7):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_history = []
        self.best_chromosome = None
        self.valid_rules_found = 0
        self.total_evaluations = 0
    
    def evolve_rules(self, indicators_df: pd.DataFrame, returns: pd.Series,
                    indicator_names: List[str], operators: List[str],
                    percentiles: List[int], progress_callback=None) -> Dict:
        """Evolve trading rules using genetic algorithm - IMPROVED"""
        
        if not operators:
            operators = ['>', '<']
        if not percentiles:
            percentiles = [10, 30, 50, 70, 90]
        if not indicator_names:
            raise ValueError("No indicators provided for genetic algorithm")
        
        # Filter valid indicators
        valid_indicators = [ind for ind in indicator_names 
                          if ind in indicators_df.columns 
                          and len(indicators_df[ind].dropna()) >= 10]
        
        if len(valid_indicators) < 2:
            return {
                'rule': "RSI_14 > 70",
                'fitness': -999,
                'history': [-999],
                'valid_rules_found': 0,
                'message': f"Not enough valid indicators (found {len(valid_indicators)})"
            }
        
        try:
            population = self._initialize_population(
                indicators_df, valid_indicators, operators, percentiles
            )
        except ValueError as e:
            return {
                'rule': "RSI_14 > 70",
                'fitness': -999,
                'history': [-999],
                'valid_rules_found': 0,
                'message': str(e)
            }
        
        # Evolution loop
        for generation in range(self.n_generations):
            valid_count = 0
            
            # Evaluate fitness
            for chromosome in population:
                if chromosome.fitness == -np.inf:
                    self.total_evaluations += 1
                    fitness = self._evaluate_fitness(chromosome, indicators_df, returns)
                    chromosome.fitness = fitness
                    
                    if fitness > -500:
                        valid_count += 1
                        self.valid_rules_found += 1
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best
            if population[0].fitness > (self.best_chromosome.fitness if self.best_chromosome else -np.inf):
                self.best_chromosome = Chromosome(
                    genes=population[0].genes.copy(),
                    fitness=population[0].fitness
                )
            
            self.fitness_history.append(population[0].fitness)
            
            # Elite selection - keep top 20%
            elite_size = max(2, self.population_size // 5)
            new_population = [Chromosome(genes=c.genes.copy(), fitness=c.fitness) 
                            for c in population[:elite_size]]
            
            # Generate offspring
            attempts = 0
            max_attempts = self.population_size * 10
            
            while len(new_population) < self.population_size and attempts < max_attempts:
                attempts += 1
                
                parent1 = self._tournament_select(population, tournament_size=3)
                parent2 = self._tournament_select(population, tournament_size=3)
                
                if np.random.rand() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = Chromosome(genes=parent1.genes.copy())
                
                if np.random.rand() < self.mutation_rate:
                    child = self._mutate(child, indicators_df, valid_indicators, 
                                       operators, percentiles)
                
                # Ensure child has genes
                if child.genes:
                    new_population.append(child)
            
            # Fill remaining with random if needed
            while len(new_population) < self.population_size:
                try:
                    random_chromo = self._create_random_chromosome(
                        indicators_df, valid_indicators, operators, percentiles
                    )
                    new_population.append(random_chromo)
                except:
                    break
            
            population = new_population[:self.population_size]
            
            if progress_callback:
                progress_callback(generation + 1, self.n_generations, 
                                population[0].fitness, valid_count)
        
        if not self.best_chromosome or not self.best_chromosome.genes:
            return {
                'rules': [],
                'fitness_scores': [],
                'history': self.fitness_history,
                'valid_rules_found': self.valid_rules_found,
                'message': "No valid rules evolved"
            }
        
        # Return TOP 10 DIVERSE rules
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        top_rules = []
        top_fitness = []
        seen_indicators = []  # Track indicators used
        
        for chromosome in population[:min(50, len(population))]:
            # STRICTER: Only rules with positive fitness
            if chromosome.fitness > 0.1 and chromosome.genes:
                rule_str = " AND ".join(f"({g})" for g in chromosome.genes)
                
                # Extract indicators from rule
                indicators_in_rule = set()
                for gene in chromosome.genes:
                    # Extract indicator name from gene (before operator)
                    parts = gene.split()
                    if len(parts) >= 1:
                        indicators_in_rule.add(parts[0])
                
                # Check if this indicator combination is new
                is_diverse = True
                for seen_set in seen_indicators:
                    if indicators_in_rule == seen_set:
                        is_diverse = False
                        break
                
                if is_diverse or len(top_rules) < 3:  # Always allow first 3
                    top_rules.append(rule_str)
                    top_fitness.append(chromosome.fitness)
                    seen_indicators.append(indicators_in_rule)
                    
                    if len(top_rules) >= 15:  # Collect more rules
                        break
        
        return {
            'rules': top_rules,
            'fitness_scores': top_fitness,
            'history': self.fitness_history,
            'valid_rules_found': self.valid_rules_found,
            'total_evaluations': self.total_evaluations
        }
    
    def _create_random_chromosome(self, indicators_df: pd.DataFrame,
                                 indicator_names: List[str], operators: List[str],
                                 percentiles: List[int]) -> Chromosome:
        """Create a single random chromosome - PREFER SIMPLE RULES"""
        # 70% chance of single condition, 30% chance of 2 conditions
        if np.random.rand() < 0.7:
            n_conditions = 1
        else:
            n_conditions = 2
            
        genes = []
        
        # Prefer common percentiles to avoid overfitting
        common_percentiles = [20, 30, 50, 70, 80]
        
        for _ in range(n_conditions):
            indicator = np.random.choice(indicator_names)
            operator = np.random.choice(operators)
            percentile = np.random.choice(common_percentiles if np.random.rand() < 0.7 else percentiles)
            
            ind_data = indicators_df[indicator].dropna()
            
            if len(ind_data) >= 10:
                threshold = ind_data.quantile(percentile / 100)
                
                if not np.isnan(threshold) and not np.isinf(threshold):
                    if abs(threshold) < 0.001:
                        threshold_str = f"{threshold:.8f}"
                    elif abs(threshold) < 1:
                        threshold_str = f"{threshold:.6f}"
                    else:
                        threshold_str = f"{threshold:.4f}"
                    
                    genes.append(f"{indicator} {operator} {threshold_str}")
        
        if not genes:
            # Fallback
            ind = indicator_names[0]
            threshold = indicators_df[ind].median()
            genes = [f"{ind} > {threshold:.6f}"]
        
        return Chromosome(genes=genes)
    
    def _initialize_population(self, indicators_df: pd.DataFrame,
                              indicator_names: List[str], operators: List[str],
                              percentiles: List[int]) -> List[Chromosome]:
        """Initialize random population with HIGH DIVERSITY"""
        population = []
        
        # Ensure we have variety in percentiles
        common_percentiles = [20, 30, 50, 70, 80]
        all_percentiles = percentiles
        
        attempts = 0
        max_attempts = self.population_size * 10
        
        # Create population with forced diversity
        for i in range(self.population_size):
            if attempts >= max_attempts:
                break
            
            attempts += 1
            
            try:
                # Vary complexity: 50% single, 40% double, 10% triple
                rand = np.random.rand()
                if rand < 0.5:
                    n_conditions = 1
                elif rand < 0.9:
                    n_conditions = 2
                else:
                    n_conditions = 3
                
                # Use different percentile strategies for diversity
                if i % 3 == 0:
                    # Extreme percentiles
                    pcts = [10, 20, 80, 90]
                elif i % 3 == 1:
                    # Middle percentiles
                    pcts = [30, 40, 50, 60, 70]
                else:
                    # All percentiles
                    pcts = all_percentiles
                
                chromosome = self._create_random_chromosome_with_percentiles(
                    indicators_df, indicator_names, operators, pcts, n_conditions
                )
                population.append(chromosome)
            except:
                continue
        
        # Fill remaining with copies if needed
        while len(population) < self.population_size:
            idx = len(population) % len(population) if population else 0
            if population:
                population.append(Chromosome(genes=population[idx].genes.copy()))
            else:
                break
        
        return population
    
    def _create_random_chromosome_with_percentiles(self, indicators_df: pd.DataFrame,
                                 indicator_names: List[str], operators: List[str],
                                 percentiles: List[int], n_conditions: int) -> Chromosome:
        """Create chromosome with specific percentiles"""
        genes = []
        
        for _ in range(n_conditions):
            indicator = np.random.choice(indicator_names)
            operator = np.random.choice(operators)
            percentile = np.random.choice(percentiles)
            
            ind_data = indicators_df[indicator].dropna()
            
            if len(ind_data) >= 10:
                threshold = ind_data.quantile(percentile / 100)
                
                if not np.isnan(threshold) and not np.isinf(threshold):
                    if abs(threshold) < 0.001:
                        threshold_str = f"{threshold:.8f}"
                    elif abs(threshold) < 1:
                        threshold_str = f"{threshold:.6f}"
                    else:
                        threshold_str = f"{threshold:.4f}"
                    
                    genes.append(f"{indicator} {operator} {threshold_str}")
        
        if not genes:
            # Fallback
            ind = indicator_names[0]
            threshold = indicators_df[ind].median()
            genes = [f"{ind} > {threshold:.6f}"]
        
        return Chromosome(genes=genes)
    
    def _evaluate_fitness(self, chromosome: Chromosome, indicators_df: pd.DataFrame,
                         returns: pd.Series) -> float:
        """Evaluate fitness - ANTI-OVERFITTING VERSION"""
        try:
            rule = " AND ".join(f"({g})" for g in chromosome.genes)
            
            parser = RobustConditionParser()
            signals = parser.evaluate_condition(rule, indicators_df)
            
            # Very lenient minimum
            if signals.sum() < 3:
                return -500
            
            # Align indices
            common_idx = indicators_df.index.intersection(returns.index)
            if len(common_idx) < 3:
                return -450
            
            aligned_signals = signals[indicators_df.index.isin(common_idx)]
            aligned_returns = returns.loc[common_idx]
            
            signal_returns = aligned_returns[aligned_signals].dropna()
            
            if len(signal_returns) < 3:
                return -400
            
            # Calculate metrics
            mean_return = signal_returns.mean()
            
            if np.isnan(mean_return) or np.isinf(mean_return):
                return -350
            
            std_return = signal_returns.std()
            if std_return < 1e-8 or np.isnan(std_return):
                std_return = 1.0
            
            sharpe = mean_return / std_return
            win_rate = (signal_returns > 0).mean()
            
            # Penalty for too few signals (overfitting indicator)
            signal_penalty = 0
            if signals.sum() < 10:
                signal_penalty = -0.5
            elif signals.sum() < 20:
                signal_penalty = -0.2
            
            # Penalty for extreme Sharpe (likely overfitting)
            extreme_penalty = 0
            if abs(sharpe) > 5:
                extreme_penalty = -1.0
            elif abs(sharpe) > 3:
                extreme_penalty = -0.5
            
            # Penalty for complex rules (more conditions = more overfitting)
            complexity_penalty = -0.1 * (len(chromosome.genes) - 1)
            
            # Reward consistency and reasonable performance
            base_fitness = sharpe * 0.4 + win_rate * 0.3 + (mean_return / 10) * 0.3
            
            # Apply penalties
            fitness = base_fitness + signal_penalty + extreme_penalty + complexity_penalty
            
            # More moderate bounds
            return max(min(fitness, 5), -5)
            
        except Exception as e:
            return -300
    
    def _tournament_select(self, population: List[Chromosome], 
                          tournament_size: int = 3) -> Chromosome:
        """Tournament selection"""
        if len(population) == 0:
            raise ValueError("Cannot select from empty population")
        
        actual_tournament_size = min(tournament_size, len(population))
        
        tournament = np.random.choice(population, actual_tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """Single-point crossover"""
        if not parent1.genes or not parent2.genes:
            return Chromosome(genes=parent1.genes if parent1.genes else parent2.genes)
        
        if len(parent1.genes) == 1 and len(parent2.genes) == 1:
            return Chromosome(genes=parent1.genes if np.random.rand() > 0.5 else parent2.genes)
        
        min_len = min(len(parent1.genes), len(parent2.genes))
        if min_len <= 1:
            return Chromosome(genes=parent1.genes + parent2.genes)
        
        split = np.random.randint(1, min_len)
        child_genes = parent1.genes[:split] + parent2.genes[split:]
        return Chromosome(genes=child_genes)
    
    def _mutate(self, chromosome: Chromosome, indicators_df: pd.DataFrame,
               indicator_names: List[str], operators: List[str],
               percentiles: List[int]) -> Chromosome:
        """Mutate chromosome"""
        if not chromosome.genes:
            indicator = np.random.choice([ind for ind in indicator_names 
                                         if ind in indicators_df.columns])
            if indicator in indicators_df.columns:
                operator = np.random.choice(operators)
                percentile = np.random.choice(percentiles)
                ind_data = indicators_df[indicator].dropna()
                
                if len(ind_data) >= 10:
                    threshold = ind_data.quantile(percentile / 100)
                    chromosome.genes.append(f"{indicator} {operator} {threshold:.6f}")
            
            chromosome.fitness = -np.inf
            return chromosome
        
        mutation_type = np.random.choice(['change', 'add', 'remove'])
        
        if mutation_type == 'change':
            idx = np.random.randint(len(chromosome.genes))
            valid_indicators = [ind for ind in indicator_names 
                              if ind in indicators_df.columns]
            
            if valid_indicators:
                indicator = np.random.choice(valid_indicators)
                operator = np.random.choice(operators)
                percentile = np.random.choice(percentiles)
                ind_data = indicators_df[indicator].dropna()
                
                if len(ind_data) >= 10:
                    threshold = ind_data.quantile(percentile / 100)
                    chromosome.genes[idx] = f"{indicator} {operator} {threshold:.6f}"
        
        elif mutation_type == 'add' and len(chromosome.genes) < 3:
            valid_indicators = [ind for ind in indicator_names 
                              if ind in indicators_df.columns]
            
            if valid_indicators:
                indicator = np.random.choice(valid_indicators)
                operator = np.random.choice(operators)
                percentile = np.random.choice(percentiles)
                ind_data = indicators_df[indicator].dropna()
                
                if len(ind_data) >= 10:
                    threshold = ind_data.quantile(percentile / 100)
                    chromosome.genes.append(f"{indicator} {operator} {threshold:.6f}")
        
        elif mutation_type == 'remove' and len(chromosome.genes) > 1:
            idx = np.random.randint(len(chromosome.genes))
            chromosome.genes.pop(idx)
        
        chromosome.fitness = -np.inf
        return chromosome


# ===================== RULE PARSING AND VALIDATION =====================

class RobustConditionParser:
    """Enhanced parser for trading rule conditions"""
    
    COMPARISON_OPS = {
        '>': lambda a, b: a > b,
        '>=': lambda a, b: a >= b,
        '<': lambda a, b: a < b,
        '<=': lambda a, b: a <= b,
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b
    }
    
    LOGICAL_OPS = {
        'AND': lambda a, b: a & b,
        'OR': lambda a, b: a | b
    }
    
    @classmethod
    def tokenize_condition(cls, condition: str) -> List[Tuple[str, str]]:
        """Tokenize a condition string"""
        condition = condition.strip()
        if condition.startswith('(') and condition.endswith(')'):
            condition = condition[1:-1]
        
        tokens = []
        logical_positions = []
        for op in ['AND', 'OR']:
            pattern = rf'\s+{op}\s+'
            for match in re.finditer(pattern, condition):
                logical_positions.append((match.start(), match.end(), op))
        
        logical_positions.sort()
        
        last_pos = 0
        for start, end, op in logical_positions:
            sub_condition = condition[last_pos:start].strip()
            if sub_condition:
                sub_condition = sub_condition.strip('()')
                tokens.append(('CONDITION', sub_condition))
            tokens.append(('LOGICAL', op))
            last_pos = end
        
        remaining = condition[last_pos:].strip()
        if remaining:
            remaining = remaining.strip('()')
            tokens.append(('CONDITION', remaining))
        
        return tokens
    
    @classmethod
    def parse_simple_condition(cls, condition: str, data_df: pd.DataFrame) -> np.ndarray:
        """Parse a simple condition"""
        try:
            pattern = r'^([A-Za-z_][A-Za-z0-9_]*)\s*([><=!]+)\s*(-?\d+\.?\d*)$'
            match = re.match(pattern, condition.strip())
            
            if not match:
                return np.zeros(len(data_df), dtype=bool)
            
            indicator, operator, value = match.groups()
            
            if operator not in cls.COMPARISON_OPS or indicator not in data_df.columns:
                return np.zeros(len(data_df), dtype=bool)
            
            indicator_values = data_df[indicator].values
            threshold = float(value)
            
            valid_mask = ~np.isnan(indicator_values)
            result = np.zeros(len(data_df), dtype=bool)
            
            comparison_func = cls.COMPARISON_OPS[operator]
            result[valid_mask] = comparison_func(indicator_values[valid_mask], threshold)
            
            return result
        except:
            return np.zeros(len(data_df), dtype=bool)
    
    @classmethod
    def evaluate_condition(cls, condition: str, data_df: pd.DataFrame) -> np.ndarray:
        """Evaluate complete condition"""
        try:
            tokens = cls.tokenize_condition(condition)
            
            if not tokens:
                return np.zeros(len(data_df), dtype=bool)
            
            if len(tokens) == 1 and tokens[0][0] == 'CONDITION':
                return cls.parse_simple_condition(tokens[0][1], data_df)
            
            result = None
            pending_logical_op = None
            
            for token_type, value in tokens:
                if token_type == 'CONDITION':
                    condition_result = cls.parse_simple_condition(value, data_df)
                    
                    if result is None:
                        result = condition_result
                    elif pending_logical_op:
                        logical_func = cls.LOGICAL_OPS[pending_logical_op]
                        result = logical_func(result, condition_result)
                        pending_logical_op = None
                
                elif token_type == 'LOGICAL':
                    pending_logical_op = value
            
            return result if result is not None else np.zeros(len(data_df), dtype=bool)
        except:
            return np.zeros(len(data_df), dtype=bool)
    
    @classmethod
    def validate_condition_syntax(cls, condition: str, available_indicators: List[str]) -> Tuple[bool, str]:
        """Validate condition syntax and REJECT impossible values"""
        try:
            if condition.count('(') != condition.count(')'):
                return False, "Unbalanced parentheses"
            
            tokens = cls.tokenize_condition(condition)
            
            if not tokens:
                return False, "Empty condition"
            
            for token_type, value in tokens:
                if token_type == 'CONDITION':
                    pattern = r'^([A-Za-z_][A-Za-z0-9_]*)\s*([><=!]+)\s*(-?\d+\.?\d*)$'
                    match = re.match(pattern, value.strip())
                    
                    if not match:
                        return False, f"Invalid condition format: {value}"
                    
                    indicator, operator, threshold_str = match.groups()
                    
                    if indicator not in available_indicators:
                        return False, f"Unknown indicator: {indicator}"
                    
                    if operator not in cls.COMPARISON_OPS:
                        return False, f"Invalid operator: {operator}"
                    
                    # CORRECTED: REJECT impossible threshold values for bounded indicators
                    try:
                        threshold = float(threshold_str)
                        indicator_upper = indicator.upper()
                        
                        # Check STOCHRSI first (before RSI)
                        if 'STOCHRSI' in indicator_upper:
                            if operator in ['>', '>='] and threshold > 100:
                                return False, f"IMPOSSIBLE: {indicator} {operator} {threshold} (max 100)"
                            if operator in ['<', '<='] and threshold < 0:
                                return False, f"IMPOSSIBLE: {indicator} {operator} {threshold} (min 0)"
                        
                        # RSI, MFI, STOCH (0-100)
                        elif any(x in indicator_upper for x in ['RSI', 'MFI']) or \
                             ('STOCH' in indicator_upper and 'STOCHRSI' not in indicator_upper):
                            if operator in ['>', '>='] and threshold > 100:
                                return False, f"IMPOSSIBLE: {indicator} {operator} {threshold} (max 100)"
                            if operator in ['<', '<='] and threshold < 0:
                                return False, f"IMPOSSIBLE: {indicator} {operator} {threshold} (min 0)"
                        
                        # Williams %R (-100 to 0)
                        elif 'WILLR' in indicator_upper:
                            if operator in ['>', '>='] and threshold > 0:
                                return False, f"IMPOSSIBLE: {indicator} {operator} {threshold} (max 0)"
                            if operator in ['<', '<='] and threshold < -100:
                                return False, f"IMPOSSIBLE: {indicator} {operator} {threshold} (min -100)"
                        
                        # ADX, AROON (0-100)
                        elif any(x in indicator_upper for x in ['ADX', 'AROON']):
                            if operator in ['>', '>='] and threshold > 100:
                                return False, f"IMPOSSIBLE: {indicator} {operator} {threshold} (max 100)"
                            if operator in ['<', '<='] and threshold < 0:
                                return False, f"IMPOSSIBLE: {indicator} {operator} {threshold} (min 0)"
                        
                        # Reject excessive decimals (likely overfitting)
                        if '.' in threshold_str and len(threshold_str.split('.')[1]) > 4:
                            return False, f"Too many decimals: {threshold_str} (max 4)"
                    
                    except ValueError:
                        return False, f"Invalid threshold: {threshold_str}"
                
                elif token_type == 'LOGICAL':
                    if value not in cls.LOGICAL_OPS:
                        return False, f"Invalid logical operator: {value}"
            
            return True, "Valid"
        except Exception as e:
            return False, f"Validation error: {str(e)}"


class StatisticalValidator:
    """Statistical significance testing"""
    
    @staticmethod
    def calculate_significance(signal_returns: pd.Series, 
                             confidence_level: float = 0.95) -> Dict:
        """Calculate statistical significance"""
        if len(signal_returns) < 5:
            return {
                'p_value': 1.0,
                'is_significant': False,
                'ci_lower': 0,
                'ci_upper': 0,
                't_statistic': 0
            }
        
        t_stat, p_value = stats.ttest_1samp(signal_returns, 0)
        
        alpha = 1 - confidence_level
        ci = stats.t.interval(confidence_level, len(signal_returns)-1,
                            loc=signal_returns.mean(),
                            scale=stats.sem(signal_returns))
        
        return {
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            't_statistic': t_stat
        }
    
    @staticmethod
    def apply_multiple_testing_correction(p_values: np.ndarray, 
                                         alpha: float = 0.05,
                                         method: str = 'bonferroni') -> np.ndarray:
        """Apply multiple testing correction"""
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            corrected_alpha = alpha / n_tests
            return p_values < corrected_alpha
        
        elif method == 'fdr':
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            
            comparisons = sorted_p <= (np.arange(1, n_tests + 1) / n_tests) * alpha
            
            if comparisons.any():
                max_idx = np.where(comparisons)[0][-1]
                threshold = sorted_p[max_idx]
                return p_values <= threshold
            else:
                return np.zeros(n_tests, dtype=bool)
        
        return np.zeros(n_tests, dtype=bool)


class ImprovedMassiveRuleGenerator:
    """
    IMPROVED Brute Force Rule Generator
    - Detects indicator types (binary, bounded, oscillator, continuous)
    - Generates meaningful thresholds based on technical analysis standards
    - Eliminates duplicates with similarity check
    - Validates rules before accepting
    """
    
    def __init__(self):
        self.accepted_rules = []
        self.indicator_types = {}
    
    def _get_indicator_type(self, ind_data: pd.Series, ind_name: str) -> str:
        """
        Classify indicator type to generate appropriate thresholds
        
        Types:
        - binary: Candlestick patterns (0, ±100)
        - bounded: RSI, Stochastic (0-100)
        - oscillator: MACD, CCI (-∞ to +∞, crosses zero)
        - continuous: Moving averages, price-based
        """
        unique_vals = ind_data.nunique()
        data_range = ind_data.max() - ind_data.min()
        min_val = ind_data.min()
        max_val = ind_data.max()
        
        # Binary (Candlestick patterns)
        if unique_vals <= 5 and data_range <= 200:
            return 'binary'
        
        # CORRECTED: Bounded (RSI, Stochastic, etc.) - check STOCHRSI FIRST before generic patterns
        # This prevents STOCHRSI from being misclassified
        if 'STOCHRSI' in ind_name.upper():
            # StochRSI is ALWAYS bounded 0-100
            return 'bounded'
        elif any(x in ind_name.upper() for x in ['RSI', 'STOCH', 'WILLR', 'MFI', 'ADX', 'AROON']):
            if 0 <= min_val and max_val <= 100:
                return 'bounded'
        
        # Oscillator (MACD, CCI, etc.) - crosses zero
        if min_val < 0 and max_val > 0:
            if any(x in ind_name.upper() for x in ['MACD', 'CCI', 'MOM', 'ROC', 'PPO', 'BOP']):
                return 'oscillator'
        
        # Continuous (everything else)
        return 'continuous'
    
    def _get_meaningful_thresholds(self, ind_data: pd.Series, ind_name: str, 
                                   ind_type: str) -> List[Tuple[str, float]]:
        """
        Generate meaningful thresholds based on indicator type and TA standards
        
        Returns: List of (operator, threshold) tuples
        """
        thresholds = []
        
        if ind_type == 'binary':
            # For candlestick patterns: only check if pattern exists
            # Pattern values are 0, 100, or -100
            thresholds = [
                ('!=', 0),      # Pattern present (any direction)
                ('>', 0),       # Bullish pattern
                ('<', 0),       # Bearish pattern
            ]
        
        elif ind_type == 'bounded':
            # For RSI, Stochastic, etc. - use standard TA levels
            # CORRECTED: Handle STOCHRSI explicitly (0-100 range)
            if 'STOCHRSI' in ind_name.upper():
                thresholds = [
                    ('<', 20),   # Extreme oversold
                    ('<', 30),   # Oversold
                    ('>', 70),   # Overbought
                    ('>', 80),   # Extreme overbought
                ]
            elif 'RSI' in ind_name.upper():
                thresholds = [
                    ('<', 20),   # Extreme oversold
                    ('<', 30),   # Oversold (standard)
                    ('>', 70),   # Overbought (standard)
                    ('>', 80),   # Extreme overbought
                ]
            elif 'STOCH' in ind_name.upper():
                thresholds = [
                    ('<', 20),   # Oversold
                    ('<', 30),
                    ('>', 70),   # Overbought
                    ('>', 80),
                ]
            elif 'WILLR' in ind_name.upper():  # Williams %R (inverted, -100 to 0)
                thresholds = [
                    ('<', -80),  # Oversold
                    ('<', -70),
                    ('>', -30),  # Overbought
                    ('>', -20),
                ]
            elif 'ADX' in ind_name.upper():  # Trend strength
                thresholds = [
                    ('>', 20),   # Trend starting
                    ('>', 25),   # Strong trend
                    ('>', 30),
                    ('>', 40),   # Very strong trend
                ]
            elif 'MFI' in ind_name.upper():  # Money Flow Index
                thresholds = [
                    ('<', 20),   # Oversold
                    ('<', 30),
                    ('>', 70),   # Overbought
                    ('>', 80),
                ]
            else:
                # Generic bounded indicator
                thresholds = [
                    ('<', 30),
                    ('>', 70),
                ]
        
        elif ind_type == 'oscillator':
            # For MACD, CCI, etc. - use percentiles but round to reasonable values
            data_clean = ind_data.dropna()
            if len(data_clean) < 50:
                return []
            
            # Use standard deviations or percentiles
            std_val = data_clean.std()
            mean_val = data_clean.mean()
            
            if 'MACD' in ind_name.upper():
                # MACD: use small multiples of std
                thresholds = [
                    ('>', std_val * 0.5),
                    ('>', std_val * 1.0),
                    ('<', -std_val * 0.5),
                    ('<', -std_val * 1.0),
                ]
            elif 'CCI' in ind_name.upper():
                # CCI: standard levels are ±100, ±200
                thresholds = [
                    ('>', 100),
                    ('>', 200),
                    ('<', -100),
                    ('<', -200),
                ]
            else:
                # Generic oscillator
                p10 = data_clean.quantile(0.10)
                p25 = data_clean.quantile(0.25)
                p75 = data_clean.quantile(0.75)
                p90 = data_clean.quantile(0.90)
                
                thresholds = [
                    ('>', p75),
                    ('>', p90),
                    ('<', p25),
                    ('<', p10),
                ]
        
        else:  # continuous
            # For price-based indicators (SMA, EMA, etc.)
            # Use percentiles but with appropriate precision
            data_clean = ind_data.dropna()
            if len(data_clean) < 50:
                return []
            
            # Calculate percentiles
            p10 = data_clean.quantile(0.10)
            p20 = data_clean.quantile(0.20)
            p30 = data_clean.quantile(0.30)
            p70 = data_clean.quantile(0.70)
            p80 = data_clean.quantile(0.80)
            p90 = data_clean.quantile(0.90)
            
            # Determine appropriate rounding based on magnitude
            magnitude = abs(data_clean.mean())
            if magnitude > 1000:
                decimals = 0
            elif magnitude > 100:
                decimals = 1
            elif magnitude > 10:
                decimals = 2
            else:
                decimals = 3
            
            thresholds = [
                ('<', round(p10, decimals)),
                ('<', round(p20, decimals)),
                ('<', round(p30, decimals)),
                ('>', round(p70, decimals)),
                ('>', round(p80, decimals)),
                ('>', round(p90, decimals)),
            ]
        
        # Filter out any thresholds that don't make sense
        valid_thresholds = []
        for op, thresh in thresholds:
            # Skip if threshold is extreme or nonsensical
            if ind_type == 'bounded':
                if thresh < -10 or thresh > 110:
                    continue
            valid_thresholds.append((op, thresh))
        
        return valid_thresholds
    
    def _is_trivial_rule(self, condition: str, ind_data: pd.Series, ind_type: str) -> bool:
        """
        Check if rule is trivial (always true or always false)
        """
        # Check for "0.000000" in patterns
        if ind_type == 'binary' and '0.000000' in condition:
            return True
        
        # Check for impossible thresholds in bounded indicators
        if ind_type == 'bounded':
            if '> 100' in condition or '< 0' in condition:
                return True
            # Check for thresholds beyond reasonable range
            import re
            numbers = re.findall(r'[<>]=?\s*([-+]?\d+\.?\d*)', condition)
            for num_str in numbers:
                num = float(num_str)
                if num > 100 or num < 0:
                    return True
        
        # Check for overly precise thresholds (likely overfit)
        if '.' in condition:
            import re
            decimals = re.findall(r'\d+\.(\d+)', condition)
            for dec in decimals:
                if len(dec) > 4:  # More than 4 decimal places
                    return True
        
        return False
    
    def _calculate_rule_similarity(self, rule1: Dict, rule2: Dict) -> float:
        """
        Calculate similarity between two rules
        Returns: similarity score (0-1)
        """
        # Extract indicators from conditions
        import re
        
        def extract_indicators(condition):
            # Remove operators and numbers, keep only indicator names
            cleaned = re.sub(r'[<>=!]+\s*[-+]?\d+\.?\d*', '', condition)
            cleaned = re.sub(r'[()]+', '', cleaned)
            cleaned = re.sub(r'\s+(AND|OR)\s+', ' ', cleaned)
            return set(cleaned.split())
        
        ind1 = extract_indicators(rule1['condition'])
        ind2 = extract_indicators(rule2['condition'])
        
        if len(ind1) == 0 or len(ind2) == 0:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(ind1 & ind2)
        union = len(ind1 | ind2)
        
        return intersection / union if union > 0 else 0.0
    
    def _is_duplicate(self, new_rule: Dict, similarity_threshold: float = 0.7) -> bool:
        """
        Check if rule is too similar to existing rules
        """
        for existing_rule in self.accepted_rules:
            similarity = self._calculate_rule_similarity(new_rule, existing_rule)
            if similarity > similarity_threshold:
                return True
        return False
    
    def generate_simple_rules(self,
                             indicators_df: pd.DataFrame,
                             percentiles: List[int],  # Now ignored - we use smart thresholds
                             operators: List[str],     # Now ignored - determined by type
                             max_indicators: int = None) -> List[Dict]:
        """
        Generate SMART simple rules with meaningful thresholds
        """
        rules = []
        self.accepted_rules = []  # Reset
        
        columns = indicators_df.columns[:max_indicators] if max_indicators else indicators_df.columns
        
        print(f"\n🔍 Generating smart rules for {len(columns)} indicators...")
        
        for idx, col in enumerate(columns):
            data = indicators_df[col].dropna()
            if len(data) < 50:
                continue
            
            # Classify indicator type
            ind_type = self._get_indicator_type(data, col)
            self.indicator_types[col] = ind_type
            
            # Get meaningful thresholds for this type
            thresholds = self._get_meaningful_thresholds(data, col, ind_type)
            
            if not thresholds:
                continue
            
            # Generate rules
            for op, threshold in thresholds:
                # Format threshold appropriately
                if ind_type == 'binary' or (ind_type == 'bounded' and threshold in [0, 20, 30, 70, 80, 100]):
                    # Use integers for standard levels
                    threshold_str = f"{int(threshold)}"
                elif ind_type == 'continuous':
                    # Already rounded in _get_meaningful_thresholds
                    if abs(threshold) > 100:
                        threshold_str = f"{threshold:.1f}"
                    elif abs(threshold) > 10:
                        threshold_str = f"{threshold:.2f}"
                    else:
                        threshold_str = f"{threshold:.3f}"
                else:
                    threshold_str = f"{threshold:.2f}"
                
                condition = f"{col} {op} {threshold_str}"
                
                # Check if trivial
                if self._is_trivial_rule(condition, data, ind_type):
                    continue
                
                # Determine trade type based on operator and indicator type
                if ind_type == 'binary':
                    if op == '>':
                        trade_type = 'BUY'
                    elif op == '<':
                        trade_type = 'SELL'
                    else:  # !=
                        trade_type = 'BUY'  # Default
                elif ind_type in ['bounded', 'oscillator']:
                    if op in ['<', '<=']:
                        trade_type = 'BUY'  # Low values = oversold = buy
                    else:
                        trade_type = 'SELL'  # High values = overbought = sell
                else:  # continuous
                    # For price-based indicators, low = buy, high = sell
                    if op in ['<', '<=']:
                        trade_type = 'BUY'
                    else:
                        trade_type = 'SELL'
                
                rule = {
                    'condition': condition,
                    'indicator': col,
                    'operator': op,
                    'threshold': threshold,
                    'threshold_str': threshold_str,
                    'type': 'simple',
                    'trade_type': trade_type,
                    'indicator_type': ind_type
                }
                
                # Check for duplicates
                if not self._is_duplicate(rule, similarity_threshold=0.9):  # Very strict for simple rules
                    rules.append(rule)
                    self.accepted_rules.append(rule)
            
            # Progress update every 20 indicators
            if (idx + 1) % 20 == 0:
                print(f"  ✓ Processed {idx + 1}/{len(columns)} indicators, {len(rules)} rules generated")
        
        print(f"✅ Generated {len(rules)} SMART simple rules (no duplicates, no trivial rules)\n")
        
        # Print statistics
        type_counts = {}
        for rule in rules:
            ind_type = rule.get('indicator_type', 'unknown')
            type_counts[ind_type] = type_counts.get(ind_type, 0) + 1
        
        print("📊 Rules by indicator type:")
        for ind_type, count in type_counts.items():
            print(f"  • {ind_type}: {count} rules")
        print()
        
        return rules
    
    def generate_compound_rules(self,
                               simple_rules: List[Dict],
                               max_depth: int = 2,
                               max_combinations: int = 10000) -> List[Dict]:
        """
        Generate SMART compound rules (avoiding similar indicator combinations)
        """
        compound_rules = []
        
        # Separate by trade type
        buy_rules = [r for r in simple_rules if r['trade_type'] == 'BUY']
        sell_rules = [r for r in simple_rules if r['trade_type'] == 'SELL']
        
        print(f"🔗 Generating compound rules from {len(buy_rules)} BUY and {len(sell_rules)} SELL rules...")
        
        # Sample if too many rules
        if len(buy_rules) > 150:
            buy_rules = np.random.choice(buy_rules, 150, replace=False).tolist()
        if len(sell_rules) > 150:
            sell_rules = np.random.choice(sell_rules, 150, replace=False).tolist()
        
        attempts = 0
        max_attempts = max_combinations * 3
        
        # Generate BUY compounds
        for r1, r2 in combinations(buy_rules, 2):
            attempts += 1
            if attempts > max_attempts:
                break
            
            # Skip if same indicator
            if r1['indicator'] == r2['indicator']:
                continue
            
            # Create AND rule
            and_rule = {
                'condition': f"({r1['condition']}) AND ({r2['condition']})",
                'type': 'compound-2',
                'logic': 'AND',
                'trade_type': 'BUY',
                'indicators': [r1['indicator'], r2['indicator']],
                'indicator_types': [r1.get('indicator_type', 'unknown'), 
                                   r2.get('indicator_type', 'unknown')]
            }
            
            # Check duplicates
            if not self._is_duplicate(and_rule, similarity_threshold=0.7):
                compound_rules.append(and_rule)
                self.accepted_rules.append(and_rule)
            
            if len(compound_rules) >= max_combinations // 2:
                break
            
            # Create OR rule
            or_rule = {
                'condition': f"({r1['condition']}) OR ({r2['condition']})",
                'type': 'compound-2',
                'logic': 'OR',
                'trade_type': 'BUY',
                'indicators': [r1['indicator'], r2['indicator']],
                'indicator_types': [r1.get('indicator_type', 'unknown'), 
                                   r2.get('indicator_type', 'unknown')]
            }
            
            if not self._is_duplicate(or_rule, similarity_threshold=0.7):
                compound_rules.append(or_rule)
                self.accepted_rules.append(or_rule)
            
            if len(compound_rules) >= max_combinations:
                break
        
        # Generate SELL compounds
        for r1, r2 in combinations(sell_rules, 2):
            if len(compound_rules) >= max_combinations:
                break
            
            if r1['indicator'] == r2['indicator']:
                continue
            
            and_rule = {
                'condition': f"({r1['condition']}) AND ({r2['condition']})",
                'type': 'compound-2',
                'logic': 'AND',
                'trade_type': 'SELL',
                'indicators': [r1['indicator'], r2['indicator']],
                'indicator_types': [r1.get('indicator_type', 'unknown'), 
                                   r2.get('indicator_type', 'unknown')]
            }
            
            if not self._is_duplicate(and_rule, similarity_threshold=0.7):
                compound_rules.append(and_rule)
                self.accepted_rules.append(and_rule)
            
            if len(compound_rules) >= max_combinations:
                break
            
            or_rule = {
                'condition': f"({r1['condition']}) OR ({r2['condition']})",
                'type': 'compound-2',
                'logic': 'OR',
                'trade_type': 'SELL',
                'indicators': [r1['indicator'], r2['indicator']],
                'indicator_types': [r1.get('indicator_type', 'unknown'), 
                                   r2.get('indicator_type', 'unknown')]
            }
            
            if not self._is_duplicate(or_rule, similarity_threshold=0.7):
                compound_rules.append(or_rule)
                self.accepted_rules.append(or_rule)
        
        print(f"✅ Generated {len(compound_rules)} SMART compound rules (no duplicates)\n")
        
        return compound_rules[:max_combinations]


class EnhancedComprehensiveValidator:
    """Enhanced validator WITHOUT look-ahead bias"""
    
    def __init__(self):
        self.parser = RobustConditionParser()
        self.stat_validator = StatisticalValidator()
    
    def validate_single_rule(self,
                           rule: Dict,
                           is_indicators: pd.DataFrame,
                           is_data: pd.DataFrame,
                           oos_indicators: pd.DataFrame,
                           oos_data: pd.DataFrame,
                           holding_period: int = 5,
                           transaction_cost: float = 0.001) -> Dict:
        """Validate a single rule - REJECT GARBAGE RULES"""
        
        is_valid, error_msg = self.parser.validate_condition_syntax(
            rule['condition'], is_indicators.columns.tolist()
        )
        
        if not is_valid:
            return None
        
        is_eval = is_indicators.copy()
        oos_eval = oos_indicators.copy()
        
        is_signals = self.parser.evaluate_condition(rule['condition'], is_eval)
        oos_signals = self.parser.evaluate_condition(rule['condition'], oos_eval)
        
        # CRITICAL: Reject rules with too few signals IMMEDIATELY
        if is_signals.sum() < 5:
            return None
        if oos_signals.sum() < 3:
            return None
        
        is_returns = is_data['Close'].pct_change(holding_period).shift(-holding_period) * 100
        oos_returns = oos_data['Close'].pct_change(holding_period).shift(-holding_period) * 100
        
        is_metrics = self._calculate_metrics(is_signals, is_returns, 
                                            rule.get('trade_type', 'BUY'),
                                            holding_period, transaction_cost)
        oos_metrics = self._calculate_metrics(oos_signals, oos_returns,
                                             rule.get('trade_type', 'BUY'),
                                             holding_period, transaction_cost)
        
        # REJECT if metrics show zero activity
        if is_metrics['signals'] == 0 or oos_metrics['signals'] == 0:
            return None
        
        # REJECT if both Sharpe ratios are near zero (useless rule)
        if abs(is_metrics['sharpe']) < 0.01 and abs(oos_metrics['sharpe']) < 0.01:
            return None
        
        return {
            'rule': rule['condition'],
            'type': rule.get('type', 'simple'),
            'trade_type': rule.get('trade_type', 'UNKNOWN'),
            **{f'IS_{k}': v for k, v in is_metrics.items()},
            **{f'OOS_{k}': v for k, v in oos_metrics.items()},
            'sharpe_degradation': (oos_metrics['sharpe'] - is_metrics['sharpe']) if is_metrics['sharpe'] != 0 else -999,
            'consistency_score': self._calculate_consistency(is_metrics, oos_metrics)
        }
    
    def validate_single_rule_multiperiod(self,
                                        rule: Dict,
                                        is_indicators: pd.DataFrame,
                                        is_data: pd.DataFrame,
                                        oos_indicators: pd.DataFrame,
                                        oos_data: pd.DataFrame,
                                        holding_periods: List[int],
                                        transaction_cost: float = 0.001) -> Dict:
        """
        Validate a rule across multiple holding periods and return the best one.
        Returns the result for the period with highest OOS Sharpe ratio.
        """
        best_result = None
        best_oos_sharpe = -999
        best_period = None
        
        for period in holding_periods:
            result = self.validate_single_rule(
                rule, is_indicators, is_data, oos_indicators, oos_data,
                period, transaction_cost
            )
            
            if result is not None:
                oos_sharpe = result.get('OOS_sharpe', -999)
                # Track best OOS Sharpe (most important for robustness)
                if oos_sharpe > best_oos_sharpe:
                    best_oos_sharpe = oos_sharpe
                    best_result = result
                    best_period = period
        
        # Add the best holding period to the result
        if best_result is not None:
            best_result['best_holding_period'] = best_period
        
        return best_result
    
    def validate_rules_batch(self,
                           rules: List[Dict],
                           is_indicators: pd.DataFrame,
                           is_data: pd.DataFrame,
                           oos_indicators: pd.DataFrame,
                           oos_data: pd.DataFrame,
                           holding_period = 5,  # Can be int or List[int]
                           transaction_cost: float = 0.001,
                           batch_size: int = 100) -> pd.DataFrame:
        """Validate multiple rules - supports single or multiple holding periods"""
        
        results = []
        total = len(rules)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check if testing multiple periods
        is_multi_period = isinstance(holding_period, list)
        
        for i in range(0, total, batch_size):
            batch = rules[i:i + batch_size]
            progress = min((i + batch_size) / total, 1.0)
            progress_bar.progress(progress)
            
            if is_multi_period:
                status_text.text(f"Testing rules {i+1} to {min(i+batch_size, total)} of {total} across {len(holding_period)} periods")
            else:
                status_text.text(f"Validating rules {i+1} to {min(i+batch_size, total)} of {total}")
            
            for rule in batch:
                if is_multi_period:
                    # Test across multiple periods and get best
                    result = self.validate_single_rule_multiperiod(
                        rule, is_indicators, is_data, oos_indicators, oos_data,
                        holding_period, transaction_cost
                    )
                else:
                    # Single period validation
                    result = self.validate_single_rule(
                        rule, is_indicators, is_data, oos_indicators, oos_data,
                        holding_period, transaction_cost
                    )
                
                if result is not None:
                    results.append(result)
        
        progress_bar.empty()
        status_text.empty()
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0 and 'IS_p_value' in results_df.columns:
            results_df['fdr_significant'] = self.stat_validator.apply_multiple_testing_correction(
                results_df['IS_p_value'].values, method='fdr'
            )
        
        return results_df
    
    def _calculate_metrics(self, signals: np.ndarray, returns: pd.Series,
                          trade_type: str, holding_period: int,
                          transaction_cost: float = 0.001) -> Dict:
        """Calculate metrics"""
        
        if signals.sum() < 5:
            return {
                'signals': 0, 'mean_return': 0, 'total_return': 0,
                'sharpe': 0, 'sortino': 0, 'profit_factor': 0,
                'win_rate': 0, 'max_dd': 0, 'p_value': 1.0,
                'is_significant': False
            }
        
        signal_returns = returns[signals].dropna()
        
        if trade_type == 'SELL':
            signal_returns = -signal_returns
        
        signal_returns = signal_returns - (2 * transaction_cost * 100)
        
        if len(signal_returns) < 5:
            return {
                'signals': int(signals.sum()), 'mean_return': 0,
                'total_return': 0, 'sharpe': 0, 'sortino': 0,
                'profit_factor': 0, 'win_rate': 0, 'max_dd': 0,
                'p_value': 1.0, 'is_significant': False
            }
        
        mean_return = signal_returns.mean()
        std_return = signal_returns.std()
        downside_std = signal_returns[signal_returns < 0].std() if len(signal_returns[signal_returns < 0]) > 0 else 1e-8
        
        cumulative = (1 + signal_returns / 100).cumprod()
        peak = cumulative.cummax()
        drawdown = ((cumulative - peak) / peak * 100)
        max_dd = drawdown.min()
        
        positive = signal_returns[signal_returns > 0].sum()
        negative = -signal_returns[signal_returns < 0].sum()
        profit_factor = positive / negative if negative > 0 else 999
        
        sig_results = self.stat_validator.calculate_significance(signal_returns)
        
        annual_factor = np.sqrt(252 / holding_period)
        
        return {
            'signals': int(signals.sum()),
            'mean_return': mean_return,
            'total_return': signal_returns.sum(),
            'sharpe': (mean_return / (std_return + 1e-8)) * annual_factor,
            'sortino': (mean_return / (downside_std + 1e-8)) * annual_factor,
            'profit_factor': min(profit_factor, 999),
            'win_rate': (signal_returns > 0).mean() * 100,
            'max_dd': max_dd,
            'calmar': abs(signal_returns.sum() / max_dd) if max_dd != 0 else 0,
            'p_value': sig_results['p_value'],
            'is_significant': sig_results['is_significant'],
            't_statistic': sig_results['t_statistic']
        }
    
    def _calculate_consistency(self, is_metrics: Dict, oos_metrics: Dict) -> float:
        """Calculate consistency score"""
        if is_metrics['signals'] == 0 or oos_metrics['signals'] == 0:
            return 0
        
        sharpe_consistency = 1 - abs(is_metrics['sharpe'] - oos_metrics['sharpe']) / (abs(is_metrics['sharpe']) + 1)
        pf_consistency = 1 - abs(is_metrics['profit_factor'] - oos_metrics['profit_factor']) / (is_metrics['profit_factor'] + 1)
        wr_consistency = 1 - abs(is_metrics['win_rate'] - oos_metrics['win_rate']) / 100
        
        consistency = (sharpe_consistency * 0.4 + pf_consistency * 0.3 + wr_consistency * 0.3)
        return max(0, min(1, consistency))


class RuleSelector:
    """Select best rules"""
    
    @staticmethod
    def filter_robust_rules(results_df: pd.DataFrame,
                          min_is_sharpe: float = 0.5,
                          min_oos_sharpe: float = 0.0,
                          max_degradation: float = 1.0,
                          min_consistency: float = 0.3,
                          min_signals_is: int = 20,
                          min_signals_oos: int = 10) -> pd.DataFrame:
        """Filter robust rules"""
        
        filtered = results_df[
            (results_df['IS_sharpe'] >= min_is_sharpe) &
            (results_df['OOS_sharpe'] >= min_oos_sharpe) &
            (results_df['sharpe_degradation'].abs() <= max_degradation) &
            (results_df['consistency_score'] >= min_consistency) &
            (results_df['IS_signals'] >= min_signals_is) &
            (results_df['OOS_signals'] >= min_signals_oos)
        ]
        
        return filtered.sort_values('consistency_score', ascending=False)
    
    @staticmethod
    def select_diverse_rules(filtered_df: pd.DataFrame, n_rules: int = 10) -> pd.DataFrame:
        """Select diverse rules"""
        
        if len(filtered_df) <= n_rules:
            return filtered_df
        
        selected_indices = []
        
        top_consistency = filtered_df.nlargest(n_rules // 3, 'consistency_score').index
        selected_indices.extend(top_consistency)
        
        top_oos = filtered_df.nlargest(n_rules // 3, 'OOS_sharpe').index
        selected_indices.extend(top_oos)
        
        top_pf = filtered_df.nlargest(n_rules // 3, 'OOS_profit_factor').index
        selected_indices.extend(top_pf)
        
        unique_indices = list(dict.fromkeys(selected_indices))[:n_rules]
        
        return filtered_df.loc[unique_indices]


# ===================== VISUALIZATION FUNCTIONS =====================

def create_optimization_convergence_plot(history: List[float], title: str) -> go.Figure:
    """Create convergence plot for optimization"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=history,
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=6, color='#764ba2'),
        name='Best Fitness'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Iteration',
        yaxis_title='Fitness Score',
        template='plotly_dark',
        height=400,
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        font=dict(color='#C9D1D9', family='Inter')
    )
    
    return fig


def create_performance_comparison_chart(results_df: pd.DataFrame) -> go.Figure:
    """Create IS vs OOS performance comparison"""
    fig = go.Figure()
    
    top_rules = results_df.nlargest(20, 'consistency_score')
    
    fig.add_trace(go.Bar(
        name='In-Sample Sharpe',
        x=list(range(len(top_rules))),
        y=top_rules['IS_sharpe'].values,
        marker=dict(color='#4ade80'),
    ))
    
    fig.add_trace(go.Bar(
        name='Out-of-Sample Sharpe',
        x=list(range(len(top_rules))),
        y=top_rules['OOS_sharpe'].values,
        marker=dict(color='#f59e0b'),
    ))
    
    fig.update_layout(
        title='Top 20 Rules: IS vs OOS Sharpe Ratio',
        xaxis_title='Rule Rank',
        yaxis_title='Sharpe Ratio',
        template='plotly_dark',
        height=500,
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        font=dict(color='#C9D1D9', family='Inter'),
        barmode='group'
    )
    
    return fig


def create_scatter_performance(results_df: pd.DataFrame) -> go.Figure:
    """Create IS vs OOS scatter plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df['IS_sharpe'],
        y=results_df['OOS_sharpe'],
        mode='markers',
        marker=dict(
            size=8,
            color=results_df['consistency_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Consistency"),
            line=dict(width=1, color='white')
        ),
        text=[f"Rule: {r[:50]}..." for r in results_df['rule']],
        hovertemplate='<b>%{text}</b><br>IS Sharpe: %{x:.2f}<br>OOS Sharpe: %{y:.2f}<extra></extra>'
    ))
    
    max_val = max(results_df['IS_sharpe'].max(), results_df['OOS_sharpe'].max())
    min_val = min(results_df['IS_sharpe'].min(), results_df['OOS_sharpe'].min())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Perfect Consistency',
        showlegend=True
    ))
    
    fig.update_layout(
        title='In-Sample vs Out-of-Sample Performance',
        xaxis_title='In-Sample Sharpe',
        yaxis_title='Out-of-Sample Sharpe',
        template='plotly_dark',
        height=600,
        paper_bgcolor='#0D1117',
        plot_bgcolor='#161B22',
        font=dict(color='#C9D1D9', family='Inter')
    )
    
    return fig


# ===================== MAIN APPLICATION =====================

def main():
    st.markdown(f"""
        <h1 class='main-header'>Advanced Quantitative Analysis Platform</h1>
        <p class='sub-header'>
            {TechnicalIndicators.get_total_count()} TECHNICAL INDICATORS · PSO & GENETIC ALGORITHMS · IS/OOS VALIDATION
        </p>
    """, unsafe_allow_html=True)
    
    # ===================== COMPREHENSIVE SESSION STATE INITIALIZATION =====================
    # Initialize ALL session state variables at once to prevent resets
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        # Basic analysis state
        st.session_state.analysis_done = False
        st.session_state.returns_data = None
        st.session_state.indicators = None
        st.session_state.data = None
        st.session_state.summary = None
        st.session_state.periods_to_test = None
        st.session_state.selected_categories = None
        
        # Advanced analysis state
        st.session_state.advanced_analysis_done = False
        st.session_state.advanced_results = None
        st.session_state.optimization_histories = None
        
        # Configuration state - preserve user inputs
        st.session_state.ticker_input = "SPY"
        st.session_state.period_input = "max"
        st.session_state.last_tab = "📊 ANALYSIS"
    
    # Restore or initialize config values
    if 'ticker_input' not in st.session_state:
        st.session_state.ticker_input = "SPY"
    if 'period_input' not in st.session_state:
        st.session_state.period_input = "max"
    
    # ===================== DATA CONFIGURATION =====================
    st.markdown("<div class='config-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Data Configuration</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
    
    with col1:
        ticker = st.text_input(
            "SYMBOL", 
            value=st.session_state.ticker_input, 
            help="Stock symbol to analyze",
            key="ticker_widget"
        )
        # Update session state when changed
        if ticker != st.session_state.ticker_input:
            st.session_state.ticker_input = ticker
    
    with col2:
        period_option = st.selectbox(
            "PERIOD",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"].index(st.session_state.period_input) if st.session_state.period_input in ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"] else 4,
            key="period_widget"
        )
        if period_option != st.session_state.period_input:
            st.session_state.period_input = period_option
    
    with col3:
        col3a, col3b = st.columns(2)
        with col3a:
            min_return_days = st.number_input("MIN DAYS", value=1, min_value=1, max_value=60, key="min_days_widget")
        with col3b:
            max_return_days = st.number_input("MAX DAYS", value=20, min_value=1, max_value=60, key="max_days_widget")
    
    with col4:
        quantiles = st.number_input("PERCENTILES", value=10, min_value=5, max_value=20, step=5, key="quantiles_widget")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===================== PERIOD CONFIGURATION =====================
    st.markdown("<div class='config-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Period Configuration</div>", unsafe_allow_html=True)
    
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
                Periods: {', '.join(map(str, periods_to_test))}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===================== INDICATOR SELECTION =====================
    st.markdown("<div class='config-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Indicator Selection</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        select_mode = st.radio("MODE", ["Presets", "Categories", "All"])
    
    with col2:
        if select_mode == "Presets":
            preset = st.selectbox(
                "CONFIGURATION",
                ["Essential (30 indicators)", "Extended (60 indicators)", 
                 "Complete (100 indicators)", "All (158+ indicators)"]
            )
            
            if "Essential" in preset:
                selected_categories = ["Superposición", "Momentum"][:1]
            elif "Extended" in preset:
                selected_categories = ["Momentum", "Volatilidad", "Volumen", "Superposición"]
            elif "Complete" in preset:
                selected_categories = list(TechnicalIndicators.CATEGORIES.keys())[:7]
            else:
                selected_categories = ["TODO"]
        
        elif select_mode == "Categories":
            selected_categories = st.multiselect(
                "SELECT CATEGORIES",
                list(TechnicalIndicators.CATEGORIES.keys()),
                default=["Momentum", "Superposición"]
            )
        else:
            selected_categories = ["TODO"]
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ===================== ANALYZE BUTTON =====================
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_button = st.button("ANALYZE", use_container_width=True, type="primary", key="analyze_main")
    
    if analyze_button and max_return_days >= min_return_days:
        with st.spinner('Processing indicators...'):
            returns_data, indicators, data, summary = calculate_all_indicators(
                ticker, period_option, quantiles, min_return_days, max_return_days,
                periods_to_test, selected_categories
            )
            
            if returns_data and indicators is not None and data is not None:
                st.session_state.analysis_done = True
                st.session_state.returns_data = returns_data
                st.session_state.indicators = indicators
                st.session_state.data = data
                st.session_state.summary = summary
                st.session_state.periods_to_test = periods_to_test
                st.session_state.selected_categories = selected_categories
                st.rerun()
    
    # ===================== RESULTS DISPLAY =====================
    if st.session_state.analysis_done:
        returns_data = st.session_state.returns_data
        indicators = st.session_state.indicators
        data = st.session_state.data
        summary = st.session_state.summary
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("INDICATORS", summary['indicators_count'])
        with col2:
            st.metric("SUCCESS RATE", f"{(summary['successful']/summary['total_attempted']*100):.1f}%")
        with col3:
            st.metric("DATA POINTS", summary['data_points'])
        with col4:
            st.metric("RANGE", summary['date_range'].split(' to ')[0])
        
        # ===================== TABS =====================
        tab1, tab2, tab3 = st.tabs(["📊 ANALYSIS", "🚀 ADVANCED RULES", "💾 EXPORT"])
        
        # TAB 1: Analysis
        with tab1:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_indicator = st.selectbox(
                    "SELECT INDICATOR",
                    sorted(indicators.columns),
                    key="indicator_select"
                )
            with col2:
                return_period = st.number_input(
                    "DAYS",
                    min_value=summary['min_return_days'],
                    max_value=summary['max_return_days'],
                    value=min(5, summary['max_return_days']),
                    key="return_period_select"
                )
            
            if selected_indicator:
                fig = create_percentile_plot(
                    indicators, returns_data, data,
                    selected_indicator, return_period, quantiles
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # TAB 2: Advanced Rules
        with tab2:
            st.markdown("### 🚀 Advanced Rule Discovery Platform")
            
            st.info("""
            **🎯 Choose Your Optimization Method:**
            
            Select ONE method that best fits your needs:
            - **Particle Swarm (PSO)** - Fast continuous optimization with optional Enhanced Multi-Swarm mode for 2-3x more diverse rules  
            - **Brute Force** - Traditional exhaustive search (~5-15 min, tests thousands of rules)
            
            **🔄 Auto-Optimization:** Each rule is tested across your selected holding period range, and the best performing period is reported in results.
            
            **⚠️ Quality Control:** Rules with <5 IS signals, <3 OOS signals, or near-zero Sharpe are automatically rejected.
            """)
            
            # Method Selection - RADIO BUTTON
            st.markdown("### 🎯 Select Optimization Method")
            optimization_method = st.radio(
                "Choose ONE method:",
                ["🔍 Brute Force Search", "🌟 Particle Swarm Optimization (PSO)"],
                index=0,
                horizontal=True,
                help="Only one method can be selected at a time",
                key="optimization_method_radio"
            )
            
            use_brute = "Brute Force" in optimization_method
            use_pso = "Particle Swarm" in optimization_method
            use_genetic = False  # Genetic Algorithm removed
            
            st.markdown("---")
            
            # Configuration Form
            with st.form("rule_config_form"):
                st.markdown("### ⚙️ Configuration")
                
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    st.markdown("**📊 Data & Trading**")
                    sample_split = st.slider("In-Sample %", 50, 80, 70, 5, key="sample_split")
                    
                    # Holding Period Range
                    st.markdown("**Holding Period Range (Auto-Optimize)**")
                    holding_period_range = st.slider(
                        "Test multiple holding periods and find the best for each rule",
                        min_value=summary['min_return_days'],
                        max_value=summary['max_return_days'],
                        value=(summary['min_return_days'], min(summary['min_return_days'] + 15, summary['max_return_days'])),
                        step=5,
                        key="holding_period_range",
                        help="System will test each rule across this range and report the best period"
                    )
                    holding_periods_to_test = list(range(holding_period_range[0], holding_period_range[1] + 1, 5))
                    if holding_period_range[1] not in holding_periods_to_test:
                        holding_periods_to_test.append(holding_period_range[1])
                    
                    st.info(f"🎯 Will test: {len(holding_periods_to_test)} periods → {holding_periods_to_test}")
                    
                    transaction_cost = st.number_input("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.01, key="tx_cost") / 100
                
                with config_col2:
                    st.markdown("**🎯 Rule Parameters**")
                    # CORRECTED: Allow using ALL indicators (no 50 limit)
                    max_indicators_opt = st.number_input("Max Indicators to Use (0 = ALL)", 0, 500, 0, key="max_ind",
                                                        help="Set to 0 to use ALL available indicators. Otherwise specify a limit.")
                    min_signals_is = st.number_input("Min Signals (IS)", 5, 100, 15, key="min_sig_is", 
                                                     help="Minimum signals required in training data (validation enforces ≥5)")
                    min_signals_oos = st.number_input("Min Signals (OOS)", 3, 50, 10, key="min_sig_oos",
                                                      help="Minimum signals required in test data (validation enforces ≥3)")
                
                # Method-specific configuration
                if use_pso:
                    st.markdown("---")
                    st.markdown("### 🌟 PSO-Specific Settings")
                    
                    # Add improved PSO option
                    use_improved_pso = st.checkbox(
                        "✨ Use Enhanced Multi-Swarm PSO (finds MORE diverse rules)",
                        value=True,
                        key="use_improved_pso",
                        help="Enhanced version with multiple swarms, diversity archive, and adaptive parameters. Finds 2-3x more unique rules!"
                    )
                    
                    pso_col1, pso_col2 = st.columns(2)
                    
                    with pso_col1:
                        pso_n_particles = st.slider("Number of Particles", 20, 150, 50, key="pso_particles")
                        pso_n_iterations = st.slider("Number of Iterations", 20, 150, 80, key="pso_iter")
                        
                        if use_improved_pso:
                            pso_n_swarms = st.slider("Number of Swarms", 2, 5, 3, key="pso_swarms",
                                                    help="Multiple swarms explore different regions")
                    
                    with pso_col2:
                        if use_improved_pso:
                            pso_inertia_max = st.slider("Max Inertia (w_max)", 0.7, 1.0, 0.9, 0.05, key="pso_w_max")
                            pso_inertia_min = st.slider("Min Inertia (w_min)", 0.2, 0.6, 0.4, 0.05, key="pso_w_min")
                        else:
                            pso_inertia = st.slider("Inertia Weight (w)", 0.1, 1.0, 0.7, 0.1, key="pso_w")
                        
                        pso_cognitive = st.slider("Cognitive (c1)", 0.5, 2.5, 1.5, 0.1, key="pso_c1")
                        pso_social = st.slider("Social (c2)", 0.5, 2.5, 1.5, 0.1, key="pso_c2")
                    
                    if use_improved_pso:
                        st.info("""
                        🚀 **Enhanced PSO Features:**
                        - Multiple swarms explore different strategies simultaneously
                        - Diversity archive maintains unique high-quality rules
                        - Adaptive parameters optimize exploration/exploitation
                        - Inter-swarm migration shares best solutions
                        - Expects to find **20-30 unique rules** (vs 10-15 in standard)
                        """)
                else:
                    use_improved_pso = False
                    pso_n_particles = 50
                    pso_n_iterations = 80
                    pso_inertia = 0.7
                    pso_inertia_max = 0.9
                    pso_inertia_min = 0.4
                    pso_cognitive = 1.5
                    pso_social = 1.5
                    pso_n_swarms = 3
                
                if use_brute:
                    st.markdown("---")
                    st.markdown("### 🔍 Brute Force Settings")
                    brute_col1, brute_col2 = st.columns(2)
                    
                    with brute_col1:
                        max_rules_test = st.number_input("Max Rules to Test", 100, 10000, 2000, 100, key="max_rules")
                        use_compound = st.checkbox("Generate Compound Rules", True, key="use_compound")
                        
                        if use_compound:
                            max_compound = st.number_input("Max Compound Rules", 100, 5000, 1000, key="max_comp")
                            compound_depth = st.select_slider("Max Conditions", [2, 3], 2, key="comp_depth")
                        else:
                            max_compound = 0
                            compound_depth = 2
                    
                    with brute_col2:
                        percentiles_use = st.multiselect(
                            "Percentile Thresholds",
                            list(range(5, 100, 5)),
                            default=[10, 20, 30, 50, 70, 80, 90],
                            key="percentiles"
                        )
                        operators = st.multiselect("Operators", ['>', '<', '>=', '<='], ['>', '<'], key="operators")
                else:
                    max_rules_test = 2000
                    use_compound = True
                    max_compound = 1000
                    compound_depth = 2
                    percentiles_use = [10, 20, 30, 50, 70, 80, 90]
                    operators = ['>', '<']
                
                st.markdown("---")
                
                # Quality Filters
                with st.expander("✅ Quality Filters", expanded=False):
                    filter_col1, filter_col2, filter_col3 = st.columns(3)
                    
                    with filter_col1:
                        min_is_sharpe = st.number_input("Min IS Sharpe", -2.0, 3.0, 0.2, 0.1, key="min_is_sharpe")
                        min_oos_sharpe = st.number_input("Min OOS Sharpe", -2.0, 3.0, -0.5, 0.1, key="min_oos_sharpe")
                    
                    with filter_col2:
                        max_degradation = st.slider("Max Sharpe Degradation", 0.5, 5.0, 3.0, 0.1, key="max_deg")
                        min_consistency = st.slider("Min Consistency", 0.0, 1.0, 0.1, 0.05, key="min_cons")
                    
                    with filter_col3:
                        if use_pso:
                            est_time = "~3-8 min"
                        else:
                            est_time = "~5-15 min"
                        
                        st.info(f"""
                        **Expected Time:**
                        {est_time}
                        """)
                
                # RULE VALIDATION SETTINGS (NEW)
                with st.expander("🔧 Rule Validation Settings", expanded=False):
                    st.markdown("""
                    **Control how strictly rules are validated:**
                    - Higher values = more lenient (keeps more rules)
                    - Lower values = stricter (filters more rules)
                    """)
                    
                    val_col1, val_col2, val_col3 = st.columns(3)
                    
                    with val_col1:
                        enable_validation = st.checkbox(
                            "Enable Rule Validation",
                            value=True,
                            help="Automatically filter rules with unrealistic thresholds"
                        )
                        
                        show_validation_details = st.checkbox(
                            "Show Filtering Details",
                            value=False,
                            help="Display which rules were filtered and why"
                        )
                    
                    with val_col2:
                        data_range_margin = st.slider(
                            "Data Range Margin (%)",
                            min_value=5,
                            max_value=50,
                            value=15,
                            step=5,
                            help="How far beyond min/max to allow thresholds (15% = default)"
                        )
                    
                    with val_col3:
                        extreme_margin = st.slider(
                            "Extreme Value Margin (%)",
                            min_value=10,
                            max_value=100,
                            value=20,
                            step=10,
                            help="How far beyond 99th percentile to allow (20% = default)"
                        )
                    
                    st.info(f"""
                    **Current Settings:**
                    - Validation: {'Enabled ✓' if enable_validation else 'Disabled ✗'}
                    - Range Margin: ±{data_range_margin}% of data range
                    - Extreme Margin: +{extreme_margin}% beyond percentiles
                    """)
                
                # SUBMIT BUTTON
                st.markdown("---")
                run_advanced = st.form_submit_button(
                    "🚀 RUN ADVANCED ANALYSIS", 
                    use_container_width=True,
                    type="primary"
                )
            
            # EXECUTION
            if run_advanced:
                # Clear previous results
                st.session_state.advanced_analysis_done = False
                st.session_state.advanced_results = None
                st.session_state.optimization_histories = None
                
                # Split data
                split_index = int(len(data) * sample_split / 100)
                is_data = data.iloc[:split_index].copy()
                oos_data = data.iloc[split_index:].copy()
                
                # Calculate indicators
                with st.spinner("Calculating indicators..."):
                    is_indicators = calculate_indicators_for_dataset(
                        is_data, st.session_state.periods_to_test,
                        st.session_state.selected_categories
                    )
                    oos_indicators = calculate_indicators_for_dataset(
                        oos_data, st.session_state.periods_to_test,
                        st.session_state.selected_categories
                    )
                
                # Prepare for PSO (uses first period in range for optimization)
                # Actual validation will test all periods
                pso_holding_period = holding_periods_to_test[0]
                is_returns = is_data['Close'].pct_change(pso_holding_period).shift(-pso_holding_period) * 100
                oos_returns = oos_data['Close'].pct_change(pso_holding_period).shift(-pso_holding_period) * 100
                
                # ============= FILTER VALID INDICATORS =============
                # Remove math functions and invalid columns
                invalid_names = {
                    'CEIL', 'FLOOR', 'SQRT', 'EXP', 'LOG', 'LOG10', 'LN',
                    'TAN', 'SIN', 'COS', 'ASIN', 'ACOS', 'ATAN', 'SINH', 'COSH', 'TANH',
                    'ABS', 'MAX', 'MIN', 'SUM', 'AVG', 'MEAN',
                    'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Date', 'Datetime', 'Timestamp'
                }
                
                # Filter to only valid indicators
                valid_indicator_names = []
                for col in is_indicators.columns:
                    # Skip if column name is in invalid set
                    col_upper = col.upper()
                    if any(inv in col_upper for inv in invalid_names):
                        continue
                    
                    # Check if indicator has enough valid data
                    col_data = is_indicators[col].dropna()
                    if len(col_data) < 20:
                        continue
                    
                    # Check if indicator has reasonable variance (not all same value)
                    if col_data.std() < 1e-10:
                        continue
                    
                    # Check for reasonable value range (not all zeros or extreme)
                    if abs(col_data.mean()) < 1e-10 and col_data.std() < 1e-10:
                        continue
                    
                    valid_indicator_names.append(col)
                
                # CORRECTED: Limit to max_indicators_opt (0 = use ALL)
                if max_indicators_opt > 0:
                    indicator_names = valid_indicator_names[:max_indicators_opt]
                else:
                    indicator_names = valid_indicator_names  # Use ALL indicators
                
                if len(indicator_names) < 3:
                    st.error(f"⚠️ Only {len(indicator_names)} valid indicators found. Need at least 3. Try selecting more indicator categories in Tab 1.")
                    st.stop()
                
                st.info(f"✅ Using {len(indicator_names)} valid indicators: {', '.join(indicator_names[:10])}{'...' if len(indicator_names) > 10 else ''}")
                
                # CRITICAL FIX: Filter to only indicators that exist in BOTH is_indicators and oos_indicators
                available_in_is = set(is_indicators.columns)
                available_in_oos = set(oos_indicators.columns)
                indicator_names = [ind for ind in indicator_names if ind in available_in_is and ind in available_in_oos]
                
                if len(indicator_names) < 3:
                    st.error(f"⚠️ Only {len(indicator_names)} indicators available in both IS and OOS periods. Need at least 3.")
                    st.info("""
                    **Possible causes:**
                    - OOS period too short to calculate some indicators
                    - Indicators require more data points than available
                    - Try increasing the train/test split ratio
                    """)
                    st.stop()
                
                st.info(f"✅ Using {len(indicator_names)} valid indicators: {', '.join(indicator_names[:10])}{'...' if len(indicator_names) > 10 else ''}")
                
                # Now safely filter the indicators DataFrames
                is_indicators = is_indicators[indicator_names].copy()
                oos_indicators = oos_indicators[indicator_names].copy()
                
                all_results = []
                optimization_histories = {}
                
                # ============= PSO OPTIMIZATION =============
                if use_pso:
                    st.markdown("### 🌟 Particle Swarm Optimization")
                    
                    if use_improved_pso:
                        st.success("✨ Using **Enhanced Multi-Swarm PSO** - Expect 20-30+ diverse rules")
                        pso = ImprovedParticleSwarmOptimizer(
                            n_particles=pso_n_particles,
                            n_iterations=pso_n_iterations,
                            w_max=pso_inertia_max,
                            w_min=pso_inertia_min,
                            c1=pso_cognitive,
                            c2=pso_social,
                            n_swarms=pso_n_swarms,
                            archive_size=50
                        )
                    else:
                        st.info("📊 Using **Standard PSO** - Expect 10-15 rules")
                        pso = ParticleSwarmOptimizer(
                            n_particles=pso_n_particles,
                            n_iterations=pso_n_iterations,
                            w=pso_inertia,
                            c1=pso_cognitive,
                            c2=pso_social
                        )
                    
                    progress_container = st.empty()
                    status_container = st.empty()
                    metrics_container = st.empty()
                    
                    def pso_callback(iteration, total, fitness, valid_count):
                        progress = iteration / total
                        progress_container.progress(progress)
                        status_container.markdown(f"""
                        <div class="progress-text">
                            🌟 PSO Iteration {iteration}/{total} | Best Fitness: {fitness:.4f} | Valid: {valid_count}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    metrics_container.info(f"🔍 Optimizing with {pso_n_particles} particles over {pso_n_iterations} iterations...")
                    
                    pso_result = pso.optimize_rule_parameters(
                        is_indicators, is_returns, indicator_names, pso_callback
                    )
                    
                    progress_container.empty()
                    status_container.empty()
                    metrics_container.empty()
                    
                    # Display PSO statistics
                    if use_improved_pso and 'archive_size' in pso_result:
                        st.info(f"📊 Enhanced PSO Stats: {pso_result.get('valid_rules_found', 0)} valid evaluations | Archive: {pso_result.get('archive_size', 0)} diverse rules | Total evaluations: {pso_result.get('total_evaluations', 0)}")
                    else:
                        st.info(f"📊 PSO Stats: {pso_result.get('valid_rules_found', 0)} valid rules found out of {pso_result.get('total_evaluations', 0)} evaluations")
                    
                    # Process multiple rules from PSO
                    if pso_result.get('rules') and len(pso_result['rules']) > 0:
                        n_rules_found = len(pso_result['rules'])
                        st.success(f"✅ PSO found {n_rules_found} diverse rules!")
                        
                        # Warning if fewer than expected
                        if use_improved_pso and n_rules_found < 15:
                            st.warning(f"⚠️ Enhanced PSO found fewer rules than expected ({n_rules_found} vs expected 20-30). This may indicate:\n"
                                     "- Limited indicator variety (try selecting more indicator categories in Tab 1)\n"
                                     "- Tight quality filters (try loosening Min Sharpe or other filters)\n"
                                     "- Small dataset (need more historical data)")
                        elif not use_improved_pso and n_rules_found < 5:
                            st.warning(f"⚠️ Standard PSO found very few rules ({n_rules_found}). Consider using Enhanced PSO or checking data quality.")
                        
                        # Show the rules found
                        with st.expander(f"View {n_rules_found} PSO Rules (Before Validation)", expanded=False):
                            for idx, (rule, fitness) in enumerate(zip(pso_result['rules'][:10], pso_result['fitness_scores'][:10])):
                                st.text(f"{idx+1}. Fitness: {fitness:.3f} | {rule}")
                        
                        validator = EnhancedComprehensiveValidator()
                        
                        # Validate with progress
                        validation_status = st.empty()
                        validation_status.info(f"🔍 Validating {len(pso_result['rules'])} PSO rules across {len(holding_periods_to_test)} periods...")
                        
                        for idx, (rule, fitness) in enumerate(zip(pso_result['rules'], pso_result['fitness_scores'])):
                            pso_validation = validator.validate_single_rule_multiperiod(
                                {'condition': rule, 'type': 'pso', 'trade_type': 'BUY'},
                                is_indicators, is_data, oos_indicators, oos_data,
                                holding_periods_to_test, transaction_cost
                            )
                            
                            if pso_validation:
                                all_results.append(pso_validation)
                        
                        validation_status.empty()
                        
                        optimization_histories['PSO'] = pso_result['history']
                        
                        if len(all_results) > 0:
                            st.info(f"📊 {len(all_results)} PSO rules passed validation (out of {len(pso_result['rules'])} found)")
                        else:
                            st.warning(f"⚠️ PSO found {len(pso_result['rules'])} rules but ALL were rejected (insufficient signals or poor quality)")
                    else:
                        st.warning(f"⚠️ PSO did not find valid rules")
                        if 'message' in pso_result:
                            st.info(f"ℹ️ {pso_result['message']}")
                    
                    # Plot convergence
                    if len(pso_result['history']) > 0:
                        fig = create_optimization_convergence_plot(
                            pso_result['history'], "PSO Convergence"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # ============= BRUTE FORCE =============
                if use_brute:
                    st.markdown("### 🔍 Brute Force Search")
                    
                    generator = ImprovedMassiveRuleGenerator()
                    
                    simple_rules = generator.generate_simple_rules(
                        is_indicators, percentiles_use, operators, max_indicators_opt
                    )
                    
                    if use_compound:
                        compound_rules = generator.generate_compound_rules(
                            simple_rules, max_depth=compound_depth, max_combinations=max_compound
                        )
                        all_rules = simple_rules + compound_rules
                    else:
                        all_rules = simple_rules
                    
                    st.info(f"Generated {len(all_rules)} rules for validation")
                    
                    validator = EnhancedComprehensiveValidator()
                    brute_results = validator.validate_rules_batch(
                        all_rules[:max_rules_test],
                        is_indicators, is_data, oos_indicators, oos_data,
                        holding_periods_to_test, transaction_cost
                    )
                    
                    if not brute_results.empty:
                        for _, row in brute_results.iterrows():
                            all_results.append(row.to_dict())
                        
                        st.success(f"✅ Validated {len(brute_results)} rules")
                
                # ============= VALIDATE AND SAVE RESULTS =============
                if all_results:
                    results_df = pd.DataFrame(all_results)
                    original_count = len(results_df)
                    
                    # === RULE VALIDATION (filters out bad thresholds) ===
                    if enable_validation:
                        st.info("🔍 Validating rule thresholds...")
                        
                        # Calculate indicator statistics
                        indicator_stats = {}
                        for col in indicators.columns:
                            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                                data = indicators[col].dropna()
                                if len(data) > 0:
                                    indicator_stats[col] = {
                                        'min': data.min(),
                                        'max': data.max(),
                                        'p01': data.quantile(0.01),
                                        'p99': data.quantile(0.99),
                                        'mean': data.mean(),
                                        'std': data.std(),
                                    }
                        
                        # Convert margins from percentage to decimal
                        range_margin_pct = data_range_margin / 100.0
                        extreme_margin_pct = extreme_margin / 100.0
                        
                        # Validation function with detailed feedback
                        def validate_rule_detailed(rule_str):
                            """Check if rule has realistic thresholds and return reason"""
                            import re
                            
                            # Parse conditions
                            conditions = re.split(r'\s+(?:AND|OR)\s+', rule_str)
                            reasons = []
                            
                            for condition in conditions:
                                condition = condition.strip('() ')
                                
                                # Extract indicator, operator, threshold
                                match = re.match(r'([A-Z_]+_\d+)\s*([><]=?)\s*([-+]?\d+\.?\d*)', condition)
                                if not match:
                                    continue
                                
                                indicator, operator, threshold = match.groups()
                                threshold = float(threshold)
                                
                                if indicator not in indicator_stats:
                                    reasons.append(f"{indicator} not in data")
                                    return False, reasons
                                
                                stats = indicator_stats[indicator]
                                data_range = stats['max'] - stats['min']
                                
                                # Check for bounded indicators (RSI, Stochastic, etc.)
                                if any(x in indicator.upper() for x in ['RSI', 'STOCH', 'WILLR']):
                                    if threshold < 0 or threshold > 100:
                                        reasons.append(f"{indicator} {operator} {threshold:.2f} (bounded 0-100)")
                                        return False, reasons
                                
                                # Check for volatility indicators (must be positive)
                                if any(x in indicator.upper() for x in ['ATR', 'NATR', 'BBANDS', 'STDDEV']):
                                    if threshold < 0:
                                        reasons.append(f"{indicator} < 0 (volatility must be positive)")
                                        return False, reasons
                                
                                # Check if threshold is way outside data range
                                margin = data_range * range_margin_pct
                                if operator in ['>', '>=']:
                                    if threshold > stats['max'] + margin:
                                        reasons.append(f"{indicator} > {threshold:.2f} (max: {stats['max']:.2f})")
                                        return False, reasons
                                    # Too extreme (beyond 99th percentile + margin)
                                    if threshold > stats['p99'] + data_range * extreme_margin_pct:
                                        reasons.append(f"{indicator} > {threshold:.2f} (99th: {stats['p99']:.2f})")
                                        return False, reasons
                                elif operator in ['<', '<=']:
                                    if threshold < stats['min'] - margin:
                                        reasons.append(f"{indicator} < {threshold:.2f} (min: {stats['min']:.2f})")
                                        return False, reasons
                                    # Too extreme (below 1st percentile - margin)
                                    if threshold < stats['p01'] - data_range * extreme_margin_pct:
                                        reasons.append(f"{indicator} < {threshold:.2f} (1st: {stats['p01']:.2f})")
                                        return False, reasons
                            
                            return True, []
                        
                        # Filter rules with detailed tracking
                        validation_results = results_df['rule'].apply(validate_rule_detailed)
                        results_df['is_valid'] = validation_results.apply(lambda x: x[0])
                        results_df['validation_reason'] = validation_results.apply(lambda x: ', '.join(x[1]))
                        
                        # Separate valid and invalid
                        valid_results = results_df[results_df['is_valid']].drop(columns=['is_valid', 'validation_reason'])
                        invalid_results = results_df[~results_df['is_valid']]
                        invalid_count = len(invalid_results)
                        
                        # Show filtering statistics
                        if invalid_count > 0:
                            st.warning(f"⚠️ Filtered out {invalid_count} rules with unrealistic thresholds "
                                     f"({invalid_count/original_count*100:.1f}% of total)")
                            
                            # Show detailed filtering info if requested
                            if show_validation_details and len(invalid_results) > 0:
                                with st.expander(f"📋 View {min(20, len(invalid_results))} Filtered Rules (Click to expand)", expanded=False):
                                    st.markdown("**Sample of filtered rules and reasons:**")
                                    
                                    # Group by reason
                                    reason_counts = invalid_results['validation_reason'].value_counts()
                                    
                                    st.markdown("**Filtering Breakdown:**")
                                    for reason, count in reason_counts.head(10).items():
                                        st.markdown(f"- `{reason}`: **{count} rules** ({count/invalid_count*100:.1f}%)")
                                    
                                    st.markdown("---")
                                    st.markdown("**Examples of filtered rules:**")
                                    
                                    for idx, row in invalid_results.head(20).iterrows():
                                        st.markdown(f"❌ `{row['rule']}`")
                                        st.markdown(f"   ↳ *{row['validation_reason']}*")
                        else:
                            st.success(f"✅ All {original_count} rules passed validation!")
                        
                        results_to_save = valid_results
                    else:
                        st.info("ℹ️ Rule validation disabled - using all generated rules")
                        results_to_save = results_df
                    
                    st.session_state.advanced_analysis_done = True
                    st.session_state.advanced_results = results_to_save
                    st.session_state.optimization_histories = optimization_histories
                    
                    if len(results_to_save) > 0:
                        st.success(f"✅ Analysis complete! {len(results_to_save)} valid rules found. Scroll down to see results.")
                    else:
                        st.warning("⚠️ No valid rules passed threshold validation.")
                        st.info("""
                        **💡 Try these adjustments:**
                        1. **Disable validation** temporarily to see all rules
                        2. **Increase margins** in validation settings (use 30-50%)
                        3. **Lower optimization thresholds** (Min IS Sharpe, Min OOS Sharpe)
                        4. Use **more historical data** for better indicator coverage
                        5. Try **different indicators** or optimization methods
                        """)
                        
                else:
                    st.session_state.advanced_analysis_done = True
                    st.session_state.advanced_results = pd.DataFrame()
                    st.session_state.optimization_histories = optimization_histories
                    
                    st.warning("⚠️ No valid rules passed all quality filters.")
                    st.info("""
                    **💡 Suggestions to find rules:**
                    1. Lower **Min IS Sharpe** to 0.0 or negative
                    2. Lower **Min OOS Sharpe** to -1.0
                    3. Increase **Min Signals** thresholds (more data = easier to find patterns)
                    4. Try **Brute Force** method (tests more combinations)
                    5. Increase holding period to 15-30 days
                    """)
                st.rerun()
            
            # ============= DISPLAY RESULTS =============
            if st.session_state.advanced_analysis_done:
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Clear button
                if st.button("🔄 Clear Results & Run New Analysis", key="clear_advanced"):
                    st.session_state.advanced_analysis_done = False
                    st.session_state.advanced_results = None
                    st.session_state.optimization_histories = None
                    st.rerun()
                
                results_df = st.session_state.advanced_results
                optimization_histories = st.session_state.optimization_histories or {}
                
                if results_df is not None and len(results_df) > 0:
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Rules Found", len(results_df))
                    
                    with col2:
                        positive_oos = len(results_df[results_df['OOS_sharpe'] > 0])
                        st.metric("Positive OOS Sharpe", positive_oos, 
                                f"{positive_oos/len(results_df)*100:.1f}%")
                    
                    with col3:
                        avg_consistency = results_df['consistency_score'].mean()
                        st.metric("Avg Consistency", f"{avg_consistency:.2%}")
                    
                    with col4:
                        selector = RuleSelector()
                        robust = selector.filter_robust_rules(
                            results_df, min_is_sharpe, min_oos_sharpe,
                            max_degradation, min_consistency, min_signals_is, min_signals_oos
                        )
                        st.metric("Robust Rules", len(robust))
                    
                    # Visualizations
                    st.markdown("### 📈 Performance Analysis")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        fig = create_performance_comparison_chart(results_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_col2:
                        fig = create_scatter_performance(results_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Results Table
                    st.markdown("### 📋 Detailed Results")
                    
                    view_option = st.radio(
                        "Show:",
                        ["Top 20 by Consistency", "Robust Rules Only", "All Results"],
                        horizontal=True,
                        key="view_option_radio"
                    )
                    
                    display_cols = [
                        'rule', 'type', 'trade_type', 'best_holding_period',
                        'IS_signals', 'IS_sharpe', 'IS_sortino', 'IS_win_rate',
                        'OOS_signals', 'OOS_sharpe', 'OOS_sortino', 'OOS_win_rate',
                        'consistency_score', 'sharpe_degradation'
                    ]
                    
                    # Filter columns that exist in results_df
                    display_cols = [col for col in display_cols if col in results_df.columns]
                    
                    if view_option == "Top 20 by Consistency":
                        display_df = results_df.nlargest(20, 'consistency_score')[display_cols]
                    elif view_option == "Robust Rules Only" and len(robust) > 0:
                        display_df = robust[display_cols]
                    else:
                        display_df = results_df[display_cols]
                    
                    st.dataframe(
                        display_df.style.format({
                            'IS_sharpe': '{:.2f}',
                            'IS_sortino': '{:.2f}',
                            'IS_win_rate': '{:.1f}',
                            'OOS_sharpe': '{:.2f}',
                            'OOS_sortino': '{:.2f}',
                            'OOS_win_rate': '{:.1f}',
                            'consistency_score': '{:.2%}',
                            'sharpe_degradation': '{:.2f}'
                        }).background_gradient(subset=['OOS_sharpe'], cmap='RdYlGn', vmin=-1, vmax=2)
                           .background_gradient(subset=['consistency_score'], cmap='YlGn', vmin=0, vmax=1),
                        use_container_width=True,
                        height=600
                    )
                    
                    # Export Options
                    st.markdown("### 💾 Export Results")
                    
                    exp_col1, exp_col2, exp_col3 = st.columns(3)
                    
                    with exp_col1:
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 All Results CSV",
                            csv,
                            f"{ticker}_advanced_rules_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with exp_col2:
                        if len(robust) > 0:
                            csv = robust.to_csv(index=False)
                            st.download_button(
                                "📥 Robust Rules CSV",
                                csv,
                                f"{ticker}_robust_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                    
                    with exp_col3:
                        if optimization_histories:
                            history_json = json.dumps(optimization_histories, indent=2)
                            st.download_button(
                                "📥 Optimization History JSON",
                                history_json,
                                f"{ticker}_opt_history_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                "application/json",
                                use_container_width=True
                            )
                
                elif results_df is not None and len(results_df) == 0:
                    st.warning("⚠️ Analysis completed but no valid rules were found. Try different parameters.")
                    if st.button("🔄 Try Again", key="retry_advanced"):
                        st.session_state.advanced_analysis_done = False
                        st.rerun()
                
                else:
                    st.error("⚠️ An error occurred during analysis. Please try again.")
                    if st.button("🔄 Retry", key="retry_error"):
                        st.session_state.advanced_analysis_done = False
                        st.rerun()
            
            elif not st.session_state.advanced_analysis_done:
                st.info("👆 Configure your parameters above and click 'RUN ADVANCED ANALYSIS' to begin.")
        
        # TAB 3: Export
        with tab3:
            st.markdown("### 💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download Indicators", key="download_indicators"):
                    csv = indicators.to_csv()
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"{ticker}_indicators_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )
            
            with col2:
                if st.button("📥 Download Data", key="download_data"):
                    csv = data.to_csv()
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"{ticker}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv"
                    )


if __name__ == "__main__":
    main()
