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

warnings.filterwarnings('ignore')

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(
    page_title="Rule-Extraction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== ESTILO ELEGANTE MODO OSCURO =====================
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
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #404040, transparent);
        margin: 2.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== CLASE DE INDICADORES TÉCNICOS =====================
class TechnicalIndicators:
    """Gestor completo de indicadores TALib (200+ indicadores)"""
    
    @staticmethod
    def calculate_single_indicator(name, h, l, c, v, o, p):
        """Calcular un indicador con manejo de errores"""
        try:
            h = np.asarray(h, dtype=np.float64)
            l = np.asarray(l, dtype=np.float64)
            c = np.asarray(c, dtype=np.float64)
            v = np.asarray(v, dtype=np.float64)
            o = np.asarray(o, dtype=np.float64)
            
            # Estudios de Superposición
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
            
            # Indicadores de Momentum
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
            
            # Indicadores de Volumen
            elif name == 'AD': return talib.AD(h, l, c, v)
            elif name == 'ADOSC': return talib.ADOSC(h, l, c, v, fastperiod=max(p//3, 2) if p else 3, slowperiod=p or 10)
            elif name == 'OBV': return talib.OBV(c, v)
            
            # Volatilidad
            elif name == 'ATR': return talib.ATR(h, l, c, timeperiod=p or 14)
            elif name == 'NATR': return talib.NATR(h, l, c, timeperiod=p or 14)
            elif name == 'TRANGE': return talib.TRANGE(h, l, c)
            
            # Indicadores de Ciclo
            elif name == 'HT_DCPERIOD': return talib.HT_DCPERIOD(c)
            elif name == 'HT_DCPHASE': return talib.HT_DCPHASE(c)
            elif name == 'HT_PHASOR':
                result = talib.HT_PHASOR(c)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'HT_SINE':
                result = talib.HT_SINE(c)
                return result[0] if isinstance(result, tuple) else result
            elif name == 'HT_TRENDMODE': return talib.HT_TRENDMODE(c)
            
            # Estadísticas
            elif name == 'BETA': return talib.BETA(h, l, timeperiod=p or 5)
            elif name == 'CORREL': return talib.CORREL(h, l, timeperiod=p or 30)
            elif name == 'LINEARREG': return talib.LINEARREG(c, timeperiod=p or 14)
            elif name == 'LINEARREG_ANGLE': return talib.LINEARREG_ANGLE(c, timeperiod=p or 14)
            elif name == 'LINEARREG_INTERCEPT': return talib.LINEARREG_INTERCEPT(c, timeperiod=p or 14)
            elif name == 'LINEARREG_SLOPE': return talib.LINEARREG_SLOPE(c, timeperiod=p or 14)
            elif name == 'STDDEV': return talib.STDDEV(c, timeperiod=p or 5)
            elif name == 'TSF': return talib.TSF(c, timeperiod=p or 14)
            elif name == 'VAR': return talib.VAR(c, timeperiod=p or 5)
            
            # Transformaciones Matemáticas
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
            
            # Operadores Matemáticos
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
            
            # Transformaciones de Precio
            elif name == 'AVGPRICE': return talib.AVGPRICE(o, h, l, c)
            elif name == 'MEDPRICE': return talib.MEDPRICE(h, l)
            elif name == 'TYPPRICE': return talib.TYPPRICE(h, l, c)
            elif name == 'WCLPRICE': return talib.WCLPRICE(h, l, c)
            
            # Patrones de velas
            elif name.startswith('CDL'):
                if hasattr(talib, name):
                    func = getattr(talib, name)
                    return func(o, h, l, c)
            
            return None
        except:
            return None
    
    # Lista completa de indicadores
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

# ===================== FLEXIBLE COMPOUND RULES ENGINE =====================
class FlexibleCompoundRulesEngine:
    """Flexible trading engine with adjustable rule generation"""
    
    @staticmethod
    def parse_simple_condition_safe(condition: str, data_df: pd.DataFrame) -> np.ndarray:
        """Parse with better error handling and NaN management"""
        try:
            parts = condition.strip().split()
            if len(parts) != 3:
                return np.zeros(len(data_df), dtype=bool)
                
            column, operator, value_str = parts
            
            if column not in data_df.columns:
                return np.zeros(len(data_df), dtype=bool)
                
            threshold = float(value_str)
            col_values = data_df[column].values
            
            # Ensure no NaN comparisons
            valid_mask = ~np.isnan(col_values)
            result = np.zeros(len(data_df), dtype=bool)
            
            if operator == '>':
                result[valid_mask] = col_values[valid_mask] > threshold
            elif operator == '>=':
                result[valid_mask] = col_values[valid_mask] >= threshold
            elif operator == '<':
                result[valid_mask] = col_values[valid_mask] < threshold
            elif operator == '<=':
                result[valid_mask] = col_values[valid_mask] <= threshold
                
            return result
        except Exception:
            return np.zeros(len(data_df), dtype=bool)
    
    @staticmethod
    def parse_compound_condition(condition: str, data_df: pd.DataFrame) -> np.ndarray:
        """Parse compound conditions with AND/OR logic"""
        condition = condition.strip()
        
        if ' AND ' not in condition and ' OR ' not in condition:
            return FlexibleCompoundRulesEngine.parse_simple_condition_safe(condition, data_df)
        
        # Handle AND logic
        if ' AND ' in condition:
            and_parts = condition.split(' AND ')
            and_results = []
            for part in and_parts:
                part_result = FlexibleCompoundRulesEngine.parse_simple_condition_safe(part.strip('()'), data_df)
                if part_result is not None:
                    and_results.append(part_result)
            
            if and_results:
                result = and_results[0]
                for r in and_results[1:]:
                    result = result & r
                return result
        
        # Handle OR logic  
        if ' OR ' in condition:
            or_parts = condition.split(' OR ')
            or_results = []
            for part in or_parts:
                part_result = FlexibleCompoundRulesEngine.parse_simple_condition_safe(part.strip('()'), data_df)
                if part_result is not None:
                    or_results.append(part_result)
            
            if or_results:
                result = or_results[0]
                for r in or_results[1:]:
                    result = result | r
                return result
        
        return None
    
    @staticmethod
    def generate_all_simple_rules(indicators_df: pd.DataFrame,
                                percentiles: List[int],
                                operators: List[str] = ['>', '<'],
                                max_indicators: int = None) -> List[Dict]:
        """Generate all simple rules without pre-filtering"""
        rules = []
        
        columns_to_use = indicators_df.columns[:max_indicators] if max_indicators else indicators_df.columns
        
        for col in columns_to_use:
            data = indicators_df[col].dropna()
            if len(data) < 100:
                continue
            
            for percentile in percentiles:
                threshold = data.quantile(percentile / 100)
                
                for operator in operators:
                    # Determine trade type based on operator and percentile
                    if operator == '<':
                        trade_type = 'BUY' if percentile <= 50 else 'SELL'
                    else:  # '>'
                        trade_type = 'SELL' if percentile >= 50 else 'BUY'
                    
                    rules.append({
                        'condition': f"{col} {operator} {threshold:.6f}",
                        'indicator': col,
                        'operator': operator,
                        'threshold': threshold,
                        'percentile': percentile,
                        'type': 'simple',
                        'trade_type': trade_type
                    })
        
        return rules
    
    @staticmethod
    def generate_compound_rules_from_simple(simple_rules: List[Dict],
                                          max_combinations: int = 1000,
                                          only_same_type: bool = True) -> List[Dict]:
        """Generate compound rules from simple rules"""
        compound_rules = []
        
        if only_same_type:
            # Separate by trade type
            buy_rules = [r for r in simple_rules if r['trade_type'] == 'BUY']
            sell_rules = [r for r in simple_rules if r['trade_type'] == 'SELL']
            
            # Generate BUY combinations
            for i, r1 in enumerate(buy_rules[:min(30, len(buy_rules))]):
                for r2 in buy_rules[i+1:min(50, len(buy_rules))]:
                    if r1['indicator'] != r2['indicator']:
                        compound_rules.append({
                            'condition': f"({r1['condition']}) AND ({r2['condition']})",
                            'type': 'compound',
                            'logic': 'AND',
                            'trade_type': 'BUY',
                            'components': [r1['indicator'], r2['indicator']]
                        })
                        
                        if len(compound_rules) >= max_combinations // 2:
                            break
                
                if len(compound_rules) >= max_combinations // 2:
                    break
            
            # Generate SELL combinations
            for i, r1 in enumerate(sell_rules[:min(30, len(sell_rules))]):
                for r2 in sell_rules[i+1:min(50, len(sell_rules))]:
                    if r1['indicator'] != r2['indicator']:
                        compound_rules.append({
                            'condition': f"({r1['condition']}) AND ({r2['condition']})",
                            'type': 'compound',
                            'logic': 'AND',
                            'trade_type': 'SELL',
                            'components': [r1['indicator'], r2['indicator']]
                        })
                        
                        if len(compound_rules) >= max_combinations:
                            break
                
                if len(compound_rules) >= max_combinations:
                    break
        
        return compound_rules
    
    @staticmethod
    def evaluate_rules_batch(rules: List[Dict],
                           data_df: pd.DataFrame,
                           indicators_df: pd.DataFrame,
                           holding_period: int = 5,
                           min_signals: int = 20,
                           batch_size: int = 100) -> pd.DataFrame:
        """Evaluate rules in batches with proper alignment"""
        
        # Ensure alignment
        common_idx = indicators_df.index.intersection(data_df.index)
        indicators_df = indicators_df.loc[common_idx]
        data_df = data_df.loc[common_idx]
        
        # Prepare evaluation dataframe
        eval_df = pd.concat([indicators_df, data_df[['Close']]], axis=1)
        
        # Pre-calculate returns
        returns = eval_df['Close'].pct_change(holding_period).shift(-holding_period) * 100
        returns_array = returns.values
        
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for batch_idx in range(0, len(rules), batch_size):
            batch = rules[batch_idx:batch_idx + batch_size]
            
            # Update progress
            progress = min((batch_idx + batch_size) / len(rules), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Evaluating rules {batch_idx+1} to {min(batch_idx+batch_size, len(rules))} of {len(rules)}")
            
            for rule in batch:
                try:
                    # Parse condition
                    signals = FlexibleCompoundRulesEngine.parse_compound_condition(
                        rule['condition'], eval_df
                    )
                    
                    if signals is None or signals.sum() < min_signals:
                        continue
                    
                    # Get signal returns
                    signal_returns = returns_array[signals]
                    
                    # Adjust for trade type
                    if rule.get('trade_type') == 'SELL':
                        signal_returns = -signal_returns
                    
                    signal_returns = signal_returns[~np.isnan(signal_returns)]
                    
                    if len(signal_returns) < 10:
                        continue
                    
                    # Calculate metrics
                    mean_ret = np.mean(signal_returns)
                    std_ret = np.std(signal_returns)
                    sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(252 / holding_period)
                    
                    # Win rate
                    win_rate = (signal_returns > 0).mean() * 100
                    
                    # Profit factor
                    pos_returns = signal_returns[signal_returns > 0].sum()
                    neg_returns = -signal_returns[signal_returns < 0].sum()
                    profit_factor = pos_returns / neg_returns if neg_returns > 0 else 999
                    
                    # Store result
                    results.append({
                        'condition': rule['condition'],
                        'type': rule.get('type', 'simple'),
                        'trade_type': rule.get('trade_type', 'UNKNOWN'),
                        'logic': rule.get('logic', 'simple'),
                        'num_signals': int(signals.sum()),
                        'mean_return': mean_ret,
                        'std_return': std_ret,
                        'sharpe': sharpe,
                        'win_rate': win_rate,
                        'profit_factor': min(profit_factor, 999),
                        'signal_pct': (signals.sum() / len(signals)) * 100
                    })
                    
                except Exception:
                    continue
        
        progress_bar.empty()
        status_text.empty()
        
        if results:
            return pd.DataFrame(results).sort_values('sharpe', ascending=False)
        return pd.DataFrame()
    
    @staticmethod
    def select_diverse_rules(results_df: pd.DataFrame,
                           n_rules: int = 10,
                           min_signals: int = 20) -> List[Dict]:
        """Select diverse rules based on multiple metrics"""
        if results_df.empty:
            return []
        
        # Filter by minimum signals
        valid_df = results_df[results_df['num_signals'] >= min_signals]
        
        if valid_df.empty:
            return []
        
        selected = []
        
        # Get top rules by different metrics
        n_per_metric = max(n_rules // 3, 1)
        
        top_sharpe = valid_df.nlargest(n_per_metric, 'sharpe')
        top_pf = valid_df.nlargest(n_per_metric, 'profit_factor')
        top_win = valid_df.nlargest(n_per_metric, 'win_rate')
        
        # Combine and deduplicate
        combined = pd.concat([top_sharpe, top_pf, top_win]).drop_duplicates(subset=['condition'])
        
        # Convert to list of dicts
        for _, row in combined.head(n_rules).iterrows():
            selected.append({
                'condition': row['condition'],
                'type': row['type'],
                'trade_type': row['trade_type'],
                'sharpe': row['sharpe'],
                'profit_factor': row['profit_factor'],
                'win_rate': row['win_rate'],
                'num_signals': row['num_signals']
            })
        
        return selected

# ===================== IMPROVED BACKTEST ENGINE =====================
class ImprovedBacktest:
    """Improved backtesting engine with proper alignment"""
    
    @staticmethod
    def backtest_rules(selected_rules: List[Dict],
                      indicators_df: pd.DataFrame,
                      data_df: pd.DataFrame,
                      initial_capital: float = 10000,
                      commission: float = 0.001) -> Tuple[pd.DataFrame, Dict]:
        """Run backtest with selected rules and proper alignment"""
        
        # Ensure alignment
        common_idx = indicators_df.index.intersection(data_df.index)
        
        if len(common_idx) == 0:
            st.error("No common dates between indicators and price data")
            return pd.DataFrame(), {}
        
        indicators_df = indicators_df.loc[common_idx]
        data_df = data_df.loc[common_idx]
        
        portfolio = pd.DataFrame(index=common_idx)
        portfolio['price'] = data_df['Close']
        portfolio['returns'] = portfolio['price'].pct_change()
        
        # Combine for evaluation
        eval_df = pd.concat([indicators_df, data_df[['Close']]], axis=1)
        
        # Initialize signals
        buy_signals = np.zeros(len(portfolio))
        sell_signals = np.zeros(len(portfolio))
        
        # Apply all rules
        for rule in selected_rules:
            try:
                signals = FlexibleCompoundRulesEngine.parse_compound_condition(
                    rule['condition'], eval_df
                )
                
                if signals is not None and signals.sum() > 0:
                    if rule.get('trade_type') == 'BUY':
                        buy_signals += signals.astype(float)
                    elif rule.get('trade_type') == 'SELL':
                        sell_signals += signals.astype(float)
            except Exception:
                continue
        
        # Generate position signal (-1, 0, 1)
        portfolio['buy_signals'] = buy_signals
        portfolio['sell_signals'] = sell_signals
        portfolio['signal'] = np.where(buy_signals > sell_signals, 1, 
                                      np.where(sell_signals > buy_signals, -1, 0))
        
        # Forward fill positions
        portfolio['position'] = portfolio['signal'].replace(0, np.nan)
        portfolio['position'] = portfolio['position'].fillna(method='ffill').fillna(0)
        
        # Calculate trades
        portfolio['trades'] = portfolio['position'].diff().fillna(0)
        
        # Calculate costs and returns
        portfolio['commission'] = np.abs(portfolio['trades']) * commission
        portfolio['strategy_returns'] = (
            portfolio['position'].shift(1) * portfolio['returns'] - portfolio['commission']
        )
        
        # Cumulative returns (properly compounded)
        portfolio['cum_returns'] = (1 + portfolio['strategy_returns'].fillna(0)).cumprod()
        portfolio['cum_market'] = (1 + portfolio['returns'].fillna(0)).cumprod()
        
        # Equity and drawdown
        portfolio['equity'] = initial_capital * portfolio['cum_returns']
        portfolio['peak'] = portfolio['equity'].cummax()
        portfolio['drawdown'] = (portfolio['equity'] / portfolio['peak'] - 1) * 100
        
        # Calculate metrics
        total_return = (portfolio['equity'].iloc[-1] / initial_capital - 1) * 100
        market_return = (portfolio['cum_market'].iloc[-1] - 1) * 100
        
        daily_returns = portfolio['strategy_returns'].dropna()
        
        if len(daily_returns) > 0:
            sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252)
            max_dd = portfolio['drawdown'].min()
            
            winning_days = (daily_returns > 0).sum()
            total_days = len(daily_returns[daily_returns != 0])
            win_rate = (winning_days / max(total_days, 1)) * 100
            
            positive_returns = daily_returns[daily_returns > 0].sum()
            negative_returns = -daily_returns[daily_returns < 0].sum()
            profit_factor = positive_returns / negative_returns if negative_returns > 0 else 999
            
            num_trades = (portfolio['trades'] != 0).sum()
        else:
            sharpe = 0
            max_dd = 0
            win_rate = 0
            profit_factor = 0
            num_trades = 0
        
        metrics = {
            'cum_return': total_return,
            'market_return': market_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'profit_factor': min(profit_factor, 999),
            'num_trades': num_trades,
            'final_equity': portfolio['equity'].iloc[-1] if len(portfolio) > 0 else initial_capital
        }
        
        return portfolio, metrics

# ===================== FUNCIONES DE CÁLCULO =====================
@st.cache_data
def download_data(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """Descargar datos históricos"""
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True, multi_level_index=False)
        if data.empty:
            st.error(f"No se encontraron datos para {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error descargando datos: {str(e)}")
        return None

@st.cache_data
def calculate_all_indicators(ticker: str, period: str, quantiles: int, min_return_days: int, 
                             max_return_days: int, periods_to_test: List[int], 
                             selected_categories: List[str]) -> Tuple:
    """Calcular todos los indicadores seleccionados"""
    
    data = download_data(ticker, period)
    if data is None:
        return None, None, None, None
    
    for i in range(min_return_days, max_return_days + 1):
        data[f'retornos_{i}_dias'] = data['Close'].pct_change(i) * 100
    
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
    
    total_calculations = sum(
        len(periods_to_test) if TechnicalIndicators.needs_period(ind) else 1 
        for ind in indicators_to_calc
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    calculation_counter = 0
    successful = 0
    
    for indicator_name in indicators_to_calc:
        if TechnicalIndicators.needs_period(indicator_name):
            for period in periods_to_test:
                calculation_counter += 1
                status_text.text(f"⏳ Calculando {indicator_name}_{period}...")
                
                result = TechnicalIndicators.calculate_indicator(
                    indicator_name, high, low, close, volume, open_prices, period
                )
                
                if result is not None:
                    indicators[f'{indicator_name}_{period}'] = result
                    successful += 1
                
                progress_bar.progress(calculation_counter / total_calculations)
        else:
            calculation_counter += 1
            status_text.text(f"⏳ Calculando {indicator_name}...")
            
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
            ✓ Calculados exitosamente {successful} de {total_calculations} configuraciones
        </div>
    """, unsafe_allow_html=True)
    
    summary = {
        'total_attempted': total_calculations,
        'successful': successful,
        'indicators_count': len(indicators.columns),
        'data_points': len(data),
        'date_range': f"{data.index[0].strftime('%Y-%m-%d')} a {data.index[-1].strftime('%Y-%m-%d')}",
        'min_return_days': min_return_days,
        'max_return_days': max_return_days
    }
    
    return returns_data, indicators, data, summary

def create_percentile_plot(indicators, returns_data, data, indicator_name, return_days, quantiles=10):
    """Crear gráficos de análisis mejorados"""
    
    if indicator_name not in indicators.columns or indicator_name not in returns_data:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>Distribución y Estadísticas</b>', '<b>Retornos por Percentil</b>',
            '<b>Correlación Móvil (126 días)</b>', '<b>Análisis de Dispersión</b>'
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
        mean_val = hist_data.mean()
        median_val = hist_data.median()
        
        fig.add_trace(
            go.Histogram(
                x=hist_data,
                nbinsx=100,
                marker=dict(
                    color='rgba(100, 150, 255, 0.4)',
                    line=dict(color='rgba(100, 150, 255, 0.6)', width=0.5)
                ),
                name='Distribución',
                showlegend=False,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        kde = gaussian_kde(hist_data.values)
        x_range = np.linspace(hist_data.min(), hist_data.max(), 200)
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
                
                fig.add_trace(
                    go.Scattergl(
                        x=x_clean,
                        y=y_clean,
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=y_clean,
                            colorscale='RdYlGn',
                            opacity=0.6,
                            line=dict(width=0),
                            showscale=True
                        ),
                        showlegend=False
                    ),
                    row=2, col=2
                )
    
    fig.update_layout(
        template="plotly_dark",
        height=800,
        title={
            'text': f"<b>{indicator_name}</b> | Análisis de Retornos a {return_days} Días",
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

# ===================== APLICACIÓN PRINCIPAL =====================
def main():
    st.markdown("""
        <h1 class='main-header'>Plataforma de Análisis Cuantitativo</h1>
        <p class='sub-header'>
            {total} INDICADORES TÉCNICOS · BACKTESTING IS/OOS · ANÁLISIS PERCENTIL
        </p>
    """.format(total=TechnicalIndicators.get_total_count()), unsafe_allow_html=True)
    
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.returns_data = None
        st.session_state.indicators = None
        st.session_state.data = None
        st.session_state.summary = None
    
    st.markdown("<div class='config-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Configuración de Datos</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])
    
    with col1:
        ticker = st.text_input("SÍMBOLO", value="SPY", help="Símbolo bursátil a analizar")
    
    with col2:
        period_option = st.selectbox(
            "PERÍODO",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=4,
            format_func=lambda x: {
                "1mo": "1 Mes", "3mo": "3 Meses", "6mo": "6 Meses",
                "1y": "1 Año", "2y": "2 Años", "5y": "5 Años",
                "10y": "10 Años", "max": "Máximo"
            }.get(x, x)
        )
    
    with col3:
        col3a, col3b = st.columns(2)
        with col3a:
            min_return_days = st.number_input("DÍAS MÍN", value=1, min_value=1, max_value=30)
        with col3b:
            max_return_days = st.number_input("DÍAS MÁX", value=10, min_value=1, max_value=30)
        
        if max_return_days < min_return_days:
            st.error("Los días máximos deben ser mayores o iguales a los días mínimos")
    
    with col4:
        quantiles = st.number_input("PERCENTILES", value=10, min_value=5, max_value=20, step=5)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='config-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Configuración de Períodos</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        min_period = st.number_input("MÍN", value=5, min_value=2, max_value=500)
    
    with col2:
        max_period = st.number_input("MÁX", value=50, min_value=5, max_value=500)
    
    with col3:
        step_period = st.number_input("PASO", value=5, min_value=1, max_value=50)
    
    with col4:
        periods_to_test = list(range(min_period, max_period + 1, step_period))
        st.markdown(f"""
            <div class="info-badge">
                Períodos: {', '.join(map(str, periods_to_test))}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='config-section'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Selección de Indicadores</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        select_mode = st.radio(
            "MODO",
            ["Presets", "Categorías", "Todo"]
        )
    
    with col2:
        if select_mode == "Presets":
            preset = st.selectbox(
                "CONFIGURACIÓN",
                ["Esencial (30 indicadores)", 
                 "Extendido (60 indicadores)", 
                 "Completo (100 indicadores)", 
                 "Todo (158+ indicadores)"]
            )
            
            if "Esencial" in preset:
                selected_categories = ["Superposición", "Momentum"][:1]
            elif "Extendido" in preset:
                selected_categories = ["Momentum", "Volatilidad", "Volumen", "Superposición"]
            elif "Completo" in preset:
                selected_categories = list(TechnicalIndicators.CATEGORIES.keys())[:7]
            else:
                selected_categories = ["TODO"]
        
        elif select_mode == "Categorías":
            selected_categories = st.multiselect(
                "SELECCIONAR CATEGORÍAS",
                list(TechnicalIndicators.CATEGORIES.keys()),
                default=["Momentum", "Superposición"]
            )
        else:
            selected_categories = ["TODO"]
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        analyze_button = st.button(
            "ANALIZAR",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button and max_return_days >= min_return_days:
        with st.spinner('Procesando indicadores...'):
            returns_data, indicators, data, summary = calculate_all_indicators(
                ticker,
                period_option,
                quantiles,
                min_return_days,
                max_return_days,
                periods_to_test,
                selected_categories
            )
            
            if returns_data and indicators is not None and data is not None:
                st.session_state.analysis_done = True
                st.session_state.returns_data = returns_data
                st.session_state.indicators = indicators
                st.session_state.data = data
                st.session_state.summary = summary
    
    if st.session_state.analysis_done:
        returns_data = st.session_state.returns_data
        indicators = st.session_state.indicators
        data = st.session_state.data
        summary = st.session_state.summary
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("INDICADORES", summary['indicators_count'])
        with col2:
            st.metric("TASA DE ÉXITO", f"{(summary['successful']/summary['total_attempted']*100):.1f}%")
        with col3:
            st.metric("PUNTOS DE DATOS", summary['data_points'])
        with col4:
            st.metric("RANGO", summary['date_range'].split(' a ')[0])
        
        tab1, tab2, tab3 = st.tabs(["📊 ANÁLISIS", "🎯 REGLAS DE TRADING", "💾 EXPORTAR"])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_indicator = st.selectbox(
                    "SELECCIONAR INDICADOR",
                    sorted(indicators.columns)
                )
            with col2:
                return_period = st.number_input(
                    "DÍAS",
                    min_value=summary['min_return_days'],
                    max_value=summary['max_return_days'],
                    value=min(5, summary['max_return_days'])
                )
            
            if selected_indicator:
                fig = create_percentile_plot(
                    indicators, returns_data, data,
                    selected_indicator, return_period, quantiles
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🎯 Advanced Trading Rules System")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sample_split = st.slider(
                    "% IN-SAMPLE",
                    min_value=50,
                    max_value=80,
                    value=70,
                    step=5
                )
            
            with col2:
                holding_period = st.selectbox(
                    "HOLDING PERIOD (days)",
                    list(range(summary['min_return_days'], summary['max_return_days'] + 1)),
                    index=min(4, summary['max_return_days'] - summary['min_return_days'])
                )
            
            with col3:
                min_signals = st.number_input(
                    "MIN SIGNALS",
                    min_value=10,
                    max_value=100,
                    value=20,
                    help="Minimum signals for valid rule"
                )
            
            with col4:
                top_rules = st.number_input(
                    "TOP RULES TO SELECT",
                    min_value=3,
                    max_value=20,
                    value=10
                )
            
            with st.expander("⚙️ Advanced Settings"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rule_generation_mode = st.selectbox(
                        "Rule Generation Mode",
                        ["Fast (200 rules)", "Normal (1000 rules)", 
                         "Thorough (5000 rules)", "Complete (All)"]
                    )
                    
                    if rule_generation_mode == "Fast (200 rules)":
                        max_rules_to_test = 200
                        percentiles_to_use = [20, 50, 80]
                    elif rule_generation_mode == "Normal (1000 rules)":
                        max_rules_to_test = 1000
                        percentiles_to_use = [10, 25, 50, 75, 90]
                    elif rule_generation_mode == "Thorough (5000 rules)":
                        max_rules_to_test = 5000
                        percentiles_to_use = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
                    else:  # Complete
                        max_rules_to_test = 50000
                        percentiles_to_use = list(range(5, 100, 5))
                
                with col2:
                    use_compound_rules = st.checkbox("Use Compound Rules", value=True)
                    if use_compound_rules:
                        max_compound_rules = st.number_input(
                            "Max Compound Rules",
                            min_value=100,
                            max_value=5000,
                            value=min(500, max_rules_to_test // 2),
                            step=100
                        )
                
                with col3:
                    max_indicators_to_analyze = st.number_input(
                        "MAX INDICATORS",
                        min_value=5,
                        max_value=100,
                        value=20,
                        help="Limit to speed up"
                    )
                
                with col4:
                    min_sharpe = st.number_input(
                        "MIN SHARPE RATIO",
                        min_value=-2.0,
                        max_value=2.0,
                        value=0.0,
                        step=0.1,
                        help="Minimum Sharpe for rule selection"
                    )
            
            if st.button("🚀 RUN ANALYSIS", use_container_width=True, type="primary"):
                with st.spinner("Generating and testing rules..."):
                    
                    # Split data
                    split_index = int(len(data) * sample_split / 100)
                    
                    in_sample_data = data.iloc[:split_index].copy()
                    out_sample_data = data.iloc[split_index:].copy()
                    
                    in_sample_indicators = indicators.iloc[:split_index].copy()
                    out_sample_indicators = indicators.iloc[split_index:].copy()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"""
                        **In-Sample**: {len(in_sample_data)} points
                        {in_sample_data.index[0].strftime('%Y-%m-%d')} to {in_sample_data.index[-1].strftime('%Y-%m-%d')}
                        """)
                    with col2:
                        st.info(f"""
                        **Out-of-Sample**: {len(out_sample_data)} points
                        {out_sample_data.index[0].strftime('%Y-%m-%d')} to {out_sample_data.index[-1].strftime('%Y-%m-%d')}
                        """)
                    
                    # Generate rules
                    st.markdown("#### 📊 Generating Trading Rules")
                    engine = FlexibleCompoundRulesEngine()
                    
                    # Generate simple rules
                    simple_rules = engine.generate_all_simple_rules(
                        in_sample_indicators,
                        percentiles_to_use,
                        ['>', '<'],
                        max_indicators_to_analyze
                    )
                    
                    st.success(f"✅ Generated {len(simple_rules)} simple rules")
                    
                    # Generate compound rules if enabled
                    if use_compound_rules:
                        compound_rules = engine.generate_compound_rules_from_simple(
                            simple_rules,
                            max_compound_rules
                        )
                        st.success(f"✅ Generated {len(compound_rules)} compound rules")
                        all_rules = simple_rules + compound_rules
                    else:
                        all_rules = simple_rules
                    
                    # Evaluate rules on IS
                    st.markdown("#### 📈 Evaluating Rules (In-Sample)")
                    is_results = engine.evaluate_rules_batch(
                        all_rules[:max_rules_to_test],
                        in_sample_data,
                        in_sample_indicators,
                        holding_period,
                        min_signals
                    )
                    
                    if not is_results.empty:
                        # Filter by minimum Sharpe
                        is_results = is_results[is_results['sharpe'] >= min_sharpe]
                        
                        if not is_results.empty:
                            # Select diverse rules
                            best_rules = engine.select_diverse_rules(
                                is_results,
                                top_rules,
                                min_signals
                            )
                            
                            st.success(f"✅ Selected {len(best_rules)} best rules")
                            
                            # Display top rules
                            st.markdown("##### Top Rules (In-Sample)")
                            display_df = is_results[['condition', 'type', 'trade_type', 'num_signals', 
                                                    'sharpe', 'profit_factor', 'win_rate']].head(15).copy()
                            display_df.columns = ['Rule', 'Type', 'Trade', 'Signals', 'Sharpe', 'PF', 'Win%']
                            
                            st.dataframe(
                                display_df.style.format({
                                    'Sharpe': '{:.2f}',
                                    'PF': '{:.2f}',
                                    'Win%': '{:.1f}'
                                }).background_gradient(subset=['Sharpe'], cmap='RdYlGn'),
                                use_container_width=True,
                                height=400
                            )
                            
                            # Evaluate on OOS
                            st.markdown("#### 📊 Evaluating Rules (Out-of-Sample)")
                            oos_results = engine.evaluate_rules_batch(
                                best_rules,
                                out_sample_data,
                                out_sample_indicators,
                                holding_period,
                                min_signals=5  # Less restrictive for OOS
                            )
                            
                            # Backtest
                            st.markdown("#### 💼 Backtesting Strategy")
                            
                            # Full period backtest
                            full_portfolio, full_metrics = ImprovedBacktest.backtest_rules(
                                best_rules[:min(5, len(best_rules))],
                                indicators,
                                data,
                                10000
                            )
                            
                            # IS backtest
                            is_portfolio, is_metrics = ImprovedBacktest.backtest_rules(
                                best_rules[:min(5, len(best_rules))],
                                in_sample_indicators,
                                in_sample_data,
                                10000
                            )
                            
                            # OOS backtest
                            oos_portfolio, oos_metrics = ImprovedBacktest.backtest_rules(
                                best_rules[:min(5, len(best_rules))],
                                out_sample_indicators,
                                out_sample_data,
                                10000
                            )
                            
                            # Performance table
                            st.markdown("##### Performance Metrics")
                            
                            metrics_table = pd.DataFrame({
                                'Metric': ['Cumulative Return (%)', 'Market Return (%)', 'Sharpe Ratio', 
                                          'Max Drawdown (%)', 'Profit Factor', 'Win Rate (%)'],
                                'In-Sample': [
                                    is_metrics['cum_return'],
                                    is_metrics.get('market_return', 0),
                                    is_metrics['sharpe'],
                                    is_metrics['max_dd'],
                                    is_metrics['profit_factor'],
                                    is_metrics['win_rate']
                                ],
                                'Out-of-Sample': [
                                    oos_metrics['cum_return'],
                                    oos_metrics.get('market_return', 0),
                                    oos_metrics['sharpe'],
                                    oos_metrics['max_dd'],
                                    oos_metrics['profit_factor'],
                                    oos_metrics['win_rate']
                                ],
                                'Full Period': [
                                    full_metrics['cum_return'],
                                    full_metrics.get('market_return', 0),
                                    full_metrics['sharpe'],
                                    full_metrics['max_dd'],
                                    full_metrics['profit_factor'],
                                    full_metrics['win_rate']
                                ]
                            })
                            
                            st.dataframe(
                                metrics_table.style.format({
                                    'In-Sample': '{:.2f}',
                                    'Out-of-Sample': '{:.2f}',
                                    'Full Period': '{:.2f}'
                                }).background_gradient(axis=1, cmap='RdYlGn'),
                                use_container_width=True
                            )
                            
                            # Equity curve
                            if len(full_portfolio) > 0:
                                fig = go.Figure()
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=full_portfolio.index,
                                        y=full_portfolio['equity'],
                                        mode='lines',
                                        name='Strategy',
                                        line=dict(color='#51CF66', width=2.5)
                                    )
                                )
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=full_portfolio.index,
                                        y=10000 * full_portfolio['cum_market'],
                                        mode='lines',
                                        name='Buy & Hold',
                                        line=dict(color='#808080', width=1.5, dash='dash')
                                    )
                                )
                                
                                # Add vertical line for IS/OOS split (fixed for timestamp issue)
                                fig.add_shape(
                                    type="line",
                                    x0=data.index[split_index],
                                    x1=data.index[split_index],
                                    y0=0,
                                    y1=1,
                                    yref="paper",
                                    line=dict(color='#FF6B6B', width=2, dash='dash')
                                )
                                fig.add_annotation(
                                    x=data.index[split_index],
                                    y=1,
                                    yref="paper",
                                    text="IS | OOS",
                                    showarrow=False,
                                    yshift=10
                                )
                                
                                fig.update_layout(
                                    height=600,
                                    template="plotly_dark",
                                    title="Equity Curve",
                                    xaxis_title="Date",
                                    yaxis_title="Equity ($)",
                                    showlegend=True,
                                    legend=dict(x=0.01, y=0.99),
                                    paper_bgcolor='#0D1117',
                                    plot_bgcolor='#161B22'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Key metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "TOTAL RETURN",
                                    f"{full_metrics['cum_return']:.1f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "SHARPE RATIO",
                                    f"{full_metrics['sharpe']:.2f}"
                                )
                            
                            with col3:
                                st.metric(
                                    "MAX DRAWDOWN",
                                    f"{full_metrics['max_dd']:.1f}%"
                                )
                            
                            with col4:
                                st.metric(
                                    "PROFIT FACTOR",
                                    f"{full_metrics['profit_factor']:.2f}"
                                )
                        else:
                            st.warning("No rules met the minimum Sharpe ratio requirement")
                    else:
                        st.warning("No valid rules found. Try adjusting parameters.")
        
        with tab3:
            st.markdown("### 💾 Opciones de Exportación")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Descargar Indicadores"):
                    csv = indicators.to_csv()
                    st.download_button(
                        "Descargar CSV",
                        data=csv,
                        file_name=f"{ticker}_indicadores_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📥 Descargar Datos"):
                    csv = data.to_csv()
                    st.download_button(
                        "Descargar CSV",
                        data=csv,
                        file_name=f"{ticker}_datos_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
