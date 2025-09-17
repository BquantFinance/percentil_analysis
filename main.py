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
from scipy.stats import spearmanr, percentileofscore
from sklearn.tree import DecisionTreeRegressor
import json

warnings.filterwarnings('ignore')

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(
    page_title="Advanced Quantitative Pattern Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== ESTILOS CSS =====================
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
        font-size: 2.8rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(99, 102, 241, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.35);
    }
    
    .pattern-card {
        background: rgba(30, 34, 56, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .stability-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 4px;
    }
    
    .high-stability {
        background: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        border: 1px solid #4CAF50;
    }
    
    .medium-stability {
        background: rgba(255, 193, 7, 0.2);
        color: #FFC107;
        border: 1px solid #FFC107;
    }
    
    .low-stability {
        background: rgba(244, 67, 54, 0.2);
        color: #F44336;
        border: 1px solid #F44336;
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== CLASE COMPLETA DE INDICADORES TÉCNICOS =====================
class TechnicalIndicators:
    """Manejador completo de TODOS los indicadores TALib (200+)"""
    
    # Configuración completa de todos los indicadores
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
    
    # Lista de todos los patrones de velas (61 patrones)
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
    
    # Categorías organizadas
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
        """Detecta qué inputs necesita cada función"""
        # Patrones de velas
        if func_name.startswith('CDL'):
            return 'ohlc'
        # Volume
        elif func_name in ['AD', 'ADOSC']:
            return 'hlcv'
        elif func_name in ['OBV']:
            return 'cv'
        # Price
        elif func_name in ['AVGPRICE']:
            return 'ohlc'
        elif func_name in ['MEDPRICE', 'MIDPRICE']:
            return 'hl'
        elif func_name in ['TYPPRICE', 'WCLPRICE']:
            return 'hlc'
        # Technical
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
        # Math operators
        elif func_name in ['ADD', 'DIV', 'MULT', 'SUB']:
            return 'cc'
        elif func_name in ['MAX', 'MAXINDEX', 'MIN', 'MININDEX', 'MINMAX', 'MINMAXINDEX', 'SUM']:
            return 'c'
        # Default
        else:
            return 'c'
    
    @classmethod
    def calculate_indicator(cls, indicator_name, high, low, close, volume, open_prices, period):
        """Calcula cualquier indicador de TALib"""
        try:
            # Patrones de velas (directo)
            if indicator_name.startswith('CDL'):
                func = getattr(talib, indicator_name)
                return func(open_prices, high, low, close)
            
            # Indicador configurado
            if indicator_name not in cls.INDICATOR_CONFIG:
                if hasattr(talib, indicator_name):
                    func = getattr(talib, indicator_name)
                    return func(close)
                return None
            
            func_name, params = cls.INDICATOR_CONFIG[indicator_name]
            func = getattr(talib, func_name)
            
            data_type = cls._get_indicator_inputs(func_name)
            
            # Preparar argumentos según tipo
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
                args = [close, close]  # Para operadores matemáticos
            else:
                args = [close]
            
            # Procesar parámetros
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
                return result[0]
            
            return result
            
        except Exception:
            return None
    
    @classmethod
    def get_all_categories(cls):
        """Retorna todas las categorías con conteo"""
        return cls.CATEGORIES
    
    @classmethod
    def needs_period(cls, indicator_name):
        """Determina si un indicador necesita período"""
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
    def get_total_indicators(cls):
        """Retorna el número total de indicadores"""
        return len(cls.INDICATOR_CONFIG) + len(cls.CANDLE_PATTERNS)

# ===================== NUEVAS FUNCIONES DE ANÁLISIS AVANZADO =====================

def identify_market_regimes(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Identificar regímenes de mercado (volatilidad, tendencia, etc.)"""
    regimes = pd.DataFrame(index=data.index)
    
    # Volatility regime
    returns = data['Close'].pct_change()
    volatility = returns.rolling(lookback).std() * np.sqrt(252)
    vol_percentiles = volatility.rolling(252).apply(lambda x: percentileofscore(x, x.iloc[-1]) if len(x) >= 252 else np.nan)
    
    regimes['volatility_regime'] = pd.cut(vol_percentiles, bins=[0, 33, 66, 100], 
                                           labels=['Low Vol', 'Normal Vol', 'High Vol'])
    
    # Trend regime
    sma_20 = data['Close'].rolling(20).mean()
    sma_50 = data['Close'].rolling(50).mean()
    sma_200 = data['Close'].rolling(200).mean()
    
    conditions = [
        (data['Close'] > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200),
        (data['Close'] < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200)
    ]
    choices = ['Strong Uptrend', 'Strong Downtrend']
    regimes['trend_regime'] = np.select(conditions, choices, default='Ranging')
    
    # Momentum regime
    rsi = talib.RSI(data['Close'].values, timeperiod=14)
    regimes['momentum_regime'] = pd.cut(rsi, bins=[0, 30, 70, 100], 
                                         labels=['Oversold', 'Neutral', 'Overbought'])
    
    return regimes

def analyze_indicator_period_enhanced(indicator_values, returns, quantiles=10, window_size=252):
    """Análisis mejorado con métricas de estabilidad y robustez"""
    try:
        temp_df = pd.DataFrame({
            'indicator': indicator_values,
            'returns': returns
        }).dropna()
        
        if len(temp_df) < quantiles * 2:
            return None
        
        # Análisis básico de percentiles
        temp_df['percentile'] = pd.qcut(temp_df['indicator'], q=quantiles, labels=False, duplicates='drop')
        percentile_returns = temp_df.groupby('percentile')['returns'].agg(['mean', 'std', 'count'])
        
        if len(percentile_returns) < quantiles * 0.8:
            return None
        
        metrics = {}
        
        # 1. SPREAD Y RETORNOS
        top_return = percentile_returns['mean'].iloc[-1]
        bottom_return = percentile_returns['mean'].iloc[0]
        metrics['spread'] = top_return - bottom_return
        metrics['top_return'] = top_return
        metrics['bottom_return'] = bottom_return
        
        # 2. DIRECCIÓN Y MONOTONÍA
        correlation, p_value = spearmanr(range(len(percentile_returns)), percentile_returns['mean'].values)
        metrics['direction'] = correlation
        metrics['p_value'] = p_value
        
        # Test de monotonicidad más estricto
        returns_array = percentile_returns['mean'].values
        monotonic_increases = sum(returns_array[i+1] > returns_array[i] for i in range(len(returns_array)-1))
        metrics['monotonicity_score'] = monotonic_increases / (len(returns_array) - 1)
        
        # 3. SHARPE Y CONSISTENCIA
        metrics['sharpe'] = abs(metrics['spread']) / (percentile_returns['std'].mean() + 1e-8)
        
        # 4. STABILITY SCORE - Nuevo
        if len(temp_df) >= window_size * 2:
            rolling_spreads = []
            for start in range(0, len(temp_df) - window_size, window_size // 4):
                window_df = temp_df.iloc[start:start + window_size]
                if len(window_df) >= quantiles * 2:
                    try:
                        window_df['window_percentile'] = pd.qcut(window_df['indicator'], q=quantiles, labels=False, duplicates='drop')
                        window_returns = window_df.groupby('window_percentile')['returns'].mean()
                        if len(window_returns) >= quantiles * 0.8:
                            window_spread = window_returns.iloc[-1] - window_returns.iloc[0]
                            rolling_spreads.append(window_spread)
                    except:
                        continue
            
            if rolling_spreads:
                metrics['stability_score'] = 1 / (np.std(rolling_spreads) + 0.001)
                metrics['stability_std'] = np.std(rolling_spreads)
                metrics['avg_rolling_spread'] = np.mean(rolling_spreads)
            else:
                metrics['stability_score'] = 0
                metrics['stability_std'] = np.nan
                metrics['avg_rolling_spread'] = metrics['spread']
        else:
            metrics['stability_score'] = 0
            metrics['stability_std'] = np.nan
            metrics['avg_rolling_spread'] = metrics['spread']
        
        # 5. PERCENTILES ÓPTIMOS
        metrics['best_long_percentile'] = percentile_returns['mean'].idxmax() + 1
        metrics['best_short_percentile'] = percentile_returns['mean'].idxmin() + 1
        
        # 6. INFORMACIÓN DE MUESTRA
        metrics['min_samples'] = percentile_returns['count'].min()
        metrics['total_samples'] = percentile_returns['count'].sum()
        
        # 7. CONFIDENCE INTERVALS - Nuevo
        # Bootstrap para intervalos de confianza
        n_bootstrap = 100
        bootstrap_spreads = []
        for _ in range(n_bootstrap):
            sample_df = temp_df.sample(n=len(temp_df), replace=True)
            try:
                sample_df['boot_percentile'] = pd.qcut(sample_df['indicator'], q=quantiles, labels=False, duplicates='drop')
                boot_returns = sample_df.groupby('boot_percentile')['returns'].mean()
                if len(boot_returns) >= quantiles * 0.8:
                    bootstrap_spreads.append(boot_returns.iloc[-1] - boot_returns.iloc[0])
            except:
                continue
        
        if bootstrap_spreads:
            metrics['spread_ci_lower'] = np.percentile(bootstrap_spreads, 5)
            metrics['spread_ci_upper'] = np.percentile(bootstrap_spreads, 95)
            metrics['spread_ci_width'] = metrics['spread_ci_upper'] - metrics['spread_ci_lower']
        else:
            metrics['spread_ci_lower'] = metrics['spread']
            metrics['spread_ci_upper'] = metrics['spread']
            metrics['spread_ci_width'] = 0
        
        return metrics
        
    except Exception as e:
        return None

def calculate_information_coefficient(indicator_values, forward_returns, window=252):
    """Calcular coeficiente de información a lo largo del tiempo"""
    df = pd.DataFrame({
        'indicator': indicator_values,
        'returns': forward_returns
    }).dropna()
    
    if len(df) < window:
        return None
    
    # IC rolling
    ic_series = df['indicator'].rolling(window).corr(df['returns'])
    
    # IC por régimen (últimos valores)
    recent_ic = ic_series.iloc[-window:] if len(ic_series) >= window else ic_series
    
    return {
        'current_ic': ic_series.iloc[-1] if not ic_series.empty else np.nan,
        'avg_ic': ic_series.mean(),
        'ic_std': ic_series.std(),
        'ic_stability': ic_series.mean() / (ic_series.std() + 1e-8),
        'ic_trend': np.polyfit(range(len(recent_ic)), recent_ic.values, 1)[0] if len(recent_ic) > 1 else 0,
        'positive_ic_pct': (ic_series > 0).mean() * 100
    }

def walk_forward_validation(indicator_name, data, period, window_size=504, step_size=126, return_days=5):
    """Validación walk-forward para probar estabilidad temporal"""
    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    close = data['Close'].values.astype(np.float64)
    volume = data['Volume'].values.astype(np.float64) if 'Volume' in data.columns else np.zeros_like(close)
    open_prices = data['Open'].values.astype(np.float64)
    
    returns = data['Close'].pct_change(return_days).shift(-return_days) * 100
    
    indicator_values = TechnicalIndicators.calculate_indicator(
        indicator_name, high, low, close, volume, open_prices, period
    )
    
    if indicator_values is None or np.all(np.isnan(indicator_values)):
        return None
    
    results = []
    
    for start_idx in range(0, len(data) - window_size, step_size):
        end_idx = start_idx + window_size
        
        window_indicator = indicator_values[start_idx:end_idx]
        window_returns = returns.values[start_idx:end_idx]
        
        metrics = analyze_indicator_period_enhanced(window_indicator, window_returns, quantiles=10, window_size=252)
        
        if metrics:
            metrics['window_start'] = data.index[start_idx]
            metrics['window_end'] = data.index[end_idx - 1]
            results.append(metrics)
    
    if not results:
        return None
    
    df_results = pd.DataFrame(results)
    
    return {
        'mean_spread': df_results['spread'].mean(),
        'std_spread': df_results['spread'].std(),
        'consistency_score': df_results['spread'].mean() / (df_results['spread'].std() + 1e-8),
        'windows_tested': len(df_results),
        'positive_spread_pct': (df_results['spread'] > 0).mean() * 100,
        'avg_stability': df_results['stability_score'].mean() if 'stability_score' in df_results else 0,
        'time_series': df_results
    }

def find_optimal_period_enhanced(indicator_name, data, periods_to_test, return_days=5):
    """Encuentra el período óptimo con análisis mejorado"""
    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    close = data['Close'].values.astype(np.float64)
    volume = data['Volume'].values.astype(np.float64) if 'Volume' in data.columns else np.zeros_like(close)
    open_prices = data['Open'].values.astype(np.float64)
    
    returns = data['Close'].pct_change(return_days).shift(-return_days) * 100
    
    results = []
    
    for period in periods_to_test:
        indicator_values = TechnicalIndicators.calculate_indicator(
            indicator_name, high, low, close, volume, open_prices, period
        )
        
        if indicator_values is None or np.all(np.isnan(indicator_values)):
            continue
        
        # Análisis mejorado con estabilidad
        metrics = analyze_indicator_period_enhanced(indicator_values, returns.values, quantiles=10)
        
        if metrics and metrics['min_samples'] >= 10:
            metrics['period'] = period
            metrics['indicator'] = indicator_name
            
            # Calcular IC
            ic_metrics = calculate_information_coefficient(indicator_values, returns.values, window=252)
            if ic_metrics:
                metrics.update({f'ic_{k}': v for k, v in ic_metrics.items()})
            
            results.append(metrics)
    
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    
    # Score compuesto mejorado
    df_results['composite_score'] = (
        abs(df_results['spread']) * 0.3 +
        df_results['sharpe'] * 10 +
        df_results.get('stability_score', 0) * 5 +
        (1 / (df_results['p_value'] + 0.001)) * 0.1 +
        df_results.get('ic_stability', 0) * 2
    )
    
    return df_results.sort_values('composite_score', ascending=False)

def discover_patterns(optimal_results: pd.DataFrame, min_spread: float = 2.0, 
                      max_p_value: float = 0.1) -> pd.DataFrame:
    """Descubre patrones estadísticamente significativos sin prescribir reglas"""
    if optimal_results.empty:
        return pd.DataFrame()
    
    patterns = []
    
    for _, row in optimal_results.iterrows():
        # Clasificar tipo de patrón
        if row['direction'] > 0.3:
            pattern_type = 'Momentum'
            pattern_strength = row['direction']
        elif row['direction'] < -0.3:
            pattern_type = 'Mean Reversion'
            pattern_strength = abs(row['direction'])
        else:
            pattern_type = 'Complex/Non-linear'
            pattern_strength = row.get('monotonicity_score', 0)
        
        # Evaluar calidad del patrón
        if row.get('stability_score', 0) > 10:
            stability_rating = 'High'
        elif row.get('stability_score', 0) > 5:
            stability_rating = 'Medium'
        else:
            stability_rating = 'Low'
        
        pattern = {
            'indicator': row['indicator'],
            'period': int(row['period']),
            'pattern_type': pattern_type,
            'pattern_strength': pattern_strength,
            'expected_spread': row['spread'],
            'spread_ci_lower': row.get('spread_ci_lower', row['spread'] * 0.8),
            'spread_ci_upper': row.get('spread_ci_upper', row['spread'] * 1.2),
            'stability_rating': stability_rating,
            'stability_score': row.get('stability_score', 0),
            'sharpe_ratio': row['sharpe'],
            'statistical_significance': 1 - row['p_value'],
            'information_coefficient': row.get('ic_current_ic', np.nan),
            'ic_stability': row.get('ic_stability', 0),
            'best_percentiles': {
                'long': int(row['best_long_percentile']),
                'short': int(row['best_short_percentile'])
            },
            'sample_size': row['total_samples'],
            'min_samples_per_percentile': row['min_samples']
        }
        
        patterns.append(pattern)
    
    patterns_df = pd.DataFrame(patterns)
    
    # Filtrar por calidad
    quality_patterns = patterns_df[
        (abs(patterns_df['expected_spread']) >= min_spread) & 
        (patterns_df['statistical_significance'] >= (1 - max_p_value))
    ]
    
    return quality_patterns.sort_values('expected_spread', ascending=False)

def create_enhanced_percentile_plots(indicators: pd.DataFrame, returns_data: Dict, 
                                    data: pd.DataFrame, indicator_name: str, 
                                    return_days: int, regimes: pd.DataFrame = None) -> go.Figure:
    """Gráficos de percentiles mejorados con intervalos de confianza y análisis de régimen"""
    
    if indicator_name not in indicators.columns or indicator_name not in returns_data:
        return None
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'<b>Distribution & KDE</b>',
            f'<b>Returns by Percentile (Bootstrap CI)</b>',
            f'<b>Rolling IC & Stability</b>',
            f'<b>Regime Analysis</b>',
            f'<b>Time Decay Analysis</b>',
            f'<b>3D Surface Plot</b>'
        ),
        row_heights=[0.33, 0.33, 0.34],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "surface"}]]
    )
    
    # 1. Histograma con KDE mejorado
    hist_data = indicators[indicator_name].dropna()
    
    fig.add_trace(
        go.Histogram(
            x=hist_data,
            nbinsx=50,
            marker=dict(
                color='rgba(102, 126, 234, 0.6)',
                line=dict(color='rgba(255,255,255,0.2)', width=0.5)
            ),
            name='Distribution',
            showlegend=False,
            hovertemplate='<b>Value:</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # KDE con mejor escala
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
                    name='PDF',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Percentiles importantes
            for p, color in zip([10, 25, 50, 75, 90], ['#FF6B6B', '#4ECDC4', '#FFD93D', '#4ECDC4', '#FF6B6B']):
                val = np.percentile(hist_data, p)
                fig.add_vline(x=val, line=dict(color=color, width=1, dash='dash'),
                             row=1, col=1, annotation_text=f'P{p}')
        except:
            pass
    
    # 2. Retornos por percentil con intervalos de confianza (Bootstrap)
    returns_col = f'returns_{return_days}_days_mean'
    if returns_col in returns_data[indicator_name].columns:
        returns_values = returns_data[indicator_name][returns_col]
        returns_std = returns_data[indicator_name][f'returns_{return_days}_days_std'] if f'returns_{return_days}_days_std' in returns_data[indicator_name].columns else None
        
        x_labels = [f'P{i+1}' for i in range(len(returns_values))]
        
        # Barras principales
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=returns_values,
                marker=dict(
                    color=returns_values,
                    colorscale='RdYlGn',
                    line=dict(color='rgba(255,255,255,0.3)', width=1)
                ),
                text=[f'{val:.2f}%' for val in returns_values],
                textposition='outside',
                name='Returns',
                showlegend=False,
                error_y=dict(
                    type='data',
                    array=returns_std * 1.96 if returns_std is not None else None,
                    visible=True if returns_std is not None else False,
                    color='rgba(255,255,255,0.5)'
                )
            ),
            row=1, col=2
        )
    
    # 3. Rolling IC y estabilidad
    if f'returns_{return_days}_days' in data.columns:
        common_index = data.index.intersection(indicators[indicator_name].index)
        if len(common_index) > 252:
            aligned_returns = data.loc[common_index, f'returns_{return_days}_days']
            aligned_indicator = indicators.loc[common_index, indicator_name]
            
            # IC rolling
            rolling_corr = aligned_returns.rolling(126).corr(aligned_indicator).dropna()
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    line=dict(color='#00D2FF', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 210, 255, 0.1)',
                    name='IC',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Añadir media móvil del IC
            ic_ma = rolling_corr.rolling(63).mean()
            fig.add_trace(
                go.Scatter(
                    x=ic_ma.index,
                    y=ic_ma.values,
                    mode='lines',
                    line=dict(color='#FFD93D', width=2, dash='dash'),
                    name='IC MA',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1), row=2, col=1)
    
    # 4. Análisis por régimen (si está disponible)
    if regimes is not None and f'returns_{return_days}_days' in data.columns:
        regime_returns = []
        for regime in regimes['volatility_regime'].dropna().unique():
            regime_mask = regimes['volatility_regime'] == regime
            regime_indicator = indicators.loc[regime_mask, indicator_name]
            regime_ret = data.loc[regime_mask, f'returns_{return_days}_days']
            
            if len(regime_indicator.dropna()) > 20:
                corr = regime_indicator.corr(regime_ret)
                regime_returns.append({
                    'regime': regime,
                    'correlation': corr,
                    'count': len(regime_indicator.dropna())
                })
        
        if regime_returns:
            regime_df = pd.DataFrame(regime_returns)
            fig.add_trace(
                go.Bar(
                    x=regime_df['regime'],
                    y=regime_df['correlation'],
                    marker=dict(color=['#4CAF50', '#FFC107', '#F44336'][:len(regime_df)]),
                    text=[f'{c:.3f}' for c in regime_df['correlation']],
                    textposition='outside',
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # 5. Time Decay Analysis
    if f'returns_{return_days}_days' in data.columns:
        # Dividir datos en cuartiles temporales
        n_periods = 4
        period_size = len(data) // n_periods
        time_decay = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(data)
            
            period_indicator = indicators[indicator_name].iloc[start_idx:end_idx]
            period_returns = data[f'returns_{return_days}_days'].iloc[start_idx:end_idx]
            
            if len(period_indicator.dropna()) > 50:
                corr = period_indicator.corr(period_returns)
                time_decay.append({
                    'period': f'Q{i+1}',
                    'correlation': corr,
                    'start_date': data.index[start_idx],
                    'end_date': data.index[end_idx-1]
                })
        
        if time_decay:
            decay_df = pd.DataFrame(time_decay)
            fig.add_trace(
                go.Scatter(
                    x=decay_df['period'],
                    y=decay_df['correlation'],
                    mode='lines+markers',
                    line=dict(color='#9C27B0', width=3),
                    marker=dict(size=10, color='#9C27B0'),
                    text=[f"{row['start_date'].strftime('%Y-%m')} to {row['end_date'].strftime('%Y-%m')}" 
                          for _, row in decay_df.iterrows()],
                    hovertemplate='%{text}<br>Correlation: %{y:.3f}<extra></extra>',
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # Añadir línea de tendencia
            if len(decay_df) > 1:
                z = np.polyfit(range(len(decay_df)), decay_df['correlation'].values, 1)
                trend_line = np.poly1d(z)(range(len(decay_df)))
                fig.add_trace(
                    go.Scatter(
                        x=decay_df['period'],
                        y=trend_line,
                        mode='lines',
                        line=dict(color='#FF5722', width=2, dash='dash'),
                        showlegend=False
                    ),
                    row=3, col=1
                )
    
    # 6. 3D Surface Plot (Indicator vs Time vs Returns)
    if f'returns_{return_days}_days' in data.columns and len(indicators[indicator_name].dropna()) > 100:
        # Preparar datos para superficie 3D
        indicator_vals = indicators[indicator_name].dropna()
        
        # Crear bins para el indicador y tiempo
        n_bins = 20
        indicator_bins = pd.qcut(indicator_vals, q=n_bins, duplicates='drop')
        time_bins = pd.qcut(range(len(indicator_vals)), q=n_bins, duplicates='drop')
        
        # Crear matriz de retornos promedio
        surface_data = pd.DataFrame({
            'indicator_bin': indicator_bins,
            'time_bin': time_bins,
            'returns': data.loc[indicator_vals.index, f'returns_{return_days}_days']
        })
        
        pivot = surface_data.pivot_table(
            index='time_bin',
            columns='indicator_bin',
            values='returns',
            aggfunc='mean'
        )
        
        if not pivot.empty:
            fig.add_trace(
                go.Surface(
                    z=pivot.values,
                    colorscale='RdYlGn',
                    showscale=False,
                    hovertemplate='Returns: %{z:.2f}%<extra></extra>'
                ),
                row=3, col=2
            )
    
    # Actualizar diseño
    fig.update_layout(
        template="plotly_dark",
        height=1200,
        showlegend=False,
        title={
            'text': f"<b>Enhanced Analysis: {indicator_name}</b>",
            'font': {'size': 26, 'color': '#E0E5FF'},
            'x': 0.5,
            'xanchor': 'center'
        },
        hovermode='closest',
        plot_bgcolor='rgba(30, 34, 56, 0.3)',
        paper_bgcolor='rgba(14, 17, 39, 0.95)'
    )
    
    # Actualizar etiquetas de ejes
    fig.update_xaxes(title_text="<b>Value</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Frequency</b>", row=1, col=1)
    
    fig.update_xaxes(title_text="<b>Percentiles</b>", row=1, col=2)
    fig.update_yaxes(title_text=f"<b>Return ({return_days}d) %</b>", row=1, col=2)
    
    fig.update_xaxes(title_text="<b>Date</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Information Coefficient</b>", row=2, col=1)
    
    fig.update_xaxes(title_text="<b>Market Regime</b>", row=2, col=2)
    fig.update_yaxes(title_text="<b>Correlation</b>", row=2, col=2)
    
    fig.update_xaxes(title_text="<b>Time Period</b>", row=3, col=1)
    fig.update_yaxes(title_text="<b>Pattern Correlation</b>", row=3, col=1)
    
    return fig

def create_pattern_discovery_visualization(patterns_df: pd.DataFrame) -> go.Figure:
    """Visualización avanzada de patrones descubiertos"""
    if patterns_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>Pattern Landscape</b>',
            '<b>Stability vs Spread</b>',
            '<b>Pattern Types Distribution</b>',
            '<b>Information Coefficient Analysis</b>'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "sunburst"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Pattern Landscape
    fig.add_trace(
        go.Scatter(
            x=patterns_df['expected_spread'],
            y=patterns_df['sharpe_ratio'],
            mode='markers+text',
            marker=dict(
                size=patterns_df['stability_score'] * 2,
                color=patterns_df['statistical_significance'] * 100,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence %", x=0.45, len=0.4)
            ),
            text=patterns_df['indicator'] + '_' + patterns_df['period'].astype(str),
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate='<b>%{text}</b><br>Spread: %{x:.2f}%<br>Sharpe: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Stability vs Spread con intervalos de confianza
    fig.add_trace(
        go.Scatter(
            x=patterns_df['stability_score'],
            y=patterns_df['expected_spread'],
            mode='markers',
            marker=dict(
                size=10,
                color=patterns_df['pattern_type'].map({
                    'Momentum': '#FF9800',
                    'Mean Reversion': '#2196F3',
                    'Complex/Non-linear': '#9E9E9E'
                }),
                line=dict(color='white', width=1)
            ),
            error_y=dict(
                type='data',
                symmetric=False,
                array=patterns_df['spread_ci_upper'] - patterns_df['expected_spread'],
                arrayminus=patterns_df['expected_spread'] - patterns_df['spread_ci_lower'],
                visible=True,
                color='rgba(255,255,255,0.3)'
            ),
            hovertemplate='<b>%{text}</b><br>Stability: %{x:.2f}<br>Spread: %{y:.2f}%<extra></extra>',
            text=patterns_df['indicator'] + '_' + patterns_df['period'].astype(str),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Sunburst de tipos de patrones
    pattern_counts = patterns_df.groupby(['pattern_type', 'stability_rating']).size().reset_index(name='count')
    
    # Crear datos para sunburst
    labels = ['All Patterns']
    parents = ['']
    values = [len(patterns_df)]
    colors = ['#ffffff']
    
    for ptype in patterns_df['pattern_type'].unique():
        labels.append(ptype)
        parents.append('All Patterns')
        values.append(len(patterns_df[patterns_df['pattern_type'] == ptype]))
        
        if ptype == 'Momentum':
            colors.append('#FF9800')
        elif ptype == 'Mean Reversion':
            colors.append('#2196F3')
        else:
            colors.append('#9E9E9E')
        
        for stability in ['High', 'Medium', 'Low']:
            mask = (patterns_df['pattern_type'] == ptype) & (patterns_df['stability_rating'] == stability)
            count = len(patterns_df[mask])
            if count > 0:
                labels.append(f'{ptype} - {stability}')
                parents.append(ptype)
                values.append(count)
                
                if stability == 'High':
                    colors.append('#4CAF50')
                elif stability == 'Medium':
                    colors.append('#FFC107')
                else:
                    colors.append('#F44336')
    
    fig.add_trace(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            textinfo="label+percent parent",
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percentParent}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. IC Analysis
    if 'information_coefficient' in patterns_df.columns:
        fig.add_trace(
            go.Scatter(
                x=patterns_df['information_coefficient'],
                y=patterns_df['ic_stability'],
                mode='markers',
                marker=dict(
                    size=abs(patterns_df['expected_spread']) * 3,
                    color=patterns_df['expected_spread'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Spread %", x=1.02, len=0.4)
                ),
                text=patterns_df['indicator'] + '_' + patterns_df['period'].astype(str),
                hovertemplate='<b>%{text}</b><br>IC: %{x:.3f}<br>IC Stability: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.add_vline(x=0, line=dict(color='gray', dash='dash'), row=2, col=2)
        fig.add_hline(y=0, line=dict(color='gray', dash='dash'), row=2, col=2)
    
    # Actualizar diseño
    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=False,
        title={
            'text': "<b>Pattern Discovery Dashboard</b>",
            'font': {'size': 24, 'color': '#E0E5FF'},
            'x': 0.5,
            'xanchor': 'center'
        }
    )
    
    # Actualizar etiquetas
    fig.update_xaxes(title_text="<b>Expected Spread (%)</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Sharpe Ratio</b>", row=1, col=1)
    
    fig.update_xaxes(title_text="<b>Stability Score</b>", row=1, col=2)
    fig.update_yaxes(title_text="<b>Expected Spread (%)</b>", row=1, col=2)
    
    fig.update_xaxes(title_text="<b>Information Coefficient</b>", row=2, col=2)
    fig.update_yaxes(title_text="<b>IC Stability</b>", row=2, col=2)
    
    return fig

# ===================== FUNCIONES DE CÁLCULO ACTUALIZADAS =====================
@st.cache_data
def download_data(ticker: str, start_date: str, end_date: datetime) -> Optional[pd.DataFrame]:
    """Descarga datos históricos"""
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            multi_level_index=False
        )
        
        if data.empty:
            st.error(f"❌ No data found for {ticker}")
            return None
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error downloading data: {str(e)}")
        return None

@st.cache_data
def calculate_indicators_batch(ticker: str, start_date: str, end_date: datetime,
                               indicators_list: List[str], quantiles: int, 
                               return_days: int, period_range: Tuple[int, int, int]) -> Tuple:
    """Calcula indicadores y análisis de percentiles en batch con métricas mejoradas"""
    
    data = download_data(ticker, start_date, end_date)
    if data is None:
        return None, None, None, None
    
    # Calcular régimenes de mercado
    regimes = identify_market_regimes(data)
    
    for i in range(1, return_days + 1):
        data[f'returns_{i}_days'] = data['Close'].pct_change(i) * 100
    
    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    close = data['Close'].values.astype(np.float64)
    volume = data['Volume'].values.astype(np.float64) if 'Volume' in data.columns else np.zeros_like(close)
    open_prices = data['Open'].values.astype(np.float64)
    
    indicators = pd.DataFrame(index=data.index)
    
    total_calculations = 0
    successful_calculations = 0
    
    progress_container = st.container()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for indicator_name in indicators_list:
        if TechnicalIndicators.needs_period(indicator_name):
            min_p, max_p, step = period_range
            periods = range(min_p, max_p + 1, step)
            
            for period in periods:
                total_calculations += 1
                status_text.text(f"Calculating {indicator_name}_{period}...")
                
                result = TechnicalIndicators.calculate_indicator(
                    indicator_name, high, low, close, volume, open_prices, period
                )
                
                if result is not None and not np.all(np.isnan(result)):
                    indicators[f'{indicator_name}_{period}'] = result
                    successful_calculations += 1
                
                progress_bar.progress(min(successful_calculations / max(total_calculations, 1), 1.0))
        else:
            total_calculations += 1
            status_text.text(f"Calculating {indicator_name}...")
            
            result = TechnicalIndicators.calculate_indicator(
                indicator_name, high, low, close, volume, open_prices, 0
            )
            
            if result is not None and not np.all(np.isnan(result)):
                indicators[indicator_name] = result
                successful_calculations += 1
            
            progress_bar.progress(min(successful_calculations / max(total_calculations, 1), 1.0))
    
    progress_bar.empty()
    status_text.empty()
    
    indicators = indicators.dropna(axis=1, how='all')
    
    # Calcular análisis de percentiles mejorado
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
                        
        except Exception:
            continue
    
    return returns_data, indicators, data, regimes

def batch_optimize_indicators_enhanced(indicator_list, data, return_days=5, quick_mode=True):
    """Optimización batch mejorada con análisis de estabilidad"""
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, indicator in enumerate(indicator_list):
        status_text.text(f"Optimizing {indicator}... ({i+1}/{len(indicator_list)})")
        
        if quick_mode:
            periods = list(range(5, 51, 5)) + [60, 75, 100, 150, 200]
        else:
            periods = list(range(5, 201, 5))
        
        if not TechnicalIndicators.needs_period(indicator):
            continue
        
        df_results = find_optimal_period_enhanced(indicator, data, periods, return_days)
        
        if not df_results.empty:
            top_periods = df_results.head(5)  # Top 5 instead of 3
            all_results.append(top_periods)
        
        progress_bar.progress((i + 1) / len(indicator_list))
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_results:
        return pd.DataFrame()
    
    combined_results = pd.concat(all_results, ignore_index=True)
    return combined_results.sort_values('composite_score', ascending=False)

# ===================== INTERFAZ PRINCIPAL MEJORADA =====================
def main():
    # Título con información mejorada
    st.markdown("""
        <h1 style='text-align: center;'>
            <span style='font-size: 1.2em;'>🔬</span> Advanced Quantitative Pattern Analyzer
        </h1>
        <p style='text-align: center; color: #8892B0; font-size: 1.2rem; margin-bottom: 2rem;'>
            Statistical Pattern Discovery with Robustness Testing & Information Coefficients
        </p>
        <p style='text-align: center; color: #667eea; font-size: 0.9rem;'>
            {total} technical indicators | Bootstrap Confidence Intervals | Walk-Forward Validation
        </p>
    """.format(total=TechnicalIndicators.get_total_indicators()), unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.returns_data = None
        st.session_state.indicators = None
        st.session_state.data = None
        st.session_state.optimal_results = None
        st.session_state.patterns = None
        st.session_state.regimes = None
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        with st.expander("📈 **MARKET DATA**", expanded=True):
            ticker = st.text_input("Symbol", value="SPY")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start",
                    value=datetime(2010, 1, 1),
                    min_value=datetime(1990, 1, 1),
                    max_value=datetime.now()
                )
            with col2:
                end_date = st.date_input(
                    "End",
                    value=datetime.now(),
                    min_value=datetime(1990, 1, 1),
                    max_value=datetime.now()
                )
        
        with st.expander("📊 **PARAMETERS**", expanded=True):
            return_days = st.select_slider(
                "Return Days",
                options=[1, 2, 3, 5, 7, 10, 14, 20, 30],
                value=5
            )
            
            quantiles = st.slider(
                "Percentiles",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                min_period = st.number_input("Period Min", value=5, min_value=2)
            with col2:
                max_period = st.number_input("Period Max", value=50, min_value=10)
            with col3:
                step_period = st.number_input("Step", value=5, min_value=1)
            
            # New parameters
            st.markdown("#### 🎯 Advanced Settings")
            
            enable_walk_forward = st.checkbox("Enable Walk-Forward Validation", value=False)
            bootstrap_samples = st.slider("Bootstrap Samples", 50, 500, 100, 50)
            min_spread_threshold = st.slider("Min Spread Threshold (%)", 0.5, 5.0, 2.0, 0.5)
            max_p_value = st.slider("Max P-Value", 0.01, 0.20, 0.10, 0.01)
        
        st.markdown("### 📐 Indicators")
        
        preset = st.selectbox(
            "Preset",
            ["📊 Essential (5)", "💫 Momentum (10)", "📈 Complete (25)", 
             "🎯 Top 50", "🚀 ALL (157+61 patterns)"]
        )
        
        all_indicators = list(TechnicalIndicators.INDICATOR_CONFIG.keys())
        
        if preset == "📊 Essential (5)":
            selected_indicators = ['RSI', 'MACD', 'CCI', 'ROC', 'ATR']
        elif preset == "💫 Momentum (10)":
            selected_indicators = ['RSI', 'MACD', 'STOCH', 'CCI', 'MFI', 
                                 'WILLR', 'MOM', 'ROC', 'ADX', 'PPO']
        elif preset == "📈 Complete (25)":
            selected_indicators = ['RSI', 'MACD', 'BBANDS', 'ATR', 'ADX',
                                 'SMA', 'EMA', 'STOCH', 'CCI', 'MFI',
                                 'WILLR', 'MOM', 'ROC', 'PPO', 'CMO',
                                 'AROON', 'ULTOSC', 'OBV', 'AD', 'NATR',
                                 'TRIX', 'TEMA', 'WMA', 'DEMA', 'KAMA']
        elif preset == "🎯 Top 50":
            selected_indicators = all_indicators[:50]
        else:
            selected_indicators = all_indicators + TechnicalIndicators.CANDLE_PATTERNS
        
        with st.expander(f"📋 {len(selected_indicators)} indicators selected"):
            categories = TechnicalIndicators.get_all_categories()
            for cat_name, cat_indicators in categories.items():
                selected_in_cat = [ind for ind in selected_indicators if ind in cat_indicators]
                if selected_in_cat:
                    st.write(f"**{cat_name}**: {len(selected_in_cat)}")
        
        analyze_button = st.button(
            "🚀 **RUN ANALYSIS**",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button:
        if not selected_indicators:
            st.error("Please select indicators")
            return
        
        with st.spinner('🔄 Processing comprehensive analysis...'):
            returns_data, indicators, data, regimes = calculate_indicators_batch(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date,
                selected_indicators,
                quantiles,
                return_days,
                (min_period, max_period, step_period)
            )
            
            if returns_data and indicators is not None and data is not None:
                st.session_state.analysis_done = True
                st.session_state.returns_data = returns_data
                st.session_state.indicators = indicators
                st.session_state.data = data
                st.session_state.regimes = regimes
                
                # Enhanced optimal period analysis
                with st.spinner('🔍 Discovering patterns with robustness testing...'):
                    st.session_state.optimal_results = batch_optimize_indicators_enhanced(
                        selected_indicators, data, return_days, quick_mode=not enable_walk_forward
                    )
                    
                    if not st.session_state.optimal_results.empty:
                        st.session_state.patterns = discover_patterns(
                            st.session_state.optimal_results,
                            min_spread=min_spread_threshold,
                            max_p_value=max_p_value
                        )
    
    if st.session_state.analysis_done:
        returns_data = st.session_state.returns_data
        indicators = st.session_state.indicators
        data = st.session_state.data
        optimal_results = st.session_state.optimal_results
        patterns = st.session_state.patterns
        regimes = st.session_state.regimes
        
        st.success(f"✅ Analysis complete: {len(indicators.columns)} configurations analyzed")
        
        # Metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        if patterns is not None and not patterns.empty:
            with col1:
                st.metric("Significant Patterns", len(patterns))
            with col2:
                high_stability = len(patterns[patterns['stability_rating'] == 'High'])
                st.metric("High Stability", high_stability)
            with col3:
                avg_spread = patterns['expected_spread'].mean()
                st.metric("Avg Spread", f"{avg_spread:.2f}%")
            with col4:
                momentum_patterns = len(patterns[patterns['pattern_type'] == 'Momentum'])
                mean_rev_patterns = len(patterns[patterns['pattern_type'] == 'Mean Reversion'])
                st.metric("Pattern Mix", f"{momentum_patterns}M/{mean_rev_patterns}R")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 **Enhanced Percentile Analysis**",
            "🔍 **Pattern Discovery**",
            "🎯 **Optimal Configurations**",
            "📊 **Robustness Testing**",
            "📥 **Export Results**"
        ])
        
        with tab1:
            st.markdown("### 📈 Enhanced Percentile Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                available = list(indicators.columns)
                if available:
                    selected_ind = st.selectbox("Indicator", available)
            
            with col2:
                sel_return = st.selectbox(
                    "Return Days",
                    list(range(1, return_days + 1)),
                    index=min(4, return_days - 1) if return_days >= 5 else 0
                )
            
            with col3:
                if 'selected_ind' in locals():
                    st.metric("Data Points", f"{len(indicators[selected_ind].dropna()):,}")
            
            if 'selected_ind' in locals() and selected_ind:
                fig = create_enhanced_percentile_plots(
                    indicators, returns_data, data,
                    selected_ind, sel_return, regimes
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🔍 Pattern Discovery Dashboard")
            
            if patterns is not None and not patterns.empty:
                # Pattern summary
                st.markdown("#### Pattern Summary")
                
                for idx, pattern in patterns.head(10).iterrows():
                    stability_color = {
                        'High': '#4CAF50',
                        'Medium': '#FFC107', 
                        'Low': '#F44336'
                    }.get(pattern['stability_rating'], '#9E9E9E')
                    
                    pattern_emoji = {
                        'Momentum': '🟠',
                        'Mean Reversion': '🔵',
                        'Complex/Non-linear': '⚪'
                    }.get(pattern['pattern_type'], '⚫')
                    
                    st.markdown(f"""
                    <div class='pattern-card'>
                        <h4>{pattern_emoji} {pattern['indicator']} (Period {pattern['period']})</h4>
                        <div style='display: flex; justify-content: space-between; margin: 10px 0;'>
                            <span><b>Type:</b> {pattern['pattern_type']}</span>
                            <span><b>Expected Spread:</b> {pattern['expected_spread']:.2f}%</span>
                            <span class='stability-badge' style='background: {stability_color}20; color: {stability_color}; border: 1px solid {stability_color};'>
                                {pattern['stability_rating']} Stability
                            </span>
                        </div>
                        <div style='display: flex; justify-content: space-between;'>
                            <span><b>CI:</b> [{pattern['spread_ci_lower']:.2f}%, {pattern['spread_ci_upper']:.2f}%]</span>
                            <span><b>Significance:</b> {pattern['statistical_significance']*100:.1f}%</span>
                            <span><b>Sharpe:</b> {pattern['sharpe_ratio']:.3f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualization
                fig = create_pattern_discovery_visualization(patterns)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No significant patterns found. Try adjusting parameters.")
        
        with tab3:
            st.markdown("### 🎯 Optimal Period Configurations")
            
            if optimal_results is not None and not optimal_results.empty:
                # Best configuration details
                best = optimal_results.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Best Configuration",
                        f"{best['indicator']}_P{int(best['period'])}",
                        f"Score: {best.get('composite_score', best.get('score', 0)):.2f}"
                    )
                with col2:
                    st.metric(
                        "Expected Spread",
                        f"{best['spread']:.2f}%",
                        f"Sharpe: {best['sharpe']:.3f}"
                    )
                with col3:
                    if 'stability_score' in best:
                        st.metric(
                            "Stability",
                            f"{best['stability_score']:.2f}",
                            f"P-value: {best['p_value']:.4f}"
                        )
                
                # Detailed results table
                display_cols = ['indicator', 'period', 'spread', 'direction', 'p_value', 
                               'sharpe', 'stability_score', 'composite_score']
                
                available_cols = [col for col in display_cols if col in optimal_results.columns]
                display_df = optimal_results[available_cols].head(20)
                
                st.dataframe(
                    display_df.style.format({
                        'spread': '{:.2f}%',
                        'direction': '{:.3f}',
                        'p_value': '{:.4f}',
                        'sharpe': '{:.3f}',
                        'stability_score': '{:.2f}',
                        'composite_score': '{:.2f}'
                    }).background_gradient(subset=['spread', 'composite_score'], cmap='RdYlGn'),
                    use_container_width=True
                )
        
        with tab4:
            st.markdown("### 📊 Robustness Testing")
            
            if optimal_results is not None and not optimal_results.empty and enable_walk_forward:
                selected_config = st.selectbox(
                    "Select configuration to test",
                    optimal_results.head(10).apply(
                        lambda x: f"{x['indicator']}_P{int(x['period'])} (Spread: {x['spread']:.2f}%)",
                        axis=1
                    ).tolist()
                )
                
                if st.button("Run Walk-Forward Validation"):
                    # Parse selection
                    config_idx = optimal_results.head(10).index[0]  # Simplified
                    config = optimal_results.loc[config_idx]
                    
                    with st.spinner('Running walk-forward validation...'):
                        validation_results = walk_forward_validation(
                            config['indicator'],
                            data,
                            int(config['period']),
                            window_size=504,
                            step_size=126,
                            return_days=return_days
                        )
                        
                        if validation_results:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Consistency Score",
                                    f"{validation_results['consistency_score']:.2f}"
                                )
                            with col2:
                                st.metric(
                                    "Positive Windows",
                                    f"{validation_results['positive_spread_pct']:.1f}%"
                                )
                            with col3:
                                st.metric(
                                    "Windows Tested",
                                    validation_results['windows_tested']
                                )
                            
                            # Plot time series of spreads
                            if 'time_series' in validation_results:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=validation_results['time_series']['window_start'],
                                    y=validation_results['time_series']['spread'],
                                    mode='lines+markers',
                                    line=dict(color='#667eea', width=2),
                                    marker=dict(size=8),
                                    name='Spread over time'
                                ))
                                
                                fig.add_hline(
                                    y=validation_results['mean_spread'],
                                    line=dict(color='green', dash='dash'),
                                    annotation_text=f"Mean: {validation_results['mean_spread']:.2f}%"
                                )
                                
                                fig.update_layout(
                                    template="plotly_dark",
                                    title="Walk-Forward Validation Results",
                                    xaxis_title="Period Start",
                                    yaxis_title="Spread (%)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Enable Walk-Forward Validation in settings to test robustness.")
        
        with tab5:
            st.markdown("### 📥 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if patterns is not None and not patterns.empty:
                    if st.button("📄 Export Patterns (CSV)"):
                        csv = patterns.to_csv(index=False)
                        st.download_button(
                            label="Download Patterns CSV",
                            data=csv,
                            file_name=f"{ticker}_patterns_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                
                if optimal_results is not None and not optimal_results.empty:
                    if st.button("📊 Export Optimal Configs (CSV)"):
                        csv = optimal_results.to_csv(index=False)
                        st.download_button(
                            label="Download Configurations CSV",
                            data=csv,
                            file_name=f"{ticker}_optimal_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if st.button("📋 Export Full Report (JSON)"):
                    report = {
                        'metadata': {
                            'ticker': ticker,
                            'start_date': start_date.strftime('%Y-%m-%d'),
                            'end_date': end_date.strftime('%Y-%m-%d'),
                            'return_days': return_days,
                            'quantiles': quantiles,
                            'indicators_analyzed': len(indicators.columns),
                            'timestamp': datetime.now().isoformat()
                        },
                        'patterns': patterns.to_dict('records') if patterns is not None else [],
                        'top_configurations': optimal_results.head(20).to_dict('records') if optimal_results is not None else [],
                        'summary_stats': {
                            'total_patterns': len(patterns) if patterns is not None else 0,
                            'high_stability_patterns': len(patterns[patterns['stability_rating'] == 'High']) if patterns is not None else 0,
                            'avg_spread': patterns['expected_spread'].mean() if patterns is not None and not patterns.empty else 0,
                            'best_sharpe': optimal_results['sharpe'].max() if optimal_results is not None and not optimal_results.empty else 0
                        }
                    }
                    
                    json_str = json.dumps(report, indent=2, default=str)
                    st.download_button(
                        label="Download Full Report (JSON)",
                        data=json_str,
                        file_name=f"{ticker}_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )

if __name__ == "__main__":
    main()
