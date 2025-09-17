import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(
    page_title="Quantitative Pattern Analyzer",
    page_icon="📊",
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
    
    .trading-rule {
        background: rgba(30, 34, 56, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .rule-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .rule-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 4px;
    }
    
    .momentum-badge {
        background: rgba(255, 152, 0, 0.2);
        color: #FF9800;
        border: 1px solid #FF9800;
    }
    
    .mean-reversion-badge {
        background: rgba(33, 150, 243, 0.2);
        color: #2196F3;
        border: 1px solid #2196F3;
    }
    
    .strong-signal {
        background: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        border: 1px solid #4CAF50;
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

# ===================== FUNCIONES DE CÁLCULO OPTIMIZADAS =====================
@st.cache_data
def download_data(ticker: str, start_date: str, end_date: datetime) -> Optional[pd.DataFrame]:
    """Descarga datos históricos"""
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
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
    """Calcula indicadores y análisis de percentiles en batch"""
    
    data = download_data(ticker, start_date, end_date)
    if data is None:
        return None, None, None
    
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
    
    # Calcular análisis de percentiles
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
    
    return returns_data, indicators, data

def analyze_indicator_period(indicator_values, returns, quantiles=10):
    """Análisis simple y rápido de un indicador"""
    try:
        temp_df = pd.DataFrame({
            'indicator': indicator_values,
            'returns': returns
        }).dropna()
        
        if len(temp_df) < quantiles * 2:
            return None
        
        temp_df['percentile'] = pd.qcut(temp_df['indicator'], q=quantiles, labels=False, duplicates='drop')
        percentile_returns = temp_df.groupby('percentile')['returns'].agg(['mean', 'std', 'count'])
        
        if len(percentile_returns) < quantiles * 0.8:
            return None
        
        metrics = {}
        
        # Core metrics
        top_return = percentile_returns['mean'].iloc[-1]
        bottom_return = percentile_returns['mean'].iloc[0]
        metrics['spread'] = top_return - bottom_return
        metrics['top_return'] = top_return
        metrics['bottom_return'] = bottom_return
        
        # Direction (momentum vs mean reversion)
        correlation, p_value = spearmanr(range(len(percentile_returns)), percentile_returns['mean'].values)
        metrics['direction'] = correlation
        metrics['p_value'] = p_value
        
        # Sharpe
        metrics['sharpe'] = abs(metrics['spread']) / (percentile_returns['std'].mean() + 1e-8)
        
        # Best percentiles
        metrics['best_long_percentile'] = percentile_returns['mean'].idxmax() + 1
        metrics['best_short_percentile'] = percentile_returns['mean'].idxmin() + 1
        
        # Sample size
        metrics['min_samples'] = percentile_returns['count'].min()
        metrics['total_samples'] = percentile_returns['count'].sum()
        
        return metrics
        
    except Exception:
        return None

def find_optimal_period(indicator_name, data, periods_to_test, return_days=5):
    """Encuentra el período óptimo para un indicador"""
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
        
        metrics = analyze_indicator_period(indicator_values, returns.values, quantiles=10)
        
        if metrics and metrics['min_samples'] >= 10:
            metrics['period'] = period
            metrics['indicator'] = indicator_name
            results.append(metrics)
    
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    
    # Score simple y efectivo
    df_results['score'] = (
        abs(df_results['spread']) * 0.5 +
        df_results['sharpe'] * 10 +
        (1 / (df_results['p_value'] + 0.001)) * 0.1
    )
    
    return df_results.sort_values('score', ascending=False)

def batch_optimize_indicators(indicator_list, data, return_days=5):
    """Optimización batch rápida"""
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Períodos a probar (reducido para velocidad)
    periods = [5, 10, 14, 20, 30, 50, 100, 200]
    
    for i, indicator in enumerate(indicator_list):
        status_text.text(f"Optimizing {indicator}... ({i+1}/{len(indicator_list)})")
        
        if not TechnicalIndicators.needs_period(indicator):
            continue
        
        df_results = find_optimal_period(indicator, data, periods, return_days)
        
        if not df_results.empty:
            top_periods = df_results.head(3)
            all_results.append(top_periods)
        
        progress_bar.progress((i + 1) / len(indicator_list))
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_results:
        return pd.DataFrame()
    
    combined_results = pd.concat(all_results, ignore_index=True)
    return combined_results.sort_values('score', ascending=False)

def extract_trading_rules(analysis_df, min_spread=2.0, max_p_value=0.1, top_n=10):
    """Extrae reglas de trading claras y simples"""
    if analysis_df.empty:
        return []
    
    quality_signals = analysis_df[
        (abs(analysis_df['spread']) >= min_spread) & 
        (analysis_df['p_value'] <= max_p_value)
    ].head(top_n)
    
    rules = []
    
    for idx, row in quality_signals.iterrows():
        # Determinar tipo de estrategia
        if row['direction'] > 0.3:
            strategy_type = "MOMENTUM"
            primary_signal = f"When {row['indicator']}_{int(row['period'])} is HIGH (top 20%)"
            primary_action = "STRONG BUY"
            secondary_signal = f"When {row['indicator']}_{int(row['period'])} is LOW (bottom 20%)"
            secondary_action = "STRONG SELL"
        elif row['direction'] < -0.3:
            strategy_type = "MEAN REVERSION"
            primary_signal = f"When {row['indicator']}_{int(row['period'])} is LOW (bottom 20%)"
            primary_action = "STRONG BUY"
            secondary_signal = f"When {row['indicator']}_{int(row['period'])} is HIGH (top 20%)"
            secondary_action = "STRONG SELL"
        else:
            strategy_type = "SELECTIVE"
            primary_signal = f"When {row['indicator']}_{int(row['period'])} is at Percentile {int(row['best_long_percentile'])}"
            primary_action = "BUY"
            secondary_signal = f"When {row['indicator']}_{int(row['period'])} is at Percentile {int(row['best_short_percentile'])}"
            secondary_action = "AVOID"
        
        rule = {
            'rank': idx + 1,
            'indicator': row['indicator'],
            'period': int(row['period']),
            'strategy_type': strategy_type,
            'primary_signal': primary_signal,
            'primary_action': primary_action,
            'secondary_signal': secondary_signal,
            'secondary_action': secondary_action,
            'expected_spread': row['spread'],
            'top_return': row['top_return'],
            'bottom_return': row['bottom_return'],
            'direction': row['direction'],
            'confidence': (1 - row['p_value']) * 100,
            'sharpe': row['sharpe'],
            'samples': row['total_samples']
        }
        
        rules.append(rule)
    
    return rules

def create_percentile_plots(indicators: pd.DataFrame, returns_data: Dict, 
                           data: pd.DataFrame, indicator_name: str, 
                           return_days: int) -> go.Figure:
    """Crea gráficos de análisis de percentiles optimizados"""
    
    if indicator_name not in indicators.columns or indicator_name not in returns_data:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'<b>Distribution of {indicator_name}</b>',
            f'<b>Returns by Percentile ({return_days} days)</b>',
            f'<b>Rolling Correlation (126 days)</b>',
            f'<b>Scatter Analysis</b>'
        ),
        row_heights=[0.5, 0.5],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 1. Histogram
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
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add mean line
    mean_val = hist_data.mean()
    fig.add_vline(x=mean_val, line=dict(color='#FF1744', width=2),
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
                name='Returns',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Rolling correlation
    if f'returns_{return_days}_days' in data.columns:
        common_index = data.index.intersection(indicators[indicator_name].index)
        if len(common_index) > 126:
            aligned_returns = data.loc[common_index, f'returns_{return_days}_days']
            aligned_indicator = indicators.loc[common_index, indicator_name]
            
            rolling_corr = aligned_returns.rolling(126).corr(aligned_indicator).dropna()
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    line=dict(color='#00D2FF', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 210, 255, 0.1)',
                    name='Correlation',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1), row=2, col=1)
    
    # 4. Scatter plot
    if f'returns_{return_days}_days' in data.columns:
        common_index = data.index.intersection(indicators[indicator_name].index)
        if len(common_index) > 0:
            x_data = indicators.loc[common_index, indicator_name]
            y_data = data.loc[common_index, f'returns_{return_days}_days']
            
            mask = ~(x_data.isna() | y_data.isna())
            x_clean = x_data[mask]
            y_clean = y_data[mask]
            
            if len(x_clean) > 1:
                fig.add_trace(
                    go.Scattergl(
                        x=x_clean,
                        y=y_clean,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=y_clean,
                            colorscale='RdYlGn',
                            opacity=0.5,
                            showscale=True
                        ),
                        name='Data',
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                # Add regression line
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        line=dict(color='#FFD93D', width=2, dash='dash'),
                        name='Trend',
                        showlegend=False
                    ),
                    row=2, col=2
                )
    
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=False,
        title={
            'text': f"<b>Percentile Analysis: {indicator_name}</b>",
            'font': {'size': 24}
        },
        hovermode='closest'
    )
    
    return fig

def create_optimal_visualization(results_df):
    """Visualización de períodos óptimos"""
    if results_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>Period vs Spread</b>',
            '<b>Strategy Direction</b>',
            '<b>Top 10 Configurations</b>',
            '<b>Sharpe Ratio Distribution</b>'
        )
    )
    
    # 1. Period vs Spread for top indicators
    top_indicators = results_df['indicator'].value_counts().head(5).index
    for ind in top_indicators:
        ind_data = results_df[results_df['indicator'] == ind]
        fig.add_trace(
            go.Scatter(
                x=ind_data['period'],
                y=ind_data['spread'],
                mode='markers+lines',
                name=ind,
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # 2. Direction analysis
    fig.add_trace(
        go.Scatter(
            x=results_df['direction'],
            y=abs(results_df['spread']),
            mode='markers',
            marker=dict(
                size=results_df['sharpe'] * 20,
                color=results_df['score'],
                colorscale='Viridis',
                showscale=True
            ),
            text=results_df['indicator'] + '_' + results_df['period'].astype(str),
            hovertemplate='%{text}<br>Direction: %{x:.3f}<br>Spread: %{y:.2f}%',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Top configurations bar chart
    top_10 = results_df.nlargest(10, 'score')
    fig.add_trace(
        go.Bar(
            x=top_10['indicator'] + '_P' + top_10['period'].astype(str),
            y=top_10['spread'],
            marker=dict(color=top_10['score'], colorscale='RdYlGn'),
            text=[f"{s:.1f}%" for s in top_10['spread']],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Sharpe distribution
    fig.add_trace(
        go.Histogram(
            x=results_df['sharpe'],
            nbinsx=30,
            marker=dict(color='rgba(102, 126, 234, 0.6)'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=True,
        title_text="<b>Optimal Period Discovery</b>"
    )
    
    return fig

# ===================== INTERFAZ PRINCIPAL =====================
def main():
    st.markdown("""
        <h1 style='text-align: center;'>
            📊 Quantitative Pattern Analyzer
        </h1>
        <p style='text-align: center; color: #8892B0; font-size: 1.2rem;'>
            Fast Percentile Analysis + Optimal Period Discovery + Trading Rules
        </p>
        <p style='text-align: center; color: #667eea; font-size: 0.9rem;'>
            {total} indicators available | Optimized for speed
        </p>
    """.format(total=TechnicalIndicators.get_total_indicators()), unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.returns_data = None
        st.session_state.indicators = None
        st.session_state.data = None
        st.session_state.optimal_results = None
        st.session_state.trading_rules = None
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        with st.expander("📈 **MARKET DATA**", expanded=True):
            ticker = st.text_input("Symbol", value="SPY")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start",
                    value=datetime(2015, 1, 1)
                )
            with col2:
                end_date = st.date_input(
                    "End",
                    value=datetime.now()
                )
        
        with st.expander("📊 **PARAMETERS**", expanded=True):
            return_days = st.select_slider(
                "Return Days",
                options=[1, 3, 5, 10, 20],
                value=5
            )
            
            quantiles = st.slider(
                "Percentiles",
                min_value=5,
                max_value=20,
                value=10,
                step=5
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                min_period = st.number_input("Min Period", value=5, min_value=2)
            with col2:
                max_period = st.number_input("Max Period", value=50, min_value=10)
            with col3:
                step_period = st.number_input("Step", value=5, min_value=1)
        
        st.markdown("### 📐 Indicators")
        
        preset = st.selectbox(
            "Quick Presets",
            ["🎯 Fast (5)", "📊 Essential (10)", "💫 Momentum (15)", "📈 Complete (25)"]
        )
        
        all_indicators = list(TechnicalIndicators.INDICATOR_CONFIG.keys())
        
        if preset == "🎯 Fast (5)":
            selected_indicators = ['RSI', 'MACD', 'CCI', 'ATR', 'ROC']
        elif preset == "📊 Essential (10)":
            selected_indicators = ['RSI', 'MACD', 'STOCH', 'CCI', 'MFI', 
                                 'WILLR', 'ADX', 'ATR', 'ROC', 'MOM']
        elif preset == "💫 Momentum (15)":
            selected_indicators = ['RSI', 'MACD', 'STOCH', 'CCI', 'MFI',
                                 'WILLR', 'MOM', 'ROC', 'ADX', 'PPO',
                                 'CMO', 'AROON', 'ULTOSC', 'TRIX', 'DX']
        else:  # Complete
            selected_indicators = ['RSI', 'MACD', 'BBANDS', 'ATR', 'ADX',
                                 'SMA', 'EMA', 'STOCH', 'CCI', 'MFI',
                                 'WILLR', 'MOM', 'ROC', 'PPO', 'CMO',
                                 'AROON', 'ULTOSC', 'OBV', 'AD', 'NATR',
                                 'TRIX', 'TEMA', 'WMA', 'DEMA', 'KAMA']
        
        st.info(f"📋 {len(selected_indicators)} indicators selected")
        
        analyze_button = st.button(
            "🚀 **RUN ANALYSIS**",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button:
        with st.spinner('Processing...'):
            returns_data, indicators, data = calculate_indicators_batch(
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
                
                # Optimize periods
                with st.spinner('Finding optimal periods...'):
                    st.session_state.optimal_results = batch_optimize_indicators(
                        selected_indicators, data, return_days
                    )
                    
                    # Extract trading rules
                    if not st.session_state.optimal_results.empty:
                        st.session_state.trading_rules = extract_trading_rules(
                            st.session_state.optimal_results,
                            min_spread=2.0,
                            max_p_value=0.1,
                            top_n=10
                        )
    
    if st.session_state.analysis_done:
        returns_data = st.session_state.returns_data
        indicators = st.session_state.indicators
        data = st.session_state.data
        optimal_results = st.session_state.optimal_results
        trading_rules = st.session_state.trading_rules
        
        st.success(f"✅ Analysis complete: {len(indicators.columns)} configurations")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 **Percentile Analysis**",
            "🏆 **Top Performers**",
            "🎯 **Optimal Periods**",
            "📋 **Trading Rules**"
        ])
        
        with tab1:
            st.markdown("### 📈 Percentile Analysis")
            
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
                fig = create_percentile_plots(
                    indicators, returns_data, data,
                    selected_ind, sel_return
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🏆 Top Performing Configurations")
            
            best_configs = []
            for ind_col in indicators.columns:
                if ind_col in returns_data:
                    ret_col = f'returns_{return_days}_days_mean'
                    if ret_col in returns_data[ind_col].columns:
                        values = returns_data[ind_col][ret_col]
                        if len(values) > 1:
                            spread = values.iloc[-1] - values.iloc[0]
                            sharpe = abs(spread) / values.std() if values.std() > 0 else 0
                            
                            best_configs.append({
                                'Indicator': ind_col,
                                'Spread': spread,
                                'Sharpe': sharpe,
                                'Top_Return': values.iloc[-1],
                                'Bottom_Return': values.iloc[0]
                            })
            
            if best_configs:
                best_df = pd.DataFrame(best_configs)
                best_df = best_df.sort_values('Spread', ascending=False).head(20)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Spread", f"{best_df['Spread'].max():.2f}%")
                with col2:
                    st.metric("Best Sharpe", f"{best_df['Sharpe'].max():.3f}")
                with col3:
                    st.metric("Configs Analyzed", len(best_df))
                
                st.dataframe(
                    best_df.style.format({
                        'Spread': '{:.2f}%',
                        'Sharpe': '{:.3f}',
                        'Top_Return': '{:.2f}%',
                        'Bottom_Return': '{:.2f}%'
                    }).background_gradient(cmap='RdYlGn', subset=['Spread', 'Sharpe']),
                    use_container_width=True
                )
        
        with tab3:
            st.markdown("### 🎯 Optimal Period Discovery")
            
            if optimal_results is not None and not optimal_results.empty:
                col1, col2, col3 = st.columns(3)
                
                best = optimal_results.iloc[0]
                
                with col1:
                    st.metric(
                        "Best Config",
                        f"{best['indicator']}_P{int(best['period'])}"
                    )
                with col2:
                    st.metric(
                        "Expected Spread",
                        f"{best['spread']:.2f}%"
                    )
                with col3:
                    st.metric(
                        "Confidence",
                        f"{(1-best['p_value'])*100:.1f}%"
                    )
                
                fig = create_optimal_visualization(optimal_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Top Configurations")
                display_df = optimal_results[['indicator', 'period', 'spread', 
                                             'direction', 'p_value', 'sharpe', 
                                             'score']].head(15)
                
                st.dataframe(
                    display_df.style.format({
                        'spread': '{:.2f}%',
                        'direction': '{:.3f}',
                        'p_value': '{:.4f}',
                        'sharpe': '{:.3f}',
                        'score': '{:.2f}'
                    }).background_gradient(subset=['spread', 'score'], cmap='RdYlGn'),
                    use_container_width=True
                )
        
        with tab4:
            st.markdown("### 📋 Trading Rules")
            
            if trading_rules:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rules", len(trading_rules))
                with col2:
                    momentum_rules = sum(1 for r in trading_rules if r['strategy_type'] == 'MOMENTUM')
                    st.metric("Momentum", momentum_rules)
                with col3:
                    mean_rev_rules = sum(1 for r in trading_rules if r['strategy_type'] == 'MEAN REVERSION')
                    st.metric("Mean Reversion", mean_rev_rules)
                with col4:
                    avg_spread = np.mean([r['expected_spread'] for r in trading_rules])
                    st.metric("Avg Spread", f"{avg_spread:.2f}%")
                
                st.markdown("---")
                
                # Display rules
                for rule in trading_rules[:10]:
                    # Determine badge colors
                    if rule['strategy_type'] == 'MOMENTUM':
                        strategy_badge = '<span class="rule-badge momentum-badge">🟠 MOMENTUM</span>'
                    elif rule['strategy_type'] == 'MEAN REVERSION':
                        strategy_badge = '<span class="rule-badge mean-reversion-badge">🔵 MEAN REVERSION</span>'
                    else:
                        strategy_badge = '<span class="rule-badge">⚪ SELECTIVE</span>'
                    
                    confidence_badge = ''
                    if rule['confidence'] >= 95:
                        confidence_badge = '<span class="rule-badge strong-signal">STRONG</span>'
                    
                    st.markdown(f"""
                    <div class="trading-rule">
                        <div class="rule-header">
                            <h4>Rule #{rule['rank']}: {rule['indicator']} (Period {rule['period']})</h4>
                            <div>{strategy_badge} {confidence_badge}</div>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            <div>
                                <p><strong>Primary Signal:</strong></p>
                                <p style="color: #4CAF50;">✓ {rule['primary_signal']}</p>
                                <p style="margin-left: 20px;">→ <strong>{rule['primary_action']}</strong></p>
                                
                                <p style="margin-top: 10px;"><strong>Secondary Signal:</strong></p>
                                <p style="color: #FF5252;">✗ {rule['secondary_signal']}</p>
                                <p style="margin-left: 20px;">→ <strong>{rule['secondary_action']}</strong></p>
                            </div>
                            <div>
                                <p><strong>Performance Metrics:</strong></p>
                                <ul style="list-style: none; padding: 0;">
                                    <li>📊 Expected Spread: <strong>{rule['expected_spread']:.2f}%</strong></li>
                                    <li>📈 Top Return: <strong>{rule['top_return']:.2f}%</strong></li>
                                    <li>📉 Bottom Return: <strong>{rule['bottom_return']:.2f}%</strong></li>
                                    <li>🎯 Confidence: <strong>{rule['confidence']:.1f}%</strong></li>
                                    <li>📐 Sharpe Ratio: <strong>{rule['sharpe']:.3f}</strong></li>
                                    <li>📅 Sample Size: <strong>{rule['samples']:,}</strong></li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Export button
                st.markdown("---")
                if st.button("📥 Export Rules as CSV"):
                    rules_df = pd.DataFrame(trading_rules)
                    csv = rules_df.to_csv(index=False)
                    st.download_button(
                        label="Download Trading Rules CSV",
                        data=csv,
                        file_name=f"{ticker}_trading_rules_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No significant trading rules found. Try adjusting parameters or adding more data.")

if __name__ == "__main__":
    main()
