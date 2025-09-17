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
import time

warnings.filterwarnings('ignore')

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(
    page_title="Comprehensive Quantitative Analyzer",
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
    
    @classmethod
    def _get_indicator_inputs(cls, func_name):
        """Detecta qué inputs necesita cada función"""
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
        """Calcula cualquier indicador de TALib"""
        try:
            if indicator_name.startswith('CDL'):
                func = getattr(talib, indicator_name)
                return func(open_prices, high, low, close)
            
            if indicator_name not in cls.INDICATOR_CONFIG:
                if hasattr(talib, indicator_name):
                    func = getattr(talib, indicator_name)
                    return func(close)
                return None
            
            func_name, params = cls.INDICATOR_CONFIG[indicator_name]
            func = getattr(talib, func_name)
            
            data_type = cls._get_indicator_inputs(func_name)
            
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
    
    @classmethod
    def get_all_indicators(cls):
        """Retorna lista de todos los indicadores"""
        return list(cls.INDICATOR_CONFIG.keys()) + cls.CANDLE_PATTERNS

# ===================== FUNCIONES DE CÁLCULO =====================
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
def calculate_all_indicators(ticker: str, start_date: str, end_date: datetime,
                           quantiles: int, return_days: int, 
                           periods_to_test: List[int], 
                           include_patterns: bool = True) -> Tuple:
    """Calcula TODOS los indicadores disponibles"""
    
    data = download_data(ticker, start_date, end_date)
    if data is None:
        return None, None, None, None
    
    # Calculate returns for different periods
    for i in range(1, return_days + 1):
        data[f'returns_{i}_days'] = data['Close'].pct_change(i) * 100
    
    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    close = data['Close'].values.astype(np.float64)
    volume = data['Volume'].values.astype(np.float64) if 'Volume' in data.columns else np.zeros_like(close)
    open_prices = data['Open'].values.astype(np.float64)
    
    indicators = pd.DataFrame(index=data.index)
    
    # Get all indicators
    all_indicators = list(TechnicalIndicators.INDICATOR_CONFIG.keys())
    if include_patterns:
        all_indicators += TechnicalIndicators.CANDLE_PATTERNS
    
    total_calculations = 0
    # Count total calculations
    for indicator_name in all_indicators:
        if TechnicalIndicators.needs_period(indicator_name):
            total_calculations += len(periods_to_test)
        else:
            total_calculations += 1
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    calculation_counter = 0
    
    # Calculate all indicators
    for indicator_name in all_indicators:
        if TechnicalIndicators.needs_period(indicator_name):
            for period in periods_to_test:
                calculation_counter += 1
                status_text.text(f"Calculating {indicator_name}_{period}... ({calculation_counter}/{total_calculations})")
                
                result = TechnicalIndicators.calculate_indicator(
                    indicator_name, high, low, close, volume, open_prices, period
                )
                
                if result is not None and not np.all(np.isnan(result)):
                    indicators[f'{indicator_name}_{period}'] = result
                
                progress_bar.progress(calculation_counter / total_calculations)
        else:
            calculation_counter += 1
            status_text.text(f"Calculating {indicator_name}... ({calculation_counter}/{total_calculations})")
            
            result = TechnicalIndicators.calculate_indicator(
                indicator_name, high, low, close, volume, open_prices, 0
            )
            
            if result is not None and not np.all(np.isnan(result)):
                indicators[indicator_name] = result
            
            progress_bar.progress(calculation_counter / total_calculations)
    
    progress_bar.empty()
    status_text.empty()
    
    indicators = indicators.dropna(axis=1, how='all')
    
    # Calculate percentile analysis for all indicators
    status_text.text("Analyzing percentile returns...")
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
    
    status_text.empty()
    
    # Find optimal configurations
    optimal_configs = find_best_configurations(indicators, returns_data, data, return_days)
    
    return returns_data, indicators, data, optimal_configs

def find_best_configurations(indicators, returns_data, data, return_days):
    """Find the best configurations across all calculated indicators"""
    configurations = []
    
    returns = data['Close'].pct_change(return_days).shift(-return_days) * 100
    
    for indicator_col in indicators.columns:
        if indicator_col in returns_data:
            ret_col = f'returns_{return_days}_days_mean'
            if ret_col in returns_data[indicator_col].columns:
                values = returns_data[indicator_col][ret_col]
                if len(values) > 1:
                    spread = values.iloc[-1] - values.iloc[0]
                    sharpe = abs(spread) / values.std() if values.std() > 0 else 0
                    
                    # Calculate direction
                    correlation, p_value = spearmanr(range(len(values)), values.values)
                    
                    configurations.append({
                        'indicator': indicator_col,
                        'spread': spread,
                        'sharpe': sharpe,
                        'direction': correlation,
                        'p_value': p_value,
                        'top_return': values.iloc[-1],
                        'bottom_return': values.iloc[0],
                        'samples': returns_data[indicator_col][f'returns_{return_days}_days_count'].sum()
                    })
    
    if configurations:
        df_configs = pd.DataFrame(configurations)
        df_configs['score'] = (
            abs(df_configs['spread']) * 0.5 +
            df_configs['sharpe'] * 10 +
            (1 / (df_configs['p_value'] + 0.001)) * 0.1
        )
        return df_configs.sort_values('score', ascending=False)
    
    return pd.DataFrame()

def extract_trading_rules(optimal_configs, min_spread=2.0, max_p_value=0.1, top_n=10):
    """Extract trading rules from all configurations"""
    if optimal_configs.empty:
        return []
    
    quality_signals = optimal_configs[
        (abs(optimal_configs['spread']) >= min_spread) & 
        (optimal_configs['p_value'] <= max_p_value)
    ].head(top_n)
    
    rules = []
    
    for idx, row in quality_signals.iterrows():
        if row['direction'] > 0.3:
            strategy_type = "MOMENTUM"
            primary_signal = f"When {row['indicator']} is HIGH (top 20%)"
            primary_action = "STRONG BUY"
            secondary_signal = f"When {row['indicator']} is LOW (bottom 20%)"
            secondary_action = "STRONG SELL"
        elif row['direction'] < -0.3:
            strategy_type = "MEAN REVERSION"
            primary_signal = f"When {row['indicator']} is LOW (bottom 20%)"
            primary_action = "STRONG BUY"
            secondary_signal = f"When {row['indicator']} is HIGH (top 20%)"
            secondary_action = "STRONG SELL"
        else:
            strategy_type = "SELECTIVE"
            primary_signal = f"When {row['indicator']} is in middle percentiles"
            primary_action = "CONSIDER ENTRY"
            secondary_signal = f"Extreme values"
            secondary_action = "AVOID"
        
        rules.append({
            'rank': idx + 1,
            'indicator': row['indicator'],
            'strategy_type': strategy_type,
            'primary_signal': primary_signal,
            'primary_action': primary_action,
            'secondary_signal': secondary_signal,
            'secondary_action': secondary_action,
            'expected_spread': row['spread'],
            'top_return': row.get('top_return', 0),
            'bottom_return': row.get('bottom_return', 0),
            'direction': row['direction'],
            'confidence': (1 - row['p_value']) * 100,
            'sharpe': row['sharpe'],
            'samples': row.get('samples', 0)
        })
    
    return rules

def create_percentile_plots(indicators: pd.DataFrame, returns_data: Dict, 
                           data: pd.DataFrame, indicator_name: str, 
                           return_days: int) -> go.Figure:
    """Create percentile analysis plots"""
    
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
    
    # 1. Histogram with KDE
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
    
    # Add KDE curve
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
                    name='KDE',
                    showlegend=False
                ),
                row=1, col=1
            )
        except:
            pass
    
    mean_val = hist_data.mean()
    std_val = hist_data.std()
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

# ===================== MAIN APPLICATION =====================
def main():
    st.markdown("""
        <h1 style='text-align: center;'>
            📊 Comprehensive Quantitative Analyzer
        </h1>
        <p style='text-align: center; color: #8892B0; font-size: 1.2rem;'>
            Calculate ALL indicators → Analyze any configuration → Extract best trading rules
        </p>
        <p style='text-align: center; color: #667eea; font-size: 0.9rem;'>
            {total} indicators available for analysis
        </p>
    """.format(total=TechnicalIndicators.get_total_indicators()), unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.returns_data = None
        st.session_state.indicators = None
        st.session_state.data = None
        st.session_state.optimal_configs = None
        st.session_state.trading_rules = None
    
    # Ensure trading_rules exists
    if 'trading_rules' not in st.session_state:
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
            
            st.markdown("#### Period Testing")
            periods_preset = st.selectbox(
                "Period Selection",
                ["Fast (8 periods)", "Standard (15 periods)", "Comprehensive (30 periods)"]
            )
            
            if periods_preset == "Fast (8 periods)":
                periods_to_test = [5, 10, 14, 20, 30, 50, 100, 200]
            elif periods_preset == "Standard (15 periods)":
                periods_to_test = [5, 8, 10, 12, 14, 16, 20, 25, 30, 40, 50, 75, 100, 150, 200]
            else:  # Comprehensive
                periods_to_test = list(range(5, 101, 5)) + [120, 150, 200, 250]
            
            st.info(f"Testing {len(periods_to_test)} periods: {periods_to_test[:5]}...{periods_to_test[-3:]}")
            
            include_patterns = st.checkbox("Include Candlestick Patterns (61)", value=False)
        
        with st.expander("🎯 **ANALYSIS SCOPE**", expanded=True):
            analysis_mode = st.radio(
                "Analysis Mode",
                ["🚀 Calculate ALL Indicators", "📊 Quick Analysis (Top 25)"]
            )
            
            if analysis_mode == "📊 Quick Analysis (Top 25)":
                selected_indicators = ['RSI', 'MACD', 'BBANDS', 'ATR', 'ADX',
                                     'SMA', 'EMA', 'STOCH', 'CCI', 'MFI',
                                     'WILLR', 'MOM', 'ROC', 'PPO', 'CMO',
                                     'AROON', 'ULTOSC', 'OBV', 'AD', 'NATR',
                                     'TRIX', 'TEMA', 'WMA', 'DEMA', 'KAMA']
                total_calcs = sum(len(periods_to_test) if TechnicalIndicators.needs_period(ind) else 1 
                                for ind in selected_indicators)
                st.warning(f"Quick mode: {len(selected_indicators)} indicators → ~{total_calcs} calculations")
            else:
                all_indicators = TechnicalIndicators.get_all_indicators()
                if not include_patterns:
                    all_indicators = [i for i in all_indicators if not i.startswith('CDL')]
                total_calcs = sum(len(periods_to_test) if TechnicalIndicators.needs_period(ind) else 1 
                                for ind in all_indicators)
                st.warning(f"⚠️ Full analysis: {len(all_indicators)} indicators → ~{total_calcs} calculations\nThis may take 2-5 minutes!")
        
        analyze_button = st.button(
            "🚀 **RUN COMPREHENSIVE ANALYSIS**",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button:
        start_time = time.time()
        
        with st.spinner('📊 Calculating all indicators...'):
            if analysis_mode == "📊 Quick Analysis (Top 25)":
                # Quick analysis with selected indicators
                from .main import calculate_indicators_batch
                returns_data, indicators, data = calculate_indicators_batch(
                    ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date,
                    selected_indicators,
                    quantiles,
                    return_days,
                    (min(periods_to_test), max(periods_to_test), 5)
                )
                optimal_configs = find_best_configurations(indicators, returns_data, data, return_days)
            else:
                # Full analysis
                returns_data, indicators, data, optimal_configs = calculate_all_indicators(
                    ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date,
                    quantiles,
                    return_days,
                    periods_to_test,
                    include_patterns
                )
            
            if returns_data and indicators is not None and data is not None:
                st.session_state.analysis_done = True
                st.session_state.returns_data = returns_data
                st.session_state.indicators = indicators
                st.session_state.data = data
                st.session_state.optimal_configs = optimal_configs
                
                # Extract trading rules from ALL configurations
                if optimal_configs is not None and not optimal_configs.empty:
                    st.session_state.trading_rules = extract_trading_rules(
                        optimal_configs,
                        min_spread=2.0,
                        max_p_value=0.1,
                        top_n=20
                    )
                
                elapsed_time = time.time() - start_time
                st.success(f"✅ Analysis complete in {elapsed_time:.1f} seconds! Analyzed {len(indicators.columns)} configurations")
    
    if st.session_state.analysis_done:
        returns_data = st.session_state.returns_data
        indicators = st.session_state.indicators
        data = st.session_state.data
        optimal_configs = st.session_state.optimal_configs
        trading_rules = st.session_state.trading_rules
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Configurations", f"{len(indicators.columns):,}")
        with col2:
            unique_indicators = len(set(col.split('_')[0] for col in indicators.columns))
            st.metric("Unique Indicators", unique_indicators)
        with col3:
            if optimal_configs is not None and not optimal_configs.empty:
                significant = len(optimal_configs[optimal_configs['p_value'] <= 0.1])
                st.metric("Significant Patterns", significant)
        with col4:
            if trading_rules:
                st.metric("Trading Rules", len(trading_rules))
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 **Individual Analysis**",
            "🏆 **Best Configurations**",
            "📋 **Trading Rules**",
            "📊 **Export Data**"
        ])
        
        with tab1:
            st.markdown("### 📈 Individual Indicator Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                available = sorted(list(indicators.columns))
                selected_ind = st.selectbox(
                    "Select indicator to analyze",
                    available,
                    help="Choose any calculated indicator configuration"
                )
            
            with col2:
                if selected_ind:
                    # Show metrics for selected indicator
                    if selected_ind in returns_data:
                        ret_col = f'returns_{return_days}_days_mean'
                        if ret_col in returns_data[selected_ind].columns:
                            values = returns_data[selected_ind][ret_col]
                            spread = values.iloc[-1] - values.iloc[0]
                            st.metric("Spread", f"{spread:.2f}%")
            
            if selected_ind:
                fig = create_percentile_plots(
                    indicators, returns_data, data,
                    selected_ind, return_days
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show specific configuration details
                with st.expander("Configuration Details"):
                    if selected_ind in optimal_configs['indicator'].values:
                        config = optimal_configs[optimal_configs['indicator'] == selected_ind].iloc[0]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Direction", f"{config['direction']:.3f}")
                            st.metric("P-value", f"{config['p_value']:.4f}")
                        with col2:
                            st.metric("Sharpe", f"{config['sharpe']:.3f}")
                            st.metric("Samples", f"{config.get('samples', 'N/A'):,}")
                        with col3:
                            st.metric("Top Return", f"{config.get('top_return', 0):.2f}%")
                            st.metric("Bottom Return", f"{config.get('bottom_return', 0):.2f}%")
        
        with tab2:
            st.markdown("### 🏆 Best Performing Configurations")
            
            if optimal_configs is not None and not optimal_configs.empty:
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_spread_filter = st.slider("Min Spread %", 0.0, 10.0, 2.0, 0.5)
                with col2:
                    max_p_filter = st.slider("Max P-value", 0.01, 0.20, 0.10, 0.01)
                with col3:
                    sort_by = st.selectbox("Sort by", ["score", "spread", "sharpe", "p_value"])
                
                # Apply filters
                filtered_configs = optimal_configs[
                    (abs(optimal_configs['spread']) >= min_spread_filter) &
                    (optimal_configs['p_value'] <= max_p_filter)
                ].sort_values(sort_by, ascending=(sort_by == 'p_value'))
                
                st.info(f"Showing {len(filtered_configs)} configurations matching criteria")
                
                # Display table
                display_columns = ['indicator', 'spread', 'direction', 'p_value', 'sharpe', 'score']
                available_columns = [col for col in display_columns if col in filtered_configs.columns]
                
                if available_columns:
                    display_df = filtered_configs[available_columns].head(50)
                    
                    format_dict = {}
                    if 'spread' in display_df.columns:
                        format_dict['spread'] = '{:.2f}%'
                    if 'direction' in display_df.columns:
                        format_dict['direction'] = '{:.3f}'
                    if 'p_value' in display_df.columns:
                        format_dict['p_value'] = '{:.4f}'
                    if 'sharpe' in display_df.columns:
                        format_dict['sharpe'] = '{:.3f}'
                    if 'score' in display_df.columns:
                        format_dict['score'] = '{:.2f}'
                    
                    gradient_cols = [col for col in ['spread', 'score'] if col in display_df.columns]
                    
                    st.dataframe(
                        display_df.style.format(format_dict).background_gradient(
                            subset=gradient_cols, cmap='RdYlGn'
                        ),
                        use_container_width=True,
                        height=600
                    )
        
        with tab3:
            st.markdown("### 📋 Extracted Trading Rules")
            
            if trading_rules:
                # Strategy type filter
                strategy_types = list(set(r['strategy_type'] for r in trading_rules))
                selected_strategy = st.multiselect(
                    "Filter by strategy type",
                    strategy_types,
                    default=strategy_types
                )
                
                filtered_rules = [r for r in trading_rules if r['strategy_type'] in selected_strategy]
                
                st.info(f"Showing {len(filtered_rules)} trading rules")
                
                for rule in filtered_rules[:20]:
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
                            <h4>Rule #{rule['rank']}: {rule['indicator']}</h4>
                            <div>{strategy_badge} {confidence_badge}</div>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            <div>
                                <p><strong>Entry Signal:</strong></p>
                                <p style="color: #4CAF50;">✓ {rule['primary_signal']}</p>
                                <p style="margin-left: 20px;">→ <strong>{rule['primary_action']}</strong></p>
                                
                                <p style="margin-top: 10px;"><strong>Exit Signal:</strong></p>
                                <p style="color: #FF5252;">✗ {rule['secondary_signal']}</p>
                                <p style="margin-left: 20px;">→ <strong>{rule['secondary_action']}</strong></p>
                            </div>
                            <div>
                                <p><strong>Expected Performance:</strong></p>
                                <ul style="list-style: none; padding: 0;">
                                    <li>📊 Spread: <strong>{rule['expected_spread']:.2f}%</strong></li>
                                    <li>🎯 Confidence: <strong>{rule['confidence']:.1f}%</strong></li>
                                    <li>📐 Sharpe: <strong>{rule['sharpe']:.3f}</strong></li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No significant trading rules found. Try adjusting filter parameters.")
        
        with tab4:
            st.markdown("### 📥 Export Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Export Configurations")
                if st.button("📊 Download Best Configurations (CSV)"):
                    csv = optimal_configs.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_configurations_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                st.markdown("#### Export Indicators")
                if st.button("📈 Download All Indicators (CSV)"):
                    # Merge indicators with dates
                    export_df = pd.concat([data[['Close']], indicators], axis=1)
                    csv = export_df.to_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_all_indicators_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                st.markdown("#### Export Trading Rules")
                if trading_rules and st.button("📋 Download Trading Rules (CSV)"):
                    rules_df = pd.DataFrame(trading_rules)
                    csv = rules_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{ticker}_trading_rules_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                st.markdown("#### Export Summary Report")
                if st.button("📄 Download Full Report (TXT)"):
                    report = f"""
QUANTITATIVE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Ticker: {ticker}
Period: {start_date} to {end_date}

SUMMARY STATISTICS
==================
Total Configurations Analyzed: {len(indicators.columns)}
Unique Indicators: {len(set(col.split('_')[0] for col in indicators.columns))}
Significant Patterns (p<0.1): {len(optimal_configs[optimal_configs['p_value'] <= 0.1]) if optimal_configs is not None else 0}
Trading Rules Generated: {len(trading_rules) if trading_rules else 0}

TOP 10 CONFIGURATIONS
====================
{optimal_configs.head(10).to_string() if optimal_configs is not None else 'No configurations found'}

TOP TRADING RULES
=================
"""
                    if trading_rules:
                        for i, rule in enumerate(trading_rules[:10], 1):
                            report += f"""
Rule #{i}: {rule['indicator']}
Strategy: {rule['strategy_type']}
Expected Spread: {rule['expected_spread']:.2f}%
Confidence: {rule['confidence']:.1f}%
Entry: {rule['primary_signal']} → {rule['primary_action']}
Exit: {rule['secondary_signal']} → {rule['secondary_action']}
---
"""
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"{ticker}_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )

if __name__ == "__main__":
    main()
