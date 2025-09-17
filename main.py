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
from scipy.signal import find_peaks
from scipy.stats import spearmanr, kurtosis

warnings.filterwarnings('ignore')

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(
    page_title="Analizador Cuantitativo de Indicadores Técnicos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== ESTILOS CSS PROFESIONALES =====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Fondo con gradiente oscuro */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #151932 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers con gradiente */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #E0E5FF;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem;
    }
    
    h3 {
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 1.3rem !important;
    }
    
    /* Métricas con efecto glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(99, 102, 241, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(99, 102, 241, 0.15);
    }
    
    /* Botones con gradiente y animación */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.02em;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.35);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 35px rgba(102, 126, 234, 0.5);
    }
    
    /* Selectbox mejorado */
    .stSelectbox > div > div {
        background: rgba(30, 34, 56, 0.9);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    /* Sidebar con glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(30, 34, 56, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Progress bar animada */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #00D2FF 50%, #764ba2 100%);
        background-size: 200% 100%;
        animation: gradient 2s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== CLASE COMPACTA PARA MANEJO DE INDICADORES =====================
class TechnicalIndicators:
    """Manejador elegante y compacto de todos los indicadores de TALib"""
    
    # Configuración de indicadores: nombre -> (función, args_especiales)
    INDICATOR_CONFIG = {
        # Overlaps
        'BBANDS': ('BBANDS', {'timeperiod': 'p', 'nbdevup': 2, 'nbdevdn': 2, 'matype': 0}),
        'DEMA': ('DEMA', {'timeperiod': 'p'}),
        'EMA': ('EMA', {'timeperiod': 'p'}),
        'HT_TRENDLINE': ('HT_TRENDLINE', {}),
        'KAMA': ('KAMA', {'timeperiod': 'p'}),
        'MA': ('MA', {'timeperiod': 'p', 'matype': 0}),
        'MAMA': ('MAMA', {'fastlimit': 0.5, 'slowlimit': 0.05}),
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
        
        # Momentum
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
        
        # Volume
        'AD': ('AD', {}),
        'ADOSC': ('ADOSC', {'fastperiod': 'max(p//3, 2)', 'slowperiod': 'p'}),
        'OBV': ('OBV', {}),
        
        # Volatility
        'ATR': ('ATR', {'timeperiod': 'p'}),
        'NATR': ('NATR', {'timeperiod': 'p'}),
        'TRANGE': ('TRANGE', {}),
        
        # Cycles
        'HT_DCPERIOD': ('HT_DCPERIOD', {}),
        'HT_DCPHASE': ('HT_DCPHASE', {}),
        'HT_PHASOR': ('HT_PHASOR', {}),
        'HT_SINE': ('HT_SINE', {}),
        'HT_TRENDMODE': ('HT_TRENDMODE', {}),
        
        # Statistics
        'BETA': ('BETA', {'timeperiod': 'p'}),
        'CORREL': ('CORREL', {'timeperiod': 'p'}),
        'LINEARREG': ('LINEARREG', {'timeperiod': 'p'}),
        'LINEARREG_ANGLE': ('LINEARREG_ANGLE', {'timeperiod': 'p'}),
        'LINEARREG_INTERCEPT': ('LINEARREG_INTERCEPT', {'timeperiod': 'p'}),
        'LINEARREG_SLOPE': ('LINEARREG_SLOPE', {'timeperiod': 'p'}),
        'STDDEV': ('STDDEV', {'timeperiod': 'p', 'nbdev': 1}),
        'TSF': ('TSF', {'timeperiod': 'p'}),
        'VAR': ('VAR', {'timeperiod': 'p', 'nbdev': 1}),
    }
    
    # Categorías de indicadores
    CATEGORIES = {
        "📈 Overlaps": ['BBANDS', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 
                       'MAMA', 'MIDPOINT', 'MIDPRICE', 'SAR', 'SAREXT', 
                       'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'],
        "💫 Momentum": ['ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP', 
                       'CCI', 'CMO', 'DX', 'MACD', 'MACDEXT', 'MACDFIX', 
                       'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 
                       'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 
                       'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX', 
                       'ULTOSC', 'WILLR'],
        "📊 Volumen": ['AD', 'ADOSC', 'OBV'],
        "📉 Volatilidad": ['ATR', 'NATR', 'TRANGE'],
        "🎯 Ciclos": ['HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR', 'HT_SINE', 'HT_TRENDMODE'],
        "📐 Estadísticas": ['BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 
                           'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR'],
    }
    
    # Indicadores de patrones de velas (generados dinámicamente)
    CANDLE_PATTERNS = [name for name in dir(talib) if name.startswith('CDL')]
    
    @classmethod
    def _get_indicator_inputs(cls, func_name):
        """Detecta automáticamente qué inputs necesita una función"""
        if func_name.startswith('CDL'):
            return 'ohlc'
        elif func_name in ['AD', 'ADOSC']:
            return 'hlcv'
        elif func_name in ['OBV']:
            return 'cv'
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
        elif func_name in ['MIDPRICE']:
            return 'hl'
        else:
            return 'c'
    
    @classmethod
    def calculate_indicator(cls, indicator_name, high, low, close, volume, open_prices, period):
        """Calcula un indicador de forma dinámica"""
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
            
            # Preparar argumentos según el tipo
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
    def get_all_categories(cls):
        """Retorna todas las categorías con sus indicadores"""
        categories = cls.CATEGORIES.copy()
        categories["🕯️ Patrones de Velas"] = cls.CANDLE_PATTERNS
        return categories
    
    @classmethod
    def needs_period(cls, indicator_name):
        """Determina si un indicador necesita período"""
        no_period = [
            'HT_TRENDLINE', 'BOP', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR',
            'HT_SINE', 'HT_TRENDMODE', 'MACDFIX', 'AD', 'OBV', 'TRANGE',
            'SAR', 'SAREXT', 'MAMA'
        ] + cls.CANDLE_PATTERNS
        
        return indicator_name not in no_period

# ===================== FUNCIONES DE ANÁLISIS DE SKEW =====================
def calculate_percentile_skew_metrics(indicator_values, returns_data, quantiles=20):
    """
    Calcula métricas avanzadas de skew y asimetría para análisis de percentiles
    """
    metrics = {}
    
    try:
        temp_df = pd.DataFrame({
            'indicator': indicator_values,
            'returns': returns_data
        }).dropna()
        
        if len(temp_df) < quantiles * 2:
            return None
        
        temp_df['percentile'] = pd.qcut(temp_df['indicator'], q=quantiles, labels=False, duplicates='drop')
        percentile_returns = temp_df.groupby('percentile')['returns'].agg(['mean', 'std', 'count'])
        
        if len(percentile_returns) < quantiles * 0.8:
            return None
        
        # 1. SKEW - Asimetría de la distribución
        metrics['return_skew'] = stats.skew(percentile_returns['mean'].values)
        
        # 2. MONOTONICITY - Correlación de Spearman
        correlation, p_value = spearmanr(range(len(percentile_returns)), percentile_returns['mean'].values)
        metrics['monotonicity'] = correlation
        metrics['monotonicity_pvalue'] = p_value
        
        # 3. EDGE RATIO - Diferencia entre extremos
        P95_return = percentile_returns['mean'].iloc[-1] if len(percentile_returns) > 0 else 0
        P5_return = percentile_returns['mean'].iloc[0] if len(percentile_returns) > 0 else 0
        middle_std = percentile_returns['std'].mean()
        metrics['edge_ratio'] = (P95_return - P5_return) / (middle_std + 1e-8)
        metrics['spread'] = P95_return - P5_return
        
        # 4. CONVEXITY - Curvatura
        returns_array = percentile_returns['mean'].values
        if len(returns_array) > 2:
            first_diff = np.diff(returns_array)
            second_diff = np.diff(first_diff)
            metrics['convexity'] = np.mean(second_diff)
        else:
            metrics['convexity'] = 0
        
        # 5. TAIL ASYMMETRY
        n_percentiles = len(percentile_returns)
        lower_tail = percentile_returns['mean'].iloc[:n_percentiles//4].mean()
        upper_tail = percentile_returns['mean'].iloc[-n_percentiles//4:].mean()
        metrics['tail_asymmetry'] = abs(upper_tail) - abs(lower_tail)
        
        # 6. KURTOSIS
        metrics['return_kurtosis'] = kurtosis(percentile_returns['mean'].values)
        
        # 7. CONSISTENCY SCORE
        metrics['consistency'] = -percentile_returns['std'].mean() / (abs(percentile_returns['mean'].mean()) + 1e-8)
        
        # 8. INFORMATION RATIO
        metrics['info_ratio'] = percentile_returns['mean'].mean() / (percentile_returns['std'].mean() + 1e-8)
        
        # 9. WIN RATE
        metrics['win_rate'] = (percentile_returns['mean'] > 0).mean()
        
        # 10. BEST/WORST PERCENTILES
        metrics['best_percentile'] = percentile_returns['mean'].idxmax()
        metrics['worst_percentile'] = percentile_returns['mean'].idxmin()
        metrics['best_return'] = percentile_returns['mean'].max()
        metrics['worst_return'] = percentile_returns['mean'].min()
        
    except Exception:
        return None
    
    return metrics

def find_optimal_periods(indicator_name, data, min_period=5, max_period=100, step=1, return_days=5):
    """
    Encuentra los períodos óptimos para un indicador
    """
    periods_analysis = {}
    
    high = data['High'].values.astype(np.float64)
    low = data['Low'].values.astype(np.float64)
    close = data['Close'].values.astype(np.float64)
    volume = data['Volume'].values.astype(np.float64) if 'Volume' in data.columns else np.zeros_like(close)
    open_prices = data['Open'].values.astype(np.float64)
    
    returns = data['Close'].pct_change(return_days).shift(-return_days) * 100
    
    for period in range(min_period, min(max_period + 1, len(data) // 4), step):
        indicator_values = TechnicalIndicators.calculate_indicator(
            indicator_name, high, low, close, volume, open_prices, period
        )
        
        if indicator_values is None or np.all(np.isnan(indicator_values)):
            continue
        
        metrics = calculate_percentile_skew_metrics(indicator_values, returns.values, quantiles=10)
        
        if metrics:
            periods_analysis[period] = metrics
            periods_analysis[period]['period'] = period
    
    if not periods_analysis:
        return pd.DataFrame()
    
    df_analysis = pd.DataFrame.from_dict(periods_analysis, orient='index')
    
    df_analysis['composite_score'] = (
        abs(df_analysis['edge_ratio']) * 0.3 +
        abs(df_analysis['monotonicity']) * 0.25 +
        abs(df_analysis['return_skew']) * 0.15 +
        df_analysis['win_rate'] * 0.15 +
        abs(df_analysis['info_ratio']) * 0.15
    )
    
    if len(df_analysis) > 3:
        scores = df_analysis['composite_score'].values
        peaks, properties = find_peaks(scores, distance=3, prominence=0.1)
        df_analysis['is_peak'] = False
        df_analysis.iloc[peaks, df_analysis.columns.get_loc('is_peak')] = True
    
    return df_analysis.sort_values('composite_score', ascending=False)

def create_skew_analysis_plot(periods_df, indicator_name):
    """
    Crea visualización del análisis de períodos óptimos
    """
    if periods_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>📈 Score Compuesto por Período</b>',
            '<b>🎯 Edge Ratio vs Monotonicity</b>',
            '<b>📊 Métricas de Skew</b>',
            '<b>🏆 Top 5 Períodos Óptimos</b>'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "table"}]]
    )
    
    # 1. Score Compuesto
    fig.add_trace(
        go.Scatter(
            x=periods_df.index,
            y=periods_df['composite_score'],
            mode='lines+markers',
            marker=dict(
                size=8,
                color=periods_df['composite_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score", x=0.45, len=0.4)
            ),
            line=dict(color='rgba(102, 126, 234, 0.8)', width=2),
            name='Score',
            hovertemplate='<b>Período:</b> %{x}<br><b>Score:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Edge Ratio vs Monotonicity
    fig.add_trace(
        go.Scatter(
            x=periods_df['monotonicity'],
            y=periods_df['edge_ratio'],
            mode='markers',
            marker=dict(
                size=10,
                color=periods_df.index,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Período", x=1.02, len=0.4)
            ),
            text=periods_df.index,
            hovertemplate='<b>Período:</b> %{text}<br><b>Monotonicity:</b> %{x:.3f}<br><b>Edge Ratio:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'), row=1, col=2)
    fig.add_vline(x=0, line=dict(color='gray', width=1, dash='dash'), row=1, col=2)
    
    # 3. Métricas de Skew
    metrics_to_plot = ['return_skew', 'tail_asymmetry', 'convexity']
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']
    
    for metric, color in zip(metrics_to_plot, colors):
        if metric in periods_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=periods_df.index,
                    y=periods_df[metric],
                    mode='lines',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color, width=2)
                ),
                row=2, col=1
            )
    
    # 4. Tabla Top 5
    top_5 = periods_df.nlargest(5, 'composite_score')
    
    table_data = []
    table_data.append(['<b>Período</b>', '<b>Score</b>', '<b>Spread</b>', '<b>Skew</b>', '<b>Edge</b>'])
    
    for idx in top_5.index:
        row = top_5.loc[idx]
        table_data.append([
            f"{idx}",
            f"{row['composite_score']:.3f}",
            f"{row['spread']:.2f}%",
            f"{row['return_skew']:.3f}",
            f"{row['edge_ratio']:.3f}"
        ])
    
    fig.add_trace(
        go.Table(
            cells=dict(
                values=list(zip(*table_data)),
                fill_color=['rgba(102, 126, 234, 0.3)'] + ['rgba(30, 34, 56, 0.6)'] * 5,
                align='center',
                font=dict(color='white', size=12),
                height=30
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=800,
        title=f"<b>🔍 Análisis de Períodos Óptimos: {indicator_name}</b>",
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='rgba(30, 34, 56, 0.3)',
        paper_bgcolor='rgba(14, 17, 39, 0.95)',
        font=dict(family="Inter, sans-serif", color='#E0E5FF', size=11)
    )
    
    fig.update_xaxes(title_text="<b>Período</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Score Compuesto</b>", row=1, col=1)
    fig.update_xaxes(title_text="<b>Monotonicity</b>", row=1, col=2)
    fig.update_yaxes(title_text="<b>Edge Ratio</b>", row=1, col=2)
    fig.update_xaxes(title_text="<b>Período</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Valor</b>", row=2, col=1)
    
    return fig

# ===================== FUNCIONES ORIGINALES DE CÁLCULO =====================
@st.cache_data
def download_data(ticker: str, start_date: str, end_date: datetime) -> Optional[pd.DataFrame]:
    """Descarga datos históricos con manejo de errores"""
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
            st.error(f"❌ No se encontraron datos para {ticker}")
            return None
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error descargando datos: {str(e)}")
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
    
    progress_container = st.container()
    progress_bar = st.progress(0)
    
    for indicator_name in indicators_list:
        if TechnicalIndicators.needs_period(indicator_name):
            min_p, max_p, step = period_range
            periods = range(min_p, max_p + 1, step)
            
            for period in periods:
                total_calculations += 1
                result = TechnicalIndicators.calculate_indicator(
                    indicator_name, high, low, close, volume, open_prices, period
                )
                
                if result is not None and not np.all(np.isnan(result)):
                    indicators[f'{indicator_name}_{period}'] = result
                    successful_calculations += 1
                
                progress_bar.progress(min(successful_calculations / max(total_calculations, 1), 1.0))
        else:
            total_calculations += 1
            result = TechnicalIndicators.calculate_indicator(
                indicator_name, high, low, close, volume, open_prices, 0
            )
            
            if result is not None and not np.all(np.isnan(result)):
                indicators[indicator_name] = result
                successful_calculations += 1
            
            progress_bar.progress(min(successful_calculations / max(total_calculations, 1), 1.0))
    
    progress_bar.empty()
    
    indicators = indicators.dropna(axis=1, how='all')
    
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

def create_percentile_plots(indicators: pd.DataFrame, returns_data: Dict, 
                           data: pd.DataFrame, indicator_name: str, 
                           return_days: int) -> go.Figure:
    """Crea los 4 gráficos de análisis de percentiles con diseño premium"""
    
    if indicator_name not in indicators.columns:
        return None
    
    if indicator_name not in returns_data:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'<b>📊 Distribución de {indicator_name}</b>',
            f'<b>📈 Retornos por Percentil ({return_days} días)</b>',
            f'<b>📉 Correlación Móvil (126 días)</b>',
            f'<b>🎯 Análisis de Dispersión</b>'
        ),
        row_heights=[0.5, 0.5],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 1. Histograma
    hist_data = indicators[indicator_name].dropna()
    
    fig.add_trace(
        go.Histogram(
            x=hist_data,
            nbinsx=70,
            marker=dict(
                color='rgba(102, 126, 234, 0.8)',
                line=dict(color='rgba(255,255,255,0.2)', width=0.5)
            ),
            name='Distribución',
            showlegend=False,
            hovertemplate='<b>Valor:</b> %{x:.2f}<br><b>Frecuencia:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    mean_val = hist_data.mean()
    median_val = hist_data.median()
    std_val = hist_data.std()
    
    fig.add_vline(x=mean_val, line=dict(color='#FF6B6B', width=2, dash='dash'),
                  row=1, col=1, annotation_text=f'μ={mean_val:.2f}', annotation_position="top")
    fig.add_vline(x=median_val, line=dict(color='#4ECDC4', width=2, dash='dot'),
                  row=1, col=1, annotation_text=f'M={median_val:.2f}', annotation_position="bottom")
    
    fig.add_vrect(x0=mean_val-std_val, x1=mean_val+std_val,
                  fillcolor="rgba(102, 126, 234, 0.1)", layer="below",
                  line_width=0, row=1, col=1)
    
    # 2. Retornos por percentil
    returns_col = f'returns_{return_days}_days_mean'
    if returns_col in returns_data[indicator_name].columns:
        returns_values = returns_data[indicator_name][returns_col]
        x_labels = [f'P{i+1}' for i in range(len(returns_values))]
        
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=returns_values,
                marker=dict(
                    color=returns_values,
                    colorscale='RdYlGn',
                    line=dict(color='rgba(255,255,255,0.3)', width=1),
                    showscale=False
                ),
                text=[f'{val:.2f}%' for val in returns_values],
                textposition='outside',
                textfont=dict(size=9, color='white'),
                name='Retornos',
                showlegend=False,
                hovertemplate='<b>Percentil:</b> %{x}<br><b>Retorno:</b> %{y:.3f}%<extra></extra>'
            ),
            row=1, col=2
        )
    
    # 3. Correlación móvil
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
                    fill='tonexty',
                    fillcolor='rgba(0, 210, 255, 0.05)',
                    name='Correlación Móvil',
                    showlegend=False,
                    hovertemplate='<b>Fecha:</b> %{x}<br><b>Correlación:</b> %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.2)', width=1),
                         row=2, col=1)
            
            overall_corr = aligned_returns.corr(aligned_indicator)
            fig.add_hline(
                y=overall_corr,
                line=dict(color='#FFD93D', width=2, dash='dash'),
                row=2, col=1,
                annotation_text=f'ρ={overall_corr:.3f}',
                annotation_position="right"
            )
    
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
                            colorscale='Spectral',
                            opacity=0.4,
                            line=dict(width=0),
                            showscale=True,
                            colorbar=dict(
                                title="Retorno %",
                                x=1.02,
                                y=0.25,
                                yanchor="middle",
                                len=0.4,
                                thickness=10
                            )
                        ),
                        name='Datos',
                        showlegend=False,
                        hovertemplate='<b>Indicador:</b> %{x:.2f}<br><b>Retorno:</b> %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=2
                )
    
    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=False,
        title={
            'text': f"<b>📊 Análisis de Percentiles: {indicator_name}</b>",
            'font': {'size': 26, 'color': '#E0E5FF', 'family': 'Inter'},
            'x': 0.5,
            'xanchor': 'center'
        },
        hovermode='closest',
        plot_bgcolor='rgba(30, 34, 56, 0.3)',
        paper_bgcolor='rgba(14, 17, 39, 0.95)',
        font=dict(family="Inter, sans-serif", color='#E0E5FF', size=11),
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(99, 102, 241, 0.08)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(99, 102, 241, 0.15)',
                row=row, col=col
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(99, 102, 241, 0.08)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(99, 102, 241, 0.15)',
                row=row, col=col
            )
    
    return fig

# ===================== INTERFAZ PRINCIPAL =====================
def main():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 0; animation: gradient 3s ease infinite;'>
            📊 Analizador Cuantitativo de Indicadores Técnicos
        </h1>
        <p style='text-align: center; color: #8892B0; font-size: 1.2rem; margin-bottom: 2rem;'>
            Análisis de Percentiles, Skew Analytics y Optimización de Períodos
        </p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 1.2rem; 
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
                        border-radius: 20px; border: 1px solid rgba(102, 126, 234, 0.3);
                        backdrop-filter: blur(10px); margin-bottom: 2rem;
                        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);'>
                <p style='margin: 0; color: #8892B0; font-size: 0.9rem; letter-spacing: 0.05em;'>
                    DESARROLLADO POR
                </p>
                <p style='margin: 0.5rem 0; font-size: 1.4rem; font-weight: 600;'>
                    <a href='https://twitter.com/Gsnchez' style='text-decoration: none;'>
                        <span style='background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%);
                                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                            @Gsnchez
                        </span>
                    </a>
                    <span style='color: #667eea; margin: 0 0.5rem;'>•</span>
                    <a href='https://bquantfinance.com' style='text-decoration: none;'>
                        <span style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                            bquantfinance.com
                        </span>
                    </a>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.returns_data = None
        st.session_state.indicators = None
        st.session_state.data = None
        st.session_state.analysis_params = None
    
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; margin-bottom: 1rem;
                        background: linear-gradient(135deg, rgba(0, 210, 255, 0.1) 0%, rgba(58, 123, 213, 0.1) 100%);
                        border-radius: 12px; border: 1px solid rgba(0, 210, 255, 0.3);'>
                <h2 style='margin: 0;'>⚙️ Configuración</h2>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📈 **DATOS DEL MERCADO**", expanded=True):
            ticker = st.text_input(
                "Símbolo Bursátil",
                value="SPY",
                help="Ingrese el ticker del activo (ej: AAPL, MSFT, BTC-USD)"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Fecha Inicio",
                    value=datetime(2020, 1, 1),
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime.now()
                )
            with col2:
                end_date = st.date_input(
                    "Fecha Fin",
                    value=datetime.now(),
                    min_value=datetime(2000, 1, 1),
                    max_value=datetime.now()
                )
        
        with st.expander("📊 **PARÁMETROS DE ANÁLISIS**", expanded=True):
            return_days = st.select_slider(
                "Días de Retorno",
                options=[1, 2, 3, 5, 7, 10, 14, 20, 30, 60],
                value=5,
                help="Período para calcular retornos"
            )
            
            quantiles = st.slider(
                "Número de Percentiles",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="División en percentiles para el análisis"
            )
            
            st.markdown("**Rango de Períodos**")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_period = st.number_input("Mín", value=5, min_value=2, max_value=200)
            with col2:
                max_period = st.number_input("Máx", value=50, min_value=2, max_value=200)
            with col3:
                step_period = st.number_input("Step", value=5, min_value=1, max_value=50)
        
        st.markdown("""
            <div style='padding: 0.8rem; margin: 1rem 0;
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
                        border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.3);'>
                <h3 style='margin: 0; text-align: center;'>📐 Indicadores Técnicos</h3>
            </div>
        """, unsafe_allow_html=True)
        
        categories = TechnicalIndicators.get_all_categories()
        
        selection_method = st.radio(
            "Método de Selección",
            ["🎯 Presets Rápidos", "📁 Por Categorías", "✏️ Selección Manual"],
            horizontal=False
        )
        
        selected_indicators = []
        
        if selection_method == "🎯 Presets Rápidos":
            preset = st.selectbox(
                "Seleccionar Preset",
                [
                    "📊 Esenciales (RSI, MACD, BB, ATR)",
                    "💫 Top Momentum (10 indicadores)",
                    "📈 Análisis Completo (25 indicadores)",
                    "🔥 Trading Profesional (15 indicadores)",
                ]
            )
            
            presets = {
                "📊 Esenciales (RSI, MACD, BB, ATR)": ['RSI', 'MACD', 'BBANDS', 'ATR'],
                "💫 Top Momentum (10 indicadores)": ['RSI', 'MACD', 'STOCH', 'CCI', 'MFI', 
                                                     'WILLR', 'MOM', 'ROC', 'ADX', 'AROONOSC'],
                "📈 Análisis Completo (25 indicadores)": ['RSI', 'MACD', 'BBANDS', 'ATR', 'ADX',
                                                          'SMA', 'EMA', 'WMA', 'TEMA', 'KAMA',
                                                          'STOCH', 'CCI', 'MFI', 'WILLR', 'MOM',
                                                          'ROC', 'PPO', 'APO', 'TRIX', 'ULTOSC',
                                                          'OBV', 'AD', 'NATR', 'STDDEV', 'TSF'],
                "🔥 Trading Profesional (15 indicadores)": ['RSI', 'MACD', 'BBANDS', 'ATR', 'ADX',
                                                            'STOCH', 'CCI', 'MFI', 'OBV', 'SAR',
                                                            'EMA', 'WILLR', 'PPO', 'AROON'],
            }
            
            selected_indicators = presets.get(preset, [])
            st.info(f"📊 {len(selected_indicators)} indicadores seleccionados")
            
        elif selection_method == "📁 Por Categorías":
            selected_categories = st.multiselect(
                "Seleccionar Categorías",
                list(categories.keys()),
                default=["💫 Momentum"]
            )
            
            for category in selected_categories:
                selected_indicators.extend(categories[category])
            
            st.info(f"📊 {len(selected_indicators)} indicadores en {len(selected_categories)} categorías")
            
        else:
            all_indicators_flat = []
            for cat_name, cat_indicators in categories.items():
                for ind in cat_indicators:
                    all_indicators_flat.append(f"{ind} ({cat_name.split()[0]})")
            
            selected_with_category = st.multiselect(
                "Seleccionar Indicadores",
                all_indicators_flat,
                default=["RSI (💫)", "MACD (💫)", "ATR (📉)"]
            )
            
            selected_indicators = [ind.split(" (")[0] for ind in selected_with_category]
            st.info(f"📊 {len(selected_indicators)} indicadores seleccionados")
        
        if selected_indicators:
            periods_count = len(range(min_period, max_period + 1, step_period))
            indicators_with_period = sum(1 for ind in selected_indicators if TechnicalIndicators.needs_period(ind))
            indicators_without_period = len(selected_indicators) - indicators_with_period
            
            total_calculations = indicators_with_period * periods_count + indicators_without_period
            
            st.markdown(f"""
                <div style='padding: 0.8rem; margin-top: 1rem;
                            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
                            border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);'>
                    <p style='margin: 0; text-align: center; color: #10b981; font-weight: 600;'>
                        📊 {total_calculations:,} configuraciones
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button(
            "🚀 **EJECUTAR ANÁLISIS**",
            use_container_width=True,
            type="primary"
        )
    
    current_params = {
        'ticker': ticker,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date,
        'indicators': selected_indicators,
        'quantiles': quantiles,
        'return_days': return_days,
        'period_range': (min_period, max_period, step_period)
    }
    
    if analyze_button:
        if not selected_indicators:
            st.error("⚠️ Por favor seleccione al menos un indicador")
            return
        
        status_container = st.container()
        
        with status_container:
            with st.spinner('🔄 Procesando análisis cuantitativo...'):
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
                    st.session_state.analysis_params = current_params
                else:
                    st.session_state.analysis_done = False
    
    if st.session_state.analysis_done:
        returns_data = st.session_state.returns_data
        indicators = st.session_state.indicators
        data = st.session_state.data
        
        st.markdown(f"""
            <div style='padding: 1rem; margin: 2rem 0;
                        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%);
                        border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.4);
                        backdrop-filter: blur(10px);'>
                <p style='margin: 0; text-align: center; color: #10b981; font-size: 1.2rem; font-weight: 600;'>
                    ✅ Análisis Completado Exitosamente
                </p>
                <p style='margin: 0.5rem 0 0 0; text-align: center; color: #E0E5FF;'>
                    📊 {len(indicators.columns)} configuraciones | 
                    📈 {len(data)} días | 
                    🎯 {len(returns_data)} con percentiles
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Los resultados permanecen visibles mientras cambias parámetros. Usa 🚀 EJECUTAR ANÁLISIS para actualizar.")
        
        tab1, tab2, tab3 = st.tabs([
            "📈 **Análisis de Percentiles**",
            "🏆 **Top Performers**",
            "🔍 **Skew Analytics & Optimización**"
        ])
        
        with tab1:
            st.markdown("### 📈 Análisis Detallado de Percentiles")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                available_indicators = list(indicators.columns)
                if available_indicators:
                    selected_indicator = st.selectbox(
                        "**Indicador**",
                        available_indicators,
                        format_func=lambda x: x.replace('_', ' ')
                    )
            
            with col2:
                selected_return = st.selectbox(
                    "**Días de Retorno**",
                    list(range(1, return_days + 1)),
                    index=min(4, return_days - 1) if return_days >= 5 else 0
                )
            
            with col3:
                st.metric("Datos Disponibles", f"{len(indicators[selected_indicator].dropna()):,}")
            
            if selected_indicator:
                fig = create_percentile_plots(
                    indicators,
                    returns_data,
                    data,
                    selected_indicator,
                    selected_return
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                if selected_indicator in returns_data:
                    with st.expander("📊 **Tabla Detallada de Percentiles**", expanded=False):
                        df_display = returns_data[selected_indicator].copy()
                        df_display.index = [f'P{i+1}' for i in range(len(df_display))]
                        st.dataframe(
                            df_display.style.format("{:.3f}").background_gradient(
                                cmap='RdYlGn',
                                subset=[col for col in df_display.columns if 'mean' in col]
                            ),
                            use_container_width=True
                        )
        
        with tab2:
            st.markdown("### 🏆 Mejores Configuraciones por Performance")
            
            best_configs = []
            for ind_col in indicators.columns:
                if ind_col in returns_data:
                    for ret_day in range(1, min(return_days + 1, 6)):
                        ret_col = f'returns_{ret_day}_days_mean'
                        if ret_col in returns_data[ind_col].columns:
                            values = returns_data[ind_col][ret_col]
                            if len(values) > 1:
                                spread = values.iloc[-1] - values.iloc[0]
                                sharpe = values.mean() / values.std() if values.std() > 0 else 0
                                
                                best_configs.append({
                                    'Indicador': ind_col,
                                    'Días': ret_day,
                                    'P_Superior': values.iloc[-1],
                                    'P_Inferior': values.iloc[0],
                                    'Spread': spread,
                                    'Sharpe': sharpe,
                                    'Promedio': values.mean()
                                })
            
            if best_configs:
                best_df = pd.DataFrame(best_configs)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mejor Spread", f"{best_df['Spread'].max():.2f}%")
                with col2:
                    st.metric("Mejor Sharpe", f"{best_df['Sharpe'].max():.3f}")
                with col3:
                    st.metric("Promedio Spread", f"{best_df['Spread'].mean():.2f}%")
                with col4:
                    st.metric("Configuraciones", len(best_df))
                
                best_df = best_df.sort_values('Spread', ascending=False).head(30)
                
                st.dataframe(
                    best_df.style.format({
                        'P_Superior': '{:.2f}%',
                        'P_Inferior': '{:.2f}%',
                        'Spread': '{:.2f}%',
                        'Sharpe': '{:.3f}',
                        'Promedio': '{:.2f}%'
                    }).background_gradient(cmap='RdYlGn', subset=['Spread', 'Sharpe']),
                    use_container_width=True
                )
        
        with tab3:
            st.markdown("### 🔍 Análisis de Skew y Períodos Óptimos")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                indicator_for_skew = st.selectbox(
                    "Seleccionar Indicador",
                    [ind for ind in selected_indicators if TechnicalIndicators.needs_period(ind)],
                    key="skew_indicator"
                )
                
                st.markdown("### Configuración")
                min_p_skew = st.number_input("Período Mínimo", value=5, min_value=2, key="min_p_skew")
                max_p_skew = st.number_input("Período Máximo", value=100, min_value=10, key="max_p_skew")
                step_skew = st.number_input("Paso", value=2, min_value=1, key="step_skew")
                
                analyze_skew = st.button("🔬 Analizar Skew", use_container_width=True)
            
            with col2:
                if analyze_skew and indicator_for_skew:
                    with st.spinner(f'Analizando {indicator_for_skew}...'):
                        periods_df = find_optimal_periods(
                            indicator_for_skew, data,
                            min_period=min_p_skew, 
                            max_period=max_p_skew, 
                            step=step_skew,
                            return_days=return_days
                        )
                    
                    if not periods_df.empty:
                        best_period = periods_df.iloc[0]
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("🎯 Mejor Período", f"{best_period.name}")
                        with col2:
                            st.metric("📊 Score", f"{best_period['composite_score']:.3f}")
                        with col3:
                            st.metric("📈 Spread", f"{best_period['spread']:.2f}%")
                        with col4:
                            st.metric("🎯 Monotonicity", f"{best_period['monotonicity']:.3f}")
                        with col5:
                            st.metric("✅ Win Rate", f"{best_period['win_rate']*100:.1f}%")
                        
                        fig = create_skew_analysis_plot(periods_df, indicator_for_skew)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("### 🎯 Recomendaciones de Trading")
                        
                        if best_period['monotonicity'] > 0.5:
                            signal = "📈 **Momentum Following**: Comprar percentiles altos, vender bajos"
                        elif best_period['monotonicity'] < -0.5:
                            signal = "📉 **Mean Reversion**: Comprar percentiles bajos, vender altos"
                        else:
                            signal = "⚠️ **No Linear**: Relación no lineal, usar con precaución"
                        
                        st.info(f"""
                        **Configuración Óptima para {indicator_for_skew}:**
                        - Período: **{best_period.name}**
                        - Estrategia: {signal}
                        - Percentil Long: **P{int(best_period['best_percentile']+1)}** (Ret: {best_period['best_return']:.2f}%)
                        - Percentil Short: **P{int(best_period['worst_percentile']+1)}** (Ret: {best_period['worst_return']:.2f}%)
                        - Holding: **{return_days} días**
                        - p-value: **{best_period['monotonicity_pvalue']:.4f}**
                        """)
    
    else:
        if st.session_state.analysis_done:
            if st.button("🔄 Limpiar Análisis", key="clear_analysis"):
                st.session_state.analysis_done = False
                st.session_state.returns_data = None
                st.session_state.indicators = None
                st.session_state.data = None
                st.session_state.analysis_params = None
                st.rerun()
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0; color: #8892B0;'>
            <p style='margin: 0;'>
                Desarrollado con 💜 por 
                <a href='https://twitter.com/Gsnchez' style='color: #00D2FF;'><b>@Gsnchez</b></a> | 
                <a href='https://bquantfinance.com' style='color: #667eea;'><b>bquantfinance.com</b></a>
            </p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.8;'>
                Analizador Cuantitativo v3.0 | TALib & Streamlit
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
