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
    page_title="Analizador Cuantitativo Completo",
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
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 34, 56, 0.6);
        border-radius: 16px;
        padding: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== CLASE DE INDICADORES TÉCNICOS =====================
class TechnicalIndicators:
    """Manejador compacto de todos los indicadores TALib"""
    
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
    
    CATEGORIES = {
        "📈 Overlaps": ['BBANDS', 'DEMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'MA', 
                       'MAMA', 'MIDPOINT', 'MIDPRICE', 'SAR', 
                       'SMA', 'T3', 'TEMA', 'TRIMA', 'WMA'],
        "💫 Momentum": ['ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 'BOP', 
                       'CCI', 'CMO', 'DX', 'MACD', 'MACDEXT', 'MACDFIX', 
                       'MFI', 'MINUS_DI', 'MINUS_DM', 'MOM', 'PLUS_DI', 
                       'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 
                       'RSI', 'STOCH', 'STOCHF', 'STOCHRSI', 'TRIX', 
                       'ULTOSC', 'WILLR'],
        "📊 Volumen": ['AD', 'ADOSC', 'OBV'],
        "📉 Volatilidad": ['ATR', 'NATR', 'TRANGE'],
        "📐 Estadísticas": ['BETA', 'CORREL', 'LINEARREG', 'LINEARREG_ANGLE', 
                           'LINEARREG_INTERCEPT', 'LINEARREG_SLOPE', 'STDDEV', 'TSF', 'VAR'],
    }
    
    @classmethod
    def _get_indicator_inputs(cls, func_name):
        """Detecta qué inputs necesita una función"""
        if func_name in ['AD', 'ADOSC']:
            return 'hlcv'
        elif func_name in ['OBV']:
            return 'cv'
        elif func_name in ['ATR', 'NATR', 'ADX', 'ADXR', 'CCI', 'DX', 'MINUS_DI', 'MINUS_DM', 'PLUS_DI', 'PLUS_DM']:
            return 'hlc'
        elif func_name == 'MFI':
            return 'hlcv'
        elif func_name in ['BOP']:
            return 'ohlc'
        elif func_name in ['SAR']:
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
        return cls.CATEGORIES
    
    @classmethod
    def needs_period(cls, indicator_name):
        """Determina si un indicador necesita período"""
        no_period = ['HT_TRENDLINE', 'BOP', 'MACDFIX', 'AD', 'OBV', 'TRANGE', 'SAR', 'MAMA']
        return indicator_name not in no_period

# ===================== FUNCIONES DE CÁLCULO ORIGINALES =====================
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
    """Crea los 4 gráficos de análisis de percentiles"""
    
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
    
    fig.add_vline(x=mean_val, line=dict(color='#FF6B6B', width=2, dash='dash'),
                  row=1, col=1, annotation_text=f'μ={mean_val:.2f}', annotation_position="top")
    fig.add_vline(x=median_val, line=dict(color='#4ECDC4', width=2, dash='dot'),
                  row=1, col=1, annotation_text=f'M={median_val:.2f}', annotation_position="bottom")
    
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
                            colorscale='Spectral',
                            opacity=0.4,
                            line=dict(width=0),
                            showscale=True,
                            colorbar=dict(
                                title="Ret %",
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
    
    return fig

# ===================== FUNCIONES DE OPTIMAL PERIOD & RULES =====================
def analyze_indicator_period(indicator_values, returns, quantiles=10):
    """Análisis simple y enfocado de un indicador en un período específico"""
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
        
        # 1. SPREAD - Métrica más importante
        top_return = percentile_returns['mean'].iloc[-1]
        bottom_return = percentile_returns['mean'].iloc[0]
        metrics['spread'] = top_return - bottom_return
        metrics['top_return'] = top_return
        metrics['bottom_return'] = bottom_return
        
        # 2. DIRECTION - Mean reversion vs momentum
        correlation, p_value = spearmanr(range(len(percentile_returns)), percentile_returns['mean'].values)
        metrics['direction'] = correlation
        metrics['p_value'] = p_value
        
        # 3. SHARPE - Consistencia
        metrics['sharpe'] = abs(metrics['spread']) / (percentile_returns['std'].mean() + 1e-8)
        
        # 4. BEST PERCENTILES
        metrics['best_long_percentile'] = percentile_returns['mean'].idxmax() + 1
        metrics['best_short_percentile'] = percentile_returns['mean'].idxmin() + 1
        
        # 5. SAMPLE SIZE
        metrics['min_samples'] = percentile_returns['count'].min()
        metrics['total_samples'] = percentile_returns['count'].sum()
        
        return metrics
        
    except Exception:
        return None

def find_optimal_period_simple(indicator_name, data, periods_to_test, return_days=5):
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
    
    # Score compuesto
    df_results['score'] = (
        abs(df_results['spread']) * 0.5 +
        df_results['sharpe'] * 10 +
        (1 / (df_results['p_value'] + 0.001)) * 0.1
    )
    
    return df_results.sort_values('score', ascending=False)

def batch_optimize_indicators(indicator_list, data, return_days=5, quick_mode=True):
    """Optimización batch para múltiples indicadores"""
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, indicator in enumerate(indicator_list):
        status_text.text(f"Optimizando {indicator}... ({i+1}/{len(indicator_list)})")
        
        if quick_mode:
            periods = list(range(5, 51, 5)) + [60, 75, 100]
        else:
            periods = list(range(5, 101, 2))
        
        if not TechnicalIndicators.needs_period(indicator):
            continue
        
        df_results = find_optimal_period_simple(indicator, data, periods, return_days)
        
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

def extract_trading_rules(analysis_df, min_spread=2.0, max_p_value=0.1, top_n=5):
    """Extrae reglas de trading claras y accionables"""
    if analysis_df.empty:
        return []
    
    quality_signals = analysis_df[
        (abs(analysis_df['spread']) >= min_spread) & 
        (analysis_df['p_value'] <= max_p_value)
    ].head(top_n)
    
    rules = []
    
    for _, row in quality_signals.iterrows():
        if row['direction'] > 0.3:
            strategy_type = "MOMENTUM"
            entry_condition = f"BUY when {row['indicator']}_{int(row['period'])} is in TOP 20% (P >= 16)"
            exit_condition = f"SELL when {row['indicator']}_{int(row['period'])} is in BOTTOM 20% (P <= 4)"
        elif row['direction'] < -0.3:
            strategy_type = "MEAN REVERSION"
            entry_condition = f"BUY when {row['indicator']}_{int(row['period'])} is in BOTTOM 20% (P <= 4)"
            exit_condition = f"SELL when {row['indicator']}_{int(row['period'])} is in TOP 20% (P >= 16)"
        else:
            strategy_type = "COMPLEX"
            entry_condition = f"BUY at Percentile {int(row['best_long_percentile'])}"
            exit_condition = f"AVOID Percentile {int(row['best_short_percentile'])}"
        
        rule = {
            'indicator': row['indicator'],
            'period': int(row['period']),
            'strategy_type': strategy_type,
            'entry_condition': entry_condition,
            'exit_condition': exit_condition,
            'expected_spread': f"{row['spread']:.2f}%",
            'direction_score': row['direction'],
            'confidence': f"{(1-row['p_value'])*100:.1f}%",
            'sharpe': row['sharpe'],
        }
        
        rules.append(rule)
    
    return rules

def create_optimal_period_visualization(results_df):
    """Visualización clara de períodos óptimos y reglas"""
    if results_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '<b>📊 Period vs Spread</b>',
            '<b>🎯 Direction Analysis</b>',
            '<b>📈 Top 10 Configurations</b>',
            '<b>💡 Score Distribution</b>'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "box"}]
        ]
    )
    
    # 1. Period vs Spread
    for ind in results_df['indicator'].unique()[:10]:
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
    
    fig.add_hline(y=2.0, line=dict(color='red', dash='dash'), 
                  annotation_text="Min 2%", row=1, col=1)
    fig.add_hline(y=-2.0, line=dict(color='red', dash='dash'), row=1, col=1)
    
    # 2. Direction scatter
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
            hovertemplate='%{text}<br>Direction: %{x:.3f}<br>Spread: %{y:.2f}%'
        ),
        row=1, col=2
    )
    
    fig.add_vline(x=-0.3, line=dict(color='blue', dash='dot'), row=1, col=2)
    fig.add_vline(x=0.3, line=dict(color='orange', dash='dot'), row=1, col=2)
    
    # 3. Top configurations
    top_10 = results_df.nlargest(10, 'score')
    
    fig.add_trace(
        go.Bar(
            x=top_10['indicator'] + '_P' + top_10['period'].astype(str),
            y=top_10['spread'],
            marker=dict(color=top_10['score'], colorscale='RdYlGn'),
            text=[f"{s:.1f}%" for s in top_10['spread']],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 4. Score distribution by indicator
    for ind in results_df['indicator'].unique()[:10]:
        ind_data = results_df[results_df['indicator'] == ind]
        fig.add_trace(
            go.Box(y=ind_data['score'], name=ind, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=True,
        title_text="<b>Optimal Period Analysis</b>",
        title_font_size=20
    )
    
    return fig

# ===================== INTERFAZ PRINCIPAL =====================
def main():
    st.markdown("""
        <h1 style='text-align: center;'>
            📊 Analizador Cuantitativo Completo
        </h1>
        <p style='text-align: center; color: #8892B0; font-size: 1.2rem; margin-bottom: 2rem;'>
            Análisis de Percentiles + Optimal Period Discovery + Rule Extraction
        </p>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
        st.session_state.returns_data = None
        st.session_state.indicators = None
        st.session_state.data = None
        st.session_state.optimal_results = None
    
    # Ensure optimal_results exists (for backwards compatibility)
    if 'optimal_results' not in st.session_state:
        st.session_state.optimal_results = None
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        
        with st.expander("📈 **DATOS DEL MERCADO**", expanded=True):
            ticker = st.text_input("Símbolo", value="SPY")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Inicio",
                    value=datetime(2020, 1, 1)
                )
            with col2:
                end_date = st.date_input(
                    "Fin",
                    value=datetime.now()
                )
        
        with st.expander("📊 **PARÁMETROS**", expanded=True):
            return_days = st.select_slider(
                "Días de Retorno",
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
        
        st.markdown("### 📐 Indicadores")
        
        preset = st.selectbox(
            "Preset",
            ["📊 Esenciales (5)", "💫 Momentum (10)", "📈 Completo (20)", "🚀 Todo (40+)"]
        )
        
        if preset == "📊 Esenciales (5)":
            selected_indicators = ['RSI', 'MACD', 'CCI', 'ROC', 'ATR']
        elif preset == "💫 Momentum (10)":
            selected_indicators = ['RSI', 'MACD', 'STOCH', 'CCI', 'MFI', 
                                 'WILLR', 'MOM', 'ROC', 'ADX', 'PPO']
        elif preset == "📈 Completo (20)":
            selected_indicators = ['RSI', 'MACD', 'BBANDS', 'ATR', 'ADX',
                                 'SMA', 'EMA', 'STOCH', 'CCI', 'MFI',
                                 'WILLR', 'MOM', 'ROC', 'PPO', 'CMO',
                                 'AROON', 'ULTOSC', 'OBV', 'AD', 'NATR']
        else:
            selected_indicators = list(TechnicalIndicators.INDICATOR_CONFIG.keys())
        
        st.info(f"📊 {len(selected_indicators)} indicadores")
        
        analyze_button = st.button(
            "🚀 **EJECUTAR ANÁLISIS**",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button:
        if not selected_indicators:
            st.error("Seleccione indicadores")
            return
        
        with st.spinner('🔄 Procesando...'):
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
                
                # Optimal period analysis
                with st.spinner('🔍 Buscando períodos óptimos...'):
                    st.session_state.optimal_results = batch_optimize_indicators(
                        selected_indicators, data, return_days, quick_mode=True
                    )
    
    if st.session_state.analysis_done:
        returns_data = st.session_state.returns_data
        indicators = st.session_state.indicators
        data = st.session_state.data
        optimal_results = st.session_state.optimal_results
        
        st.success(f"✅ Análisis completado: {len(indicators.columns)} configuraciones")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 **Análisis Percentiles**",
            "🏆 **Top Performers**",
            "🎯 **Optimal Periods**",
            "📋 **Trading Rules**"
        ])
        
        with tab1:
            st.markdown("### 📈 Análisis de Percentiles")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                available = list(indicators.columns)
                if available:
                    selected_ind = st.selectbox("Indicador", available)
            
            with col2:
                sel_return = st.selectbox(
                    "Días Retorno",
                    list(range(1, return_days + 1)),
                    index=min(4, return_days - 1) if return_days >= 5 else 0
                )
            
            with col3:
                if 'selected_ind' in locals():
                    st.metric("Datos", f"{len(indicators[selected_ind].dropna()):,}")
            
            if 'selected_ind' in locals() and selected_ind:
                fig = create_percentile_plots(
                    indicators, returns_data, data,
                    selected_ind, sel_return
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🏆 Top Performers")
            
            best_configs = []
            for ind_col in indicators.columns:
                if ind_col in returns_data:
                    ret_col = f'returns_{return_days}_days_mean'
                    if ret_col in returns_data[ind_col].columns:
                        values = returns_data[ind_col][ret_col]
                        if len(values) > 1:
                            spread = values.iloc[-1] - values.iloc[0]
                            sharpe = values.mean() / values.std() if values.std() > 0 else 0
                            
                            best_configs.append({
                                'Indicador': ind_col,
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
                    st.metric("Mejor Spread", f"{best_df['Spread'].max():.2f}%")
                with col2:
                    st.metric("Mejor Sharpe", f"{best_df['Sharpe'].max():.3f}")
                with col3:
                    st.metric("Total Configs", len(best_df))
                
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
                col1, col2, col3, col4 = st.columns(4)
                
                best = optimal_results.iloc[0]
                significant = optimal_results[optimal_results['p_value'] <= 0.1]
                tradeable = optimal_results[abs(optimal_results['spread']) >= 2.0]
                
                with col1:
                    st.metric(
                        "Best Config",
                        f"{best['indicator']}_P{int(best['period'])}",
                        f"Spread: {best['spread']:.2f}%"
                    )
                with col2:
                    st.metric("Significant", len(significant), "p < 0.10")
                with col3:
                    st.metric("Tradeable", len(tradeable), "Spread > 2%")
                with col4:
                    momentum = len(optimal_results[optimal_results['direction'] > 0.3])
                    meanrev = len(optimal_results[optimal_results['direction'] < -0.3])
                    st.metric("Strategy Mix", f"{momentum}M/{meanrev}R")
                
                fig = create_optimal_period_visualization(optimal_results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📊 Detailed Results"):
                    display_df = optimal_results[['indicator', 'period', 'spread', 'direction', 
                                                 'p_value', 'sharpe', 'score']].head(20)
                    
                    display_df['strategy'] = display_df['direction'].apply(
                        lambda x: 'MOMENTUM' if x > 0.3 else ('MEAN REV' if x < -0.3 else 'COMPLEX')
                    )
                    
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
            else:
                st.info("Run analysis to discover optimal periods for indicators")
        
        with tab4:
            st.markdown("### 📋 Extracted Trading Rules")
            
            if not optimal_results.empty:
                rules = extract_trading_rules(optimal_results, min_spread=2.0, max_p_value=0.1, top_n=10)
                
                if rules:
                    for i, rule in enumerate(rules[:5], 1):
                        color = "🟠" if rule['strategy_type'] == "MOMENTUM" else "🔵" if rule['strategy_type'] == "MEAN REVERSION" else "⚪"
                        
                        st.markdown(f"""
                        #### {color} Rule #{i}: {rule['indicator']} (Period {rule['period']})
                        - **Type:** {rule['strategy_type']}
                        - **Entry:** {rule['entry_condition']}
                        - **Exit:** {rule['exit_condition']}
                        - **Expected Spread:** {rule['expected_spread']}
                        - **Confidence:** {rule['confidence']}
                        - **Direction Score:** {rule['direction_score']:.3f}
                        """)
                    
                    # Export button
                    if st.button("📥 Export Rules as CSV"):
                        rules_df = pd.DataFrame(rules)
                        csv = rules_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"{ticker}_rules_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("No significant rules found. Try adjusting parameters.")
    
    else:
        if st.session_state.get('analysis_done'):
            if st.button("🔄 Clear Analysis"):
                for key in ['analysis_done', 'returns_data', 'indicators', 'data', 'optimal_results']:
                    st.session_state[key] = False if key == 'analysis_done' else None
                st.rerun()

if __name__ == "__main__":
    main()
