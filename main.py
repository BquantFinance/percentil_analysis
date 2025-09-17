import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página con tema oscuro
st.set_page_config(
    page_title="Analizador de Percentiles - Indicadores Técnicos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para diseño ultra-estético
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #151932 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers con gradiente */
    h1 {
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #E0E5FF;
        font-weight: 600;
    }
    
    h3 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    /* Métricas estilizadas */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    /* Botones con gradiente */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox y inputs estilizados */
    .stSelectbox > div > div {
        background: rgba(30, 34, 56, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input {
        background: rgba(30, 34, 56, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        color: #E0E5FF;
    }
    
    /* Tabs con estilo moderno */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 34, 56, 0.5);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0 24px;
        background: transparent;
        border-radius: 8px;
        color: #8892B0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(99, 102, 241, 0.1);
        color: #E0E5FF;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: white !important;
    }
    
    /* Sidebar estilizada */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2238 0%, #252943 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%);
    }
    
    /* Expander con estilo */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        color: #E0E5FF;
    }
    
    /* Links estilizados */
    a {
        color: #00D2FF !important;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    a:hover {
        color: #3A7BD5 !important;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(185, 28, 28, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# Paleta de colores profesional
COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#00D2FF',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'dark': '#1e2238',
    'light': '#E0E5FF',
    'gradient_1': ['#667eea', '#764ba2'],
    'gradient_2': ['#00D2FF', '#3A7BD5'],
    'gradient_3': ['#f093fb', '#f5576c'],
    'gradient_4': ['#4facfe', '#00f2fe'],
    'gradient_5': ['#43e97b', '#38f9d7'],
    'gradient_6': ['#fa709a', '#fee140'],
    'gradient_7': ['#30cfd0', '#330867'],
    'gradient_8': ['#a8edea', '#fed6e3']
}

@st.cache_data
def calculate_selected_indicators_returns(ticker, start_date='2000-01-01', end_date=None, 
                                         indicators_to_calculate=None, quantiles=50, 
                                         return_days=5):
    """
    Calcula indicadores y retornos - implementación exacta del original
    Calcula indicadores para períodos 1 a 100
    """
    if end_date is None:
        end_date = datetime.now()
    
    # Descargar datos
    data = pd.DataFrame(yf.download(ticker, start=start_date, end=end_date, 
                                   progress=False, auto_adjust=True))
    
    if data.empty:
        return None, None, None
    
    indicators = pd.DataFrame(index=data.index)
    
    # Calcular retornos para los días especificados
    for i in range(1, return_days + 1):
        data[f'returns_{i}_days'] = data['Close'].pct_change(i) * 100
    
    data = data.dropna()
    
    # Convertir a arrays numpy para talib
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    volume = data['Volume'].values
    open_prices = data['Open'].values
    
    # Definir indicadores disponibles usando talib
    available_indicators = {
        'WILLR': lambda i: talib.WILLR(high, low, close, timeperiod=i),
        'CCI': lambda i: talib.CCI(high, low, close, timeperiod=i),
        'RSI': lambda i: talib.RSI(close, timeperiod=i),
        'ATR': lambda i: talib.ATR(high, low, close, timeperiod=i),
        'ADX': lambda i: talib.ADX(high, low, close, timeperiod=i),
        'STOCH_K': lambda i: talib.STOCH(high, low, close, fastk_period=i)[0],
        'STOCH_D': lambda i: talib.STOCH(high, low, close, fastk_period=i)[1],
        'DONCHIAN': lambda i: (talib.MAX(high, timeperiod=i) + talib.MIN(low, timeperiod=i)) / 2,
        'SLOPE': lambda i: talib.LINEARREG_SLOPE(close, timeperiod=i),
        'ZSCORE': lambda i: (close - talib.SMA(close, timeperiod=i)) / talib.STDDEV(close, timeperiod=i),
        'MACD': lambda i: talib.MACD(close, fastperiod=max(i//2, 2), slowperiod=i, signalperiod=9)[0],
        'EMA': lambda i: talib.EMA(close, timeperiod=i),
        'SMA': lambda i: talib.SMA(close, timeperiod=i),
        'KAMA': lambda i: talib.KAMA(close, timeperiod=i),
        'PLUS_DI': lambda i: talib.PLUS_DI(high, low, close, timeperiod=i),
        'MINUS_DI': lambda i: talib.MINUS_DI(high, low, close, timeperiod=i),
        'APO': lambda i: talib.APO(close, fastperiod=max(i//2, 2), slowperiod=i),
        'HMA': lambda i: talib.WMA(2 * talib.WMA(close, timeperiod=i//2) - talib.WMA(close, timeperiod=i), timeperiod=int(np.sqrt(i))),
        'MFI': lambda i: talib.MFI(high, low, close, volume, timeperiod=i),
        'OBV': lambda i: talib.OBV(close, volume),
        'NATR': lambda i: talib.NATR(high, low, close, timeperiod=i),
        'STDDEV': lambda i: talib.STDDEV(close, timeperiod=i),
        'TSF': lambda i: talib.TSF(close, timeperiod=i),
        'VAR': lambda i: talib.VAR(close, timeperiod=i),
        'LINEARREG': lambda i: talib.LINEARREG(close, timeperiod=i),
        'ROC': lambda i: talib.ROC(close, timeperiod=i),
        'ROCP': lambda i: talib.ROCP(close, timeperiod=i),
        'MOM': lambda i: talib.MOM(close, timeperiod=i),
        'BBANDS_UPPER': lambda i: talib.BBANDS(close, timeperiod=i)[0],
        'BBANDS_MIDDLE': lambda i: talib.BBANDS(close, timeperiod=i)[1],
        'BBANDS_LOWER': lambda i: talib.BBANDS(close, timeperiod=i)[2],
        'DEMA': lambda i: talib.DEMA(close, timeperiod=i),
        'TEMA': lambda i: talib.TEMA(close, timeperiod=i),
        'WMA': lambda i: talib.WMA(close, timeperiod=i),
        'CMO': lambda i: talib.CMO(close, timeperiod=i),
        'PPO': lambda i: talib.PPO(close, fastperiod=max(i//2, 2), slowperiod=i),
        'AROON_UP': lambda i: talib.AROON(high, low, timeperiod=i)[0],
        'AROON_DOWN': lambda i: talib.AROON(high, low, timeperiod=i)[1],
        'AROONOSC': lambda i: talib.AROONOSC(high, low, timeperiod=i),
        'DX': lambda i: talib.DX(high, low, close, timeperiod=i),
        'TRIX': lambda i: talib.TRIX(close, timeperiod=i),
        'T3': lambda i: talib.T3(close, timeperiod=i, vfactor=0),
    }
    
    if indicators_to_calculate is None:
        indicators_to_calculate = list(available_indicators.keys())
    
    # Calcular indicadores para períodos 1 a 100 con barra de progreso elegante
    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_calculations = len(indicators_to_calculate) * 100
    current_calculation = 0
    
    for indicator in indicators_to_calculate:
        if indicator in available_indicators:
            for i in range(1, 101, 1):
                try:
                    result = available_indicators[indicator](i)
                    if result is not None:
                        indicators[f'{indicator}{i}'] = pd.Series(result, index=data.index)
                except:
                    pass
                
                current_calculation += 1
                progress_bar.progress(current_calculation / total_calculations)
                progress_text.text(f'⚡ Procesando: {indicator} - Período {i}/100')
    
    progress_bar.empty()
    progress_text.empty()
    
    indicators = indicators.dropna()
    data_clean = data.drop(['Open', 'High', 'Low', 'Volume', 'Close'], axis=1, errors='ignore')
    
    # Calcular datos de retornos para cada indicador
    returns_data = {}
    for indicator_name in indicators.columns:
        deciles_column = f'{indicator_name}_Deciles'
        returns_data[indicator_name] = pd.DataFrame()
        
        try:
            data_clean[deciles_column] = pd.qcut(indicators[indicator_name], 
                                                 q=quantiles, precision=2, duplicates='drop')
            
            for i in range(1, return_days + 1):
                returns_df = data_clean.groupby(data_clean[deciles_column])[f'returns_{i}_days'].describe(
                    percentiles=[0.1, 0.2, 0.8, 0.9]
                )
                returns_data[indicator_name][f'returns_{i}_days_mean'] = returns_df['mean']
        except:
            continue
    
    return returns_data, indicators, data_clean

def plot_integrated_indicators_returns(indicators, returns_data, data, feature_name, 
                                      feature_length=None, return_days=None):
    """
    Crear los 4 gráficos con diseño ultra-estético usando Plotly
    """
    indicator_col = f'{feature_name}{feature_length}'
    
    if indicator_col not in indicators.columns:
        st.error(f"❌ Indicador {indicator_col} no encontrado")
        return None
    
    if indicator_col not in returns_data:
        st.error(f"❌ Datos de retorno para {indicator_col} no encontrados")
        return None
    
    # Crear subplots con diseño mejorado
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'<b>{feature_name}{feature_length} - Distribución</b>',
            f'<b>Retornos Promedio por Percentil ({return_days} días)</b>',
            f'<b>Correlación Móvil (ventana: 126 días)</b>',
            f'<b>Análisis de Dispersión - {feature_name}{feature_length}</b>'
        ),
        row_heights=[0.5, 0.5],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 1. HISTOGRAMA ESTÉTICO con gradiente
    hist_data = indicators[indicator_col].dropna()
    
    # Crear bins más estéticos
    counts, bins = np.histogram(hist_data, bins=70)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Normalizar colores para gradiente
    colors = [f'rgba({102 + i*2}, {126 - i}, {234 - i*2}, 0.8)' for i in range(len(counts))]
    
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            marker=dict(
                color=counts,
                colorscale='Viridis',
                line=dict(color='rgba(255,255,255,0.3)', width=0.5)
            ),
            name='Distribución',
            showlegend=False,
            hovertemplate='<b>Valor:</b> %{x:.2f}<br><b>Frecuencia:</b> %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Línea de media con estilo
    mean_val = hist_data.mean()
    std_val = hist_data.std()
    
    fig.add_vline(
        x=mean_val,
        line=dict(color='#FF6B6B', width=2, dash='dash'),
        row=1, col=1
    )
    
    # Añadir área de desviación estándar
    fig.add_vrect(
        x0=mean_val - std_val,
        x1=mean_val + std_val,
        fillcolor="rgba(255, 107, 107, 0.1)",
        layer="below",
        line_width=0,
        row=1, col=1
    )
    
    # Anotación elegante
    fig.add_annotation(
        x=mean_val,
        y=max(counts) * 1.05,
        text=f'<b>μ = {mean_val:.2f}</b><br>σ = {std_val:.2f}',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#FF6B6B",
        ax=30,
        ay=-30,
        bgcolor="rgba(30, 34, 56, 0.9)",
        bordercolor="#FF6B6B",
        borderwidth=1,
        font=dict(color='white', size=10),
        row=1, col=1
    )
    
    # 2. GRÁFICO DE BARRAS DE RETORNOS con gradiente
    returns_col = f'returns_{return_days}_days_mean'
    if returns_col in returns_data[indicator_col].columns:
        returns_values = returns_data[indicator_col][returns_col]
        x_labels = [f'Q{i+1}' for i in range(len(returns_values))]
        
        # Crear gradiente de colores basado en valores
        norm_returns = (returns_values - returns_values.min()) / (returns_values.max() - returns_values.min() + 0.0001)
        colors_bar = [f'rgba({int(30 + r*225)}, {int(100 - r*50)}, {int(100 + r*50)}, 0.8)' 
                     for r in norm_returns]
        
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=returns_values,
                marker=dict(
                    color=returns_values,
                    colorscale=[
                        [0, '#FF6B6B'],
                        [0.5, '#FFD93D'],
                        [1, '#6BCF7F']
                    ],
                    line=dict(color='rgba(255,255,255,0.3)', width=1),
                    colorbar=dict(
                        title="Retorno %",
                        titleside="right",
                        tickmode="linear",
                        tick0=returns_values.min(),
                        dtick=(returns_values.max() - returns_values.min()) / 5,
                        len=0.4,
                        y=0.75,
                        yanchor="middle"
                    )
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
        
        # Línea de tendencia
        x_numeric = list(range(len(returns_values)))
        z = np.polyfit(x_numeric, returns_values, 2)
        p = np.poly1d(z)
        trend_line = p(x_numeric)
        
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=trend_line,
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.5)', width=2, dash='dot'),
                name='Tendencia',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. CORRELACIÓN MÓVIL con área sombreada
    if f'returns_{return_days}_days' in data.columns:
        overall_corr = data[f'returns_{return_days}_days'].corr(indicators[indicator_col])
        rolling_corr = data[f'returns_{return_days}_days'].rolling(126).corr(indicators[indicator_col]).dropna()
        
        # Gráfico de área con gradiente
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode='lines',
                line=dict(color='#00D2FF', width=2),
                fill='tonexty',
                fillcolor='rgba(0, 210, 255, 0.1)',
                name='Correlación Móvil',
                showlegend=False,
                hovertemplate='<b>Fecha:</b> %{x}<br><b>Correlación:</b> %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Añadir banda de confianza
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=[0] * len(rolling_corr),
                mode='lines',
                line=dict(color='rgba(255,255,255,0.1)', width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Línea de correlación general
        fig.add_hline(
            y=overall_corr,
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            row=2, col=1
        )
        
        # Anotación estilizada
        fig.add_annotation(
            x=rolling_corr.index[-1],
            y=overall_corr,
            text=f'<b>ρ = {overall_corr:.3f}</b>',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FF6B6B",
            ax=-50,
            ay=0,
            bgcolor="rgba(30, 34, 56, 0.9)",
            bordercolor="#FF6B6B",
            borderwidth=1,
            font=dict(color='white', size=11),
            row=2, col=1
        )
    
    # 4. SCATTER PLOT con hexbin effect
    if f'returns_{return_days}_days' in data.columns:
        valid_dates = data.index[data.index >= indicators[indicator_col].index[0]]
        x_data = indicators[indicator_col]
        y_data = data[f'returns_{return_days}_days'].loc[valid_dates]
        
        mask = ~(x_data.isna() | y_data.isna())
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) > 1:
            # Scatter con gradiente de densidad
            fig.add_trace(
                go.Scatter(
                    x=x_clean,
                    y=y_clean,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=y_clean,
                        colorscale='Plasma',
                        opacity=0.6,
                        line=dict(width=0),
                        colorbar=dict(
                            title="Retorno %",
                            titleside="right",
                            tickmode="linear",
                            len=0.4,
                            y=0.25,
                            yanchor="middle"
                        )
                    ),
                    name='Datos',
                    showlegend=False,
                    hovertemplate='<b>Indicador:</b> %{x:.2f}<br><b>Retorno:</b> %{y:.2f}%<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Ajuste polinómico con intervalo de confianza
            try:
                poly_coefficients = np.polyfit(x_clean, y_clean, deg=2)
                poly_curve = np.polyval(poly_coefficients, x_clean)
                
                # Línea de ajuste principal
                fig.add_trace(
                    go.Scatter(
                        x=x_clean,
                        y=poly_curve,
                        mode='lines',
                        line=dict(color='#FFD93D', width=3),
                        name='Ajuste Cuadrático',
                        showlegend=True,
                        hovertemplate='<b>Ajuste</b><extra></extra>'
                    ),
                    row=2, col=2
                )
                
                # Intervalo de confianza
                residuals = y_clean - poly_curve
                std_residuals = np.std(residuals)
                
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_clean, x_clean[::-1]]),
                        y=np.concatenate([poly_curve + 1.96*std_residuals, 
                                        (poly_curve - 1.96*std_residuals)[::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 217, 61, 0.1)',
                        line=dict(color='rgba(255, 217, 61, 0)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=2, col=2
                )
            except:
                pass
    
    # Actualizar diseño general
    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=True,
        title={
            'text': f"<b>📊 Análisis de Percentiles: {feature_name}{feature_length}</b>",
            'font': {'size': 24, 'color': '#E0E5FF'},
            'x': 0.5,
            'xanchor': 'center'
        },
        hovermode='closest',
        plot_bgcolor='rgba(30, 34, 56, 0.3)',
        paper_bgcolor='rgba(14, 17, 39, 0.8)',
        font=dict(family="Inter, sans-serif", color='#E0E5FF'),
        legend=dict(
            bgcolor='rgba(30, 34, 56, 0.8)',
            bordercolor='rgba(99, 102, 241, 0.3)',
            borderwidth=1,
            font=dict(size=11)
        )
    )
    
    # Actualizar ejes con estilo
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(99, 102, 241, 0.1)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(99, 102, 241, 0.2)',
                row=row, col=col
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(99, 102, 241, 0.1)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(99, 102, 241, 0.2)',
                row=row, col=col
            )
    
    # Etiquetas específicas de ejes
    fig.update_xaxes(title_text="<b>Valores del Indicador</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Frecuencia</b>", row=1, col=1)
    
    fig.update_xaxes(title_text="<b>Percentiles</b>", tickangle=0, row=1, col=2)
    fig.update_yaxes(title_text=f"<b>Retorno Promedio ({return_days}d) %</b>", row=1, col=2)
    
    fig.update_xaxes(title_text="<b>Fecha</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Correlación</b>", row=2, col=1)
    
    fig.update_xaxes(title_text=f"<b>{feature_name}{feature_length}</b>", row=2, col=2)
    fig.update_yaxes(title_text=f"<b>Retornos ({return_days}d) %</b>", row=2, col=2)
    
    return fig

def main():
    # Header con gradiente
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 0;'>
            📊 Analizador de Performance por Percentiles
        </h1>
        <h3 style='text-align: center; margin-top: 0; margin-bottom: 2rem;'>
            Indicadores Técnicos con Análisis Cuantitativo
        </h3>
    """, unsafe_allow_html=True)
    
    # Créditos con estilo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%); border-radius: 15px; border: 1px solid rgba(99, 102, 241, 0.3); margin-bottom: 2rem;'>
                <p style='margin: 0; color: #E0E5FF; font-size: 0.9rem;'>Desarrollado por</p>
                <p style='margin: 0.5rem 0; font-size: 1.2rem;'>
                    <a href='https://twitter.com/Gsnchez' style='text-decoration: none;'>
                        <b>@Gsnchez</b>
                    </a> | 
                    <a href='https://bquantfinance.com' style='text-decoration: none;'>
                        <b>bquantfinance.com</b>
                    </a>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Configuración en la barra lateral
    with st.sidebar:
        st.markdown("""
            <div style='padding: 1rem; background: linear-gradient(135deg, rgba(0, 210, 255, 0.1) 0%, rgba(58, 123, 213, 0.1) 100%); border-radius: 10px; margin-bottom: 1rem;'>
                <h2 style='margin: 0; text-align: center;'>⚙️ Configuración</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Ticker con estilo
        ticker = st.text_input("🎯 **Símbolo Bursátil**", value="SPY", 
                              help="Ingrese un símbolo bursátil válido")
        
        # Fechas con contenedor estilizado
        st.markdown("### 📅 **Rango de Fechas**")
        use_default_dates = st.checkbox("Usar rango completo (2000-Hoy)", value=True)
        
        if use_default_dates:
            start_date = "2000-01-01"
            end_date = datetime.now()
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Inicio", value=datetime(2000, 1, 1))
            with col2:
                end_date = st.date_input("Fin", value=datetime.now())
            start_date = start_date.strftime('%Y-%m-%d')
        
        # Configuración de retornos
        st.markdown("### 📈 **Análisis de Retornos**")
        return_days = st.slider(
            "Días máximos de retorno",
            min_value=1, max_value=30, value=5,
            help="Calcular retornos hasta esta cantidad de días"
        )
        
        # Cuantiles
        quantiles = st.slider(
            "Número de percentiles",
            min_value=5, max_value=100, value=50,
            help="División en percentiles para el análisis"
        )
        
        # Selección de indicadores mejorada
        st.markdown("### 📊 **Indicadores Técnicos**")
        
        # Categorías con íconos
        momentum_indicators = ['RSI', 'WILLR', 'CCI', 'STOCH_K', 'STOCH_D', 'CMO', 'MOM', 'ROC', 'TRIX']
        trend_indicators = ['ADX', 'DX', 'PLUS_DI', 'MINUS_DI', 'AROON_UP', 'AROON_DOWN', 'MACD']
        volatility_indicators = ['ATR', 'NATR', 'STDDEV', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER']
        moving_averages = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'KAMA', 'T3', 'HMA']
        
        selection_method = st.radio(
            "Método de selección",
            ["⚡ Rápido", "📁 Por Categoría", "🎨 Personalizado"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if selection_method == "⚡ Rápido":
            quick_option = st.selectbox(
                "Configuración preestablecida",
                ["🔬 Paper Original (RSI)", "🌟 Top 5 Populares", "💫 Momentum", "📈 Tendencia"]
            )
            
            if "Paper Original" in quick_option:
                selected_indicators = ['RSI']
            elif "Top 5" in quick_option:
                selected_indicators = ['RSI', 'MACD', 'BBANDS_UPPER', 'ATR', 'ADX']
            elif "Momentum" in quick_option:
                selected_indicators = momentum_indicators[:5]
            else:
                selected_indicators = trend_indicators[:5]
                
        elif selection_method == "📁 Por Categoría":
            categories = st.multiselect(
                "Seleccionar categorías",
                ["💫 Momentum", "📈 Tendencia", "📊 Volatilidad", "📉 Medias Móviles"],
                default=["💫 Momentum"]
            )
            
            selected_indicators = []
            if "💫 Momentum" in categories:
                selected_indicators.extend(momentum_indicators)
            if "📈 Tendencia" in categories:
                selected_indicators.extend(trend_indicators)
            if "📊 Volatilidad" in categories:
                selected_indicators.extend(volatility_indicators)
            if "📉 Medias Móviles" in categories:
                selected_indicators.extend(moving_averages)
        else:
            all_indicators = momentum_indicators + trend_indicators + volatility_indicators + moving_averages
            selected_indicators = st.multiselect(
                "Seleccionar indicadores",
                options=all_indicators,
                default=['RSI']
            )
        
        # Información con estilo
        st.markdown(f"""
            <div style='padding: 0.75rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);'>
                <p style='margin: 0; color: #10b981; text-align: center;'>
                    <b>{len(selected_indicators)}</b> indicadores × <b>100</b> períodos = <b>{len(selected_indicators) * 100:,}</b> cálculos
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Botón de análisis con estilo
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🚀 **EJECUTAR ANÁLISIS**", use_container_width=True, type="primary")
    
    # Área principal
    if analyze_button:
        if not selected_indicators:
            st.error("⚠️ Por favor seleccione al menos un indicador")
            return
        
        # Spinner personalizado
        with st.spinner('🔄 Procesando análisis cuantitativo...'):
            returns_data, indicators, data = calculate_selected_indicators_returns(
                ticker, 
                start_date=start_date,
                end_date=end_date,
                indicators_to_calculate=selected_indicators,
                quantiles=quantiles,
                return_days=return_days
            )
        
        if returns_data is not None and indicators is not None and data is not None:
            # Success message con estilo
            st.markdown(f"""
                <div style='padding: 1rem; background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); border-radius: 10px; border: 1px solid rgba(16, 185, 129, 0.3); margin-bottom: 2rem;'>
                    <p style='margin: 0; color: #10b981; text-align: center; font-size: 1.1rem;'>
                        ✅ <b>Análisis completado:</b> {len(indicators.columns)} configuraciones procesadas
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Tabs mejorados
            tab1, tab2, tab3 = st.tabs([
                "📈 **Análisis de Percentiles**",
                "🏆 **Top Performers**",
                "📊 **Dashboard Estadístico**"
            ])
            
            with tab1:
                # Selección de indicador para análisis
                available_indicators = list(indicators.columns)
                
                if available_indicators:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        indicator_names = list(set([col.rstrip('0123456789') for col in available_indicators]))
                        indicator_names.sort()
                        selected_indicator = st.selectbox(
                            "**Indicador**",
                            options=indicator_names
                        )
                    
                    with col2:
                        available_periods = []
                        for col in available_indicators:
                            if col.startswith(selected_indicator):
                                try:
                                    period = int(col.replace(selected_indicator, ''))
                                    available_periods.append(period)
                                except:
                                    pass
                        available_periods.sort()
                        
                        if available_periods:
                            default_idx = available_periods.index(10) if 10 in available_periods else 0
                            selected_period = st.selectbox(
                                "**Período**",
                                options=available_periods,
                                index=default_idx
                            )
                        else:
                            selected_period = 10
                    
                    with col3:
                        selected_return_days = st.selectbox(
                            "**Días de Retorno**",
                            options=list(range(1, return_days + 1)),
                            index=min(4, return_days-1) if return_days >= 5 else 0
                        )
                    
                    # Generar gráficos estéticos
                    fig = plot_integrated_indicators_returns(
                        indicators, 
                        returns_data, 
                        data, 
                        selected_indicator,
                        feature_length=selected_period,
                        return_days=selected_return_days
                    )
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("### 🏆 **Mejores Configuraciones por Spread**")
                
                # Análisis de mejores performers
                best_configs = []
                for ind_col in indicators.columns[:300]:
                    if ind_col in returns_data:
                        for ret_day in range(1, min(return_days + 1, 6)):
                            ret_col = f'returns_{ret_day}_days_mean'
                            if ret_col in returns_data[ind_col].columns:
                                returns_df = returns_data[ind_col][ret_col]
                                if len(returns_df) > 1:
                                    spread = returns_df.iloc[-1] - returns_df.iloc[0]
                                    best_configs.append({
                                        'Indicador': ind_col,
                                        'Días': ret_day,
                                        'Q_Superior': returns_df.iloc[-1],
                                        'Q_Inferior': returns_df.iloc[0],
                                        'Spread': spread
                                    })
                
                if best_configs:
                    best_df = pd.DataFrame(best_configs).sort_values('Spread', ascending=False).head(20)
                    
                    # Gráfico de barras horizontal
                    fig_top = go.Figure()
                    fig_top.add_trace(go.Bar(
                        y=best_df['Indicador'],
                        x=best_df['Spread'],
                        orientation='h',
                        marker=dict(
                            color=best_df['Spread'],
                            colorscale='Viridis',
                            line=dict(color='rgba(255,255,255,0.3)', width=1)
                        ),
                        text=[f'{s:.2f}%' for s in best_df['Spread']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Spread: %{x:.3f}%<extra></extra>'
                    ))
                    
                    fig_top.update_layout(
                        template="plotly_dark",
                        height=600,
                        title="<b>Top 20 Indicadores por Spread de Retorno</b>",
                        xaxis_title="<b>Spread (%)</b>",
                        yaxis_title="",
                        plot_bgcolor='rgba(30, 34, 56, 0.3)',
                        paper_bgcolor='rgba(14, 17, 39, 0.8)',
                        font=dict(family="Inter, sans-serif", color='#E0E5FF')
                    )
                    
                    st.plotly_chart(fig_top, use_container_width=True)
            
            with tab3:
                st.markdown("### 📊 **Resumen Estadístico**")
                
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📅 Total Días", f"{len(data):,}")
                with col2:
                    st.metric("📈 Indicadores", f"{len(selected_indicators)}")
                with col3:
                    st.metric("⚙️ Configuraciones", f"{len(indicators.columns):,}")
                with col4:
                    if f'returns_{return_days}_days' in data.columns:
                        avg_ret = data[f'returns_{return_days}_days'].mean()
                        st.metric(f"📊 μ Retorno {return_days}d", f"{avg_ret:.2f}%")
                
                # Distribución de retornos
                st.markdown("### 📉 **Distribución de Retornos**")
                
                fig_dist = go.Figure()
                colors_dist = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
                
                for idx, i in enumerate([1, 5, 10, 20][:min(4, return_days)]):
                    if f'returns_{i}_days' in data.columns:
                        returns_series = data[f'returns_{i}_days'].dropna()
                        
                        fig_dist.add_trace(go.Histogram(
                            x=returns_series,
                            name=f'{i} días',
                            opacity=0.7,
                            nbinsx=50,
                            marker_color=colors_dist[idx % len(colors_dist)],
                            hovertemplate=f'<b>{i} días</b><br>Retorno: %{{x:.2f}}%<br>Frecuencia: %{{y}}<extra></extra>'
                        ))
                
                fig_dist.update_layout(
                    template="plotly_dark",
                    title="<b>Distribución de Retornos por Período</b>",
                    xaxis_title="<b>Retorno (%)</b>",
                    yaxis_title="<b>Frecuencia</b>",
                    barmode='overlay',
                    height=400,
                    plot_bgcolor='rgba(30, 34, 56, 0.3)',
                    paper_bgcolor='rgba(14, 17, 39, 0.8)',
                    font=dict(family="Inter, sans-serif", color='#E0E5FF'),
                    showlegend=True,
                    legend=dict(
                        bgcolor='rgba(30, 34, 56, 0.8)',
                        bordercolor='rgba(99, 102, 241, 0.3)',
                        borderwidth=1
                    )
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # Footer estilizado
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0; color: #8892B0;'>
            <p>Desarrollado con 💜 por <a href='https://twitter.com/Gsnchez'><b>@Gsnchez</b></a> | <a href='https://bquantfinance.com'><b>bquantfinance.com</b></a></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
