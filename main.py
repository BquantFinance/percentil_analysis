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
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def calculate_selected_indicators_returns(ticker, start_date='2000-01-01', end_date=None, 
                                         indicators_to_calculate=None, quantiles=50, 
                                         return_days=5):
    """
    Calcula indicadores y retornos - implementación corregida
    """
    try:
        if end_date is None:
            end_date = datetime.now()
        
        # Descargar datos
        st.info(f"📥 Descargando datos para {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty:
            st.error(f"❌ No se encontraron datos para {ticker}")
            return None, None, None
        
        # Asegurarse de que es DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        st.success(f"✅ Datos descargados: {len(data)} días")
        
        # Calcular retornos
        for i in range(1, return_days + 1):
            data[f'returns_{i}_days'] = data['Close'].pct_change(i) * 100
        
        # Eliminar las primeras filas con NaN en retornos
        data = data.iloc[return_days:]
        
        # Preparar arrays para indicadores
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        volume = data['Volume'].values if 'Volume' in data.columns else None
        open_prices = data['Open'].values
        
        # DataFrame para almacenar indicadores
        indicators = pd.DataFrame(index=data.index)
        
        # Definir indicadores disponibles
        available_indicators = {
            'RSI': lambda i: talib.RSI(close, timeperiod=i) if i >= 2 else None,
            'WILLR': lambda i: talib.WILLR(high, low, close, timeperiod=i) if i >= 2 else None,
            'CCI': lambda i: talib.CCI(high, low, close, timeperiod=i) if i >= 2 else None,
            'ATR': lambda i: talib.ATR(high, low, close, timeperiod=i) if i >= 2 else None,
            'ADX': lambda i: talib.ADX(high, low, close, timeperiod=i) if i >= 2 else None,
            'STOCH_K': lambda i: talib.STOCH(high, low, close, fastk_period=i)[0] if i >= 2 else None,
            'STOCH_D': lambda i: talib.STOCH(high, low, close, fastk_period=i)[1] if i >= 2 else None,
            'SMA': lambda i: talib.SMA(close, timeperiod=i) if i >= 2 else None,
            'EMA': lambda i: talib.EMA(close, timeperiod=i) if i >= 2 else None,
            'WMA': lambda i: talib.WMA(close, timeperiod=i) if i >= 2 else None,
            'DEMA': lambda i: talib.DEMA(close, timeperiod=i) if i >= 2 else None,
            'TEMA': lambda i: talib.TEMA(close, timeperiod=i) if i >= 2 else None,
            'KAMA': lambda i: talib.KAMA(close, timeperiod=i) if i >= 2 else None,
            'MACD': lambda i: talib.MACD(close, fastperiod=max(i//2, 2), slowperiod=i, signalperiod=9)[0] if i >= 4 else None,
            'PPO': lambda i: talib.PPO(close, fastperiod=max(i//2, 2), slowperiod=i) if i >= 4 else None,
            'APO': lambda i: talib.APO(close, fastperiod=max(i//2, 2), slowperiod=i) if i >= 4 else None,
            'MOM': lambda i: talib.MOM(close, timeperiod=i) if i >= 2 else None,
            'ROC': lambda i: talib.ROC(close, timeperiod=i) if i >= 2 else None,
            'ROCP': lambda i: talib.ROCP(close, timeperiod=i) if i >= 2 else None,
            'AROONOSC': lambda i: talib.AROONOSC(high, low, timeperiod=i) if i >= 2 else None,
            'MFI': lambda i: talib.MFI(high, low, close, volume, timeperiod=i) if i >= 2 and volume is not None else None,
            'BBANDS_UPPER': lambda i: talib.BBANDS(close, timeperiod=i)[0] if i >= 2 else None,
            'BBANDS_MIDDLE': lambda i: talib.BBANDS(close, timeperiod=i)[1] if i >= 2 else None,
            'BBANDS_LOWER': lambda i: talib.BBANDS(close, timeperiod=i)[2] if i >= 2 else None,
            'STDDEV': lambda i: talib.STDDEV(close, timeperiod=i) if i >= 2 else None,
            'VAR': lambda i: talib.VAR(close, timeperiod=i) if i >= 2 else None,
            'NATR': lambda i: talib.NATR(high, low, close, timeperiod=i) if i >= 2 else None,
            'CMO': lambda i: talib.CMO(close, timeperiod=i) if i >= 2 else None,
            'DX': lambda i: talib.DX(high, low, close, timeperiod=i) if i >= 2 else None,
            'PLUS_DI': lambda i: talib.PLUS_DI(high, low, close, timeperiod=i) if i >= 2 else None,
            'MINUS_DI': lambda i: talib.MINUS_DI(high, low, close, timeperiod=i) if i >= 2 else None,
            'TRIX': lambda i: talib.TRIX(close, timeperiod=i) if i >= 2 else None,
            'ULTOSC': lambda i: talib.ULTOSC(high, low, close, timeperiod1=max(i//3, 2), timeperiod2=max(i//2, 3), timeperiod3=i) if i >= 7 else None,
            'TSF': lambda i: talib.TSF(close, timeperiod=i) if i >= 2 else None,
            'T3': lambda i: talib.T3(close, timeperiod=i, vfactor=0) if i >= 2 else None,
            'HMA': lambda i: talib.WMA(2 * talib.WMA(close, timeperiod=max(i//2, 2)) - talib.WMA(close, timeperiod=i), timeperiod=int(np.sqrt(i))) if i >= 4 else None,
        }
        
        if indicators_to_calculate is None:
            indicators_to_calculate = list(available_indicators.keys())
        
        # Filtrar solo indicadores disponibles
        indicators_to_calculate = [ind for ind in indicators_to_calculate if ind in available_indicators]
        
        if not indicators_to_calculate:
            st.error("❌ No hay indicadores válidos para calcular")
            return None, None, None
        
        # Calcular indicadores con progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        total_calculations = len(indicators_to_calculate) * 100
        successful_calculations = 0
        current_calculation = 0
        
        for indicator in indicators_to_calculate:
            indicator_func = available_indicators[indicator]
            successful_periods = 0
            
            for period in range(1, 101):
                try:
                    result = indicator_func(period)
                    if result is not None and not np.all(np.isnan(result)):
                        indicators[f'{indicator}{period}'] = pd.Series(result, index=data.index)
                        successful_periods += 1
                        successful_calculations += 1
                except Exception as e:
                    pass
                
                current_calculation += 1
                progress_bar.progress(min(current_calculation / total_calculations, 1.0))
                progress_text.text(f'⚡ Calculando: {indicator} - Período {period}/100 - Exitosos: {successful_calculations}')
        
        progress_bar.empty()
        progress_text.empty()
        
        # Verificar que hay indicadores calculados
        if indicators.empty or len(indicators.columns) == 0:
            st.error("❌ No se pudo calcular ningún indicador")
            return None, None, None
        
        st.info(f"📊 Indicadores calculados: {len(indicators.columns)} configuraciones")
        
        # Eliminar columnas con todos NaN
        indicators = indicators.dropna(axis=1, how='all')
        
        # Calcular percentiles para cada indicador
        returns_data = {}
        successful_percentiles = 0
        
        for indicator_name in indicators.columns:
            try:
                # Crear DataFrame temporal con indicador y retornos
                temp_df = pd.DataFrame({
                    'indicator': indicators[indicator_name],
                })
                
                # Agregar columnas de retornos
                for i in range(1, return_days + 1):
                    ret_col = f'returns_{i}_days'
                    if ret_col in data.columns:
                        temp_df[ret_col] = data[ret_col]
                
                # Eliminar filas con NaN
                temp_df = temp_df.dropna()
                
                if len(temp_df) < quantiles:
                    continue
                
                # Crear cuantiles
                temp_df['quantile'] = pd.qcut(temp_df['indicator'], q=quantiles, duplicates='drop')
                
                # Calcular estadísticas por cuantil
                returns_data[indicator_name] = pd.DataFrame()
                
                for i in range(1, return_days + 1):
                    ret_col = f'returns_{i}_days'
                    if ret_col in temp_df.columns:
                        grouped = temp_df.groupby('quantile')[ret_col].agg(['mean', 'std', 'count'])
                        returns_data[indicator_name][f'returns_{i}_days_mean'] = grouped['mean']
                        
                successful_percentiles += 1
                
            except Exception as e:
                continue
        
        st.success(f"✅ Análisis de percentiles completado: {successful_percentiles} configuraciones")
        
        # Retornar los datos limpios
        data_with_indicators = data.copy()
        
        return returns_data, indicators, data_with_indicators
        
    except Exception as e:
        st.error(f"❌ Error en el cálculo: {str(e)}")
        return None, None, None

def plot_integrated_indicators_returns(indicators, returns_data, data, feature_name, 
                                      feature_length=None, return_days=None):
    """
    Crear los 4 gráficos con diseño ultra-estético
    """
    indicator_col = f'{feature_name}{feature_length}'
    
    if indicator_col not in indicators.columns:
        st.error(f"❌ Indicador {indicator_col} no encontrado")
        return None
    
    if indicator_col not in returns_data:
        st.warning(f"⚠️ No hay datos de percentiles para {indicator_col}")
        return None
    
    # Crear subplots
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
        vertical_spacing=0.15
    )
    
    # 1. HISTOGRAMA
    hist_data = indicators[indicator_col].dropna()
    
    if len(hist_data) > 0:
        fig.add_trace(
            go.Histogram(
                x=hist_data,
                nbinsx=70,
                marker=dict(
                    color='rgba(102, 126, 234, 0.7)',
                    line=dict(color='rgba(255,255,255,0.3)', width=0.5)
                ),
                name='Distribución',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Línea de media
        mean_val = hist_data.mean()
        fig.add_vline(
            x=mean_val,
            line=dict(color='#FF6B6B', width=2, dash='dash'),
            row=1, col=1,
            annotation_text=f'Media: {mean_val:.2f}',
            annotation_position="top"
        )
    
    # 2. RETORNOS POR PERCENTIL
    returns_col = f'returns_{return_days}_days_mean'
    if returns_col in returns_data[indicator_col].columns:
        returns_values = returns_data[indicator_col][returns_col]
        x_labels = [f'P{i+1}' for i in range(len(returns_values))]
        
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
                name='Retornos',
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. CORRELACIÓN MÓVIL
    if f'returns_{return_days}_days' in data.columns:
        # Alinear datos
        common_index = data.index.intersection(indicators[indicator_col].index)
        
        if len(common_index) > 126:
            aligned_returns = data.loc[common_index, f'returns_{return_days}_days']
            aligned_indicator = indicators.loc[common_index, indicator_col]
            
            rolling_corr = aligned_returns.rolling(126).corr(aligned_indicator).dropna()
            overall_corr = aligned_returns.corr(aligned_indicator)
            
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    line=dict(color='#00D2FF', width=2),
                    name='Correlación Móvil',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.add_hline(
                y=overall_corr,
                line=dict(color='#FF6B6B', width=2, dash='dash'),
                row=2, col=1,
                annotation_text=f'Correlación: {overall_corr:.3f}',
                annotation_position="right"
            )
    
    # 4. SCATTER PLOT
    if f'returns_{return_days}_days' in data.columns:
        common_index = data.index.intersection(indicators[indicator_col].index)
        
        if len(common_index) > 0:
            x_data = indicators.loc[common_index, indicator_col]
            y_data = data.loc[common_index, f'returns_{return_days}_days']
            
            mask = ~(x_data.isna() | y_data.isna())
            x_clean = x_data[mask]
            y_clean = y_data[mask]
            
            if len(x_clean) > 1:
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
                            line=dict(width=0)
                        ),
                        name='Datos',
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                # Línea de tendencia
                try:
                    z = np.polyfit(x_clean, y_clean, 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x_clean.min(), x_clean.max(), 100)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_trend,
                            y=p(x_trend),
                            mode='lines',
                            line=dict(color='#FFD93D', width=2),
                            name='Tendencia',
                            showlegend=False
                        ),
                        row=2, col=2
                    )
                except:
                    pass
    
    # Actualizar diseño
    fig.update_layout(
        template="plotly_dark",
        height=900,
        showlegend=False,
        title={
            'text': f"<b>📊 Análisis de Percentiles: {feature_name}{feature_length}</b>",
            'font': {'size': 24, 'color': '#E0E5FF'},
            'x': 0.5,
            'xanchor': 'center'
        },
        plot_bgcolor='rgba(30, 34, 56, 0.3)',
        paper_bgcolor='rgba(14, 17, 39, 0.8)'
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text="<b>Valores</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Frecuencia</b>", row=1, col=1)
    
    fig.update_xaxes(title_text="<b>Percentiles</b>", row=1, col=2)
    fig.update_yaxes(title_text="<b>Retorno %</b>", row=1, col=2)
    
    fig.update_xaxes(title_text="<b>Fecha</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Correlación</b>", row=2, col=1)
    
    fig.update_xaxes(title_text=f"<b>{feature_name}{feature_length}</b>", row=2, col=2)
    fig.update_yaxes(title_text="<b>Retorno %</b>", row=2, col=2)
    
    return fig

def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: 0;'>
            📊 Analizador de Performance por Percentiles
        </h1>
        <h3 style='text-align: center; margin-top: 0; margin-bottom: 2rem;'>
            Indicadores Técnicos con Análisis Cuantitativo
        </h3>
    """, unsafe_allow_html=True)
    
    # Créditos
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
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ **Configuración**")
        
        ticker = st.text_input("🎯 **Símbolo Bursátil**", value="SPY")
        
        st.markdown("### 📅 **Rango de Fechas**")
        use_default = st.checkbox("Usar 2000-Hoy", value=True)
        
        if use_default:
            start_date = "2000-01-01"
            end_date = datetime.now()
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Inicio", value=datetime(2000, 1, 1))
            with col2:
                end_date = st.date_input("Fin", value=datetime.now())
            start_date = start_date.strftime('%Y-%m-%d')
        
        st.markdown("### 📈 **Parámetros**")
        return_days = st.slider("Días de retorno", 1, 30, 5)
        quantiles = st.slider("Número de percentiles", 5, 100, 50)
        
        st.markdown("### 📊 **Indicadores**")
        
        # Lista de indicadores disponibles
        all_indicators = ['RSI', 'SMA', 'EMA', 'WILLR', 'CCI', 'ATR', 'ADX', 
                         'STOCH_K', 'MACD', 'BBANDS_UPPER', 'MOM', 'ROC']
        
        selected_indicators = st.multiselect(
            "Seleccionar indicadores",
            options=all_indicators,
            default=['RSI']
        )
        
        st.info(f"📊 {len(selected_indicators)} indicadores seleccionados")
        
        analyze_button = st.button("🚀 **EJECUTAR ANÁLISIS**", use_container_width=True)
    
    # Análisis
    if analyze_button:
        if not selected_indicators:
            st.error("⚠️ Seleccione al menos un indicador")
            return
        
        with st.spinner('🔄 Procesando...'):
            returns_data, indicators, data = calculate_selected_indicators_returns(
                ticker,
                start_date=start_date,
                end_date=end_date,
                indicators_to_calculate=selected_indicators,
                quantiles=quantiles,
                return_days=return_days
            )
        
        if returns_data and indicators is not None and data is not None:
            if len(indicators.columns) > 0:
                st.success(f"✅ **Análisis completado:** {len(indicators.columns)} configuraciones procesadas")
                
                # Tabs
                tab1, tab2 = st.tabs(["📈 **Análisis de Percentiles**", "📊 **Resumen**"])
                
                with tab1:
                    available = list(indicators.columns)
                    
                    if available:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Extraer nombres de indicadores
                            indicator_names = []
                            for col in available:
                                # Extraer el nombre base del indicador
                                base_name = ''.join([c for c in col if not c.isdigit()])
                                if base_name not in indicator_names:
                                    indicator_names.append(base_name)
                            
                            selected_indicator = st.selectbox("**Indicador**", indicator_names)
                        
                        with col2:
                            # Obtener períodos disponibles
                            periods = []
                            for col in available:
                                if col.startswith(selected_indicator):
                                    try:
                                        period = int(col.replace(selected_indicator, ''))
                                        periods.append(period)
                                    except:
                                        pass
                            
                            if periods:
                                periods.sort()
                                selected_period = st.selectbox("**Período**", periods)
                            else:
                                selected_period = 10
                        
                        with col3:
                            selected_return = st.selectbox(
                                "**Días Retorno**",
                                list(range(1, return_days + 1)),
                                index=min(4, return_days-1)
                            )
                        
                        # Generar gráficos
                        fig = plot_integrated_indicators_returns(
                            indicators,
                            returns_data,
                            data,
                            selected_indicator,
                            selected_period,
                            selected_return
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📅 Total días", f"{len(data):,}")
                    with col2:
                        st.metric("📊 Configuraciones", f"{len(indicators.columns)}")
                    with col3:
                        st.metric("✅ Percentiles", quantiles)
            else:
                st.error("❌ No se pudieron calcular indicadores. Verifica los datos del ticker.")
        else:
            st.error("❌ Error en el análisis. Verifica el ticker y el rango de fechas.")

if __name__ == "__main__":
    main()
