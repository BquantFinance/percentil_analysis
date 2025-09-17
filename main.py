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

# Page configuration with dark theme
st.set_page_config(
    page_title="Technical Indicators Percentile Performance Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark theme CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-container {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def calculate_selected_indicators_returns(ticker, start_date='2000-01-01', end_date=None, 
                                         indicators_to_calculate=None, quantiles=50, 
                                         return_days=5):
    """
    Calculate indicators and returns - exact implementation from original
    Calculates indicators for periods 1 to 100
    """
    if end_date is None:
        end_date = datetime.now()
    
    # Download data
    data = pd.DataFrame(yf.download(ticker, start=start_date, end=end_date, 
                                   progress=False, auto_adjust=True))
    
    if data.empty:
        return None, None, None
    
    indicators = pd.DataFrame(index=data.index)
    
    # Calculate returns for specified days
    for i in range(1, return_days + 1):
        data[f'returns_{i}_days'] = data['Close'].pct_change(i) * 100
    
    data = data.dropna()
    
    # Convert to numpy arrays for talib
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    volume = data['Volume'].values
    open_prices = data['Open'].values
    
    # Define available indicators using talib
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
        'KURTOSIS': lambda i: talib.VAR(close, timeperiod=i),  # Using VAR as proxy
        'MFI': lambda i: talib.MFI(high, low, close, volume, timeperiod=i),
        'OBV': lambda i: talib.OBV(close, volume),
        'NATR': lambda i: talib.NATR(high, low, close, timeperiod=i),
        'TRANGE': lambda i: talib.TRANGE(high, low, close),
        'STDDEV': lambda i: talib.STDDEV(close, timeperiod=i),
        'TSF': lambda i: talib.TSF(close, timeperiod=i),
        'VAR': lambda i: talib.VAR(close, timeperiod=i),
        'LINEARREG': lambda i: talib.LINEARREG(close, timeperiod=i),
        'LINEARREG_ANGLE': lambda i: talib.LINEARREG_ANGLE(close, timeperiod=i),
        'LINEARREG_INTERCEPT': lambda i: talib.LINEARREG_INTERCEPT(close, timeperiod=i),
        'CORREL': lambda i: talib.CORREL(high, low, timeperiod=i),
        'BETA': lambda i: talib.BETA(high, low, timeperiod=i),
        'ROC': lambda i: talib.ROC(close, timeperiod=i),
        'ROCP': lambda i: talib.ROCP(close, timeperiod=i),
        'ROCR': lambda i: talib.ROCR(close, timeperiod=i),
        'MOM': lambda i: talib.MOM(close, timeperiod=i),
        'BBANDS_UPPER': lambda i: talib.BBANDS(close, timeperiod=i)[0],
        'BBANDS_MIDDLE': lambda i: talib.BBANDS(close, timeperiod=i)[1],
        'BBANDS_LOWER': lambda i: talib.BBANDS(close, timeperiod=i)[2],
        'DEMA': lambda i: talib.DEMA(close, timeperiod=i),
        'TEMA': lambda i: talib.TEMA(close, timeperiod=i),
        'TRIMA': lambda i: talib.TRIMA(close, timeperiod=i),
        'WMA': lambda i: talib.WMA(close, timeperiod=i),
        'MIDPOINT': lambda i: talib.MIDPOINT(close, timeperiod=i),
        'MIDPRICE': lambda i: talib.MIDPRICE(high, low, timeperiod=i),
        'CMO': lambda i: talib.CMO(close, timeperiod=i),
        'PPO': lambda i: talib.PPO(close, fastperiod=max(i//2, 2), slowperiod=i),
        'AROON_UP': lambda i: talib.AROON(high, low, timeperiod=i)[0],
        'AROON_DOWN': lambda i: talib.AROON(high, low, timeperiod=i)[1],
        'AROONOSC': lambda i: talib.AROONOSC(high, low, timeperiod=i),
        'ULTOSC': lambda i: talib.ULTOSC(high, low, close, timeperiod1=max(i//3, 2), timeperiod2=max(i//2, 3), timeperiod3=i),
        'DX': lambda i: talib.DX(high, low, close, timeperiod=i),
        'ADXR': lambda i: talib.ADXR(high, low, close, timeperiod=i),
        'TRIX': lambda i: talib.TRIX(close, timeperiod=i),
        'BOP': lambda i: talib.BOP(open_prices, high, low, close),
        'SAR': lambda i: talib.SAR(high, low, acceleration=0.02, maximum=0.2),
        'T3': lambda i: talib.T3(close, timeperiod=i, vfactor=0),
        'HT_TRENDLINE': lambda i: talib.HT_TRENDLINE(close),
        'HT_DCPERIOD': lambda i: talib.HT_DCPERIOD(close),
        'HT_DCPHASE': lambda i: talib.HT_DCPHASE(close),
        'HT_TRENDMODE': lambda i: talib.HT_TRENDMODE(close),
    }
    
    if indicators_to_calculate is None:
        indicators_to_calculate = list(available_indicators.keys())
    
    # Calculate indicators for periods 1 to 100 (exact as original)
    progress_bar = st.progress(0)
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
    
    progress_bar.empty()
    
    indicators = indicators.dropna()
    data_clean = data.drop(['Open', 'High', 'Low', 'Volume', 'Close'], axis=1, errors='ignore')
    
    # Calculate returns data for each indicator - exact as original
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
    Create the exact 4 plots as in the original implementation using Plotly
    """
    indicator_col = f'{feature_name}{feature_length}'
    
    if indicator_col not in indicators.columns:
        st.error(f"Indicator {indicator_col} not found")
        return None
    
    if indicator_col not in returns_data:
        st.error(f"Returns data for {indicator_col} not found")
        return None
    
    # Create subplots - exact layout as original
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f'{feature_name}{feature_length}',
            f'Average Returns ({return_days}-day) for {feature_name}{feature_length}',
            f'Rolling Correlation(126) between {feature_name}{feature_length} and returns',
            f'{feature_name}{feature_length} vs Returns ({return_days}-day)'
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25],
        vertical_spacing=0.12
    )
    
    # 1. Histogram - exact as original
    fig.add_trace(
        go.Histogram(
            x=indicators[indicator_col],
            nbinsx=70,
            marker_color='blue',
            opacity=0.6,
            name='Distribution',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add mean line
    mean_val = indicators[indicator_col].mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        line_width=1.5,
        row=1, col=1
    )
    
    # Add annotation for mean
    fig.add_annotation(
        x=mean_val,
        y=0,
        text=f'{feature_name}{feature_length} Mean = {mean_val:.2f}',
        showarrow=True,
        arrowhead=2,
        row=1, col=1,
        yref="paper",
        yshift=10
    )
    
    # 2. Average Returns by Quantile - exact as original
    returns_col = f'returns_{return_days}_days_mean'
    if returns_col in returns_data[indicator_col].columns:
        fig.add_trace(
            go.Bar(
                x=[str(idx) for idx in returns_data[indicator_col].index],
                y=returns_data[indicator_col][returns_col],
                marker_color='red',
                opacity=0.5,
                name='Avg Returns',
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 3. Rolling Correlation - exact as original
    if f'returns_{return_days}_days' in data.columns:
        # Calculate overall correlation
        overall_corr = data[f'returns_{return_days}_days'].corr(indicators[indicator_col])
        
        # Calculate rolling correlation with 126 window
        rolling_corr = data[f'returns_{return_days}_days'].rolling(126).corr(indicators[indicator_col]).dropna()
        
        fig.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr.values,
                mode='lines',
                line=dict(color='yellow'),
                name='Rolling Corr',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Add overall correlation line
        fig.add_hline(
            y=overall_corr,
            line_dash="dash",
            line_color="red",
            line_width=2,
            row=3, col=1
        )
        
        # Add annotation for correlation
        fig.add_annotation(
            x=0.98,
            y=overall_corr,
            text=f'Correlation = {overall_corr:.2f}',
            showarrow=False,
            xref="paper",
            row=3, col=1,
            bgcolor="rgba(255,0,0,0.2)",
            font=dict(color="white")
        )
    
    # 4. Scatter plot with Polynomial Fit - exact as original
    if f'returns_{return_days}_days' in data.columns:
        valid_dates = data.index[data.index >= indicators[indicator_col].index[0]]
        x_data = indicators[indicator_col]
        y_data = data[f'returns_{return_days}_days'].loc[valid_dates]
        
        # Remove NaN values
        mask = ~(x_data.isna() | y_data.isna())
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) > 1:
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=x_clean,
                    y=y_clean,
                    mode='markers',
                    marker=dict(color='blue', size=3, opacity=0.6),
                    name='Data Points',
                    showlegend=False
                ),
                row=4, col=1
            )
            
            # Add polynomial fit (degree 1 as in original)
            try:
                poly_coefficients = np.polyfit(x_clean, y_clean, deg=1)
                poly_curve = np.polyval(poly_coefficients, x_clean)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_clean,
                        y=poly_curve,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Polynomial Fit',
                        showlegend=True
                    ),
                    row=4, col=1
                )
            except:
                pass
    
    # Update layout - matching original style
    fig.update_layout(
        template="plotly_dark",
        height=1200,
        showlegend=False,
        title_text=f"Percentile Analysis: {feature_name}{feature_length}",
        title_font_size=20,
        hovermode='x unified'
    )
    
    # Update axes labels - exact as original
    fig.update_xaxes(title_text="Values", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    fig.update_xaxes(title_text="Quantiles", tickangle=45, row=2, col=1)
    fig.update_yaxes(title_text=f"Average Return ({return_days}-day)", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Correlation", row=3, col=1)
    
    fig.update_xaxes(title_text=f"{feature_name}{feature_length}", row=4, col=1)
    fig.update_yaxes(title_text=f"Returns ({return_days}-day)", row=4, col=1)
    
    return fig

def main():
    st.title("📊 Technical Indicators Percentile Performance Analyzer")
    st.markdown("### Exact Implementation with TALib - 100 Periods Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Ticker selection
        ticker = st.text_input("Stock Ticker", value="SPY", 
                              help="Enter a valid stock ticker symbol")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            use_default_dates = st.checkbox("Use Default (2000-Now)", value=True)
        
        if use_default_dates:
            start_date = "2000-01-01"
            end_date = datetime.now()
        else:
            with col1:
                start_date = st.date_input("Start Date", 
                                          value=datetime(2000, 1, 1))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())
            
            start_date = start_date.strftime('%Y-%m-%d')
        
        # Return days configuration
        st.subheader("📈 Return Configuration")
        return_days = st.slider("Max Return Days", 
                               min_value=1, max_value=30, value=5,
                               help="Calculate returns up to this many days")
        
        # Quantiles configuration
        quantiles = st.slider("Number of Quantiles", 
                            min_value=5, max_value=100, value=50,
                            help="Number of quantiles for percentile analysis")
        
        # Indicator selection
        st.subheader("📊 Indicator Selection")
        
        # Grouped indicators
        momentum_indicators = ['RSI', 'WILLR', 'CCI', 'STOCH_K', 'STOCH_D', 'CMO', 
                              'MOM', 'ROC', 'ROCP', 'ROCR', 'TRIX', 'ULTOSC']
        
        trend_indicators = ['ADX', 'ADXR', 'DX', 'PLUS_DI', 'MINUS_DI', 'AROON_UP', 
                          'AROON_DOWN', 'AROONOSC', 'SAR', 'APO', 'PPO', 'MACD']
        
        volatility_indicators = ['ATR', 'NATR', 'TRANGE', 'STDDEV', 'VAR', 
                                'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER']
        
        moving_averages = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 
                         'T3', 'HMA', 'MIDPOINT', 'MIDPRICE']
        
        statistics_indicators = ['LINEARREG', 'LINEARREG_ANGLE', 'LINEARREG_INTERCEPT', 
                                'LINEARREG_SLOPE', 'TSF', 'BETA', 'CORREL', 'SLOPE', 'ZSCORE']
        
        other_indicators = ['DONCHIAN', 'BOP', 'HT_TRENDLINE', 'HT_DCPERIOD', 
                          'HT_DCPHASE', 'HT_TRENDMODE', 'MFI', 'OBV']
        
        all_available = (momentum_indicators + trend_indicators + volatility_indicators + 
                        moving_averages + statistics_indicators + other_indicators)
        
        # Selection method
        selection_method = st.radio(
            "Selection Method",
            ["Quick Select", "By Category", "Custom"],
            horizontal=True
        )
        
        if selection_method == "Quick Select":
            quick_option = st.selectbox(
                "Choose preset",
                ["Original Paper (RSI)", "Popular 5", "Momentum", "Trend", "All"]
            )
            
            if quick_option == "Original Paper (RSI)":
                selected_indicators = ['RSI']
            elif quick_option == "Popular 5":
                selected_indicators = ['RSI', 'MACD', 'BBANDS_UPPER', 'ATR', 'ADX']
            elif quick_option == "Momentum":
                selected_indicators = momentum_indicators[:5]
            elif quick_option == "Trend":
                selected_indicators = trend_indicators[:5]
            else:
                selected_indicators = all_available[:20]
                
        elif selection_method == "By Category":
            categories = st.multiselect(
                "Select Categories",
                ["Momentum", "Trend", "Volatility", "Moving Averages", "Statistics", "Other"],
                default=["Momentum"]
            )
            
            selected_indicators = []
            if "Momentum" in categories:
                selected_indicators.extend(momentum_indicators)
            if "Trend" in categories:
                selected_indicators.extend(trend_indicators)
            if "Volatility" in categories:
                selected_indicators.extend(volatility_indicators)
            if "Moving Averages" in categories:
                selected_indicators.extend(moving_averages)
            if "Statistics" in categories:
                selected_indicators.extend(statistics_indicators)
            if "Other" in categories:
                selected_indicators.extend(other_indicators)
                
        else:  # Custom
            selected_indicators = st.multiselect(
                "Select indicators to analyze",
                options=all_available,
                default=['RSI']
            )
        
        st.info(f"Selected {len(selected_indicators)} indicators")
        st.caption("Each indicator will be calculated for periods 1-100")
        
        # Analysis button
        analyze_button = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    
    # Main content area
    if analyze_button:
        if not selected_indicators:
            st.error("Please select at least one indicator")
            return
            
        with st.spinner(f"Calculating {len(selected_indicators)} indicators for 100 periods each..."):
            # Calculate indicators and returns
            returns_data, indicators, data = calculate_selected_indicators_returns(
                ticker, 
                start_date=start_date,
                end_date=end_date,
                indicators_to_calculate=selected_indicators,
                quantiles=quantiles,
                return_days=return_days
            )
        
        if returns_data is not None and indicators is not None and data is not None:
            st.success(f"✅ Analysis complete! Calculated {len(indicators.columns)} indicator configurations")
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Percentile Analysis", 
                "🎯 Best Performers", 
                "📊 Comparison Matrix",
                "📋 Data Overview"
            ])
            
            with tab1:
                st.header("Percentile Analysis - Exact Original Implementation")
                
                # Get available indicators
                available_indicators = list(indicators.columns)
                
                if available_indicators:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Extract unique indicator names
                        indicator_names = list(set([col.rstrip('0123456789') for col in available_indicators]))
                        indicator_names.sort()
                        selected_indicator = st.selectbox(
                            "Select Indicator",
                            options=indicator_names,
                            index=0
                        )
                    
                    with col2:
                        # Get available periods for selected indicator
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
                            # Default to period 10 if available, otherwise first
                            default_idx = available_periods.index(10) if 10 in available_periods else 0
                            selected_period = st.selectbox(
                                "Select Period",
                                options=available_periods,
                                index=default_idx
                            )
                        else:
                            selected_period = 10
                            st.warning("No periods found")
                    
                    with col3:
                        selected_return_days = st.selectbox(
                            "Return Days",
                            options=list(range(1, return_days + 1)),
                            index=min(4, return_days-1) if return_days >= 5 else 0
                        )
                    
                    # Create the 4 plots - exact as original
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
                    
                    # Show quantile statistics
                    with st.expander("📊 Quantile Statistics"):
                        indicator_col = f'{selected_indicator}{selected_period}'
                        if indicator_col in returns_data:
                            st.dataframe(
                                returns_data[indicator_col],
                                use_container_width=True
                            )
            
            with tab2:
                st.header("🎯 Best Performing Indicators")
                
                # Find best configurations
                best_configs = []
                
                for ind_col in indicators.columns[:500]:  # Limit to first 500 for performance
                    if ind_col in returns_data:
                        for ret_day in range(1, min(return_days + 1, 6)):
                            ret_col = f'returns_{ret_day}_days_mean'
                            if ret_col in returns_data[ind_col].columns:
                                # Get spread between top and bottom quantile
                                returns_df = returns_data[ind_col][ret_col]
                                if len(returns_df) > 1:
                                    spread = returns_df.iloc[-1] - returns_df.iloc[0]
                                    best_configs.append({
                                        'Indicator': ind_col,
                                        'Return_Days': ret_day,
                                        'Top_Quantile': returns_df.iloc[-1],
                                        'Bottom_Quantile': returns_df.iloc[0],
                                        'Spread': spread
                                    })
                
                if best_configs:
                    best_df = pd.DataFrame(best_configs)
                    best_df = best_df.sort_values('Spread', ascending=False).head(30)
                    
                    # Display top performers
                    st.subheader("Top 30 Configurations by Return Spread")
                    
                    # Format and display
                    display_df = best_df.copy()
                    display_df['Top_Quantile'] = display_df['Top_Quantile'].apply(lambda x: f"{x:.3f}%")
                    display_df['Bottom_Quantile'] = display_df['Bottom_Quantile'].apply(lambda x: f"{x:.3f}%")
                    display_df['Spread'] = display_df['Spread'].apply(lambda x: f"{x:.3f}%")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Plot spread distribution
                    fig_spread = go.Figure()
                    fig_spread.add_trace(go.Bar(
                        x=best_df['Indicator'].head(20),
                        y=best_df['Spread'].head(20),
                        marker_color='green'
                    ))
                    fig_spread.update_layout(
                        template="plotly_dark",
                        title="Top 20 Indicators by Return Spread",
                        xaxis_title="Indicator",
                        yaxis_title="Spread (%)",
                        xaxis_tickangle=45,
                        height=500
                    )
                    st.plotly_chart(fig_spread, use_container_width=True)
            
            with tab3:
                st.header("📊 Indicator Comparison Matrix")
                
                # Select indicators to compare
                comparison_indicators = st.multiselect(
                    "Select indicators to compare (with specific periods)",
                    options=available_indicators[:200],
                    default=available_indicators[:min(5, len(available_indicators))]
                )
                
                if comparison_indicators and len(comparison_indicators) > 1:
                    # Create correlation matrix
                    corr_matrix = indicators[comparison_indicators].corr()
                    
                    # Create heatmap
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10}
                    ))
                    
                    fig_corr.update_layout(
                        template="plotly_dark",
                        title="Indicator Correlation Matrix",
                        height=600,
                        xaxis_tickangle=45
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Performance comparison
                    st.subheader("Performance Comparison")
                    
                    comparison_data = []
                    for ind in comparison_indicators:
                        if ind in returns_data:
                            for ret_day in [1, 5, 10, 20][:min(4, return_days)]:
                                ret_col = f'returns_{ret_day}_days_mean'
                                if ret_col in returns_data[ind].columns:
                                    returns_df = returns_data[ind][ret_col]
                                    if len(returns_df) > 1:
                                        comparison_data.append({
                                            'Indicator': ind,
                                            f'{ret_day}d_Spread': returns_df.iloc[-1] - returns_df.iloc[0]
                                        })
                    
                    if comparison_data:
                        # Aggregate by indicator
                        comp_df = pd.DataFrame(comparison_data)
                        comp_pivot = comp_df.pivot_table(index='Indicator', values=[col for col in comp_df.columns if 'd_Spread' in col])
                        
                        st.dataframe(
                            comp_pivot.style.format("{:.3f}%").background_gradient(cmap='RdYlGn'),
                            use_container_width=True
                        )
            
            with tab4:
                st.header("📋 Data Overview")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Ticker", ticker)
                    st.metric("Total Days", f"{len(data):,}")
                    st.metric("Date Range", f"{(end_date - pd.to_datetime(start_date)).days} days" if not use_default_dates else "Max Available")
                
                with col2:
                    st.metric("Indicators Calculated", f"{len(selected_indicators)}")
                    st.metric("Total Configurations", f"{len(indicators.columns):,}")
                    st.metric("Quantiles", quantiles)
                
                with col3:
                    st.metric("Return Days", return_days)
                    st.metric("Valid Data Points", f"{len(indicators):,}")
                    
                    # Calculate average return
                    if f'returns_{return_days}_days' in data.columns:
                        avg_return = data[f'returns_{return_days}_days'].mean()
                        st.metric(f"Avg {return_days}-Day Return", f"{avg_return:.3f}%")
                
                # Show sample of indicators data
                st.subheader("Sample Indicator Values")
                
                sample_cols = indicators.columns[:10] if len(indicators.columns) > 10 else indicators.columns
                st.dataframe(
                    indicators[sample_cols].tail(20),
                    use_container_width=True
                )
                
                # Distribution of returns
                st.subheader("Returns Distribution")
                
                fig_dist = go.Figure()
                
                for i in [1, 5, 10, 20][:min(4, return_days)]:
                    if f'returns_{i}_days' in data.columns:
                        fig_dist.add_trace(go.Histogram(
                            x=data[f'returns_{i}_days'].dropna(),
                            name=f'{i}-Day Returns',
                            opacity=0.7,
                            nbinsx=50
                        ))
                
                fig_dist.update_layout(
                    template="plotly_dark",
                    title="Distribution of Returns",
                    xaxis_title="Return (%)",
                    yaxis_title="Frequency",
                    barmode='overlay',
                    height=400
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()
