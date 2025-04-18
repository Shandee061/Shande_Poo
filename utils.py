"""
Utility functions for the WINFUT trading robot.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import datetime
import calendar

def plot_price_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive price chart with indicators.
    
    Args:
        df: DataFrame with OHLCV data and indicators
        
    Returns:
        Plotly figure object
    """
    # Create subplots: 1 for price, 1 for volume, 1 for indicators
    fig = make_subplots(
        rows=3, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.6, 0.15, 0.25],
        subplot_titles=("Price", "Volume", "Indicators")
    )
    
    # Add candlestick chart for price
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['sma_fast'],
                name=f"SMA Fast",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['sma_slow'],
                name=f"SMA Slow",
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands if available
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                name="BB Upper",
                line=dict(color='rgba(0,128,0,0.3)', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                name="BB Lower",
                line=dict(color='rgba(0,128,0,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0,128,0,0.05)'
            ),
            row=1, col=1
        )
    
    # Add volume bars
    colors = ['red' if row['close'] < row['open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volume",
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Add indicators: RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name="RSI",
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_shape(
            type='line',
            x0=df.index[0],
            x1=df.index[-1],
            y0=70,
            y1=70,
            line=dict(color='red', width=1, dash='dash'),
            row=3, col=1
        )
        
        fig.add_shape(
            type='line',
            x0=df.index[0],
            x1=df.index[-1],
            y0=30,
            y1=30,
            line=dict(color='green', width=1, dash='dash'),
            row=3, col=1
        )
    
    # Add MACD if available
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd'],
                name="MACD",
                line=dict(color='blue', width=1)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd_signal'],
                name="Signal",
                line=dict(color='red', width=1)
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        title="WINFUT Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False
    )
    
    # Update y-axis for RSI
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    
    # Update y-axis for volume
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def plot_equity_curve(equity_df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive equity curve chart with drawdowns.
    
    Args:
        equity_df: DataFrame with equity and drawdown data
        
    Returns:
        Plotly figure object
    """
    # Ensure equity_df has a datetime index
    if not isinstance(equity_df.index, pd.DatetimeIndex):
        try:
            if 'timestamp' in equity_df.columns:
                equity_df = equity_df.set_index('timestamp')
            else:
                equity_df.index = pd.to_datetime(equity_df.index)
        except:
            pass

    # Create subplots
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3],
        subplot_titles=("Curva de Equity e Retorno Acumulado", "Drawdown")
    )
    
    # Calculate cumulative returns if it doesn't exist
    if 'returns' not in equity_df.columns and 'equity' in equity_df.columns:
        # Calculate returns from equity
        initial_equity = equity_df['equity'].iloc[0]
        equity_df['returns'] = equity_df['equity'] / initial_equity - 1
    
    # Add equity curve
    if 'equity' in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                name="Equity",
                line=dict(color='rgb(0, 100, 200)', width=2)
            ),
            row=1, col=1
        )
    
    # Add returns curve
    if 'returns' in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['returns'] * 100,  # Convert to percentage
                name="Retorno %",
                line=dict(color='rgb(0, 150, 50)', width=2, dash='dash'),
                yaxis="y2"
            ),
            row=1, col=1
        )
    
    # Add benchmark if available
    if 'benchmark' in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['benchmark'],
                name="Benchmark",
                line=dict(color='rgb(150, 150, 150)', width=1.5, dash='dot')
            ),
            row=1, col=1
        )
    
    # Add drawdown
    if 'drawdown' in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df.index,
                y=equity_df['drawdown'] * 100,  # Convert to percentage
                name="Drawdown",
                line=dict(color='rgba(255, 50, 50, 0.8)', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255, 50, 50, 0.2)'
            ),
            row=2, col=1
        )

    # Add position markers if available
    if 'position' in equity_df.columns and 'equity' in equity_df.columns:
        # Find points where position changes from 0 to 1 (entries)
        long_entries = equity_df[
            (equity_df['position'] == 1) & 
            (equity_df['position'].shift(1) != 1)
        ]
        
        # Find points where position changes from 0 to -1 (entries)
        short_entries = equity_df[
            (equity_df['position'] == -1) & 
            (equity_df['position'].shift(1) != -1)
        ]
        
        # Find points where position changes from 1 or -1 to 0 (exits)
        exits = equity_df[
            (equity_df['position'] == 0) & 
            ((equity_df['position'].shift(1) == 1) | (equity_df['position'].shift(1) == -1))
        ]
        
        # Add markers for long entries
        if not long_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=long_entries.index,
                    y=long_entries['equity'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green',
                        line=dict(width=1, color='darkgreen')
                    ),
                    name='Compra'
                ),
                row=1, col=1
            )
        
        # Add markers for short entries
        if not short_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=short_entries.index,
                    y=short_entries['equity'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red',
                        line=dict(width=1, color='darkred')
                    ),
                    name='Venda'
                ),
                row=1, col=1
            )
        
        # Add markers for exits
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits.index,
                    y=exits['equity'],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color='gray',
                        line=dict(width=1, color='darkgray')
                    ),
                    name='Saída'
                ),
                row=1, col=1
            )
    
    # Update layout with improved styling
    fig.update_layout(
        height=700,
        template="plotly_white",
        title={
            'text': "Resultado do Backtest - WINFUT",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 22, 'color': 'rgb(50, 50, 100)'}
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(200, 200, 200, 0.7)',
            borderwidth=1
        ),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        )
    )
    
    # Update axes
    fig.update_yaxes(
        title_text="Capital (R$)",
        row=1, col=1,
        showgrid=True,
        gridcolor='rgba(230, 230, 230, 0.8)',
        zeroline=True,
        zerolinecolor='rgba(50, 50, 50, 0.2)',
        zerolinewidth=1,
        tickprefix="R$ "
    )
    
    # Add second y-axis for returns
    fig.update_layout(
        yaxis2=dict(
            title="Retorno (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            ticksuffix="%"
        )
    )
    
    # Update drawdown y-axis
    fig.update_yaxes(
        title_text="Drawdown (%)",
        row=2, col=1,
        showgrid=True,
        gridcolor='rgba(230, 230, 230, 0.8)',
        range=[
            min(-0.5, equity_df.get('drawdown', pd.Series(0)).min() * 110), 
            0.5
        ],
        ticksuffix="%"
    )
    
    return fig


def get_current_winfut_contract() -> str:
    """
    Determina o contrato atual do Mini Índice (WINFUT) com base na data atual.
    
    Os contratos do Mini Índice vencem a cada dois meses (ciclo par):
    Fevereiro (G), Abril (J), Junho (M), Agosto (Q), Outubro (V), Dezembro (Z)
    
    Retorna:
        Código do contrato ativo atual (ex: WINM25 para Junho/2025)
    """
    # Códigos dos meses para os contratos
    month_codes = {
        2: 'G',  # Fevereiro
        4: 'J',  # Abril
        6: 'M',  # Junho
        8: 'Q',  # Agosto
        10: 'V', # Outubro
        12: 'Z'  # Dezembro
    }
    
    now = datetime.datetime.now()
    current_year = now.year
    current_month = now.month
    
    # Próximo mês de vencimento
    next_expiry_month = None
    
    # Encontra o próximo mês de vencimento
    for month in sorted(month_codes.keys()):
        if month >= current_month:
            next_expiry_month = month
            break
    
    # Se estamos em novembro ou dezembro, o próximo vencimento será em fevereiro do próximo ano
    if next_expiry_month is None:
        next_expiry_month = 2
        current_year += 1
    
    # Precisamos verificar se estamos no mês de vencimento e se já passou o dia de vencimento
    # Normalmente o vencimento é na quarta-feira mais próxima do dia 15
    if next_expiry_month == current_month:
        # Calcula o dia de vencimento (quarta-feira mais próxima do dia 15)
        c = calendar.monthcalendar(current_year, current_month)
        # Encontra a quarta-feira (weekday=2) mais próxima do dia 15
        third_week = c[2]  # Terceira semana do mês
        if third_week[2] != 0:  # Se tem quarta-feira na terceira semana
            expiry_day = third_week[2]
        else:  # Caso contrário, usa a quarta da segunda semana
            expiry_day = c[1][2]
        
        # Se já passou o vencimento, aponta para o próximo
        if now.day > expiry_day:
            # Encontra o próximo mês de vencimento
            next_months = [m for m in sorted(month_codes.keys()) if m > current_month]
            if next_months:
                next_expiry_month = next_months[0]
            else:
                next_expiry_month = 2  # Fevereiro do próximo ano
                current_year += 1
    
    # Formata o código do contrato (ex: WINM25 para Junho/2025)
    month_code = month_codes[next_expiry_month]
    year_code = str(current_year)[-2:]  # Últimos 2 dígitos do ano
    
    return f"WIN{month_code}{year_code}"


def get_available_winfut_contracts() -> List[str]:
    """
    Retorna uma lista de contratos do Mini Índice (WINFUT) disponíveis para negociação.
    
    Retorna:
        Lista de códigos de contratos (ex: ["WINM25", "WINQ25", "WINV25", "WINZ25", "WING26"])
    """
    month_codes = {
        2: 'G',  # Fevereiro
        4: 'J',  # Abril
        6: 'M',  # Junho
        8: 'Q',  # Agosto
        10: 'V', # Outubro
        12: 'Z'  # Dezembro
    }
    
    now = datetime.datetime.now()
    current_year = now.year
    current_month = now.month
    
    contracts = []
    
    # Adiciona contratos para o ano atual
    for month in sorted(month_codes.keys()):
        if month >= current_month:
            month_code = month_codes[month]
            year_code = str(current_year)[-2:]
            contracts.append(f"WIN{month_code}{year_code}")
    
    # Adiciona os primeiros contratos do próximo ano
    next_year = current_year + 1
    for month in sorted(month_codes.keys())[:2]:  # Apenas os dois primeiros vencimentos do próximo ano
        month_code = month_codes[month]
        year_code = str(next_year)[-2:]
        contracts.append(f"WIN{month_code}{year_code}")
    
    return contracts


def select_winfut_contract_for_trading(symbol: Optional[str] = None) -> str:
    """
    Seleciona o contrato WINFUT para negociação.
    
    Args:
        symbol: Símbolo específico (se fornecido)
        
    Returns:
        Código do contrato para negociação
    """
    # Se um símbolo foi fornecido e parece válido (começa com WIN), usa-o
    if symbol and symbol.upper().startswith("WIN"):
        return symbol.upper()
    
    # Caso contrário, retorna o contrato atual
    return get_current_winfut_contract()


def format_number(number, precision=2, decimals=None):
    """
    Format a number for display with proper formatting.
    
    Args:
        number: Number to format
        precision: Decimal precision
        decimals: If provided, overrides the precision parameter
        
    Returns:
        Formatted string
    """
    if number is None:
        return "N/A"
    
    try:
        number = float(number)
        # If decimals parameter is provided, use it instead of precision
        if decimals is not None:
            precision = decimals
            
        if abs(number) >= 1000000:
            return f"{number/1000000:.{precision}f}M"
        elif abs(number) >= 1000:
            return f"{number/1000:.{precision}f}K"
        else:
            return f"{number:.{precision}f}"
    except (ValueError, TypeError):
        return str(number)


def display_trade_list(trades, detailed=False):
    """
    Display a list of trades in a formatted table.
    
    Args:
        trades: List of trade dictionaries
        detailed: Whether to display detailed trade information
    """
    if not trades:
        st.info("Nenhum trade para exibir")
        return
    
    # Extract relevant fields for display
    trade_data = []
    for i, trade in enumerate(trades):
        # Handle different trade record formats
        if "position_type" in trade:  # Open position format
            trade_type = "Long" if trade["position_type"] == 1 else "Short"
            entry_time = trade.get("entry_time", "N/A")
            entry_price = trade.get("entry_price", "N/A")
            exit_time = "Open"
            exit_price = "N/A"
            pnl = "N/A"
            status = "Open"
            exit_reason = "N/A"
            position = trade.get("position_type", 0)
        elif "side" in trade:  # Order format
            trade_type = trade.get("side", "N/A")
            entry_time = trade.get("timestamp", "N/A")
            entry_price = trade.get("price", "N/A")
            exit_time = "N/A"
            exit_price = "N/A"
            pnl = "N/A"
            status = trade.get("status", "N/A")
            exit_reason = "N/A"
            position = 1 if trade_type == "BUY" else -1
        elif "position" in trade:  # Trade history format (backtest)
            position = trade.get("position", 0)
            trade_type = "Long" if position == 1 else "Short"
            entry_time = trade.get("entry_time", "N/A")
            entry_price = trade.get("entry_price", "N/A")
            exit_time = trade.get("exit_time", "N/A")
            exit_price = trade.get("exit_price", "N/A")
            pnl = trade.get("pnl", "N/A")
            status = "Closed"
            exit_reason = trade.get("exit_reason", "N/A")
        else:  # Other trade history format
            trade_type = trade.get("type", "N/A")
            entry_time = trade.get("entry_time", "N/A")
            entry_price = trade.get("entry_price", "N/A")
            exit_time = trade.get("exit_time", "N/A")
            exit_price = trade.get("exit_price", "N/A")
            pnl = trade.get("pnl", "N/A")
            status = "Closed"
            exit_reason = trade.get("exit_reason", "N/A")
            position = 1 if trade_type == "Long" else -1
        
        # Calculate trade duration
        duration = "N/A"
        if isinstance(entry_time, (datetime.datetime, pd.Timestamp)) and isinstance(exit_time, (datetime.datetime, pd.Timestamp)):
            delta = exit_time - entry_time
            if delta.days > 0:
                duration = f"{delta.days}d {delta.seconds // 3600}h"
            elif delta.seconds // 3600 > 0:
                duration = f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
            else:
                duration = f"{(delta.seconds % 3600) // 60}m {delta.seconds % 60}s"
        
        # Format times
        entry_time_str = entry_time
        if isinstance(entry_time, (datetime.datetime, pd.Timestamp)):
            entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M:%S")
        
        exit_time_str = exit_time
        if isinstance(exit_time, (datetime.datetime, pd.Timestamp)):
            exit_time_str = exit_time.strftime("%Y-%m-%d %H:%M:%S")
            
        # Determine if trade was profitable
        result = "N/A"
        if isinstance(pnl, (int, float)):
            result = "Ganho" if pnl > 0 else "Perda" if pnl < 0 else "Neutro"
        
        # Create trade record with basic information
        trade_record = {
            "ID": i+1,
            "Tipo": trade_type,
            "Entrada": entry_time_str,
            "Preço Entrada": format_number(entry_price),
            "Saída": exit_time_str,
            "Preço Saída": format_number(exit_price),
            "Duração": duration,
            "Resultado": result,
            "P&L": format_number(pnl),
            "Status": status
        }
        
        # Add detailed information if requested
        if detailed:
            trade_record.update({
                "Motivo de Saída": exit_reason.replace("_", " ").title(),
                "Barras": trade.get("bars_held", "N/A"),
                "Stop Loss": format_number(trade.get("stop_loss", "N/A")),
                "Take Profit": format_number(trade.get("take_profit", "N/A"))
            })
        
        trade_data.append(trade_record)
    
    # Convert to DataFrame
    trade_df = pd.DataFrame(trade_data)
    
    # Add styling
    def highlight_pnl(val):
        if isinstance(val, str) and val.startswith("R$"):
            try:
                val_float = float(val.replace("R$", "").replace(",", "").strip())
                if val_float > 0:
                    return 'background-color: rgba(0, 200, 0, 0.2); color: green;'
                elif val_float < 0:
                    return 'background-color: rgba(255, 0, 0, 0.2); color: red;'
            except:
                pass
        return ''
    
    def highlight_result(val):
        if val == "Ganho":
            return 'background-color: rgba(0, 200, 0, 0.2); color: green;'
        elif val == "Perda":
            return 'background-color: rgba(255, 0, 0, 0.2); color: red;'
        return ''
    
    # Display the dataframe
    st.dataframe(trade_df.style.applymap(highlight_pnl, subset=['P&L'])
                                 .applymap(highlight_result, subset=['Resultado']), 
                 use_container_width=True)
    
    return trade_df


def plot_trade_distribution(trades):
    """
    Create an interactive visualization of trade distribution.
    
    Args:
        trades: List of trade dictionaries
    
    Returns:
        Plotly figure object or None if trades is empty
    """
    if not trades:
        return None
    
    # Convert trades to DataFrame if it's not already
    if not isinstance(trades, pd.DataFrame):
        # Create DataFrame from trades
        trade_df = display_trade_list(trades, detailed=False)
        if trade_df is None:
            return None
    else:
        trade_df = trades
    
    # Ensure we have a P&L column
    if 'P&L' not in trade_df.columns and 'pnl' in trade_df.columns:
        trade_df['P&L'] = trade_df['pnl']
    
    # Clean P&L column if it's a string
    if trade_df['P&L'].dtype == 'object':
        try:
            trade_df['P&L_num'] = trade_df['P&L'].str.replace('R$', '').str.replace(',', '.').astype(float)
        except:
            st.warning("Não foi possível converter os valores de P&L para análise")
            return None
    else:
        trade_df['P&L_num'] = trade_df['P&L']
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}], 
               [{"type": "histogram"}, {"type": "scatter"}]],
        subplot_titles=("P&L por Trade", "Distribuição de Resultados", 
                        "Histograma de P&L", "P&L Acumulado"),
        vertical_spacing=0.12,
        horizontal_spacing=0.07
    )
    
    # 1. P&L by Trade (Bar Chart)
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in trade_df['P&L_num']]
    
    fig.add_trace(
        go.Bar(
            x=trade_df.index,
            y=trade_df['P&L_num'],
            marker_color=colors,
            name="P&L"
        ),
        row=1, col=1
    )
    
    # 2. Win/Loss Distribution (Pie Chart)
    win_count = sum(trade_df['P&L_num'] > 0)
    loss_count = sum(trade_df['P&L_num'] < 0)
    neutral_count = sum(trade_df['P&L_num'] == 0)
    
    fig.add_trace(
        go.Pie(
            labels=['Ganhos', 'Perdas', 'Neutros'],
            values=[win_count, loss_count, neutral_count],
            marker_colors=['green', 'red', 'gray'],
            textinfo='percent+value',
            hole=0.3
        ),
        row=1, col=2
    )
    
    # 3. P&L Distribution (Histogram)
    fig.add_trace(
        go.Histogram(
            x=trade_df['P&L_num'],
            nbinsx=20,
            marker_color='blue',
            opacity=0.7,
            name="Distribuição"
        ),
        row=2, col=1
    )
    
    # 4. Cumulative P&L (Line Chart)
    cumulative_pnl = trade_df['P&L_num'].cumsum()
    
    fig.add_trace(
        go.Scatter(
            x=trade_df.index,
            y=cumulative_pnl,
            mode='lines+markers',
            line=dict(width=2, color='purple'),
            name="P&L Acumulado"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        title={
            'text': "Análise de Trades - WINFUT",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text="ID do Trade", row=1, col=1)
    fig.update_yaxes(title_text="R$", row=1, col=1)
    
    fig.update_xaxes(title_text="P&L (R$)", row=2, col=1)
    fig.update_yaxes(title_text="Frequência", row=2, col=1)
    
    fig.update_xaxes(title_text="ID do Trade", row=2, col=2)
    fig.update_yaxes(title_text="P&L Acumulado (R$)", row=2, col=2)
    
    return fig


def render_technical_indicators(df):
    """
    Render the current status of technical indicators.
    
    Args:
        df: DataFrame with indicator data
        
    Returns:
        Markdown string with indicator status
    """
    if df.empty:
        return "No indicator data available"
    
    # Get the latest values
    latest = df.iloc[-1]
    
    # Create indicator status
    indicators = []
    
    # Moving Averages
    if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
        sma_signal = "Bullish" if latest['sma_fast'] > latest['sma_slow'] else "Bearish"
        sma_color = "green" if sma_signal == "Bullish" else "red"
        indicators.append(f"**SMA Crossover:** <span style='color:{sma_color}'>{sma_signal}</span>")
    
    if 'ema_fast' in df.columns and 'ema_slow' in df.columns:
        ema_signal = "Bullish" if latest['ema_fast'] > latest['ema_slow'] else "Bearish"
        ema_color = "green" if ema_signal == "Bullish" else "red"
        indicators.append(f"**EMA Crossover:** <span style='color:{ema_color}'>{ema_signal}</span>")
    
    # RSI
    if 'rsi' in df.columns:
        rsi_value = latest['rsi']
        if rsi_value > 70:
            rsi_signal = "Overbought"
            rsi_color = "red"
        elif rsi_value < 30:
            rsi_signal = "Oversold"
            rsi_color = "green"
        else:
            rsi_signal = "Neutral"
            rsi_color = "gray"
        indicators.append(f"**RSI ({rsi_value:.2f}):** <span style='color:{rsi_color}'>{rsi_signal}</span>")
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd_signal = "Bullish" if latest['macd'] > latest['macd_signal'] else "Bearish"
        macd_color = "green" if macd_signal == "Bullish" else "red"
        indicators.append(f"**MACD:** <span style='color:{macd_color}'>{macd_signal}</span>")
    
    # Bollinger Bands
    if 'close' in df.columns and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        if latest['close'] > latest['bb_upper']:
            bb_signal = "Overbought"
            bb_color = "red"
        elif latest['close'] < latest['bb_lower']:
            bb_signal = "Oversold"
            bb_color = "green"
        else:
            bb_signal = "Neutral"
            bb_color = "gray"
        indicators.append(f"**Bollinger Bands:** <span style='color:{bb_color}'>{bb_signal}</span>")
    
    # Stochastic
    if 'slowk' in df.columns and 'slowd' in df.columns:
        if latest['slowk'] > 80 and latest['slowd'] > 80:
            stoch_signal = "Overbought"
            stoch_color = "red"
        elif latest['slowk'] < 20 and latest['slowd'] < 20:
            stoch_signal = "Oversold"
            stoch_color = "green"
        else:
            stoch_signal = "Neutral"
            stoch_color = "gray"
        indicators.append(f"**Stochastic:** <span style='color:{stoch_color}'>{stoch_signal}</span>")
    
    # ADX
    if 'adx' in df.columns:
        adx_value = latest['adx']
        if adx_value > 25:
            adx_signal = "Strong Trend"
            adx_color = "blue"
        else:
            adx_signal = "Weak Trend"
            adx_color = "gray"
        indicators.append(f"**ADX ({adx_value:.2f}):** <span style='color:{adx_color}'>{adx_signal}</span>")
    
    # Format as a markdown table
    indicator_str = " | ".join(indicators)
    return indicator_str


def create_position_summary(position):
    """
    Create a formatted summary of a position.
    
    Args:
        position: Position dictionary
        
    Returns:
        Markdown string with position summary
    """
    position_type = "Long" if position["position_type"] == 1 else "Short"
    
    # Determine position color
    position_color = "green" if position_type == "Long" else "red"
    
    # Format entry time
    entry_time = position.get("entry_time", "N/A")
    if isinstance(entry_time, (datetime.datetime, pd.Timestamp)):
        entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        entry_time_str = str(entry_time)
    
    # Calculate position duration
    if isinstance(entry_time, (datetime.datetime, pd.Timestamp)):
        duration = datetime.datetime.now() - entry_time
        duration_str = f"{duration.seconds // 60} minutes"
    else:
        duration_str = "N/A"
    
    summary = f"""
    **Position Type:** <span style='color:{position_color}'>{position_type}</span>
    
    **Entry Price:** {format_number(position["entry_price"])}
    
    **Quantity:** {position.get("quantity", 1)}
    
    **Entry Time:** {entry_time_str}
    
    **Duration:** {duration_str}
    
    **Stop Loss:** {format_number(position.get("stop_loss", "N/A"))}
    
    **Take Profit:** {format_number(position.get("take_profit", "N/A"))}
    """
    
    return summary


def create_backtest_summary(backtest_results):
    """
    Create a formatted summary of backtest results.
    
    Args:
        backtest_results: Dictionary with backtest results
        
    Returns:
        Markdown string with backtest summary
    """
    if not backtest_results:
        return "Nenhum resultado de backtest disponível. Execute um backtest primeiro."
    
    # Extract key metrics
    initial_capital = backtest_results.get("initial_capital", 0)
    final_capital = backtest_results.get("final_capital", 0)
    total_return = backtest_results.get("total_return", 0)
    annual_return = backtest_results.get("annual_return", 0)
    max_drawdown = backtest_results.get("max_drawdown", 0)
    sharpe_ratio = backtest_results.get("sharpe_ratio", 0)
    win_rate = backtest_results.get("win_rate", 0)
    profit_factor = backtest_results.get("profit_factor", 0)
    total_trades = backtest_results.get("total_trades", 0)
    
    # Calculate additional metrics
    avg_profit = backtest_results.get("avg_profit", 0)
    avg_loss = backtest_results.get("avg_loss", 0)
    max_consecutive_wins = backtest_results.get("max_consecutive_wins", 0)
    max_consecutive_losses = backtest_results.get("max_consecutive_losses", 0)
    
    # Determine overall performance color
    if total_return > 0:
        performance_color = "green"
        performance_icon = "📈"
    else:
        performance_color = "red"
        performance_icon = "📉"
    
    # Format the summary with more visual styling
    summary = f"""
    <div style="text-align: center; margin-bottom: 25px;">
        <h2 style="color: #2a3b4c;">Resultado do Backtest {performance_icon}</h2>
        <p style="font-size: 14px; color: #555;">Período: {backtest_results.get('start_date', 'N/A')} a {backtest_results.get('end_date', 'N/A')}</p>
    </div>
    
    <div style="border-left: 4px solid {performance_color}; padding-left: 15px; margin-bottom: 20px;">
        <h3 style="color: {performance_color};">Desempenho Geral</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Capital Inicial:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">R$ {format_number(initial_capital)}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Capital Final:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">R$ {format_number(final_capital)}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Retorno Total:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; color: {performance_color}; font-weight: bold;">{total_return:.2%}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Retorno Anual:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{annual_return:.2%}</td>
            </tr>
        </table>
    </div>
    
    <div style="border-left: 4px solid #f39c12; padding-left: 15px; margin-bottom: 20px;">
        <h3 style="color: #d35400;">Métricas de Risco</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Drawdown Máximo:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; color: #e74c3c;">{max_drawdown:.2%}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Sharpe Ratio:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{sharpe_ratio:.2f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Profit Factor:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{profit_factor:.2f}</td>
            </tr>
        </table>
    </div>
    
    <div style="border-left: 4px solid #3498db; padding-left: 15px;">
        <h3 style="color: #2980b9;">Estatísticas de Negociação</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Total de Trades:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{total_trades}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Taxa de Acerto:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{win_rate:.2%}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Lucro Médio:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; color: green;">R$ {format_number(avg_profit)}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Perda Média:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right; color: red;">R$ {format_number(avg_loss)}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #eee;"><strong>Sequência Máxima de Ganhos:</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{max_consecutive_wins}</td>
            </tr>
            <tr>
                <td style="padding: 8px;"><strong>Sequência Máxima de Perdas:</strong></td>
                <td style="padding: 8px; text-align: right;">{max_consecutive_losses}</td>
            </tr>
        </table>
    </div>
    """
    
    return summary
