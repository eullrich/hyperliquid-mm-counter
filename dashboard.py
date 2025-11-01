"""
Hyperliquid Data Dashboard
Shows aggregated stats by token category: OI, orderbook spread, funding rates
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import time

from config import DB_CONFIG
from token_categories import get_token_category, get_all_categories

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Hyperliquid Category Dashboard"

# Database connection
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

# Fetch category aggregated data
def fetch_category_stats():
    """Fetch latest stats aggregated by category with trading metrics"""
    conn = get_db_connection()

    # Get latest data + 24h volume + momentum + volume indicators
    query = """
    WITH latest_data AS (
        SELECT DISTINCT ON (coin)
            coin,
            timestamp,
            close as price,
            volume as current_volume
        FROM candles
        WHERE interval = '1h'
        ORDER BY coin, timestamp DESC
    ),
    previous_candle AS (
        SELECT DISTINCT ON (coin)
            coin,
            volume as prev_volume
        FROM candles
        WHERE interval = '1h'
          AND timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '2 hours') * 1000
          AND timestamp < EXTRACT(EPOCH FROM NOW() - INTERVAL '30 minutes') * 1000
        ORDER BY coin, timestamp DESC
    ),
    volume_ma AS (
        SELECT
            coin,
            AVG(volume) as vol_ma_20
        FROM (
            SELECT DISTINCT ON (coin, timestamp)
                coin,
                timestamp,
                volume
            FROM candles
            WHERE interval = '1h'
              AND timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '21 hours') * 1000
            ORDER BY coin, timestamp DESC
        ) sub
        GROUP BY coin
    ),
    price_24h_ago AS (
        SELECT DISTINCT ON (coin)
            coin,
            close as price_24h
        FROM candles
        WHERE interval = '1h'
          AND timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '25 hours') * 1000
          AND timestamp < EXTRACT(EPOCH FROM NOW() - INTERVAL '23 hours') * 1000
        ORDER BY coin, timestamp DESC
    ),
    volume_24h AS (
        SELECT
            coin,
            SUM(volume) as volume_24h
        FROM candles
        WHERE interval = '1h'
          AND timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours') * 1000
        GROUP BY coin
    ),
    volatility_24h AS (
        SELECT
            coin,
            STDDEV(close) / AVG(close) as volatility_pct
        FROM candles
        WHERE interval = '1h'
          AND timestamp > EXTRACT(EPOCH FROM NOW() - INTERVAL '24 hours') * 1000
        GROUP BY coin
    ),
    latest_oi AS (
        SELECT DISTINCT ON (coin)
            coin,
            value as oi,
            timestamp
        FROM open_interest
        ORDER BY coin, timestamp DESC
    ),
    latest_spread AS (
        SELECT DISTINCT ON (coin)
            coin,
            spread_bps,
            timestamp
        FROM orderbook
        ORDER BY coin, timestamp DESC
    ),
    latest_funding AS (
        SELECT DISTINCT ON (coin)
            coin,
            rate as funding_rate,
            timestamp
        FROM funding_rates
        ORDER BY coin, timestamp DESC
    )
    SELECT
        ld.coin,
        ld.price,
        ld.current_volume,
        COALESCE(pc.prev_volume, ld.current_volume) as prev_volume,
        COALESCE(vma.vol_ma_20, ld.current_volume) as vol_ma_20,
        COALESCE(p24.price_24h, ld.price) as price_24h,
        COALESCE(v24.volume_24h, 0) as volume_24h,
        COALESCE(vol.volatility_pct, 0) as volatility_pct,
        COALESCE(lo.oi, 0) as oi,
        COALESCE(ls.spread_bps, 0) as spread_bps,
        COALESCE(lf.funding_rate, 0) as funding_rate,
        ld.timestamp as price_ts
    FROM latest_data ld
    LEFT JOIN previous_candle pc ON ld.coin = pc.coin
    LEFT JOIN volume_ma vma ON ld.coin = vma.coin
    LEFT JOIN price_24h_ago p24 ON ld.coin = p24.coin
    LEFT JOIN volume_24h v24 ON ld.coin = v24.coin
    LEFT JOIN volatility_24h vol ON ld.coin = vol.coin
    LEFT JOIN latest_oi lo ON ld.coin = lo.coin
    LEFT JOIN latest_spread ls ON ld.coin = ls.coin
    LEFT JOIN latest_funding lf ON ld.coin = lf.coin
    ORDER BY ld.coin;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate price change %
    df['price_change_24h_pct'] = ((df['price'] - df['price_24h']) / df['price_24h']) * 100

    # Calculate Volume Rate of Change (VROC)
    df['vroc'] = ((df['current_volume'] - df['prev_volume']) / df['prev_volume'].replace(0, 1)) * 100

    # Calculate Volume Ratio vs MA
    df['vol_ratio'] = df['current_volume'] / df['vol_ma_20'].replace(0, 1)

    # Calculate Volume Momentum Score
    df['vol_momentum'] = df['vroc'] * df['vol_ratio']

    # Add category column
    df['category'] = df['coin'].apply(get_token_category)

    return df

def aggregate_by_category(df):
    """Aggregate stats by category"""
    agg_stats = df.groupby('category').agg({
        'coin': 'count',
        'oi': 'sum',
        'volume_24h': 'sum',
        'spread_bps': 'mean',
        'funding_rate': 'mean',
        'volatility_pct': 'mean',
        'price_change_24h_pct': 'mean',
        'vroc': 'mean',
        'vol_ratio': 'mean',
        'vol_momentum': 'mean'
    }).reset_index()

    agg_stats.columns = ['category', 'token_count', 'total_oi', 'total_volume_24h', 'avg_spread_bps', 'avg_funding_rate', 'avg_volatility_pct', 'avg_price_change_24h_pct', 'avg_vroc', 'avg_vol_ratio', 'avg_vol_momentum']
    agg_stats = agg_stats.sort_values('total_volume_24h', ascending=False)

    return agg_stats

def fetch_volume_price_data(hours=24, interval='1h'):
    """Fetch volume and price data by category"""
    conn = get_db_connection()

    cutoff_time = int((time.time() - hours * 3600) * 1000)

    query = f"""
    SELECT
        coin,
        timestamp,
        close as price,
        volume,
        open,
        high,
        low
    FROM candles
    WHERE timestamp > {cutoff_time}
      AND interval = '{interval}'
    ORDER BY timestamp;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df['category'] = df['coin'].apply(get_token_category)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

def fetch_spread_timeseries(hours=24):
    """Fetch spread timeseries by category"""
    conn = get_db_connection()

    cutoff_time = int((time.time() - hours * 3600) * 1000)

    query = f"""
    SELECT
        coin,
        timestamp,
        spread_bps
    FROM orderbook
    WHERE timestamp > {cutoff_time}
    ORDER BY timestamp;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df['category'] = df['coin'].apply(get_token_category)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Aggregate by category and time
    df_agg = df.groupby(['category', 'datetime'])['spread_bps'].mean().reset_index()

    return df_agg

# Layout
app.layout = html.Div([
    html.H1("Hyperliquid Category Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),

    # Refresh interval
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every 60 seconds
        n_intervals=0
    ),

    # Last updated timestamp
    html.Div(id='last-updated', style={'textAlign': 'center', 'marginBottom': 20}),

    # Summary cards - Row 1
    html.Div([
        html.Div([
            html.H3("24h Volume", style={'textAlign': 'center'}),
            html.H2(id='total-volume', style={'textAlign': 'center', 'color': '#2196F3'})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '1%', 'borderRadius': '5px'}),

        html.Div([
            html.H3("Total OI", style={'textAlign': 'center'}),
            html.H2(id='total-oi', style={'textAlign': 'center', 'color': '#00BCD4'})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '1%', 'borderRadius': '5px'}),

        html.Div([
            html.H3("Avg Volatility", style={'textAlign': 'center'}),
            html.H2(id='avg-volatility', style={'textAlign': 'center', 'color': '#FF5722'})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '1%', 'borderRadius': '5px'}),

        html.Div([
            html.H3("Avg 24h Change", style={'textAlign': 'center'}),
            html.H2(id='avg-price-change', style={'textAlign': 'center', 'color': '#4CAF50'})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '1%', 'borderRadius': '5px'}),
    ], style={'marginBottom': 10}),

    # Summary cards - Row 2
    html.Div([
        html.Div([
            html.H3("Avg Vol Momentum", style={'textAlign': 'center'}),
            html.H2(id='avg-vol-momentum', style={'textAlign': 'center', 'color': '#FF5722'})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '1%', 'borderRadius': '5px'}),

        html.Div([
            html.H3("Avg Spread (bps)", style={'textAlign': 'center'}),
            html.H2(id='avg-spread', style={'textAlign': 'center', 'color': '#FF9800'})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '1%', 'borderRadius': '5px'}),

        html.Div([
            html.H3("Avg Funding Rate", style={'textAlign': 'center'}),
            html.H2(id='avg-funding', style={'textAlign': 'center', 'color': '#9C27B0'})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '1%', 'borderRadius': '5px'}),

        html.Div([
            html.H3("Active Tokens", style={'textAlign': 'center'}),
            html.H2(id='total-tokens', style={'textAlign': 'center', 'color': '#607D8B'})
        ], style={'width': '23%', 'display': 'inline-block', 'padding': '20px', 'border': '1px solid #ddd', 'margin': '1%', 'borderRadius': '5px'}),
    ], style={'marginBottom': 30}),

    # Charts row 1 - Volume and Price Change
    html.Div([
        html.Div([
            dcc.Graph(id='volume-by-category')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),

        html.Div([
            dcc.Graph(id='price-change-by-category')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),
    ]),

    # Charts row 2 - Volatility and OI
    html.Div([
        html.Div([
            dcc.Graph(id='volatility-by-category')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),

        html.Div([
            dcc.Graph(id='oi-by-category')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),
    ]),

    # Charts row 3 - Volume Momentum and VROC
    html.Div([
        html.Div([
            dcc.Graph(id='vol-momentum-by-category')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),

        html.Div([
            dcc.Graph(id='vroc-by-category')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),
    ]),

    # Charts row 4 - Spread and Funding
    html.Div([
        html.Div([
            dcc.Graph(id='spread-by-category')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),

        html.Div([
            dcc.Graph(id='funding-by-category')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),
    ]),

    # Time series charts with controls
    html.H2("Volume & Price Trends", style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20}),

    # Controls for charts
    html.Div([
        html.Label("Select Timeframe:", style={'fontWeight': 'bold', 'marginRight': 10}),
        dcc.Dropdown(
            id='timeframe-dropdown',
            options=[
                {'label': '6 Hours', 'value': 6},
                {'label': '12 Hours', 'value': 12},
                {'label': '24 Hours', 'value': 24},
                {'label': '3 Days', 'value': 72},
                {'label': '7 Days', 'value': 168},
            ],
            value=24,
            style={'width': '150px', 'display': 'inline-block', 'marginRight': 20}
        ),
        html.Label("Candle Interval:", style={'fontWeight': 'bold', 'marginRight': 10}),
        dcc.Dropdown(
            id='interval-dropdown',
            options=[
                {'label': '5 Minutes', 'value': '5m'},
                {'label': '1 Hour', 'value': '1h'},
                {'label': '4 Hours', 'value': '4h'},
                {'label': '1 Day', 'value': '1d'},
            ],
            value='1h',
            style={'width': '150px', 'display': 'inline-block', 'marginRight': 20}
        ),
        html.Label("Category Filter:", style={'fontWeight': 'bold', 'marginRight': 10}),
        dcc.Dropdown(
            id='category-filter',
            options=[{'label': 'All Categories', 'value': 'all'}],
            value='all',
            multi=True,
            style={'width': '300px', 'display': 'inline-block'}
        ),
    ], style={'textAlign': 'center', 'marginBottom': 20, 'padding': '20px', 'backgroundColor': '#f5f5f5', 'borderRadius': '5px'}),

    html.Div([
        html.Div([
            dcc.Graph(id='volume-timeseries')
        ], style={'width': '100%', 'padding': '1%'}),
    ]),

    html.Div([
        html.Div([
            dcc.Graph(id='price-volume-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),

        html.Div([
            dcc.Graph(id='spread-timeseries')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '1%'}),
    ]),

    # Data table
    html.H2("Category Summary Table", style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20}),
    html.Div([
        dash_table.DataTable(
            id='category-table',
            columns=[
                {'name': 'Category', 'id': 'category'},
                {'name': '# Tokens', 'id': 'token_count'},
                {'name': '24h Volume', 'id': 'total_volume_24h', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                {'name': 'Vol Momentum', 'id': 'avg_vol_momentum', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'VROC %', 'id': 'avg_vroc', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                {'name': 'Vol Ratio', 'id': 'avg_vol_ratio', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'Total OI', 'id': 'total_oi', 'type': 'numeric', 'format': {'specifier': ',.0f'}},
                {'name': 'Avg Volatility %', 'id': 'avg_volatility_pct', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                {'name': 'Avg 24h Change %', 'id': 'avg_price_change_24h_pct', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'Avg Spread (bps)', 'id': 'avg_spread_bps', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                {'name': 'Avg Funding Rate', 'id': 'avg_funding_rate', 'type': 'numeric', 'format': {'specifier': '.6f'}},
            ],
            style_table={'overflowX': 'auto', 'margin': '0 auto', 'width': '95%'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': '#2196F3',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'
                },
                {
                    'if': {
                        'filter_query': '{avg_price_change_24h_pct} > 0',
                        'column_id': 'avg_price_change_24h_pct'
                    },
                    'color': '#4CAF50',
                    'fontWeight': 'bold'
                },
                {
                    'if': {
                        'filter_query': '{avg_price_change_24h_pct} < 0',
                        'column_id': 'avg_price_change_24h_pct'
                    },
                    'color': '#F44336',
                    'fontWeight': 'bold'
                }
            ],
            sort_action='native',
            filter_action='native',
        )
    ], style={'marginBottom': 50})
], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})

# Callback to populate category filter options
@app.callback(
    Output('category-filter', 'options'),
    [Input('interval-component', 'n_intervals')]
)
def update_category_options(n):
    categories = get_all_categories()
    options = [{'label': 'All Categories', 'value': 'all'}]
    options.extend([{'label': cat, 'value': cat} for cat in sorted(categories)])
    return options

# Main dashboard callback
@app.callback(
    [
        Output('last-updated', 'children'),
        Output('total-volume', 'children'),
        Output('total-oi', 'children'),
        Output('avg-volatility', 'children'),
        Output('avg-price-change', 'children'),
        Output('avg-vol-momentum', 'children'),
        Output('avg-spread', 'children'),
        Output('avg-funding', 'children'),
        Output('total-tokens', 'children'),
        Output('volume-by-category', 'figure'),
        Output('price-change-by-category', 'figure'),
        Output('volatility-by-category', 'figure'),
        Output('oi-by-category', 'figure'),
        Output('vol-momentum-by-category', 'figure'),
        Output('vroc-by-category', 'figure'),
        Output('spread-by-category', 'figure'),
        Output('funding-by-category', 'figure'),
        Output('spread-timeseries', 'figure'),
        Output('category-table', 'data'),
    ],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Fetch data
    df = fetch_category_stats()
    agg_df = aggregate_by_category(df)

    # Summary stats
    total_volume = df['volume_24h'].sum()
    total_oi = df['oi'].sum()
    avg_volatility = df['volatility_pct'].mean()
    avg_price_change = df['price_change_24h_pct'].mean()
    avg_vol_momentum = df['vol_momentum'].mean()
    avg_spread = df['spread_bps'].mean()
    avg_funding = df['funding_rate'].mean()
    total_tokens = len(df)

    last_updated = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Volume by category bar chart
    fig_volume = px.bar(
        agg_df,
        x='category',
        y='total_volume_24h',
        title='24h Volume by Category',
        labels={'total_volume_24h': '24h Volume', 'category': 'Category'},
        color='total_volume_24h',
        color_continuous_scale='Blues'
    )
    fig_volume.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Price change by category
    fig_price_change = px.bar(
        agg_df,
        x='category',
        y='avg_price_change_24h_pct',
        title='Average 24h Price Change by Category',
        labels={'avg_price_change_24h_pct': '24h Change %', 'category': 'Category'},
        color='avg_price_change_24h_pct',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0
    )
    fig_price_change.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Volatility by category
    fig_volatility = px.bar(
        agg_df,
        x='category',
        y='avg_volatility_pct',
        title='Average Volatility by Category',
        labels={'avg_volatility_pct': 'Volatility %', 'category': 'Category'},
        color='avg_volatility_pct',
        color_continuous_scale='Reds'
    )
    fig_volatility.update_layout(showlegend=False, xaxis_tickangle=-45)

    # OI by category bar chart
    fig_oi = px.bar(
        agg_df,
        x='category',
        y='total_oi',
        title='Total Open Interest by Category',
        labels={'total_oi': 'Open Interest', 'category': 'Category'},
        color='total_oi',
        color_continuous_scale='Teal'
    )
    fig_oi.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Spread by category
    fig_spread = px.bar(
        agg_df,
        x='category',
        y='avg_spread_bps',
        title='Average Spread (bps) by Category',
        labels={'avg_spread_bps': 'Spread (bps)', 'category': 'Category'},
        color='avg_spread_bps',
        color_continuous_scale='Oranges'
    )
    fig_spread.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Volume Momentum by category
    fig_vol_momentum = px.bar(
        agg_df,
        x='category',
        y='avg_vol_momentum',
        title='Volume Momentum Score by Category',
        labels={'avg_vol_momentum': 'Vol Momentum (VROC × Vol Ratio)', 'category': 'Category'},
        color='avg_vol_momentum',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0
    )
    fig_vol_momentum.update_layout(showlegend=False, xaxis_tickangle=-45)

    # VROC by category
    fig_vroc = px.bar(
        agg_df,
        x='category',
        y='avg_vroc',
        title='Volume Rate of Change (VROC %) by Category',
        labels={'avg_vroc': 'VROC %', 'category': 'Category'},
        color='avg_vroc',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0
    )
    fig_vroc.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Spread by category
    fig_spread = px.bar(
        agg_df,
        x='category',
        y='avg_spread_bps',
        title='Average Spread (bps) by Category',
        labels={'avg_spread_bps': 'Spread (bps)', 'category': 'Category'},
        color='avg_spread_bps',
        color_continuous_scale='Oranges'
    )
    fig_spread.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Funding by category
    fig_funding = px.bar(
        agg_df,
        x='category',
        y='avg_funding_rate',
        title='Average Funding Rate by Category',
        labels={'avg_funding_rate': 'Funding Rate', 'category': 'Category'},
        color='avg_funding_rate',
        color_continuous_scale='Purples'
    )
    fig_funding.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Spread time series
    spread_ts = fetch_spread_timeseries(hours=24)
    fig_spread_ts = px.line(
        spread_ts,
        x='datetime',
        y='spread_bps',
        color='category',
        title='Spread Trend (24h)',
        labels={'spread_bps': 'Spread (bps)', 'datetime': 'Time'}
    )
    fig_spread_ts.update_layout(xaxis_title='Time', yaxis_title='Spread (bps)')

    # Format table data
    table_data = agg_df.to_dict('records')

    return (
        last_updated,
        f"${total_volume:,.0f}",
        f"${total_oi:,.0f}",
        f"{avg_volatility:.2%}",
        f"{avg_price_change:+.2f}%",
        f"{avg_vol_momentum:.2f}",
        f"{avg_spread:.2f}",
        f"{avg_funding:.4%}",
        str(total_tokens),
        fig_volume,
        fig_price_change,
        fig_volatility,
        fig_oi,
        fig_vol_momentum,
        fig_vroc,
        fig_spread,
        fig_funding,
        fig_spread_ts,
        table_data
    )

# Callback for interactive volume/price charts
@app.callback(
    [
        Output('volume-timeseries', 'figure'),
        Output('price-volume-chart', 'figure'),
    ],
    [
        Input('timeframe-dropdown', 'value'),
        Input('interval-dropdown', 'value'),
        Input('category-filter', 'value'),
    ]
)
def update_volume_charts(hours, interval, categories):
    from plotly.subplots import make_subplots

    # Fetch data
    df = fetch_volume_price_data(hours=hours, interval=interval)

    # Filter by categories
    if categories and 'all' not in categories:
        df = df[df['category'].isin(categories)]

    # If no data, return empty charts
    if len(df) == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return empty_fig, empty_fig

    # Aggregate by category and time
    vol_agg = df.groupby(['category', 'datetime']).agg({
        'volume': 'sum',
        'price': 'mean'
    }).reset_index()

    # Volume Line Chart
    fig_vol = px.line(
        vol_agg,
        x='datetime',
        y='volume',
        color='category',
        title=f'Trading Volume by Category ({hours}h - {interval} candles)',
        labels={'volume': 'Volume', 'datetime': 'Time'}
    )
    fig_vol.update_layout(
        xaxis_title='Time',
        yaxis_title='Volume',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    # Price/Volume Overlay Chart with dual axis
    # Get top 5 categories by volume
    cat_avg_vol = vol_agg.groupby('category')['volume'].mean().sort_values(ascending=False)
    top_cats = cat_avg_vol.head(5).index.tolist()
    df_top = vol_agg[vol_agg['category'].isin(top_cats)]

    fig_price_vol = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'Average Price Trend (Top 5 Categories)', 'Volume')
    )

    # Add price traces for top categories
    for cat in top_cats:
        cat_data = df_top[df_top['category'] == cat]
        fig_price_vol.add_trace(
            go.Scatter(
                x=cat_data['datetime'],
                y=cat_data['price'],
                name=cat,
                mode='lines',
                line=dict(width=2)
            ),
            row=1, col=1
        )

    # Add volume bars
    for cat in top_cats:
        cat_data = df_top[df_top['category'] == cat]
        fig_price_vol.add_trace(
            go.Bar(
                x=cat_data['datetime'],
                y=cat_data['volume'],
                name=f'{cat} Vol',
                showlegend=False,
                opacity=0.6
            ),
            row=2, col=1
        )

    fig_price_vol.update_layout(
        height=600,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig_price_vol.update_xaxes(title_text='Time', row=2, col=1)
    fig_price_vol.update_yaxes(title_text='Price', row=1, col=1)
    fig_price_vol.update_yaxes(title_text='Volume', row=2, col=1)

    return fig_vol, fig_price_vol

if __name__ == '__main__':
    print("Starting Hyperliquid Category Dashboard...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run_server(debug=True, host='0.0.0.0', port=8050)
