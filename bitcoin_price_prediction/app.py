from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load dataset
DATA_PATH = 'data/BitstampData_sample.csv'
df = pd.read_csv(DATA_PATH)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Store raw data statistics
raw_stats = {
    'description': df.describe(),
    'info': df.info(buf=io.StringIO()),
    'null_counts': df.isnull().sum(),
    'shape': df.shape
}


def preprocess_data(df):
    # Create a copy of raw data
    processed_df = df.copy()

    # Convert timestamp
    processed_df['Timestamp'] = pd.to_datetime(
        processed_df['Timestamp'], unit='s')

    # Add time-based features
    processed_df['Hour'] = processed_df['Timestamp'].dt.hour
    processed_df['Day'] = processed_df['Timestamp'].dt.day
    processed_df['Month'] = processed_df['Timestamp'].dt.month
    processed_df['Year'] = processed_df['Timestamp'].dt.year
    processed_df['DayOfWeek'] = processed_df['Timestamp'].dt.dayofweek

    # Calculate price changes
    processed_df['Price_Change'] = processed_df['Close'].pct_change()
    processed_df['Price_Change_24h'] = processed_df['Close'].pct_change(24)

    # Calculate moving averages
    processed_df['MA7'] = processed_df['Close'].rolling(window=7).mean()
    processed_df['MA30'] = processed_df['Close'].rolling(window=30).mean()

    # Calculate volatility
    processed_df['Volatility'] = processed_df['Price_Change'].rolling(
        window=24).std()

    # Create target variable (1 for price increase, 0 for decrease)
    processed_df['target'] = np.where(
        processed_df['Close'].shift(-1) > processed_df['Close'], 1, 0)

    return processed_df


# Process data
df_processed = preprocess_data(df)

# Store processed data statistics
processed_stats = {
    'description': df_processed.describe(),
    'info': df_processed.info(buf=io.StringIO()),
    'null_counts': df_processed.isnull().sum(),
    'shape': df_processed.shape,
    'new_features': [col for col in df_processed.columns if col not in df.columns]
}


@app.route('/')
def index():
    # Basic statistics
    raw_desc = raw_stats['description']
    processed_desc = processed_stats['description']

    # Correlation matrices
    raw_corr = df.corr().round(2)
    processed_corr = df_processed.corr().round(2)

    # Data info
    data_info = {
        'raw_shape': raw_stats['shape'],
        'processed_shape': processed_stats['shape'],
        'new_features': processed_stats['new_features']
    }

    return render_template('index.html',
                           raw_desc=raw_desc.to_html(
                               classes='table table-striped'),
                           processed_desc=processed_desc.to_html(
                               classes='table table-striped'),
                           raw_corr=raw_corr.to_html(
                               classes='table table-striped'),
                           processed_corr=processed_corr.to_html(
                               classes='table table-striped'),
                           data_info=data_info)


@app.route('/visualizations')
def visualizations():
    # 1. Price Trend with Moving Averages
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df_processed['Timestamp'], y=df_processed['Close'],
                                   name='Close Price', line=dict(color='#2ecc71')))
    fig_trend.add_trace(go.Scatter(x=df_processed['Timestamp'], y=df_processed['MA7'],
                                   name='7-day MA', line=dict(color='#3498db')))
    fig_trend.add_trace(go.Scatter(x=df_processed['Timestamp'], y=df_processed['MA30'],
                                   name='30-day MA', line=dict(color='#e74c3c')))
    fig_trend.update_layout(title='Bitcoin Price Trend with Moving Averages',
                            template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)')

    # 2. Enhanced Candlestick Chart with Volume
    fig_candlestick = go.Figure()
    fig_candlestick.add_trace(go.Candlestick(x=df_processed['Timestamp'],
                                             open=df_processed['Open'],
                                             high=df_processed['High'],
                                             low=df_processed['Low'],
                                             close=df_processed['Close'],
                                             name='OHLC'))
    fig_candlestick.add_trace(go.Bar(x=df_processed['Timestamp'],
                                     y=df_processed['Volume_(BTC)'],
                                     name='Volume',
                                     yaxis='y2'))
    fig_candlestick.update_layout(
        title='Candlestick Chart with Volume',
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # 3. Volatility Over Time
    fig_volatility = px.line(df_processed, x='Timestamp', y='Volatility',
                             title='Bitcoin Price Volatility Over Time')
    fig_volatility.update_layout(template='plotly_dark',
                                 paper_bgcolor='rgba(0,0,0,0)',
                                 plot_bgcolor='rgba(0,0,0,0)')

    # 4. Price Distribution by Year
    fig_dist = px.box(df_processed, x='Year', y='Close',
                      title='Price Distribution by Year')
    fig_dist.update_layout(template='plotly_dark',
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)')

    # 5. Average Price by Hour of Day
    hourly_avg = df_processed.groupby('Hour')['Close'].mean().reset_index()
    fig_hourly = px.line(hourly_avg, x='Hour', y='Close',
                         title='Average Bitcoin Price by Hour of Day')
    fig_hourly.update_layout(template='plotly_dark',
                             paper_bgcolor='rgba(0,0,0,0)',
                             plot_bgcolor='rgba(0,0,0,0)')

    # Convert all figures to JSON
    visualizations = {
        'trend': pio.to_json(fig_trend),
        'candlestick': pio.to_json(fig_candlestick),
        'volatility': pio.to_json(fig_volatility),
        'distribution': pio.to_json(fig_dist),
        'hourly': pio.to_json(fig_hourly)
    }

    return render_template('visualizations.html', visualizations=visualizations)


@app.route('/predict', methods=['POST'])
def predict():
    features = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)']

    # Filter out rows with missing values
    valid_data = df_processed.dropna(subset=features + ['target'])

    X = valid_data[features]
    y = valid_data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    input_data = request.json
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df[features])[0]
    prediction_proba = model.predict_proba(input_df[features])[0]

    return jsonify({
        'prediction': 'Buy' if prediction == 1 else 'Sell',
        'confidence': float(max(prediction_proba)),
        'model_accuracy': float(accuracy)
    })


if __name__ == '__main__':
    app.run(debug=True)
