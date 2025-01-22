# Bitcoin Price Prediction

A Flask-based web application for analyzing and predicting Bitcoin price movements using historical trading data.

## Features
- **Data Analysis**: Explore raw and processed Bitcoin trading data with summary statistics and correlation matrices.
- **Interactive Visualizations**: View price trends, candlestick charts, volatility analysis, and more using Plotly.
- **Price Prediction**: Predict whether the Bitcoin price will increase (Buy) or decrease (Sell) using a Random Forest model.
- **User-Friendly Interface**: Built with Flask, HTML, CSS, anda JavaScript for a seamless user experience.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Abig12/bitcoin-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd bitcoin-price-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   python app.py
   ```
5. Open your browser and go to `http://127.0.0.1:5000` to view the application.

## Usage
- **Homepage**: View summary statistics and correlation matrices for raw and processed data.
- **Visualizations**: Explore interactive charts for price trends, candlestick patterns, volatility, and more.
- **Prediction**: Submit input data (Open, High, Low, Close, Volume) to get a Buy/Sell prediction.

## Dataset
The dataset used in this project is `BitstampData_sample.csv`, which contains historical Bitcoin trading data from the Bitstamp exchange. It includes the following columns:
- `Timestamp`: UNIX timestamp of the trade.
- `Open`, `High`, `Low`, `Close`: Bitcoin price data.
- `Volume_(BTC)`: Trading volume in Bitcoin.
- `Volume_(Currency)`: Trading volume in USD.
- `Weighted_Price`: Weighted average price.

## Technologies Used
- **Backend**: Flask, Python
- **Frontend**: HTML, CSS, JavaScript
- **Data Visualization**: Plotly
- **Machine Learning**: Scikit-learn (Random Forest)

## Screenshots
### Homepage
![Image](https://github.com/user-attachments/assets/39e5adfa-3733-4bd6-8366-d45a40c1888e)

### Visualizations
![Image](https://github.com/user-attachments/assets/f073d091-b550-4a1d-9980-aabd48376f08)

### Prediction Form
![Image](https://github.com/user-attachments/assets/2087346b-b14e-4b34-aa33-9974c4ec3f1f)
