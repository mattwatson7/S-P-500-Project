# S&P 500 Investment Signal Generator

This project is a scalable machine learning system that analyzes all S&P 500 stocks and provides actionable Buy/Sell investment signals based on predicted vs. actual closing prices. It uses data fetched live from Yahoo Finance, builds per-stock linear regression models, and offers a ranked recommendation list of the most promising trades.

---

## Features

- Live data fetching from Yahoo Finance  
- Per-stock Linear Regression model training  
- Buy/Sell signal generation based on price prediction  
- Market capitalization insights included  
- Multiprocessing for faster performance
- Configurable threadpool for downloading data
- Tracks most recent data update for each stock  

---

## Requirements

Make sure you have Python 3.8+ and install the dependencies listed in
`requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mattwatson7/S-P-500-Project.git
   cd S-P-500-Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the project:
   ```bash
   python "S&P500_Project.py"
   ```

---

## Configuration

This script requires no user configuration. It automatically:
- Pulls the latest list of S&P 500 tickers from Wikipedia
- Fetches 5 years of historical data
- Processes each stock individually in parallel
- Outputs Buy/Sell suggestions ranked by predicted return

---

## Output

The script prints two tables to the console:

### Top 10 Companies to Buy:
Based on the largest positive difference between predicted and actual prices.

### Top 10 Companies to Sell:
Based on the largest negative difference between predicted and actual prices.

Each entry shows:
- Ticker and Company Name
- Predicted Price vs. Actual Price
- Buy/Sell Signal
- Market Cap
- Last Data Update

Example:
```
Top 10 Companies to Buy based on Predicted vs Actual Price Difference:

     Ticker             Pred. Price      Act. Price     Difference
------------------     ------------     ------------    ------------
AAPL (Apple Inc.)         186.32           181.75           4.57
NVDA (NVIDIA Corp.)       720.11           705.50          14.61
...
```

---

## How It Works

1. Ticker Fetching – Grabs the latest list of S&P 500 companies.  
2. Data Retrieval – Downloads 5 years of price data for each stock using `yfinance`.  
3. Model Training – Trains a `LinearRegression` model per stock on features: `Open`, `High`, `Low`, `Volume`.  
4. Prediction – Uses today’s data to predict the closing price and compares it to the actual.  
5. Signal Generation – Issues a Buy if the prediction is higher than the actual, Sell otherwise.  
6. Ranking – Ranks all tickers by prediction delta and displays the top 10 buy/sell opportunities.  

---
