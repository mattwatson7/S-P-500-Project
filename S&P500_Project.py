import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pytz
import multiprocessing as mp

# Suppress specific warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Function to check if the U.S. stock market is open
def is_market_open():
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False, now
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close, market_close

# Function to fetch data for a single ticker
def fetch_ticker_data(ticker, five_years_ago):
    try:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get('marketCap', 0)  # Fetch market cap
        ticker_data = yf.download(ticker, start=five_years_ago.strftime('%Y-%m-%d'), progress=False, timeout=10)
        ticker_data['Ticker'] = ticker
        ticker_data['Date'] = ticker_data.index
        return ticker_data, market_cap
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None, None

# Function to process each ticker's data (training and prediction)
def process_ticker_data(ticker_data, market_cap, ticker_to_company):
    if ticker_data.empty:  # Check if the DataFrame is empty
        return []

    features = ['Open', 'High', 'Low', 'Volume']
    investment_suggestions = []
    
    ticker = ticker_data['Ticker'].iloc[0]  # Safe because of the empty check
    company_name = ticker_to_company.get(ticker, 'Unknown Company')

    # Filter data for the current ticker
    ticker_data = ticker_data.reset_index(drop=True)

    # Define X and y for the current ticker
    X = ticker_data[features]
    y = ticker_data['Close']

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model using NumPy arrays
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Get today's data for the ticker
    today_data = ticker_data.iloc[-1]  # Get the latest row for today
    
    if not today_data.empty:
        # Get features for today's prediction
        today_features = today_data[features].values.reshape(1, -1)
        today_features_scaled = scaler.transform(today_features)

        # Predict today's closing price
        predicted_close = model.predict(today_features_scaled)[0]
        actual_close = today_data['Close']

        # Generate investment suggestion based on the predicted vs. actual close price
        signal = 'Buy' if predicted_close > actual_close else 'Sell'

        # Get the last date the data was updated for this ticker in mm/dd/yyyy format
        last_date = today_data['Date'].strftime('%m/%d/%Y')

        # Add the suggestion, predicted price, actual price, market cap, and last update date to the list
        investment_suggestions.append({
            'Ticker': f'{ticker} ({company_name})',
            'Signal': signal,
            'Pred. Price': predicted_close,
            'Act. Price': actual_close,
            'Difference': predicted_close - actual_close,  # Difference between predicted and actual price
            'Market Cap': market_cap,
            'Last Updated': last_date
        })

    return investment_suggestions

# Function to fetch historical data for a list of tickers
def fetch_historical_data(tickers, start_date):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(fetch_ticker_data, [(ticker, start_date) for ticker in tickers])
    return [(data, cap) for data, cap in results if data is not None]

# Main function to fetch stock data and make predictions in parallel
def main():
    # Fetch S&P 500 ticker symbols and company names from Wikipedia
    sp500_tickers_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    ticker_to_company = dict(zip(sp500_tickers_df['Symbol'], sp500_tickers_df['Security']))

    # Check if the market is open or closed
    market_open, last_market_close_time = is_market_open()
    if market_open:
        print("MARKET IS CURRENTLY OPEN.")
    else:
        time_since_close = datetime.now(pytz.timezone('US/Eastern')) - last_market_close_time
        hours, remainder = divmod(time_since_close.total_seconds(), 3600)
        minutes = remainder // 60
        print(f"MARKET IS CLOSED. It was last open {int(hours)} hours and {int(minutes)} minutes ago.")

    # Set the start date for fetching historical data (5 years ago)
    five_years_ago = datetime.now() - timedelta(days=5 * 365)

    # Fetch stock data in parallel using the new function
    stock_data_and_caps = fetch_historical_data(ticker_to_company.keys(), five_years_ago)

    # Process the stock data and make predictions in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        investment_suggestions_lists = pool.starmap(process_ticker_data, [(data, cap, ticker_to_company) for data, cap in stock_data_and_caps])

    # Combine all investment suggestions into a single DataFrame
    investment_suggestions = [item for sublist in investment_suggestions_lists for item in sublist]
    investment_suggestions_df = pd.DataFrame(investment_suggestions)

    # Sort the suggestions alphabetically by Ticker
    investment_suggestions_df = investment_suggestions_df.sort_values(by='Ticker', ascending=True)

    # Format numbers with commas and display the DataFrame
    investment_suggestions_df[['Pred. Price', 'Act. Price', 'Market Cap']] = investment_suggestions_df[['Pred. Price', 'Act. Price', 'Market Cap']].applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

    # Display the investment suggestions (Ticker, Buy/Sell Signal, Predicted/Actual Price, Market Cap, Last Updated)
    # print(investment_suggestions_df[['Ticker', 'Signal', 'Pred. Price', 'Act. Price', 'Market Cap', 'Last Updated']].to_string(index=False, col_space=15))

    # Generate recommendations for the top 10 "Buy" and "Sell" companies based on the price difference
    top_10_buy = investment_suggestions_df[investment_suggestions_df['Signal'] == 'Buy'].nlargest(10, 'Difference')
    top_10_sell = investment_suggestions_df[investment_suggestions_df['Signal'] == 'Sell'].nsmallest(10, 'Difference')

    print("\nTop 10 Companies to Buy based on Predicted vs Actual Price Difference:\n")
    print(top_10_buy[['Ticker', 'Pred. Price', 'Act. Price', 'Difference']].to_string(index=False, col_space = 15))

    print("\nTop 10 Companies to Sell based on Predicted vs Actual Price Difference:\n")
    print(top_10_sell[['Ticker', 'Pred. Price', 'Act. Price', 'Difference']].to_string(index=False, col_space = 15))


if __name__ == '__main__':
    main()
