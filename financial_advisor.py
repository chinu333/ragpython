import yfinance as yf 


def get_historical_stock_price(ticker):

    # Get the last 5 years of data
    stock_data = yf.download(ticker, period="5y")
    print("Ticker sysbol :: ", ticker) 

    # Print the closing prices for the last 5 years 
    # print(stock_data["Close"].to_json())
    return stock_data["Close"].to_json()