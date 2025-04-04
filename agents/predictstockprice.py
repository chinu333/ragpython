from datetime import timedelta
from datetime import date as dt
import pandas as pd
import yfinance as yf
import numpy as np
from dotenv import load_dotenv
import os
import requests


env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

stock_prediction_model_url = os.getenv("STOCK_PREDICTION_MODEL_URL")
stock_prediction_key = os.getenv("STOCK_PREDICTION_MODEL_KEY")

def predict_stock_price(future_date):
    ticker = "MSFT"
    data = yf.download(ticker, period="1y", multi_level_index = False)
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    # Display the first few rows of the DataFrame
    data["Previous Close"] = data["Close"].shift(1)
    data["High - Low"] = data["High"] - data["Low"]
    data["Open - Close"] = data["Open"] - data["Close"]
    data["Volume"] = data["Volume"]
    data["MA10"] = data["Close"].rolling(window=10).mean()
    data["MA50"] = data["Close"].rolling(window=20).mean()
    data = data.dropna()
    # print(data.tail())

    last_date = data.index[-1]  # Assuming 'Date' is the index of the DataFrame
    # future_dates = [last_date + timedelta(days=i) for i in range(1, 10)]
    # print("Last Date: ", last_date)
    given_date_obj = dt.fromisoformat(future_date)
    difference = given_date_obj - last_date.date()
    # print("Difference in days: ", difference.days)

    future_dates = [last_date + timedelta(days=i) for i in range(1, difference.days + 1)]
    
    last_row = data.iloc[-1]
    # print("Last Row Data: ", last_row)
    # print("Future Data: ") 
    # print(future_data)
    current_close = last_row['Close']
    high_low = last_row['High'] - last_row['Low']
    open_close = last_row['Open'] - last_row['Close']
    volume = last_row['Volume']
    prediction = ""
    
    for date in future_dates:
        dates = [date]
        future_data = pd.DataFrame(index=dates, columns=['Previous Close', 'High - Low', 'Open - Close', 'Volume', 'MA10', 'MA50'])
        future_data.loc[date, 'Previous Close'] = current_close
        future_data.loc[date, 'High - Low'] = high_low
        future_data.loc[date, 'Open - Close'] = open_close
        future_data.loc[date, 'Volume'] = volume
        future_data.loc[date, 'MA10'] = data["Close"].rolling(window=10).mean().iloc[-1]
        future_data.loc[date, 'MA50'] = data["Close"].rolling(window=50).mean().iloc[-1]

        headers = {
            "Content-Type": "application/json",
            "x-functions-key": stock_prediction_key
        }

        payload = {
            'features': [future_data['Previous Close'].values[0], 
                        future_data['High - Low'].values[0], 
                        future_data['Open - Close'].values[0], 
                        future_data['Volume'].values[0], 
                        future_data['MA10'].values[0], 
                        future_data['MA50'].values[0]]
        }

        print("Payload: ", payload)

        response = requests.post(stock_prediction_model_url, headers=headers, json=payload)
        resstr = str(response.content)
        print("Response Status Code: ", response.status_code)
        print("Response Content: ", resstr)
        index_of_cl = resstr.index(":")
        index_of_curly_brace_close = resstr.index("}")
        predicted_value = resstr[(index_of_cl + 1):(index_of_curly_brace_close - 1)]
        print("Predicted Value: ", predicted_value)

        
        current_close = float(predicted_value)

        maxhigh = data["High"].rolling(window=30).max().iloc[-1]
        minhigh = data["High"].rolling(window=30).min().iloc[-1]
        maxlow = data["Low"].rolling(window=30).max().iloc[-1]
        minlow = data["Low"].rolling(window=30).min().iloc[-1]

        high_low = np.random.uniform(minhigh, maxhigh) - np.random.uniform(minlow, maxlow)

        maxopen = data["Open"].rolling(window=30).max().iloc[-1]
        minopen = data["Open"].rolling(window=30).min().iloc[-1]
        maxclose = data["Close"].rolling(window=30).max().iloc[-1]
        minclose = data["Close"].rolling(window=30).min().iloc[-1]

        open_close = np.random.uniform(minopen, maxopen) - np.random.uniform(minclose, maxclose)

        prediction = f"Ticker: {ticker}, Date: {date.strftime('%Y-%m-%d')}, Predicted Close Price: ${predicted_value}."
    
    print(prediction)     
    return prediction + " The MSE (Mean Squared Error) of the model is 4.836365252614313."