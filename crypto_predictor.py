import requests
from sklearn.linear_model import LinearRegression
import numpy as np

def fetch_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=7"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        prices = [point[1] for point in data['prices']]
        return prices[-50:]
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return []

def predict_next_price(prices):
    if len(prices) < 2:
        return None
    X = np.array(range(len(prices))).reshape(-1, 1)
    y = np.array(prices)
    model = LinearRegression()
    model.fit(X, y)
    next_index = len(prices)
    next_price = model.predict([[next_index]])[0]
    return next_price

def main():
    print("Fetching Bitcoin price data...")
    prices = fetch_crypto_data()
    
    if prices:
        next_price = predict_next_price(prices)
        current_price = prices[-1]
        print(f"Current Bitcoin Price: ${current_price:.2f}")
        print(f"Predicted Next Price: ${next_price:.2f}")
        if next_price > current_price:
            print("Trend: Bullish (Price may increase)")
        elif next_price < current_price:
            print("Trend: Bearish (Price may decrease)")
        else:
            print("Trend: Neutral")
    else:
        print("No data available for prediction.")

if __name__ == "__main__":
    main()