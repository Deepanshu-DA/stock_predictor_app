# Stock Sentiment Prediction Flask App

This Flask web application predicts stock price trends by combining historical stock data from Yahoo Finance and sentiment analysis from recent news headlines using FinBERT.

## Features

- Fetches 6 months of historical stock price data using yFinance
- Retrieves recent news headlines related to the stock symbol via NewsAPI
- Applies FinBERT model to analyze sentiment of news headlines
- Predicts next price movement based on sentiment score
- Displays stock symbol, recent news, sentiment score, and predicted price
- Simple and clean UI with Flask templates

## Requirements

- Python 3.8+
- Flask
- Flask-CORS
- yfinance
- newsapi-python
- transformers
- torch
- numpy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stock-sentiment-flask.git
   cd stock-sentiment-flask
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Add your NewsAPI API key in the app.py:
   ```bash
   newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')
## Usage

Run the Flask app:
```bash
python app.py
```

Open your browser at http://127.0.0.1:5000 and enter a stock symbol to see the prediction.

## How It Works

- Downloads 6 months of closing prices for the requested stock symbol.
- Retrieves up to 5 recent news headlines related to the stock.
- Runs FinBERT sentiment analysis on headlines to calculate a sentiment score.
- Uses the sentiment score to forecast a simple adjusted predicted price.

## Notes

-The prediction model is a simple heuristic for demonstration purposes.
-NewsAPI has request limits depending on your subscription plan.
- FinBERT model loading requires PyTorch and can take some time on first run.

## License
MIT License
