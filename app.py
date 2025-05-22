from flask import Flask, render_template, request
from flask_cors import CORS
import yfinance as yf
import datetime
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = Flask(__name__)
CORS(app)

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# NewsAPI (insert your key)
newsapi = NewsApiClient(api_key='YOUR_NEWSAPI_KEY')

def get_sentiment_score(headlines):
    if not headlines:
        return 0.0
    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
    return round(np.mean(scores[:, 2] - scores[:, 0]), 2)  # Positive - Negative

def get_news_headlines(symbol):
    try:
        articles = newsapi.get_everything(q=symbol, language='en', sort_by='publishedAt', page_size=5)
        return [a['title'] for a in articles['articles']]
    except:
        return []

@app.route('/', methods=['GET', 'POST'])
def index():
    stock_symbol = None
    news_headlines = []
    sentiment_score = 0.0
    predicted_price = None
    error_message = None

    if request.method == 'POST':
        stock_symbol = request.form.get('stockSymbol', '').strip().upper()
        try:
            df = yf.download(stock_symbol, period="6mo")
            if df.empty:
                raise ValueError("No stock data found.")
            last_price = df["Close"].iloc[-1]

            # News & sentiment
            news_headlines = get_news_headlines(stock_symbol)
            sentiment_score = get_sentiment_score(news_headlines)

            # Simple forecast
            predicted_price = round(last_price * (1 + sentiment_score * 0.05), 2)

        except Exception as e:
            error_message = str(e)

    return render_template("index.html",
                           stock_symbol=stock_symbol,
                           news_headlines=news_headlines,
                           sentiment_score=sentiment_score,
                           predicted_price=predicted_price,
                           error_message=error_message,
                           current_year=datetime.datetime.now().year)

if __name__ == '__main__':
    app.run(debug=True)
