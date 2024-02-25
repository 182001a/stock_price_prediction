import yfinance as yf

ticker = yf.Ticker('7731.T')    # ニコン
print(ticker.history(start="2008-01-01"))