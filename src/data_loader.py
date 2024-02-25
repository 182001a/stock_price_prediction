import yfinance as yf

# ticker = yf.Ticker('6752.T')  # パナソニック
ticker = yf.Ticker('7731.T')    # ニコン

hist = ticker.history(start="2008-01-01")
hist = hist[hist['Volume'] != 0]
hist.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True, errors='ignore')

train, test = hist[:-300], hist[-300:]
train.to_csv('../data/nicon_train.csv')
test.to_csv('../data/nicon_test.csv')