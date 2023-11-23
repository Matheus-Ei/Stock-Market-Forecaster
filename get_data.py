# Imports
import yahooquery as yq
import yfinance as yf
import pandas as pd
import requests

def get_events(ticker, past_days = 1):
    # Get the corporate events for the ticker
    tk = yq.Ticker(ticker)
    df = tk.corporate_events
    df_reset = df.reset_index() # Reset the index to get the last row

    description = df_reset['description'] # Get the description
    description_length = len(description) # Get the length of the description

    last_description = description[description_length - past_days] # Get the description
    return last_description # Return the last description


def get_news(ticker):
    url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={}&interval=5min&apikey=1RRNP1ITPIQ7QFV8".format(ticker)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        data = data['feed']

        sentiment = []

        for e in data:
            if e['overall_sentiment_score'] > 0.15:
                sentiment.append(1)
            elif e['overall_sentiment_score'] > 0.35:
                sentiment.append(2)
            elif e['overall_sentiment_score'] < -0.15:
                sentiment.append(-1)
            elif e['overall_sentiment_score'] < -0.35:
                sentiment.append(-2)
            elif e['overall_sentiment_score'] < 0.15 and e['overall_sentiment_score'] > -0.15:
                sentiment.append(0)

        sentiment = sum(sentiment) / len(sentiment)
        return sentiment
    else:
        return None


def get_direction(ticker):
    # Get the corporate events for the ticker
    tk = yq.Ticker(ticker)
    df = tk.technical_insights
    direction = df[ticker]['instrumentInfo']['technicalEvents']
    short_term = direction['shortTermOutlook']
    intermediate_term = direction['intermediateTermOutlook']
    long_term = direction['longTermOutlook']

    short_term_sector = short_term['sectorDirection']
    short_term_sector_score = short_term['sectorScore']
    short_term_stock = short_term['direction']
    short_term_stock_score = short_term['score']

    intermediate_term_sector = intermediate_term['sectorDirection']
    intermediate_term_sector_score = intermediate_term['sectorScore']
    intermediate_term_stock = intermediate_term['direction']
    intermediate_term_stock_score = intermediate_term['score']

    long_term_sector = long_term['sectorDirection']
    long_term_sector_score = long_term['sectorScore']
    long_term_stock = long_term['direction']
    long_term_stock_score = long_term['score']

    return [{'stock': [short_term_sector, short_term_sector_score], 'market': [short_term_stock, short_term_stock_score]}, 
            {'stock': [intermediate_term_sector, intermediate_term_sector_score], 'market': [intermediate_term_stock, intermediate_term_stock_score]}, 
            {'stock': [long_term_sector, long_term_sector_score], 'market': [long_term_stock, long_term_stock_score]}]


def get_recomendation(ticker):
    # Get the corporate events for the ticker
    tk = yq.Ticker(ticker)
    df = tk.technical_insights
    recomendation = df[ticker]['recommendation']['rating']
    if recomendation == "BUY":
        recomendation = 1
    elif recomendation == "HOLD":
        recomendation = 0
    elif recomendation == "SELL":
        recomendation = -1

    return recomendation


def get_basic(ticker):
    data = yf.download(ticker, start=None, end=None, interval="1d") # Get Data From Yahoo Finance
    data.reset_index(inplace=True) # Remove the Index

    i = 365 # Get the last 90 days
    open, close, high, low, volume = [], [], [], [], []

    while i >= 1:
        row = data.iloc[-i] # Get the last row
        open.append(row['Open'])
        close.append(row['Close'])
        low.append(row['Low'])
        high.append(row['High'])
        volume.append(row['Volume'])
        i = i - 1

    return {'open': open, 'close': close, 'high': high, 'low': low, 'volume': volume}


def get_average(ticker):
    basic = get_basic(ticker)
    close = basic['close'] # Get the close prices

    sma7 = pd.Series(close).rolling(window=7).mean() # Calculate the 7 day SMA
    sma14 = pd.Series(close).rolling(window=14).mean() # Calculate the 14 day SMA
    sma30 = pd.Series(close).rolling(window=30).mean() # Calculate the 30 day SMA
    sma60 = pd.Series(close).rolling(window=60).mean() # Calculate the 60 day SMA
    sma90 = pd.Series(close).rolling(window=90).mean() # Calculate the 90 day SMA
    sma120 = pd.Series(close).rolling(window=120).mean() # Calculate the 120 day SMA

    return {'sma7': sma7, 'sma14': sma14, 'sma30': sma30, 'sma60': sma60, 'sma90': sma90, 'sma120': sma120}


def get_expo_average(ticker):
    basic = get_basic(ticker)
    close = basic['close'] # Get the close prices

    ema7 = pd.Series(close).ewm(span=7, adjust=True).mean() # Calculate the 7 day EMA
    ema12 = pd.Series(close).ewm(span=12, adjust=True).mean() # Calculate the 12 day EMA
    ema26 = pd.Series(close).ewm(span=26, adjust=True).mean() # Calculate the 26 day EMA
    ema60 = pd.Series(close).ewm(span=60, adjust=True).mean() # Calculate the 60 day EMA
    ema90 = pd.Series(close).ewm(span=90, adjust=True).mean() # Calculate the 90 day EMA
    ema120 = pd.Series(close).ewm(span=120, adjust=True).mean() # Calculate the 120 day EMA

    return {'ema7': ema7, 'ema12': ema12, 'ema26': ema26, 'ema60': ema60, 'ema90': ema90, 'ema120': ema120}


def get_MACD(ticker):
    ret = get_expo_average(ticker)
    ema12 = ret['ema12']
    ema26 = ret['ema26']

    i = 0
    macd = []
    for e in ema12:
        macd.append(round(e - ema26[i], 2))
        i = i + 1
        
    return macd


def get_RSI(ticker):
    basic = get_basic(ticker)
    close = basic['close'] # Get the close prices
    close = pd.Series(close) # Convert to a Series

    delta = close.diff() # Get the difference in price from previous step
    delta = delta[1:] # Remove the first row
    up, down = delta.copy(), delta.copy() # Make a copy of the delta dataframe

    up[up < 0] = 0 # Set the up values to 0 if negative
    down[down > 0] = 0 # Set the down values to 0 if positive

    roll_up = up.ewm(span=14).mean() # Calculate the EWMA
    roll_down = down.abs().ewm(span=14).mean() # Calculate the EWMA

    RS = roll_up / roll_down # Calculate the RS
    RSI = 100.0 - (100.0 / (1.0 + RS)) # Calculate the RSI

    rsi = RSI.tolist() 

    return rsi


def get_candle(ticker):
    basic = get_basic(ticker)

    open = basic['open']
    close = basic['close']
    high = basic['high']
    low = basic['low']

    i, ii = 0, 0
    variation, amplitude, color, trail_up, trail_down = [], [], [], [], []

    for e in open:
        variation.append(((close[i]/e)-1) * 100)
        i = i + 1
    for e in low:
        amplitude.append(((high[ii]/e)-1) * 100)
        ii = ii + 1

    iiii = 0
    for e in variation:
        if e > 0:
            color.append(1)

            upp = ((high[iiii]/close[iiii])-1)*100
            downn = ((open[iiii]/low[iiii])-1)*100

            trail_up.append(upp)
            trail_down.append(downn)
        elif e < 0:
            color.append(-1)

            upp = ((high[iiii]/open[iiii])-1)*100
            downn = ((close[iiii]/low[iiii])-1)*100

            trail_up.append(upp)
            trail_down.append(downn)
        else:
            color.append(0)

            upp = ((high[iiii]/close[iiii])-1)*100
            downn = ((open[iiii]/low[iiii])-1)*100

            trail_up.append(upp)
            trail_down.append(downn)

        iiii = iiii + 1

    candle = []
    iii = 0
    for e in close:
        candle.append([variation[iii], amplitude[iii], color[iii], trail_up[iii], trail_down[iii]])
        iii = iii + 1

    return candle


def get_last_30_days_pattern(ticker):
    candle = get_candle(ticker)
    candle = candle[-30:]

    candles = []

    i = 0
    while i <= 30:
        try:
            can1 = candle[i-2]
            can2 = candle[i-1]
            can3 = candle[i]
            can4 = candle[i+1]
            can5 = candle[i+2]

            candles.append([can1, can2, can3, can4, can5])
        except IndexError:
            pass
        i = i + 1

    return candles


# Execute only if the namespace == main
if __name__ == "__main__":
    ret = get_news("AAPL")
    print(ret)