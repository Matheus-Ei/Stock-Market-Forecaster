# Imports
import get_data as gd

def main(ticker):
    rate_up = 100
    rate_down = 100
    rate_neutral = 100

    rec = gd.get_recomendation(ticker)
    news_sent = gd.get_news(ticker)

    average = gd.get_average(ticker)
    expo_average = gd.get_expo_average(ticker)

    macd = gd.get_MACD(ticker)
    rsi = gd.get_RSI(ticker)

    if rec == 1:
        rate_up = rate_up * 1.25
        rate_down = rate_down * 0.8
    elif rec == -1:
        rate_down = rate_down * 1.25
        rate_up = rate_up * 0.8
    elif rec == 0:
        rate_up = rate_up * 1.25
        rate_down = rate_down * 0.8

    if macd[-1] > 0:
        if macd[-1] > macd[-2]:
            rate_up = rate_up * 1.25
            rate_down = rate_down * 0.9
            rate_neutral = rate_neutral * 1.01
        else:
            rate_up = rate_up * 1.05
            rate_down = rate_down * 0.95
            rate_neutral = rate_neutral * 1.05
    elif macd[-1] < 0:
        if macd[-1] < macd[-2]:
            rate_down = rate_down * 1.25
            rate_up = rate_up * 0.9
            rate_neutral = rate_neutral * 1.01
        else:
            rate_down = rate_down * 1.05
            rate_up = rate_up * 0.95
            rate_neutral = rate_neutral * 1.05
    elif macd[-1] == 0:
        rate_down = rate_down * 0.95
        rate_up = rate_up * 0.95
        rate_neutral = rate_neutral * 1.05


    if rsi[-1] > 70:
        rate_down = rate_down * 1.25
        rate_up = rate_up * 0.9
        rate_neutral = rate_neutral * 1.01
    elif rsi[-1] < 30:
        rate_up = rate_up * 1.25
        rate_down = rate_down * 0.9
        rate_neutral = rate_neutral * 1.01
    elif rsi[-1] > 50:
        rate_up = rate_up * 1.05
        rate_down = rate_down * 0.95
        rate_neutral = rate_neutral * 1.05
    elif rsi[-1] < 50: 
        rate_down = rate_down * 1.05
        rate_up = rate_up * 0.95
        rate_neutral = rate_neutral * 1.05

        if news_sent > 0.35:
            rate_up = rate_up * 1.25
            rate_down = rate_down * 0.9
            rate_neutral = rate_neutral * 1.01
        elif news_sent < -0.35:
            rate_down = rate_down * 1.25
            rate_up = rate_up * 0.9
            rate_neutral = rate_neutral * 1.01
        elif news_sent > 0.15:
            rate_up = rate_up * 1.05
            rate_down = rate_down * 0.95
            rate_neutral = rate_neutral * 1.05
        elif news_sent < -0.15:
            rate_down = rate_down * 1.05
            rate_up = rate_up * 0.95
            rate_neutral = rate_neutral * 1.05
        elif news_sent > -0.05 and news_sent < 0.05:
            rate_down = rate_down * 0.95
            rate_up = rate_up * 0.95
            rate_neutral = rate_neutral * 1.10

    print("Rate Up: ", int(rate_up))
    print("Rate Down: ", int(rate_down))
    print("Rate Neutral: ", int(rate_neutral))

    return [rate_up, rate_down, rate_neutral]

# Execute only if the namespace == main
if __name__ == "__main__":
    main("AAPL")