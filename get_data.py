# Imports
import yahooquery as yq

def get_events(ticker, past_days = 1):
    # Get the corporate events for the ticker
    tk = yq.Ticker(ticker)
    df = tk.corporate_events
    df_reset = df.reset_index() # Reset the index to get the last row

    description = df_reset['description'] # Get the description
    description_length = len(description) # Get the length of the description

    last_description = description[description_length - past_days] # Get the description
    return last_description # Return the last description


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

    return recomendation


# Execute only if the namespace == main
if __name__ == "__main__":
    direction_array = get_recomendation("AAPL")
    print(direction_array)