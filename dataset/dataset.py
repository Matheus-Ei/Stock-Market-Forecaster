# Imports the Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
try:
    import processor as prc
except ModuleNotFoundError:
    import dataset.processor as prc

# Funcion to Create the Dataset
def create(ticker_list):
    all_data = pd.DataFrame() # Create a Dataframe to Fill With the Data
    
    for ticker in ticker_list: # Create a for to go in all elements of ticker list
        data = yf.download(ticker, start=None, end=None, interval="1d") # Get Data From Yahoo Finance

        selected_data = data.copy() # Choose All the selected data
        selected_data['Ticker'] = ticker # Add the Ticker Collum
        selected_data.reset_index(inplace=True) # Remove the Index

        all_data = pd.concat([all_data, selected_data], ignore_index=True) # Concatenate the Data
    all_data.to_csv(r"dataset\default_data.csv", index=False) # Save Data in File

    '''fred = Fred(api_key="0fd3ecf67328f292d90794fe1630c40f")
    federal_funds_rate = fred.get_series('FEDFUNDS')

    last_fed = []

    for e in data['Date']: # For to get the Date
        date = int(int(e.split("-")[0])*365) + (int(e.split("-")[1])*30) + (int(e.split("-")[2]))

    for dt_fed, fed in federal_funds_rate: # For to get the Date
        fed_date = int(int(dt_fed.split("-")[0])*365) + (int(dt_fed.split("-")[1])*30) + (int(dt_fed.split("-")[2]))

        if date == fed_date:
            last_fed.append(fed)'''


# Funcion to Get the Data from the dataset
def get_data(data, i, k):
    # Try to Get the Data
    try:
        close = float(data['Close'][i + k]) # Get the Close Value
        open = float(data['Open'][i + k]) # Get the Open Value
        high = float(data['High'][i + k]) # Get the High Value
        low = float(data['Low'][i + k]) # Get the Low Value
        volume = float(data['Volume'][i + k]) # Get the Volume Value
    # If the Data is not Available and gets a key error
    except KeyError:
        close = float(data['Close'][i]) # Get the Close Value
        open = float(data['Open'][i]) # Get the Open Value
        high = float(data['High'][i]) # Get the High Value
        low = float(data['Low'][i]) # Get the Low Value
        volume = float(data['Volume'][i]) # Get the Volume Value

    return(open, close, high, low, volume) # Return the Values



# Funcion to get the Undimensional Data that will be used to add in x_train and y_train
def undimentional_x(data, i, k):
    (open, close, high, low, volume) = get_data(data, i, k) # Get the dimensiona data from the file
    (last_open, last_close, last_high, last_low, last_volume) = get_data(data, i, k-1) # Get the dimensiona data from the file
    
    # Get the Close Diference
    try: # Get the division from open and close
        percent_variation = float(close) / float(open)
    except ZeroDivisionError: # If the open is 0
        percent_variation = 1
    
    # Get the Volume Diference
    try: # Get the division from last volume and current volume
        negociation_percent_volume = float(volume) / float(last_volume)
    except ZeroDivisionError: # If the open is 0
        negociation_percent_volume = 1
    
    # Get the High Diference
    try: # Get the division from high and low
        index_volatility = float(high) / float(low)
    except ZeroDivisionError: # If the low is 0
        index_volatility = 1

    close_avarage_7_percent = prc.get_avarage_7(data, i, "Open", "Close")
    candle_balance = prc.get_candle_balance(data, i, "Open", "Close")

    # Divide the Values by 2 and Round
    percent_variation = round((float(percent_variation/2)), 3)
    negociation_percent_volume = round((float(negociation_percent_volume/2)), 3)
    index_volatility = round((float(index_volatility/2)), 3)
    close_avarage_7_percent = round((float(close_avarage_7_percent/2)), 3)
    candle_balance = round((float(((candle_balance/2)/2)/2)), 3)

    no_arr = [percent_variation, index_volatility, close_avarage_7_percent, candle_balance, negociation_percent_volume] # Create a List with the Values
    numpy_arr = np.array(no_arr) # Convert the List to Numpy Array
    return(numpy_arr) # Return the Values


# Funcion to get the Undimensional Data that will be used to add in x_train and y_train
def undimentional_y(data, i, k):
    (open, close, high, low, volume) = get_data(data, i, k) # Get the dimensiona data from the file
    
    # Get the Close Diference
    try: # Get the division from open and close
        percent_variation = float(close) / float(open)
    except ZeroDivisionError: # If the open is 0
        percent_variation = 1

    # Divide the Values by 2
    percent_variation = round((float(percent_variation/2)), 3)

    no_arr = [percent_variation] # Create a List with the Values
    numpy_arr = np.array(no_arr) # Convert the List to Numpy Array
    return(numpy_arr) # Return the Values


# Funcion to return the x_train and y_train to the model
def fit(dataset_name="dataset/default_data.csv"):
    data = pd.read_csv(dataset_name) # Read the Data

    x_train, y_train = [], [] # Defining the x_train and y_train
    red, green = 0, 0 # Defining the Starter Red Candle and Green

    for number_el, ticker_key in enumerate(data['Ticker']): # To count the number of elements + an - in the list
        (t_close_dif) = undimentional_y(data, number_el, 1)
        if t_close_dif<=0.5:
            red = red + 1
        elif t_close_dif>0.5:
            green = green + 1

    print("Red: ", red)
    print("Green: ", green)


    for i, ticker_key in enumerate(data['Ticker']):
        day = undimentional_x(data, i, 0)
        x_train.append(day) # Append the Values to x_train

        next_day = undimentional_y(data, i, 1)
        y_train.append(next_day) # Append the Values to y_train

    print("The Dataset has been Fitted") # Print the Message
    print("Last X Train: ", x_train[-1])
    print("Last Y Train: ",y_train[-1])
    return(x_train, y_train) # Return the Values


# To Test
if __name__ == "__main__":
    create(["AAPL"])
    fit()