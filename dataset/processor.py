def get_avarage_7(data, i, var1, var2):
    # Function to Calculate the Average
    def med(array):
        som_ret_arr = 0
        for item in array:
            som_ret_arr = item + som_ret_arr
        ret_arr = som_ret_arr / len(array)
        return ret_arr

    try:
        close = float(data[var2][i])
    except KeyError:
        close = float(data[var2][i])
    try:
        close1 = float(data[var2][i - 1])
    except KeyError:
        close1 = float(data[var2][i])
    try:
        close2 = float(data[var2][i - 2])
    except KeyError:
        close2 = float(data[var2][i])
    try:  
        close3 = float(data[var2][i - 3])
    except KeyError:
        close3 = float(data[var2][i])
    try:
        close4 = float(data[var2][i - 4])
    except KeyError:
        close4 = float(data[var2][i])
    try:
        close5 = float(data[var2][i - 5])
    except KeyError:
        close5 = float(data[var2][i])
    try:
        close6 = float(data[var2][i - 6])
    except KeyError:
        close6 = float(data[var2][i])

    try:
        open = float(data[var1][i])
    except KeyError:
        open = float(data[var1][i])
    try:
        open1 = float(data[var1][i - 1])
    except KeyError:
        open1 = float(data[var1][i])
    try:
        open2 = float(data[var1][i - 2])
    except KeyError:
        open2 = float(data[var1][i])
    try:  
        open3 = float(data[var1][i - 3]) 
    except KeyError:
        open3 = float(data[var1][i])
    try:
        open4 = float(data[var1][i - 4]) 
    except KeyError:
        open4 = float(data[var1][i])
    try:
        open5 = float(data[var1][i - 5]) 
    except KeyError:
        open5 = float(data[var1][i])
    try:
        open6 = float(data[var1][i - 6]) 
    except KeyError:
        open6 = float(data[var1][i])

    arr_close = [close, close1, close2, close3, close4, close5, close6]
    arr_open = [open, open1, open2, open3, open4, open5, open6]

    ee = 0
    ret_arr = []
    for e in arr_close:
        try:
            med_7_d = e/arr_open[ee]
            ret_arr.append(med_7_d)
            ee = ee+1
        except ZeroDivisionError:
            ret_arr.append(1)
            ee = ee+1
        
    ret_arr = med(ret_arr)
    return(ret_arr)



def get_candle_balance(data, i, var1, var2):
    try:
        close = float(data[var2][i])
    except KeyError:
        close = float(data[var2][i])
    try:
        close1 = float(data[var2][i - 1])
    except KeyError:
        close1 = float(data[var2][i])
    try:
        close2 = float(data[var2][i - 2])
    except KeyError:
        close2 = float(data[var2][i])
    try:  
        close3 = float(data[var2][i - 3])
    except KeyError:
        close3 = float(data[var2][i])
    try:
        close4 = float(data[var2][i - 4])
    except KeyError:
        close4 = float(data[var2][i])
    try:
        close5 = float(data[var2][i - 5])
    except KeyError:
        close5 = float(data[var2][i])
    try:
        close6 = float(data[var2][i - 6])
    except KeyError:
        close6 = float(data[var2][i])

    try:
        open = float(data[var1][i])
    except KeyError:
        open = float(data[var1][i])
    try:
        open1 = float(data[var1][i - 1])
    except KeyError:
        open1 = float(data[var1][i])
    try:
        open2 = float(data[var1][i - 2])
    except KeyError:
        open2 = float(data[var1][i])
    try:  
        open3 = float(data[var1][i - 3]) 
    except KeyError:
        open3 = float(data[var1][i])
    try:
        open4 = float(data[var1][i - 4]) 
    except KeyError:
        open4 = float(data[var1][i])
    try:
        open5 = float(data[var1][i - 5]) 
    except KeyError:
        open5 = float(data[var1][i])
    try:
        open6 = float(data[var1][i - 6]) 
    except KeyError:
        open6 = float(data[var1][i])

    arr_close = [close, close1, close2, close3, close4, close5, close6]
    arr_open = [open, open1, open2, open3, open4, open5, open6]
    i, last_color = 0, 0
    arr_candle_color = []

    for e in arr_open:
        # Get the color of the candle
        if arr_close[0]>e: 
            candle_color = 1 # Green
        elif arr_close[0]<=e:
            candle_color = 0 # Red

        arr_candle_color.append(candle_color)
        i = i + 1

    for e in arr_candle_color:
        last_color = last_color + e

    return(last_color)