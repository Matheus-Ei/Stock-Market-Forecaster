# Import the Libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import json

# Import the Modules
import dataset.dataset as dts
import logs.logs as lgs
import train as trn


#Function to Predict
def predict(loaded_model = 'checkpoints\\best_model.h5'):
    with tf.keras.utils.custom_object_scope({"price_loss": trn.price_loss}, {'price_division': trn.price_division}, {'price_diference': trn.price_diference}):
        loaded_model = tf.keras.models.load_model(f"{loaded_model}")
        
    counter, kkk, iii = 0, 0, 0
    while counter <= 0:
        # Defining the Variables
        x_trat, y_trat = [], []
        x_trat_1, y_trat_1 = [], []
        x, y = [], []
        zero, time, converts, real_converts = [], [], [], []

        # Function to Calculate the Average
        def med(array):
            som_ret_arr = 0
            for item in array:
                som_ret_arr = item + som_ret_arr
            ret_arr = som_ret_arr / len(array)
            return ret_arr

        # Open the Config File
        with open("config\config.json", "r") as config:
            config_data = json.load(config)

        # Defining the Variables of Test
        ticker = config_data["test_ticker"]
        number_of_tests = config_data["number_of_tests"]

        # Defining the Variables Base of Dataset
        number_of_parameters = config_data["number_of_parameters"]
        number_of_subparameters = config_data["number_of_subparameters"]

        dts.create(ticker) # Create the File
        x_data, y_data = dts.fit() # Load the Data

        i = 0
        # Loop to Predict
        while i != number_of_tests:
            last_sequence = x_data[-i] # Get the Last Sequence
            last_sequence = np.array(last_sequence).reshape(-1, number_of_parameters, number_of_subparameters) # Converting to numpy array 

            next_price = loaded_model.predict(last_sequence) # Predict the Next Price
            arr_next_price = next_price
            next_price = med(next_price[0]) # Calculate the Average

            # Append the Values into the Lists
            x.append(next_price[0]*2)
            y.append((y_data[-i][0])*2)
            zero.append(0)
            time.append(i)
            i = i + 1 

        print("#######--------------------------------------------------@ Last Sequence @--------------------------------------------------#######")
        print(last_sequence)
        print("#######--------------------------------------------------@ Avarege Predict @--------------------------------------------------#######")
        print(next_price*2)
        print("#######--------------------------------------------------@ Correct Predict @--------------------------------------------------#######")
        print((y_data[0])*2)
        print("#######----------------------------------------------------------------------------------------------------#######")

        # Treat the Values of the Lists to First Graph
        for e in x:
            e = e - 1
            e = e * 100
            x_trat.append(round(e, 3))
        for ee in y:
            ee = ee - 1
            ee = ee * 100
            y_trat.append(round(ee, 3))
        while kkk != len(time):
            ele_x = x_trat[kkk]
            ele_y = y_trat[kkk]
            if ele_x == ele_y:
                converts.append([ele_x, time[kkk]])
            kkk = kkk + 1
        while iii != len(time):
            ele_x = x_trat[iii]
            ele_y = y_trat[iii]
            if ((ele_x < 0) and (ele_y < 0)) or ((ele_x > 0) and (ele_y > 0)) or ((ele_x == 0) and (ele_y == 0)):
                real_converts.append([time[iii]])
            iii = iii + 1

        # Treat the Values of the Lists to Second Graph
        last_x = 1
        for e in x:
            e = e*last_x
            last_x = e
            x_trat_1.append(round(e, 3))
        last_y = 1
        for e in y:
            e = e*last_y
            last_y = e
            y_trat_1.append(round(e, 3))


        plt.figure(figsize=(15, 7.5)) # Create and Set the Size of the Graph

        # Create the First Graph
        plt.subplot(2, 1, 1) 
        plt.plot(time, zero, color="black", label="Zero")
        plt.scatter(0, 0, color='purple', label='Conversões')
        for e_temp in converts:
            plt.scatter(e_temp[1], e_temp[0], color='purple')
        for e_temp in real_converts:
            plt.axvline(e_temp, color='gray', linestyle='--')
        plt.plot(time, x_trat, color="red", label="Previsão")
        plt.plot(time, y_trat, color="blue", label="Valor Real")
        
        # Defining the labels of First Graph
        plt.title("Variação de Porcentagens")
        plt.xlabel("Tempo")
        plt.ylabel("Valor")
        plt.legend()

        # Create the Second Graph
        plt.subplot(2, 1, 2)
        plt.plot(time, x_trat_1, color="red", label="Previsão")
        plt.plot(time, y_trat_1, color="blue", label="Valor Real")
        # Defining the labels of Second Graph
        plt.title("Variação Progressiva")
        plt.xlabel("Tempo")
        plt.ylabel("Valor")
        plt.legend()

        # Show the Graphs
        plt.tight_layout()
        plt.savefig(f'graphs\\Graph_{counter}.png') # save the plot as a PNG image
        log_value = [("Number of Predictions: " + str(number_of_tests)), 
                     ("Number of Correct Acurate Predictions: " + str(len(converts))),
                     ("Percentage of Correct Acurate Predictions: " + str((len(converts)/number_of_tests)*100) + "%"),
                     ("Number of Correct Aproximated Predictions: " + str(len(real_converts))),
                     ("Percentage of Correct Aproximated Predictions: " + str((len(real_converts)/number_of_tests)*100) + "%")]
        
        print("#######--------------------------------------------------@ Graph Avaliation @--------------------------------------------------#######")
        for value in log_value:
            print(value)
        print("#######------------------------------------------------------------------#######")
        counter = counter + 1

        lgs.add_use(log_value) # Add the Use Log
    #plt.show()


# To Run the Code
if __name__ == "__main__":
    predict()