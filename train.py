# Imports Libraries
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras import metrics
from io import StringIO
from keras.callbacks import ModelCheckpoint

# Import Modules
import dataset.dataset as dts
import use as us
import logs.logs as lgs


# Function to Calculate the price_dif metric
def price_division(y_true, y_pred):
    try:
        loss_var = y_pred/y_true
    except ZeroDivisionError:
        loss_var = y_pred/0.5
    loss_var = (loss_var-1)*100
    loss_var = abs(loss_var)
    return loss_var


# Function to Calculate the price_dif metric
def price_diference(y_true, y_pred):
    try:
        loss_var = y_pred/y_true
    except ZeroDivisionError:
        loss_var = y_pred/0.5
    loss_var = (loss_var-1)*100
    return loss_var


# Function to Calculate the Loss
def price_loss(y_true, y_pred):
    try:
        loss_var = y_pred/y_true
    except ZeroDivisionError:
        loss_var = y_pred/0.5
    loss_var = (loss_var-1)*100
    loss_var = abs(loss_var)
    return loss_var


def train():
    # Open the Config File
    with open("config\config.json", "r") as config:
        config_data = json.load(config)

    # Defining the Variables of Train
    ticker = config_data["train_ticker"]
    n_epochs = config_data["number_of_epochs"]
    num_bach_size = config_data["bach_size"]
    val_split = config_data["validation_split"]

    # Defining the Variables Base of Dataset
    number_of_parameters = config_data["number_of_parameters"]
    number_of_subparameters = config_data["number_of_subparameters"]
    number_of_outputs = config_data["number_of_outputs"]
    number_of_suboutputs = config_data["number_of_suboutputs"]

    checkpoint_filepath = 'checkpoints\\best_model.h5' # Define the checkpoint filepath

    # Defining the Variables X and Y Train
    x_train = []
    y_train = []

    dts.create(ticker) # Create the File
    x_train, y_train = dts.fit() # Load the Data

    # Converting to numpy array
    x_train = np.array(x_train).reshape(-1, number_of_parameters, number_of_subparameters)
    y_train = np.array(y_train).reshape(-1, number_of_outputs, number_of_suboutputs)

    print("@@@@@@@@@@@-----------------------------------------------------------------------------------------------------@@@@@@@@@@@")
    print(x_train)
    print("-----------------------------------------------------------------------------------------------------")
    print(y_train)
    print("@@@@@@@@@@@-----------------------------------------------------------------------------------------------------@@@@@@@@@@@")


    # Defining Early Stopping
    early_stopping = EarlyStopping(
        monitor='price_division',  # Metric to be monitored
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        mode='min',  # Mode of the monitor
        verbose=1  # Shows the epoch that the training stopped
    )

    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        save_best_only=True,
        monitor='price_division',
        mode='min',
        verbose=1
    )

    # Crating the Model
    model = Sequential([
        Dense(8192, input_shape=(number_of_parameters, number_of_subparameters)),
        Dense(1024),
        Dense(128),
        Dense(16)
    ])

    model.compile(optimizer=RMSprop(learning_rate = 0.001), loss="mean_squared_error", metrics=[metrics.MeanSquaredError(), price_division, price_diference]) # Compile the Model
    history = model.fit(x_train, y_train, validation_split = val_split, epochs=n_epochs, batch_size=num_bach_size, callbacks=[early_stopping, checkpoint_callback]) # Fit the Model

    buffer = StringIO() # Create a string buffer to capture the output
    model.summary(print_fn=lambda x: buffer.write(x + '\n')) # Redirect the output to the buffer
    model_summary = buffer.getvalue() # Get the output from the buffer
    model.summary()

    model.save('models\model_01.h5') # Save the Model;
    lgs.add_train([model_summary, history, n_epochs]) # Add the Logs

    us.predict() # Predict the Model

# If the Namespace is main
if __name__ == "__main__":
    train()