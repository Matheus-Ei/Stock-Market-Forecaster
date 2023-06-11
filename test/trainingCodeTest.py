import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from datetime import datetime, timedelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.linear_model import LinearRegression

# Carregar dados históricos
data = pd.read_csv("dados.csv")  # Essa parte do codigo le o arquivo
pre_prices = data[['Close']].to_numpy().reshape(-1, 1) # Define que "prices" sera atribuido a coluna close do arquivo e faz um reshape que significa que ele vai mudar a forma, nesse caso para que tenha apenas uma coluna
print(pre_prices[1, 0])
print(len(data))

scaler = MinMaxScaler(feature_range=(0, 1)) 
prices = scaler.fit_transform(pre_prices) 

# Número de dias para o histórico
historical_days = 30

# Criação de x_train e y_train
x_train = []
y_train = []

for i in range(len(prices) - historical_days - 1):
    x_train.append(prices[i:i+historical_days].reshape(historical_days, 1))
    y_train.append(prices[i+historical_days])

x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train)
print("-----------------------------------------------------------------------------------------------------")
print(y_train)


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(1024, return_sequences=True, input_shape=(historical_days, 1)),
    tf.keras.layers.Dropout(0.8),

    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.Dropout(0.8),

    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.LSTM(16, return_sequences=True),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(8),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(1),
])



@tf.autograph.experimental.do_not_convert
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())



# Criar um objeto EarlyStopping
early_stopping = EarlyStopping(
    monitor='r_squared',  # Métrica para monitorar e decidir quando interromper o treinamento
    patience=3,  # Número de épocas sem melhoria após as quais o treinamento é interrompido
    mode='max',  # Modo da métrica (por exemplo, 'min' para minimizar a métrica)
    verbose=1  # Exibir mensagens de status durante a parada antecipada
)



# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', r_squared])

# Treinar o modelo
model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[early_stopping])

# Salvar o modelo
model.save('modelo.h5')


