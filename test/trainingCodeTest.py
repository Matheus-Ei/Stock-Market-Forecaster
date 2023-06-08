import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from datetime import datetime, timedelta

# Carregar dados históricos
data = pd.read_csv(r'test\test.csv')  # Essa parte do codigo le o arquivo
prices = data[['Close']].values.reshape(-1, 1) # Define que "prices" sera atribuido a coluna close do arquivo e faz um reshape que significa que ele vai mudar a forma, nesse caso para que tenha apenas uma coluna
print(prices[1, 0])  # Imprime o "Date" do segundo registro
print(len(data))

# Pré-processamento dos dados
scaler = MinMaxScaler(feature_range=(0, 1)) # Cria um objeto scaler da classe MinMaxScaler, estamos especificando que queremos normalizar os dados para o intervalo entre 0 e 1.
scaled_prices = scaler.fit_transform(prices)


def create_dataset(data):
    i=0
    entrada, saidaEsperada = [], []
    while i != len(data) - 1:
        entrada.append([float(prices[i, 0])])
        saidaEsperada.append([float(prices[i + 1, 0])])
        i = i + 1
        print(i)
    return np.array(entrada), np.array(saidaEsperada)

X_train, Y_train = create_dataset(data)

# Remodelar a entrada para ter uma dimensão adicional
entrada = np.expand_dims(X_train, axis=1)
saidaEsperada = np.expand_dims(Y_train, axis=1)

print(entrada)
print("-----------------------------------------------------------------------------------------------------")
print(saidaEsperada)


# Construir o modelo de rede neural
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1),

    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.Dropout(0.9),

    tf.keras.layers.LSTM(512, return_sequences=True),
    tf.keras.layers.Dropout(0.9),

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
    tf.keras.layers.Dense(1),
])

@tf.autograph.experimental.do_not_convert
def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', r_squared])

# Treinar o modelo
model.fit(entrada, saidaEsperada, epochs=100, batch_size=32)

# Salvar o modelo
model.save('modelo.h5')


