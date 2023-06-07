import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K

# Carregar dados históricos
data = pd.read_csv('dados.csv')  # Substitua 'dados.csv' pelo caminho do seu arquivo de dados
prices = data['Close'].values.reshape(-1, 1)

# Pré-processamento dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Dividir os dados em conjuntos de treinamento e teste
train_size = int(len(scaled_prices) * 0.8)
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

# Preparar dados de treinamento e teste
def create_dataset(data, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, 0])
        Y.append(data[i+lookback, 0])
    return np.array(X), np.array(Y)

lookback = 30  # Número de períodos anteriores a serem considerados
X_train, Y_train = create_dataset(train_data, lookback)

# Construir o modelo de rede neural
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1)
])

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', r_squared])

# Treinar o modelo
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Salvar o modelo
model.save('modelo.h5')
