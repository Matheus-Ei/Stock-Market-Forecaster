import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Carregar dados históricos
data = pd.read_csv('dados.csv')  # Substitua 'dados.csv' pelo caminho do seu arquivo de dados
prices = data['Preço'].values.reshape(-1, 1)

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
    tf.keras.layers.LSTM(50, input_shape=(lookback, 1)),
    tf.keras.layers.Dense(1)
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# Salvar o modelo
model.save('modelo.h5')
