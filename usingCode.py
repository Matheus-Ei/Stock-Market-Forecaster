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

# Preparar dados de teste
lookback = 30  # Número de períodos anteriores a serem considerados
X_test, Y_test = create_dataset(test_data, lookback)

# Carregar o modelo
loaded_model = tf.keras.models.load_model('modelo.h5')

# Usar o modelo carregado para fazer previsões
predictions = loaded_model.predict(X_test)

# Desfazer a escala dos dados
predictions = scaler.inverse_transform(predictions)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Avaliar o desempenho do modelo
mse = np.mean((predictions - Y_test)**2)
print(f"MSE: {mse}")

# Plotar os resultados
import matplotlib.pyplot as plt
plt.plot(Y_test, label='Valor Real')
plt.plot(predictions, label='Previsão')
plt.legend()
plt.show()
