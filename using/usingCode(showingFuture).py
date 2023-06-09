import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Definir a função create_dataset
def create_dataset(data, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback, 0])
        Y.append(data[i+lookback, 0])
    return np.array(X), np.array(Y)

# Registrar a métrica personalizada
def r_squared(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# Carregar dados históricos
data = pd.read_csv('using\dados.csv')  # Substitua 'dados.csv' pelo caminho do seu arquivo de dados
prices = data['Close'].values.reshape(-1, 1)

# Pré-processamento dos dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Preparar dados de teste
lookback = 7  # Número de períodos anteriores a serem considerados
X_test, Y_test = create_dataset(scaled_prices, lookback)

# Carregar o modelo com a métrica personalizada registrada
with tf.keras.utils.custom_object_scope({'r_squared': r_squared}):
    loaded_model = tf.keras.models.load_model('modelo.h5')

# Fazer previsões para os próximos preços
num_predictions = 7  # Número de previsões para os próximos dias
last_sequence = X_test[-1]  # Última sequência de entrada conhecida

predicted_prices = []  # Lista para armazenar as previsões

for _ in range(num_predictions):
    # Fazer a previsão para a próxima entrada
    next_price = loaded_model.predict(last_sequence.reshape(1, lookback, 1))
    predicted_prices.append(scaler.inverse_transform(next_price)[0][0])

    # Atualizar a sequência de entrada conhecida com a nova previsão
    last_sequence = np.append(last_sequence[1:], next_price)

# Desfazer a escala dos dados de teste
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1)) 

# Plotar os resultados
plt.plot(Y_test, label='Valor Real')
plt.plot(range(len(Y_test), len(Y_test) + num_predictions), predicted_prices, label='Previsão')
plt.legend()
plt.show()
