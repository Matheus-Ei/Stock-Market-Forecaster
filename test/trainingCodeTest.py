import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from datetime import datetime, timedelta
from keras.callbacks import ModelCheckpoint, EarlyStopping

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
        entrada.append([float(scaled_prices[i, 0])])
        saidaEsperada.append([float(scaled_prices[i + 1, 0])])
        i = i + 1
        print(i)
    return np.array(entrada), np.array(saidaEsperada)

entrada, saidaEsperada = create_dataset(data)

print(entrada)
print("-----------------------------------------------------------------------------------------------------")
print(saidaEsperada)


# Definir o caminho onde deseja salvar os checkpoints
checkpoint_path = 'test\check\modelo-{epoch:02d}.h5'

# Criar um objeto ModelCheckpoint
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='loss',  # Métrica para monitorar e decidir quando salvar o modelo
    save_best_only=True,  # Salvar apenas o melhor modelo com base na métrica monitorada
    mode='min',  # Modo da métrica (por exemplo, 'min' para minimizar a métrica)
    verbose=1  # Exibir mensagens de status durante o salvamento do modelo
)

# Criar um objeto EarlyStopping
early_stopping = EarlyStopping(
    monitor='loss',  # Métrica para monitorar e decidir quando interromper o treinamento
    patience=5,  # Número de épocas sem melhoria após as quais o treinamento é interrompido
    mode='min',  # Modo da métrica (por exemplo, 'min' para minimizar a métrica)
    verbose=1  # Exibir mensagens de status durante a parada antecipada
)


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(1024, return_sequences=True, input_shape=(3, 1)),
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

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse', 'mae', r_squared])

# Treinar o modelo
model.fit(entrada, saidaEsperada, epochs=50, batch_size=32, callbacks=[checkpoint, early_stopping])

# Salvar o modelo
model.save('modelo.h5')


