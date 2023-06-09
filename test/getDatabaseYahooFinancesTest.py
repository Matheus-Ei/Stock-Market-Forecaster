import yfinance as yf
import pandas as pd

# Definir os tickers dos ativos desejados
tickers = ["AAPL"]  # Substitua pelos tickers dos ativos desejados

# Definir o intervalo de datas
start_date = "2003-01-01"
end_date = "2023-01-01"

# Criar um DataFrame vazio para armazenar os dados
all_data = pd.DataFrame()

# Obter os dados do Yahoo Finance para cada ticker
for ticker in tickers:
    # Obter os dados do Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Selecionar apenas as colunas desejadas
    selected_data = data[["Close"]].copy()
    
    # Adicionar uma coluna para o ticker
    selected_data['Ticker'] = ticker
    
    # Resetar o índice e adicionar uma coluna de data
    selected_data.reset_index(inplace=True)
    
    # Concatenar os dados ao DataFrame geral
    all_data = pd.concat([all_data, selected_data], ignore_index=True)

# Salvar os dados em um arquivo CSV
all_data.to_csv("dados.csv", index=False)
