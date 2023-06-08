import yfinance as yf

# Definir o ticker do ativo
ticker = "MSFT"  # Substitua pelo ticker do ativo desejado

# Definir o intervalo de datas
start_date = "2020-01-01"
end_date = "2023-06-05"

# Obter os dados do Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Selecionar apenas as colunas desejadas
selected_data = data[["Close", "Volume"]].copy()
selected_data.reset_index(inplace=True)

# Salvar os dados em um arquivo CSV
selected_data.to_csv(r"using\dados.csv", index=False)
selected_data.to_csv(r"training\dados.csv", index=False)

