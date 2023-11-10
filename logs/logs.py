# Funcion to Add Logs
def add_train(log_value):
    # Open the File in Read Mode
    with open(r'logs\number_of_trains.txt', 'r') as f:
        number_of_test = f.read()
        number_of_test = int(number_of_test)
        print(number_of_test)

    number_of_test += 1 # Add 1 to the Number of Tests

    # Open the File in Write Mode
    with open(r'logs\number_of_trains.txt', 'w') as f:
        f.write(str(number_of_test))

    # Create the String to Save in the File
    last_e = ""
    for e in log_value:
        last_e = str(str(last_e) + str(e) + "\n")
    last_e = last_e + "\n" + "\n"

    # Open the File in Write Mode
    with open(r'logs\train_logs.txt', 'a') as f:
        f.write(last_e)

# Funcion to Add Logs
def add_use(log_value):
    # Open the File in Read Mode
    with open(r'logs\use_number.txt', 'r') as f:
        number_of_test = f.read()
        number_of_test = int(number_of_test)
        print(number_of_test)

    number_of_test += 1 # Add 1 to the Number of Tests

    # Open the File in Write Mode
    with open(r'logs\use_number.txt', 'w') as f:
        f.write(str(number_of_test))

    last_e = ""
    for e in log_value:
        last_e = last_e + e + "\n"
    last_e = last_e + "\n" + "\n"

    # Open the File in Write Mode
    with open(r'logs\use_logs.txt', 'a') as f:
        f.write(last_e)