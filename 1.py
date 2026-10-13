import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url)
data = df.iloc[:, 1].values.astype("float32").reshape(-1, 1)

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

look_back = 12
X, y = [], []
for i in range(len(data) - look_back):
    X.append(data[i:i + look_back])
    y.append(data[i + look_back])
X, y = np.array(X), np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


def run(layer, name):
    model = Sequential([layer(50, input_shape=(look_back, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=100, batch_size=8,
              validation_data=(X_test, y_test), verbose=0)
    pred = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual = scaler.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    print(name, "test RMSE:", round(float(rmse), 2))
    return rmse


rnn_rmse = run(SimpleRNN, "SimpleRNN")
lstm_rmse = run(LSTM, "LSTM")
gru_rmse = run(GRU, "GRU")
