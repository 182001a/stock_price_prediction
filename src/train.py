from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from lstm_model import LSTMS

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step -1):
        a = data[i:(i + time_step)]
        X.append(a)
        Y.append(data[i + time_step])

    return np.array(X), np.array(Y)

def main():
    df = pd.read_csv('../data/nicon_train.csv')
    close_prices = df['Close'].values.reshape(-1,1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices)

    time_step = 1
    X, Y = create_dataset(scaled_close_prices, time_step)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LSTMS(input_shape=(time_step, 1)).build()  # 入力形状を(time_steps, 1)に変更
    model.compile(optimizer='adam', loss='mean_squared_error')
    callback = [EarlyStopping(patience=5)]
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, callbacks=callback)
    model.save('../models/nicon_model_close_only.h5')

if __name__ == '__main__':
    main()
