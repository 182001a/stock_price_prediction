import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

class LSTMS:
    def __init__(self, input_shape, output_size=1,
                 lstm_units=100, dropout_rate=0.2,
                 dense_units=50):
        # LSTMモデルパラメータ
        self.input_shape = input_shape  # (time_steps, features)
        self.output_size = output_size
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units

    def build(self):
        # 入力層
        input_layer = Input(shape=self.input_shape, name='input_layer')
        
        # LSTM層
        x = LSTM(self.lstm_units, return_sequences=True, activation='relu', name='lstm_layer1')(input_layer)
        x = Dropout(self.dropout_rate, name='dropout_layer1')(x)
        
        x = LSTM(self.lstm_units, return_sequences=True, activation='relu', name='lstm_layer2')(x)
        x = Dropout(self.dropout_rate, name='dropout_layer2')(x)

        x = LSTM(self.lstm_units, return_sequences=False, activation='relu', name='lstm_layer3')(x)
        x = Dropout(self.dropout_rate, name='dropout_layer3')(x)
        
        # 全結合層
        x = Dense(self.dense_units,  name='dense_layer')(x)
        
        # 出力層
        output_layer = Dense(self.output_size, activation='sigmoid', name='output_layer')(x)
        
        return Model(inputs=input_layer, outputs=output_layer, name='lstm_stock_predictor')
