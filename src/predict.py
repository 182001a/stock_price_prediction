from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import pandas as pd

from utils import parse_args

def create_dataset(data, time_step=60):
    X = []
    for i in range(len(data) - time_step -1):
        a = data[i:(i + time_step)]
        X.append(a)
    return np.array(X)

def main():
    args = parse_args()

    df = pd.read_csv(args.test_data)
    
    # 'Close'価格のみを抽出
    close_prices = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices)
    
    time_step = 1
    X_test = create_dataset(scaled_close_prices, time_step)
    
    model = load_model(args.model_path)
    
    y_pred = model.predict(X_test)
    # 予測値のスケーリングを元に戻す
    y_pred = scaler.inverse_transform(y_pred)
    
    # 実測値の取得（予測と同じ範囲にするための調整）
    y_actual = close_prices[time_step+1:]  # create_datasetの処理を考慮
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual, label='実測値')
    plt.plot(y_pred, label='予測値')
    plt.title('LSTMSによる株価予測')
    plt.xlabel('日')
    plt.ylabel('株価')
    plt.legend()
    plt.savefig('../results/sample_close_only.png')
    
if __name__ == '__main__':
    main()
