import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 설정 (Configuration)
# ---------------------------------------------------------
CONFIG = {
    'ticker': 'GOOGL',        # 구글 주가
    'start_date': '2020-01-01',
    'end_date': '2025-11-30',
    'seq_length': 10,         # 시퀀스 길이
    'input_dim': 6,           # Feature 개수
    'hidden_dim': 64,
    'output_dim': 1,
    'num_layers': 3,
    'epochs': 100,
    'learning_rate': 0.001,
    'train_split': 0.8
}

# ---------------------------------------------------------
# 1. 데이터 처리 클래스 (Data Processing)
# ---------------------------------------------------------
class StockDataProcessor:
    def __init__(self, ticker, start, end):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.df = None
        
    def download_and_process(self):
        print(f"Downloading data for {self.ticker}...")
        self.df = yf.download(self.ticker, start=self.start, end=self.end)
        
        if self.df.empty:
            raise ValueError("데이터를 다운로드할 수 없습니다.")

        if isinstance(self.df.columns, pd.MultiIndex):
            close_col = self.df['Close'][self.ticker]
        else:
            close_col = self.df['Close']
            
        self.df['MA5'] = close_col.rolling(window=5).mean()
        self.df = self.df.dropna()
        
        print(f"Data loaded. Shape: {self.df.shape}")
        
    def get_features_and_target(self):
        try:
            open_p = self.df['Open'][self.ticker]
            high_p = self.df['High'][self.ticker]
            low_p = self.df['Low'][self.ticker]
            close_p = self.df['Close'][self.ticker]
            vol_p = self.df['Volume'][self.ticker]
        except KeyError:
            open_p = self.df['Open']
            high_p = self.df['High']
            low_p = self.df['Low']
            close_p = self.df['Close']
            vol_p = self.df['Volume']
            
        ma5_p = self.df['MA5']

        features_df = pd.DataFrame({
            'Open': open_p,
            'High': high_p,
            'Low': low_p,
            'Close': close_p,
            'Volume': vol_p,
            'MA5': ma5_p
        })
        
        x_values = features_df.values
        y_values = features_df[['Close']].values
        
        return x_values, y_values

    def scale_data(self, x_values, y_values):
        x_scaled = self.scaler_x.fit_transform(x_values)
        y_scaled = self.scaler_y.fit_transform(y_values)
        return x_scaled, y_scaled

    def create_sequences(self, x_data, y_data, seq_length):
        xs, ys = [], []
        for i in range(len(x_data) - seq_length):
            x_window = x_data[i : i+seq_length]
            y_label = y_data[i + seq_length]
            xs.append(x_window)
            ys.append(y_label)
        return np.array(xs), np.array(ys)
    
    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)

# ---------------------------------------------------------
# 2. 모델 정의 (Stacked LSTM) - 변경됨
# ---------------------------------------------------------
class StackedStockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StackedStockLSTM, self).__init__()
        
        # [변경 1] nn.RNN -> nn.LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # [변경 2] LSTM은 (hidden state, cell state) 튜플을 반환
        # out: (Batch, Seq, Hidden)
        # h_n: (NumLayers, Batch, Hidden)
        # c_n: (NumLayers, Batch, Hidden) - Cell State
        out, (h_n, c_n) = self.lstm(x)
        
        # 마지막 층의 마지막 시점 Hidden State 추출
        # h_n은 (Layer, Batch, Hidden) 형태이므로 마지막 Layer([-1])를 선택
        top_layer_last_hidden = h_n[-1] 
        
        output = self.fc(top_layer_last_hidden)
        return output

# ---------------------------------------------------------
# 3. 학습 및 평가 함수
# ---------------------------------------------------------
def train_model(model, X_train, y_train, config):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    model.train()
    print("\nStarting Training (LSTM)...")
    
    loss_history = []
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {loss.item():.6f}")
            
    return loss_history

def evaluate_model(model, X_test, y_test, processor):
    model.eval()
    with torch.no_grad():
        predicted = model(X_test)
    
    predicted_price = processor.inverse_transform_y(predicted.numpy())
    real_price = processor.inverse_transform_y(y_test.numpy())
    
    return predicted_price, real_price

# ---------------------------------------------------------
# 4. 메인 실행 블록
# ---------------------------------------------------------
def main():
    processor = StockDataProcessor(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'])
    processor.download_and_process()
    
    raw_x, raw_y = processor.get_features_and_target()
    scaled_x, scaled_y = processor.scale_data(raw_x, raw_y)
    
    X, y = processor.create_sequences(scaled_x, scaled_y, CONFIG['seq_length'])
    
    train_size = int(len(X) * CONFIG['train_split'])
    X_train_np, y_train_np = X[:train_size], y[:train_size]
    X_test_np, y_test_np = X[train_size:], y[train_size:]
    
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # [변경 3] 모델 생성 시 StackedStockLSTM 사용
    model = StackedStockLSTM(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim'],
        num_layers=CONFIG['num_layers']
    )
    print(model)

    train_model(model, X_train, y_train, CONFIG)
    
    pred_price, real_price = evaluate_model(model, X_test, y_test, processor)
    
    plt.figure(figsize=(12, 6))
    plt.plot(real_price, label='Real Price (GOOGL)', color='blue')
    plt.plot(pred_price, label='Predicted Price (LSTM)', color='green', linestyle='--')
    plt.title(f"{CONFIG['ticker']} Stock Prediction (LSTM)")
    plt.xlabel('Time (Test Data Index)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
    
    