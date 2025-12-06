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
    'seq_length': 10,         # 시퀀스 길이 (10일치 데이터로 다음날 예측)
    'input_dim': 6,           # Feature 개수 (Open, High, Low, Close, Volume, MA5)
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
        
        # 데이터가 제대로 받아지지 않았을 경우 처리
        if self.df.empty:
            raise ValueError("데이터를 다운로드할 수 없습니다. 티커나 기간을 확인하세요.")

        # 5일 이동평균선(MA5) 추가
        # yfinance 최신 버전에서는 컬럼이 MultiIndex일 수 있으므로 'Close' 접근 시 주의
        if isinstance(self.df.columns, pd.MultiIndex):
            close_col = self.df['Close'][self.ticker]
        else:
            close_col = self.df['Close']
            
        self.df['MA5'] = close_col.rolling(window=5).mean()
        self.df = self.df.dropna() # 이동평균으로 인한 NaN 제거
        
        print(f"Data loaded. Shape: {self.df.shape}")
        
    def get_features_and_target(self):
        # 사용할 Feature: 시가, 고가, 저가, 종가, 거래량, MA5
        # yfinance 구조에 따라 컬럼 선택 방식이 다를 수 있어 안전하게 처리
        try:
            # MultiIndex 처리 (최신 yfinance)
            open_p = self.df['Open'][self.ticker]
            high_p = self.df['High'][self.ticker]
            low_p = self.df['Low'][self.ticker]
            close_p = self.df['Close'][self.ticker]
            vol_p = self.df['Volume'][self.ticker]
        except KeyError:
            # 단일 Index 처리
            open_p = self.df['Open']
            high_p = self.df['High']
            low_p = self.df['Low']
            close_p = self.df['Close']
            vol_p = self.df['Volume']
            
        ma5_p = self.df['MA5'] # 위에서 이미 생성함

        # DataFrame으로 합치기
        features_df = pd.DataFrame({
            'Open': open_p,
            'High': high_p,
            'Low': low_p,
            'Close': close_p,
            'Volume': vol_p,
            'MA5': ma5_p
        })
        
        # X 데이터 (모든 Feature)
        x_values = features_df.values
        # y 데이터 (종가 Close) - 예측 대상
        y_values = features_df[['Close']].values
        
        return x_values, y_values

    def scale_data(self, x_values, y_values):
        # 정규화 (0~1 사이 값으로 변환)
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
        # 예측 후 원래 가격으로 되돌리기 위한 함수
        return self.scaler_y.inverse_transform(y_scaled)

# ---------------------------------------------------------
# 2. 모델 정의 (Stacked RNN)
# ---------------------------------------------------------
class StackedStockRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StackedStockRNN, self).__init__()
        
        # 다층 RNN
        self.rnn = nn.RNN(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 출력층
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (Batch, Seq, Feature)
        # out: (Batch, Seq, Hidden), h_n: (NumLayers, Batch, Hidden)
        out, h_n = self.rnn(x)
        
        # 마지막 층(top layer)의 마지막 시점(last time step) 은닉 상태 사용
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
    print("\nStarting Training...")
    
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
    
    # 정규화된 값을 원래 주가로 복원
    predicted_price = processor.inverse_transform_y(predicted.numpy())
    real_price = processor.inverse_transform_y(y_test.numpy())
    
    return predicted_price, real_price

# ---------------------------------------------------------
# 4. 메인 실행 블록
# ---------------------------------------------------------
def main():
    # A. 데이터 준비
    processor = StockDataProcessor(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'])
    processor.download_and_process()
    
    raw_x, raw_y = processor.get_features_and_target()
    scaled_x, scaled_y = processor.scale_data(raw_x, raw_y)
    
    # 시퀀스 데이터 생성
    X, y = processor.create_sequences(scaled_x, scaled_y, CONFIG['seq_length'])
    
    # 학습/테스트 분리
    train_size = int(len(X) * CONFIG['train_split'])
    X_train_np, y_train_np = X[:train_size], y[:train_size]
    X_test_np, y_test_np = X[train_size:], y[train_size:]
    
    # 텐서 변환
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # B. 모델 생성
    model = StackedStockRNN(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim'],
        num_layers=CONFIG['num_layers']
    )
    print(model)

    # C. 학습
    train_model(model, X_train, y_train, CONFIG)
    
    # D. 평가 및 시각화
    pred_price, real_price = evaluate_model(model, X_test, y_test, processor)
    
    plt.figure(figsize=(12, 6))
    plt.plot(real_price, label='Real Price (GOOGL)', color='blue')
    plt.plot(pred_price, label='Predicted Price', color='red', linestyle='--')
    plt.title(f"{CONFIG['ticker']} Stock Prediction with {CONFIG['input_dim']} Features")
    plt.xlabel('Time (Test Data Index)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()