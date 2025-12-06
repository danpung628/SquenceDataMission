import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------
# 설정 (Configuration)
# ---------------------------------------------------------
CONFIG = {
    'ticker': 'GOOGL',        
    'start_date': '2020-01-01',
    'end_date': '2025-11-30',
    'seq_length': 10,         
    'input_dim': 6,           # Feature 개수
    'd_model': 64,            # Transformer 내부 차원 (Hidden Dim)
    'nhead': 4,               # Multi-head Attention 헤드 개수 (64 / 4 = 16)
    'num_layers': 3,          # Encoder Layer 개수
    'output_dim': 1,
    'epochs': 100,
    'learning_rate': 0.001,
    'train_split': 0.8,
    'dropout': 0.1
}

# ---------------------------------------------------------
# 1. 데이터 처리 클래스 (기존과 동일)
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
# 2. Positional Encoding (순서 정보 주입)
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # PE 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (Batch, Seq, Feature) 형태에 맞추기 위해 차원 추가 -> (1, MaxLen, D_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq, D_model)
        # x의 시퀀스 길이만큼 PE를 잘라서 더함
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ---------------------------------------------------------
# 3. Transformer Encoder 모델 정의
# ---------------------------------------------------------
class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super(StockTransformer, self).__init__()
        
        # 1. Input Embedding: 입력 Feature(6) -> d_model(64)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # 3. Transformer Encoder Layer
        # batch_first=True: 입력이 (Batch, Seq, Feature) 형태임을 명시
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, # 보통 d_model의 4배 사용
            dropout=dropout,
            batch_first=True 
        )
        
        # 4. Encoder Stacking
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output Layer
        self.decoder = nn.Linear(d_model, output_dim)
        
        self.d_model = d_model

    def forward(self, x):
        # x: (Batch, Seq, Input_Dim)
        
        # Embedding & Scaling (논문에 따라 sqrt(d_model)을 곱해주기도 함)
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Positional Encoding 더하기
        x = self.pos_encoder(x)
        
        # Transformer Encoder 통과
        # output: (Batch, Seq, d_model)
        output = self.transformer_encoder(x)
        
        # 예측을 위해 마지막 시점(t)의 output만 사용
        # output[:, -1, :] -> (Batch, d_model)
        last_output = output[:, -1, :]
        
        # 최종 예측값
        prediction = self.decoder(last_output)
        
        return prediction

# ---------------------------------------------------------
# 4. 학습 및 평가 함수
# ---------------------------------------------------------
def train_model(model, X_train, y_train, config):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    model.train()
    print("\nStarting Training (Transformer)...")
    
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
# 5. 메인 실행 블록
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
    
    # 모델 생성
    model = StockTransformer(
        input_dim=CONFIG['input_dim'],
        d_model=CONFIG['d_model'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        output_dim=CONFIG['output_dim'],
        dropout=CONFIG['dropout']
    )
    print(model)

    train_model(model, X_train, y_train, CONFIG)
    
    pred_price, real_price = evaluate_model(model, X_test, y_test, processor)
    
    plt.figure(figsize=(12, 6))
    plt.plot(real_price, label='Real Price (GOOGL)', color='blue')
    plt.plot(pred_price, label='Predicted Price (Transformer)', color='purple', linestyle='--')
    plt.title(f"{CONFIG['ticker']} Stock Prediction (Transformer Encoder)")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()