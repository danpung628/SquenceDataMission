"""
하이브리드 주가 예측 모델
- 시계열 데이터: RNN/LSTM/Transformer
- 고정 데이터: Linear Layer
- 두 출력을 Concatenate하여 최종 예측
"""
import torch
import torch.nn as nn
import math


# ---------------------------------------------------------
# 1. 하이브리드 RNN 모델
# ---------------------------------------------------------
class HybridRNN(nn.Module):
    """
    시계열 데이터 (RNN) + 고정 데이터 (Linear) 결합 모델

    구조:
    [시퀀스] → RNN → hidden_dim
                          ↘
                           Concat → FC → 예측값
                          ↗
    [고정]   → Linear → static_hidden_dim
    """

    def __init__(self, seq_input_dim=6, hidden_dim=64, num_layers=3,
                 static_input_dim=3, static_hidden_dim=16, output_dim=1, dropout=0.2):
        super(HybridRNN, self).__init__()

        # 시계열 처리: RNN
        self.rnn = nn.RNN(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 고정 데이터 처리: Linear
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 결합 후 출력층
        combined_dim = hidden_dim + static_hidden_dim
        self.output_fc = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, seq_x, static_x):
        """
        Args:
            seq_x: [batch, seq_len, seq_input_dim] - 시계열 데이터
            static_x: [batch, static_input_dim] - 고정 데이터 (PER, PBR, ROE)

        Returns:
            output: [batch, output_dim] - 예측값
        """
        # 시계열 처리
        _, h_n = self.rnn(seq_x)  # h_n: [num_layers, batch, hidden]
        seq_out = h_n[-1]  # 마지막 레이어의 hidden state [batch, hidden]

        # 고정 데이터 처리
        static_out = self.static_fc(static_x)  # [batch, static_hidden]

        # Concatenate
        combined = torch.cat([seq_out, static_out], dim=1)  # [batch, hidden + static_hidden]

        # 최종 예측
        output = self.output_fc(combined)
        return output


# ---------------------------------------------------------
# 2. 하이브리드 LSTM 모델
# ---------------------------------------------------------
class HybridLSTM(nn.Module):
    """
    시계열 데이터 (LSTM) + 고정 데이터 (Linear) 결합 모델
    """

    def __init__(self, seq_input_dim=6, hidden_dim=64, num_layers=3,
                 static_input_dim=3, static_hidden_dim=16, output_dim=1, dropout=0.2):
        super(HybridLSTM, self).__init__()

        # 시계열 처리: LSTM
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 고정 데이터 처리: Linear
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 결합 후 출력층
        combined_dim = hidden_dim + static_hidden_dim
        self.output_fc = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, seq_x, static_x):
        # 시계열 처리 (LSTM은 hidden, cell state 반환)
        _, (h_n, _) = self.lstm(seq_x)
        seq_out = h_n[-1]  # [batch, hidden]

        # 고정 데이터 처리
        static_out = self.static_fc(static_x)

        # Concatenate & 출력
        combined = torch.cat([seq_out, static_out], dim=1)
        output = self.output_fc(combined)
        return output


# ---------------------------------------------------------
# 3. Positional Encoding (Transformer용)
# ---------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------
# 4. 하이브리드 Transformer 모델
# ---------------------------------------------------------
class HybridTransformer(nn.Module):
    """
    시계열 데이터 (Transformer Encoder) + 고정 데이터 (Linear) 결합 모델
    """

    def __init__(self, seq_input_dim=6, d_model=64, nhead=4, num_layers=3,
                 static_input_dim=3, static_hidden_dim=16, output_dim=1, dropout=0.1):
        super(HybridTransformer, self).__init__()

        self.d_model = d_model

        # 입력 임베딩
        self.input_embedding = nn.Linear(seq_input_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 고정 데이터 처리: Linear
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 결합 후 출력층
        combined_dim = d_model + static_hidden_dim
        self.output_fc = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, seq_x, static_x):
        # 입력 임베딩 & 스케일링
        x = self.input_embedding(seq_x) * math.sqrt(self.d_model)

        # Positional Encoding
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # 마지막 시점 출력 사용
        seq_out = x[:, -1, :]  # [batch, d_model]

        # 고정 데이터 처리
        static_out = self.static_fc(static_x)

        # Concatenate & 출력
        combined = torch.cat([seq_out, static_out], dim=1)
        output = self.output_fc(combined)
        return output


# ---------------------------------------------------------
# 모델 생성 헬퍼 함수
# ---------------------------------------------------------
def create_model(model_type, config=None):
    """
    모델 타입에 따라 모델 인스턴스 생성

    Args:
        model_type: 'rnn', 'lstm', 'transformer'
        config: 모델 설정 딕셔너리

    Returns:
        model: 생성된 모델
    """
    if config is None:
        config = {
            'seq_input_dim': 6,
            'hidden_dim': 64,
            'num_layers': 3,
            'static_input_dim': 3,
            'static_hidden_dim': 16,
            'output_dim': 1,
            'dropout': 0.2
        }

    model_type = model_type.lower()

    if model_type == 'rnn':
        return HybridRNN(
            seq_input_dim=config['seq_input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            static_input_dim=config['static_input_dim'],
            static_hidden_dim=config['static_hidden_dim'],
            output_dim=config['output_dim'],
            dropout=config['dropout']
        )
    elif model_type == 'lstm':
        return HybridLSTM(
            seq_input_dim=config['seq_input_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            static_input_dim=config['static_input_dim'],
            static_hidden_dim=config['static_hidden_dim'],
            output_dim=config['output_dim'],
            dropout=config['dropout']
        )
    elif model_type == 'transformer':
        return HybridTransformer(
            seq_input_dim=config['seq_input_dim'],
            d_model=config['hidden_dim'],
            nhead=config.get('nhead', 4),
            num_layers=config['num_layers'],
            static_input_dim=config['static_input_dim'],
            static_hidden_dim=config['static_hidden_dim'],
            output_dim=config['output_dim'],
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# 테스트 코드
if __name__ == "__main__":
    # 더미 데이터
    batch_size = 32
    seq_len = 10
    seq_input_dim = 6
    static_input_dim = 3

    seq_x = torch.randn(batch_size, seq_len, seq_input_dim)
    static_x = torch.randn(batch_size, static_input_dim)

    print("=" * 50)
    print("모델 테스트")
    print("=" * 50)

    for model_type in ['rnn', 'lstm', 'transformer']:
        model = create_model(model_type)
        output = model(seq_x, static_x)
        print(f"\n{model_type.upper()}")
        print(f"  입력 - 시퀀스: {seq_x.shape}, 고정: {static_x.shape}")
        print(f"  출력: {output.shape}")

        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  파라미터 수: {total_params:,}")
