"""
테스트 종목 2개에 대한 상세 예측 검증 스크립트
- NVDA (엔비디아)
- AMD
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

from models import create_model

# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
CONFIG = {
    'data_dir': './data',
    'results_dir': './results',
    'seq_length': 10,

    # 모델 설정 (train.py와 동일)
    'seq_input_dim': 6,
    'hidden_dim': 64,
    'num_layers': 3,
    'static_input_dim': 3,
    'static_hidden_dim': 16,
    'output_dim': 1,
    'dropout': 0.2,
    'nhead': 4,
}

# 검증할 테스트 종목
TARGET_STOCKS = ['NVDA', 'AMD']


# ---------------------------------------------------------
# 데이터 로드 및 전처리
# ---------------------------------------------------------
def load_stock_data(ticker, data_dir):
    """특정 종목의 주가 데이터 로드"""
    file_path = os.path.join(data_dir, 'stock_prices', f'{ticker}.csv')
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


def load_fundamentals(ticker, data_dir):
    """특정 종목의 펀다멘탈 데이터 로드"""
    fund_df = pd.read_csv(os.path.join(data_dir, 'fundamentals.csv'))
    row = fund_df[fund_df['ticker'] == ticker]
    if len(row) == 0:
        raise ValueError(f"{ticker} not found in fundamentals")
    return row[['PER', 'PBR', 'ROE']].values[0]


def prepare_scalers(data_dir):
    """
    학습 데이터 기반으로 스케일러 생성
    (train.py에서 학습할 때 사용한 것과 동일한 스케일러 재현)
    """
    train_tickers = pd.read_csv(os.path.join(data_dir, 'train_tickers.csv'))['ticker'].tolist()
    fund_df = pd.read_csv(os.path.join(data_dir, 'fundamentals.csv'))

    all_X = []
    all_y = []
    all_static = []

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5']

    for ticker in train_tickers:
        try:
            df = load_stock_data(ticker, data_dir)
            X_data = df[features].values
            y_data = df[['Close']].values

            fund_row = fund_df[fund_df['ticker'] == ticker]
            if len(fund_row) > 0:
                static_data = fund_row[['PER', 'PBR', 'ROE']].values[0]
                all_X.append(X_data)
                all_y.append(y_data)
                all_static.append(static_data)
        except:
            continue

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_static = MinMaxScaler()

    scaler_X.fit(np.vstack(all_X))
    scaler_y.fit(np.vstack(all_y))
    scaler_static.fit(np.array(all_static))

    return scaler_X, scaler_y, scaler_static


def create_sequences(df, static_data, scaler_X, scaler_y, scaler_static, seq_length):
    """시퀀스 데이터 생성"""
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5']

    X_data = df[features].values
    y_data = df[['Close']].values

    # 정규화
    X_scaled = scaler_X.transform(X_data)
    y_scaled = scaler_y.transform(y_data)
    static_scaled = scaler_static.transform(static_data.reshape(1, -1))[0]

    # 시퀀스 생성
    sequences = []
    targets = []
    dates = []

    for i in range(len(X_scaled) - seq_length):
        seq_x = X_scaled[i:i + seq_length]
        seq_y = y_scaled[i + seq_length]

        sequences.append(seq_x)
        targets.append(seq_y)
        dates.append(df.index[i + seq_length])

    sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
    targets = torch.tensor(np.array(targets), dtype=torch.float32)
    static_tensor = torch.tensor(static_scaled, dtype=torch.float32).unsqueeze(0).repeat(len(sequences), 1)

    return sequences, static_tensor, targets, dates


# ---------------------------------------------------------
# 모델 예측
# ---------------------------------------------------------
def load_model(model_type, config, results_dir, device):
    """저장된 모델 로드"""
    model = create_model(model_type, config).to(device)
    model_path = os.path.join(results_dir, f'{model_type}_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict(model, sequences, static_data, device):
    """모델 예측 수행"""
    with torch.no_grad():
        sequences = sequences.to(device)
        static_data = static_data.to(device)
        predictions = model(sequences, static_data)
    return predictions.cpu().numpy()


# ---------------------------------------------------------
# 시각화
# ---------------------------------------------------------
def plot_stock_prediction(ticker, dates, actual_prices, predictions_dict, save_path=None):
    """
    특정 종목에 대한 3개 모델 예측 비교 그래프
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    colors = {'rnn': 'red', 'lstm': 'green', 'transformer': 'purple'}

    for ax, (model_type, preds) in zip(axes, predictions_dict.items()):
        ax.plot(dates, actual_prices, label='Actual', color='blue', linewidth=1.5, alpha=0.8)
        ax.plot(dates, preds, label=f'Predicted ({model_type.upper()})',
                color=colors[model_type], linestyle='--', linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price ($)')
        ax.set_title(f'{ticker} - {model_type.upper()} Model Prediction')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # x축 날짜 포맷
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(f'{ticker} Stock Price Prediction Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.show()


def plot_combined_comparison(ticker, dates, actual_prices, predictions_dict, save_path=None):
    """
    모든 모델 예측을 하나의 그래프에 표시
    """
    plt.figure(figsize=(14, 6))

    colors = {'rnn': 'red', 'lstm': 'green', 'transformer': 'purple'}

    # 실제 가격
    plt.plot(dates, actual_prices, label='Actual', color='blue', linewidth=2)

    # 각 모델 예측
    for model_type, preds in predictions_dict.items():
        plt.plot(dates, preds, label=f'{model_type.upper()}',
                color=colors[model_type], linestyle='--', linewidth=1.5, alpha=0.7)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.title(f'{ticker} - All Models Prediction Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"그래프 저장: {save_path}")

    plt.show()


def calculate_metrics(actual, predicted):
    """예측 성능 지표 계산"""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }


def print_metrics_table(ticker, metrics_dict):
    """성능 지표 테이블 출력"""
    print(f"\n{'='*60}")
    print(f" {ticker} 예측 성능 비교")
    print(f"{'='*60}")
    print(f"{'모델':<15} {'MSE':<15} {'RMSE':<15} {'MAE':<15} {'MAPE(%)':<15}")
    print("-" * 75)

    for model_type, metrics in metrics_dict.items():
        print(f"{model_type.upper():<15} {metrics['MSE']:<15.4f} {metrics['RMSE']:<15.4f} "
              f"{metrics['MAE']:<15.4f} {metrics['MAPE']:<15.2f}")
    print("-" * 75)


# ---------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print(" 테스트 종목 예측 검증")
    print(f" 대상 종목: {TARGET_STOCKS}")
    print("=" * 60)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 스케일러 준비
    print("\n스케일러 준비 중...")
    scaler_X, scaler_y, scaler_static = prepare_scalers(CONFIG['data_dir'])

    # 모델 로드
    print("모델 로드 중...")
    models = {}
    for model_type in ['rnn', 'lstm', 'transformer']:
        models[model_type] = load_model(model_type, CONFIG, CONFIG['results_dir'], device)
        print(f"  {model_type.upper()} 모델 로드 완료")

    # 각 종목별 예측 수행
    all_results = {}

    for ticker in TARGET_STOCKS:
        print(f"\n{'='*60}")
        print(f" {ticker} 예측 수행")
        print("=" * 60)

        # 데이터 로드
        df = load_stock_data(ticker, CONFIG['data_dir'])
        static_data = load_fundamentals(ticker, CONFIG['data_dir'])

        print(f"  데이터 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  총 거래일: {len(df)}")
        print(f"  펀다멘탈: PER={static_data[0]:.2f}, PBR={static_data[1]:.2f}, ROE={static_data[2]:.4f}")

        # 시퀀스 생성
        sequences, static_tensor, targets, dates = create_sequences(
            df, static_data, scaler_X, scaler_y, scaler_static, CONFIG['seq_length']
        )

        print(f"  시퀀스 수: {len(sequences)}")

        # 각 모델로 예측
        predictions_dict = {}
        metrics_dict = {}

        # 실제 가격 (역정규화)
        actual_prices = scaler_y.inverse_transform(targets.numpy()).flatten()

        for model_type, model in models.items():
            preds = predict(model, sequences, static_tensor, device)
            preds_prices = scaler_y.inverse_transform(preds).flatten()
            predictions_dict[model_type] = preds_prices

            # 성능 지표 계산
            metrics_dict[model_type] = calculate_metrics(actual_prices, preds_prices)

        # 성능 지표 출력
        print_metrics_table(ticker, metrics_dict)

        # 시각화 - 개별 모델 비교
        plot_stock_prediction(
            ticker, dates, actual_prices, predictions_dict,
            save_path=os.path.join(CONFIG['results_dir'], f'{ticker}_prediction_detail.png')
        )

        # 시각화 - 통합 비교
        plot_combined_comparison(
            ticker, dates, actual_prices, predictions_dict,
            save_path=os.path.join(CONFIG['results_dir'], f'{ticker}_prediction_combined.png')
        )

        all_results[ticker] = {
            'dates': dates,
            'actual': actual_prices,
            'predictions': predictions_dict,
            'metrics': metrics_dict
        }

    # 최종 요약
    print("\n" + "=" * 60)
    print(" 최종 요약")
    print("=" * 60)

    for ticker in TARGET_STOCKS:
        print(f"\n[{ticker}]")
        metrics = all_results[ticker]['metrics']
        best_model = min(metrics.keys(), key=lambda x: metrics[x]['RMSE'])
        print(f"  최고 성능 모델: {best_model.upper()} (RMSE: {metrics[best_model]['RMSE']:.4f})")

    print("\n결과 파일 저장 완료!")
    print(f"  저장 위치: {CONFIG['results_dir']}")

    return all_results


if __name__ == "__main__":
    results = main()
