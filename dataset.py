"""
주가 예측을 위한 PyTorch Dataset 및 DataLoader
- 시계열 데이터 (주가)와 고정 데이터 (펀다멘탈)를 함께 처리
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


class StockDataset(Dataset):
    """
    주가 예측을 위한 커스텀 Dataset

    각 샘플 구성:
    - sequence_data: [seq_length, num_features] - 시계열 주가 데이터
    - static_data: [3] - 고정 펀다멘탈 지표 (PER, PBR, ROE)
    - target: [1] - 예측 대상 (다음날 종가)
    """

    def __init__(self, tickers, data_dir, fundamentals_df, seq_length=10, scaler_X=None, scaler_y=None, scaler_static=None, fit_scaler=True):
        """
        Args:
            tickers: 사용할 종목 리스트
            data_dir: 데이터 폴더 경로
            fundamentals_df: 펀다멘탈 데이터 DataFrame
            seq_length: 시퀀스 길이 (기본 10일)
            scaler_X: 시계열 데이터 스케일러 (None이면 새로 생성)
            scaler_y: 타겟 스케일러 (None이면 새로 생성)
            scaler_static: 고정 데이터 스케일러 (None이면 새로 생성)
            fit_scaler: 스케일러를 fit할지 여부 (학습 데이터만 True)
        """
        self.seq_length = seq_length
        self.data_dir = data_dir

        # 스케일러 초기화
        self.scaler_X = scaler_X if scaler_X else MinMaxScaler()
        self.scaler_y = scaler_y if scaler_y else MinMaxScaler()
        self.scaler_static = scaler_static if scaler_static else MinMaxScaler()

        # 데이터 로드 및 전처리
        self.samples = []
        self._load_data(tickers, fundamentals_df, fit_scaler)

    def _load_data(self, tickers, fundamentals_df, fit_scaler):
        """모든 종목의 데이터를 로드하고 시퀀스를 생성합니다."""

        all_X = []  # 시계열 피처
        all_y = []  # 타겟 (종가)
        all_static = []  # 고정 데이터

        stock_dir = os.path.join(self.data_dir, 'stock_prices')

        for ticker in tickers:
            try:
                # 주가 데이터 로드
                file_path = os.path.join(stock_dir, f'{ticker}.csv')
                df = pd.read_csv(file_path, index_col=0)

                # 사용할 피처: Open, High, Low, Close, Volume, MA5
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5']
                X_data = df[features].values
                y_data = df[['Close']].values

                # 펀다멘탈 데이터 가져오기
                fund_row = fundamentals_df[fundamentals_df['ticker'] == ticker]
                if len(fund_row) == 0:
                    continue
                static_data = fund_row[['PER', 'PBR', 'ROE']].values[0]

                all_X.append(X_data)
                all_y.append(y_data)
                all_static.append((static_data, len(X_data)))  # (고정데이터, 데이터길이)

            except Exception as e:
                print(f"  {ticker} 로드 실패: {e}")
                continue

        # 스케일러 fit (학습 데이터에서만)
        if fit_scaler:
            all_X_concat = np.vstack(all_X)
            all_y_concat = np.vstack(all_y)
            all_static_only = np.array([s[0] for s in all_static])

            self.scaler_X.fit(all_X_concat)
            self.scaler_y.fit(all_y_concat)
            self.scaler_static.fit(all_static_only)

        # 시퀀스 생성
        for i, (X_data, y_data, (static_data, _)) in enumerate(zip(all_X, all_y, all_static)):
            # 정규화
            X_scaled = self.scaler_X.transform(X_data)
            y_scaled = self.scaler_y.transform(y_data)
            static_scaled = self.scaler_static.transform(static_data.reshape(1, -1))[0]

            # 시퀀스 생성
            for j in range(len(X_scaled) - self.seq_length):
                seq_x = X_scaled[j:j + self.seq_length]
                seq_y = y_scaled[j + self.seq_length]

                self.samples.append({
                    'sequence': torch.tensor(seq_x, dtype=torch.float32),
                    'static': torch.tensor(static_scaled, dtype=torch.float32),
                    'target': torch.tensor(seq_y, dtype=torch.float32)
                })

        print(f"  총 {len(self.samples)}개 샘플 생성 완료")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['sequence'], sample['static'], sample['target']

    def inverse_transform_y(self, y_scaled):
        """정규화된 예측값을 원래 가격으로 변환"""
        if isinstance(y_scaled, torch.Tensor):
            y_scaled = y_scaled.detach().cpu().numpy()
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1))


def create_dataloaders(data_dir, batch_size=64, seq_length=10, num_workers=0):
    """
    학습용/테스트용 DataLoader를 생성합니다.

    Args:
        data_dir: 데이터 폴더 경로
        batch_size: 배치 크기
        seq_length: 시퀀스 길이
        num_workers: 데이터 로딩 워커 수

    Returns:
        train_loader, test_loader, train_dataset, test_dataset
    """
    print("=" * 50)
    print("DataLoader 생성 중...")
    print("=" * 50)

    # 종목 리스트 로드
    train_tickers = pd.read_csv(os.path.join(data_dir, 'train_tickers.csv'))['ticker'].tolist()
    test_tickers = pd.read_csv(os.path.join(data_dir, 'test_tickers.csv'))['ticker'].tolist()

    # 펀다멘탈 데이터 로드
    fundamentals_df = pd.read_csv(os.path.join(data_dir, 'fundamentals.csv'))

    print(f"\n학습 종목: {len(train_tickers)}개")
    print(f"테스트 종목: {len(test_tickers)}개")

    # 학습 데이터셋 생성 (스케일러 fit)
    print(f"\n[학습 데이터셋 생성]")
    train_dataset = StockDataset(
        tickers=train_tickers,
        data_dir=data_dir,
        fundamentals_df=fundamentals_df,
        seq_length=seq_length,
        fit_scaler=True
    )

    # 테스트 데이터셋 생성 (학습 데이터의 스케일러 사용)
    print(f"\n[테스트 데이터셋 생성]")
    test_dataset = StockDataset(
        tickers=test_tickers,
        data_dir=data_dir,
        fundamentals_df=fundamentals_df,
        seq_length=seq_length,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y,
        scaler_static=train_dataset.scaler_static,
        fit_scaler=False
    )

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"\n[DataLoader 정보]")
    print(f"  배치 크기: {batch_size}")
    print(f"  시퀀스 길이: {seq_length}")
    print(f"  학습 배치 수: {len(train_loader)}")
    print(f"  테스트 배치 수: {len(test_loader)}")

    # 샘플 데이터 shape 확인
    seq, static, target = train_dataset[0]
    print(f"\n[샘플 데이터 Shape]")
    print(f"  시퀀스 (sequence): {seq.shape}")  # [seq_length, 6]
    print(f"  고정 데이터 (static): {static.shape}")  # [3]
    print(f"  타겟 (target): {target.shape}")  # [1]

    return train_loader, test_loader, train_dataset, test_dataset


# 테스트 코드
if __name__ == "__main__":
    data_dir = "./data"

    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        seq_length=10
    )

    # 배치 샘플 확인
    print("\n[배치 샘플 확인]")
    for batch_idx, (seq, static, target) in enumerate(train_loader):
        print(f"  배치 {batch_idx}: seq={seq.shape}, static={static.shape}, target={target.shape}")
        if batch_idx >= 2:
            break
