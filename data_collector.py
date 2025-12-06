"""
S&P 500 IT 섹터 주가 및 펀다멘탈 데이터 수집
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import time
import os

# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
CONFIG = {
    'start_date': '2021-01-01',
    'end_date': '2025-11-30',
    'train_ratio': 0.8,
    'data_dir': './data'
}

# ---------------------------------------------------------
# 1. S&P 500 IT 섹터 종목 리스트 가져오기
# ---------------------------------------------------------
def get_sp500_it_tickers():
    """Wikipedia에서 S&P 500 IT 섹터 종목 리스트를 가져옵니다."""
    print("S&P 500 IT 섹터 종목 리스트 가져오는 중...")

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    sp500 = tables[0]

    # IT 섹터 필터링
    it_sector = sp500[sp500['GICS Sector'] == 'Information Technology']
    tickers = it_sector['Symbol'].tolist()

    # 특수문자 처리 (예: BRK.B -> BRK-B)
    tickers = [t.replace('.', '-') for t in tickers]

    print(f"총 {len(tickers)}개 IT 섹터 종목 확인")
    return tickers, it_sector[['Symbol', 'Security']].reset_index(drop=True)

# ---------------------------------------------------------
# 2. 주가 데이터 수집
# ---------------------------------------------------------
def download_stock_data(tickers, start_date, end_date):
    """
    여러 종목의 주가 데이터를 다운로드합니다.

    Returns:
        dict: {ticker: DataFrame} 형태
    """
    print(f"\n주가 데이터 다운로드 중... ({start_date} ~ {end_date})")

    stock_data = {}
    failed_tickers = []

    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if len(hist) > 100:  # 최소 100일 이상 데이터가 있는 종목만
                # 5일 이동평균 추가
                hist['MA5'] = hist['Close'].rolling(window=5).mean()
                hist = hist.dropna()
                stock_data[ticker] = hist
                print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(hist)} 거래일 ✓")
            else:
                print(f"  [{i+1}/{len(tickers)}] {ticker}: 데이터 부족 (skip)")
                failed_tickers.append(ticker)

        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {ticker}: 에러 - {str(e)[:50]}")
            failed_tickers.append(ticker)

        # API 제한 방지
        if (i + 1) % 10 == 0:
            time.sleep(1)

    print(f"\n주가 데이터 수집 완료: {len(stock_data)}개 종목")
    if failed_tickers:
        print(f"수집 실패: {failed_tickers}")

    return stock_data

# ---------------------------------------------------------
# 3. 펀다멘탈 지표 수집 (PER, PBR, ROE)
# ---------------------------------------------------------
def download_fundamental_data(tickers):
    """
    펀다멘탈 지표 (PER, PBR, ROE)를 수집합니다.

    Returns:
        DataFrame: 종목별 펀다멘탈 지표
    """
    print("\n펀다멘탈 지표 수집 중...")

    fundamentals = []

    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            pe_ratio = info.get('trailingPE', None)
            pb_ratio = info.get('priceToBook', None)
            roe = info.get('returnOnEquity', None)

            fundamentals.append({
                'ticker': ticker,
                'PER': pe_ratio,
                'PBR': pb_ratio,
                'ROE': roe
            })

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(tickers)} 완료...")

        except Exception as e:
            print(f"  {ticker}: 에러 - {str(e)[:30]}")
            fundamentals.append({
                'ticker': ticker,
                'PER': None,
                'PBR': None,
                'ROE': None
            })

        # API 제한 방지
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    df = pd.DataFrame(fundamentals)

    # 결측값 처리 (중앙값으로 대체)
    for col in ['PER', 'PBR', 'ROE']:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    print(f"\n펀다멘탈 지표 수집 완료")
    print(df.describe())

    return df

# ---------------------------------------------------------
# 4. 학습/테스트 종목 분리
# ---------------------------------------------------------
def split_train_test_tickers(tickers, train_ratio=0.8, random_seed=42):
    """
    종목을 학습용과 테스트용으로 분리합니다.
    """
    np.random.seed(random_seed)
    shuffled = np.random.permutation(tickers)

    train_size = int(len(shuffled) * train_ratio)
    train_tickers = list(shuffled[:train_size])
    test_tickers = list(shuffled[train_size:])

    print(f"\n종목 분리 완료:")
    print(f"  학습용: {len(train_tickers)}개")
    print(f"  테스트용: {len(test_tickers)}개")
    print(f"\n테스트 종목: {test_tickers}")

    return train_tickers, test_tickers

# ---------------------------------------------------------
# 5. 데이터 저장
# ---------------------------------------------------------
def save_data(stock_data, fundamentals, train_tickers, test_tickers, data_dir):
    """수집한 데이터를 파일로 저장합니다."""

    os.makedirs(data_dir, exist_ok=True)

    # 1) 주가 데이터 저장 (각 종목별 CSV)
    stock_dir = os.path.join(data_dir, 'stock_prices')
    os.makedirs(stock_dir, exist_ok=True)

    for ticker, df in stock_data.items():
        df.to_csv(os.path.join(stock_dir, f'{ticker}.csv'))

    # 2) 펀다멘탈 데이터 저장
    fundamentals.to_csv(os.path.join(data_dir, 'fundamentals.csv'), index=False)

    # 3) 종목 리스트 저장
    pd.DataFrame({'ticker': train_tickers}).to_csv(
        os.path.join(data_dir, 'train_tickers.csv'), index=False
    )
    pd.DataFrame({'ticker': test_tickers}).to_csv(
        os.path.join(data_dir, 'test_tickers.csv'), index=False
    )

    print(f"\n데이터 저장 완료: {data_dir}")

# ---------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("S&P 500 IT 섹터 데이터 수집 시작")
    print("=" * 60)

    # 1. 종목 리스트 가져오기
    tickers, ticker_info = get_sp500_it_tickers()

    # 2. 주가 데이터 수집
    stock_data = download_stock_data(
        tickers,
        CONFIG['start_date'],
        CONFIG['end_date']
    )

    # 성공적으로 수집된 종목만 사용
    valid_tickers = list(stock_data.keys())

    # 3. 펀다멘탈 데이터 수집
    fundamentals = download_fundamental_data(valid_tickers)

    # 4. 학습/테스트 분리
    train_tickers, test_tickers = split_train_test_tickers(
        valid_tickers,
        CONFIG['train_ratio']
    )

    # 5. 데이터 저장
    save_data(stock_data, fundamentals, train_tickers, test_tickers, CONFIG['data_dir'])

    print("\n" + "=" * 60)
    print("데이터 수집 완료!")
    print("=" * 60)

    # 요약 정보 출력
    print(f"\n[요약]")
    print(f"  - 총 종목 수: {len(valid_tickers)}")
    print(f"  - 학습 종목: {len(train_tickers)}")
    print(f"  - 테스트 종목: {len(test_tickers)}")
    print(f"  - 기간: {CONFIG['start_date']} ~ {CONFIG['end_date']}")
    print(f"  - 저장 위치: {CONFIG['data_dir']}")

    return stock_data, fundamentals, train_tickers, test_tickers

if __name__ == "__main__":
    stock_data, fundamentals, train_tickers, test_tickers = main()
