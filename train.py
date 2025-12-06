"""
하이브리드 모델 학습 및 비교 스크립트
- RNN, LSTM, Transformer 모델 학습
- 학습 시간, Loss, 예측 성능 비교
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from dataset import create_dataloaders
from models import create_model


# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------
CONFIG = {
    # 데이터
    'data_dir': './data',
    'seq_length': 10,
    'batch_size': 64,

    # 모델
    'seq_input_dim': 6,      # Open, High, Low, Close, Volume, MA5
    'hidden_dim': 64,
    'num_layers': 3,
    'static_input_dim': 3,   # PER, PBR, ROE
    'static_hidden_dim': 16,
    'output_dim': 1,
    'dropout': 0.2,
    'nhead': 4,              # Transformer용

    # 학습
    'epochs': 50,
    'learning_rate': 0.001,

    # 저장
    'save_dir': './results'
}


# ---------------------------------------------------------
# 학습 함수
# ---------------------------------------------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    num_batches = 0

    for seq_x, static_x, target in train_loader:
        seq_x = seq_x.to(device)
        static_x = static_x.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(seq_x, static_x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, test_loader, criterion, device):
    """평가"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for seq_x, static_x, target in test_loader:
            seq_x = seq_x.to(device)
            static_x = static_x.to(device)
            target = target.to(device)

            output = model(seq_x, static_x)
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    avg_loss = total_loss / num_batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return avg_loss, all_preds, all_targets


def train_model(model_type, train_loader, test_loader, config, device):
    """
    모델 학습 및 평가

    Returns:
        results: 학습 결과 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f" {model_type.upper()} 모델 학습")
    print(f"{'='*60}")

    # 모델 생성
    model = create_model(model_type, config).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 손실 함수 & 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 학습 기록
    train_losses = []
    test_losses = []

    # 학습 시작
    start_time = time.time()

    for epoch in range(config['epochs']):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, _, _ = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{config['epochs']}] "
                  f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

    end_time = time.time()
    training_time = end_time - start_time

    # 최종 평가
    final_test_loss, preds, targets = evaluate(model, test_loader, criterion, device)

    print(f"\n  학습 완료!")
    print(f"  총 학습 시간: {training_time:.2f}초")
    print(f"  최종 Test Loss: {final_test_loss:.6f}")

    return {
        'model_type': model_type,
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_time': training_time,
        'final_test_loss': final_test_loss,
        'predictions': preds,
        'targets': targets
    }


# ---------------------------------------------------------
# 시각화 함수
# ---------------------------------------------------------
def plot_loss_comparison(results_list, save_path=None):
    """Loss 비교 그래프"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'rnn': 'red', 'lstm': 'green', 'transformer': 'purple'}

    # Train Loss
    ax1 = axes[0]
    for result in results_list:
        model_type = result['model_type']
        ax1.plot(result['train_losses'], label=model_type.upper(), color=colors[model_type])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)

    # Test Loss
    ax2 = axes[1]
    for result in results_list:
        model_type = result['model_type']
        ax2.plot(result['test_losses'], label=model_type.upper(), color=colors[model_type])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss 그래프 저장: {save_path}")

    plt.show()


def plot_prediction_comparison(results_list, test_dataset, num_samples=200, save_path=None):
    """예측 vs 실제 비교 그래프"""
    fig, axes = plt.subplots(len(results_list), 1, figsize=(14, 4*len(results_list)))

    if len(results_list) == 1:
        axes = [axes]

    colors = {'rnn': 'red', 'lstm': 'green', 'transformer': 'purple'}

    for ax, result in zip(axes, results_list):
        model_type = result['model_type']
        preds = result['predictions'][:num_samples].numpy()
        targets = result['targets'][:num_samples].numpy()

        # 원래 가격으로 변환
        preds_price = test_dataset.inverse_transform_y(preds)
        targets_price = test_dataset.inverse_transform_y(targets)

        ax.plot(targets_price, label='Actual', color='blue', linewidth=1.5)
        ax.plot(preds_price, label=f'Predicted ({model_type.upper()})',
                color=colors[model_type], linestyle='--', linewidth=1.5)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Stock Price')
        ax.set_title(f'{model_type.upper()} - Actual vs Predicted')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"예측 그래프 저장: {save_path}")

    plt.show()


def print_summary(results_list):
    """학습 결과 요약 출력"""
    print("\n" + "=" * 60)
    print(" 학습 결과 요약")
    print("=" * 60)

    print(f"\n{'모델':<15} {'학습 시간(초)':<15} {'최종 Test Loss':<20} {'파라미터 수':<15}")
    print("-" * 65)

    for result in results_list:
        model_type = result['model_type'].upper()
        training_time = result['training_time']
        final_loss = result['final_test_loss']
        num_params = sum(p.numel() for p in result['model'].parameters())

        print(f"{model_type:<15} {training_time:<15.2f} {final_loss:<20.6f} {num_params:<15,}")

    print("-" * 65)


# ---------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------
def main():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 결과 저장 폴더 생성
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # DataLoader 생성
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        data_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        seq_length=CONFIG['seq_length']
    )

    # 3가지 모델 학습
    model_types = ['rnn', 'lstm', 'transformer']
    results_list = []

    for model_type in model_types:
        result = train_model(model_type, train_loader, test_loader, CONFIG, device)
        results_list.append(result)

        # 모델 저장
        model_path = os.path.join(CONFIG['save_dir'], f'{model_type}_model.pt')
        torch.save(result['model'].state_dict(), model_path)
        print(f"  모델 저장: {model_path}")

    # 결과 요약
    print_summary(results_list)

    # 시각화
    plot_loss_comparison(
        results_list,
        save_path=os.path.join(CONFIG['save_dir'], 'loss_comparison.png')
    )

    plot_prediction_comparison(
        results_list,
        test_dataset,
        num_samples=200,
        save_path=os.path.join(CONFIG['save_dir'], 'prediction_comparison.png')
    )

    return results_list


if __name__ == "__main__":
    results = main()
