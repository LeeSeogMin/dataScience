# 10-5-deepsurv.py
"""
DeepSurv 딥러닝 생존 분석 실습
- PyTorch 기반 Cox 부분 우도 손실 구현
- 신경망을 통한 비선형 위험 함수 학습
- 충분히 큰 합성 데이터로 딥러닝의 장점 확인
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter

warnings.filterwarnings('ignore')

# 폰트 설정 (크로스 플랫폼)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리 설정
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DIAGRAM_DIR = Path(__file__).parents[3] / "diagram"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

# 재현성
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def generate_synthetic_survival_data(n_samples=5000, n_features=20, seed=42):
    """
    비선형 관계를 포함한 합성 생존 데이터 생성
    - 딥러닝이 선형 모형보다 유리한 패턴 포함
    - 다층 상호작용 및 비선형 효과 반영
    """
    np.random.seed(seed)

    # 특성 생성
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # 비선형 위험 함수 설계
    # 1. 비선형 주효과
    log_hazard = (
        0.5 * np.sin(2 * X[:, 0]) +                    # 사인파 비선형
        0.3 * X[:, 1]**2 +                              # 이차 효과
        0.4 * np.tanh(X[:, 2]) +                        # S자 곡선
        0.3 * np.abs(X[:, 3]) +                         # 절대값 효과
        0.2 * X[:, 4] * np.sign(X[:, 5]) +             # 부호 상호작용
        0.25 * (X[:, 0] * X[:, 1]) +                   # 2차 상호작용
        0.2 * (X[:, 2] * X[:, 3] * X[:, 4]) +          # 3차 상호작용
        0.15 * np.maximum(0, X[:, 6] - 0.5) +          # ReLU 유사 효과
        0.1 * np.exp(-X[:, 7]**2) +                    # 가우시안 효과
        0.1 * X[:, 8]                                   # 선형 효과 일부 포함
    )

    # 기저 위험 스케일 조정
    baseline_hazard = 0.02
    scale = np.exp(-log_hazard) / baseline_hazard

    # Weibull 분포에서 생존 시간 생성
    shape = 1.5
    survival_time = scale * np.random.weibull(shape, n_samples)

    # 중도절단 시간 (지수 분포)
    censoring_time = np.random.exponential(np.median(survival_time) * 1.2, n_samples)

    # 관측 시간 및 사건 상태
    observed_time = np.minimum(survival_time, censoring_time).astype(np.float32)
    event = (survival_time <= censoring_time).astype(np.float32)

    # 극단값 제한
    max_time = np.percentile(observed_time, 99)
    observed_time = np.clip(observed_time, 0.01, max_time)

    # DataFrame 생성
    feature_cols = [f'x{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['time'] = observed_time
    df['event'] = event.astype(int)

    return df, feature_cols


class DeepSurv(nn.Module):
    """DeepSurv 신경망 모델"""

    def __init__(self, in_features, hidden_layers=[64, 32, 16], dropout=0.2):
        super(DeepSurv, self).__init__()

        layers = []
        prev_dim = in_features

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 최종 출력: 단일 위험 점수
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def cox_partial_likelihood_loss(risk_scores, time, event):
    """
    Cox 부분 우도 손실 함수
    - risk_scores: 모델이 예측한 위험 점수 (log hazard)
    - time: 관측 시간
    - event: 사건 발생 여부 (1=사건, 0=중도절단)
    """
    # 시간 순으로 정렬
    sorted_idx = torch.argsort(time, descending=True)
    risk_sorted = risk_scores[sorted_idx]
    event_sorted = event[sorted_idx]

    # 위험 집합의 로그 합 (역순 누적 합)
    hazard_ratio = torch.exp(risk_sorted)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))

    # 사건이 발생한 경우만 손실 계산
    uncensored_likelihood = risk_sorted - log_risk
    censored_likelihood = uncensored_likelihood * event_sorted

    # 사건 수로 정규화
    num_events = event_sorted.sum()
    if num_events > 0:
        loss = -censored_likelihood.sum() / num_events
    else:
        loss = torch.tensor(0.0)

    return loss


def train_deepsurv(model, train_loader, val_X, val_time, val_event,
                   epochs=150, lr=0.001, weight_decay=1e-4):
    """DeepSurv 모델 학습"""

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )

    history = {'train_loss': [], 'val_c_index': []}
    best_c_index = 0
    best_state = None
    patience_counter = 0
    early_stop_patience = 30

    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0

        for batch_X, batch_time, batch_event in train_loader:
            optimizer.zero_grad()
            risk_scores = model(batch_X)
            loss = cox_partial_likelihood_loss(risk_scores, batch_time, batch_event)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # 검증
        model.eval()
        with torch.no_grad():
            val_risk = model(torch.tensor(val_X)).numpy()

        c_index = concordance_index_censored(
            val_event.astype(bool), val_time, val_risk
        )[0]
        history['val_c_index'].append(c_index)

        scheduler.step(c_index)

        if c_index > best_c_index:
            best_c_index = c_index
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, Val C-index={c_index:.4f}")

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # 최적 가중치 복원
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_c_index


def train_cox_baseline(train_df, val_df, feature_cols):
    """Cox PH 모형 기준선 학습"""
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df[feature_cols + ['time', 'event']],
            duration_col='time', event_col='event')

    # 검증 성능
    val_risk = cph.predict_partial_hazard(val_df[feature_cols]).values.flatten()
    val_c_index = concordance_index_censored(
        val_df['event'].astype(bool),
        val_df['time'].values,
        val_risk
    )[0]

    return cph, val_c_index


def main():
    """DeepSurv 메인 함수"""

    print("=" * 60)
    print("10.5 DeepSurv 딥러닝 생존 분석 실습")
    print("=" * 60)

    # 1. 합성 데이터 생성
    N_SAMPLES = 5000
    N_FEATURES = 20

    print(f"\n[합성 데이터 생성]")
    print(f"- 표본 수: {N_SAMPLES}명")
    print(f"- 특성 수: {N_FEATURES}개")
    print(f"- 비선형 관계 및 다층 상호작용 포함")

    data, feature_cols = generate_synthetic_survival_data(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        seed=SEED
    )

    event_rate = data['event'].mean()
    print(f"- 사건 발생률: {event_rate:.1%}")

    # 2. 데이터 분할
    train_df, temp_df = train_test_split(data, test_size=0.3, random_state=SEED,
                                          stratify=data['event'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED,
                                        stratify=temp_df['event'])

    print(f"\n[데이터 분할]")
    print(f"- 학습: {len(train_df)}명 (사건: {train_df['event'].sum()})")
    print(f"- 검증: {len(val_df)}명 (사건: {val_df['event'].sum()})")
    print(f"- 테스트: {len(test_df)}명 (사건: {test_df['event'].sum()})")

    # 3. 데이터 준비
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    time_train = train_df['time'].values.astype(np.float32)
    time_val = val_df['time'].values.astype(np.float32)
    time_test = test_df['time'].values.astype(np.float32)

    event_train = train_df['event'].values.astype(np.float32)
    event_val = val_df['event'].values.astype(np.float32)
    event_test = test_df['event'].values.astype(np.float32)

    # 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Cox PH 기준선
    print("\n[Cox PH 기준선 모형]")
    train_df_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    train_df_scaled['time'] = time_train
    train_df_scaled['event'] = event_train.astype(int)

    val_df_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
    val_df_scaled['time'] = time_val
    val_df_scaled['event'] = event_val.astype(int)

    test_df_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    test_df_scaled['time'] = time_test
    test_df_scaled['event'] = event_test.astype(int)

    cph, cox_val_c_index = train_cox_baseline(train_df_scaled, val_df_scaled, feature_cols)

    cox_test_risk = cph.predict_partial_hazard(test_df_scaled[feature_cols]).values.flatten()
    cox_test_c_index = concordance_index_censored(
        event_test.astype(bool), time_test, cox_test_risk
    )[0]

    print(f"- Cox 검증 C-index: {cox_val_c_index:.4f}")
    print(f"- Cox 테스트 C-index: {cox_test_c_index:.4f}")

    # 5. DataLoader 생성
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled),
        torch.tensor(time_train),
        torch.tensor(event_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 6. DeepSurv 모델 초기화
    print("\n[DeepSurv 모델]")
    model = DeepSurv(
        in_features=N_FEATURES,
        hidden_layers=[64, 32, 16],
        dropout=0.2
    )
    print(model)

    # 7. 학습
    print("\n[모델 학습]")
    model, history, best_val_c_index = train_deepsurv(
        model, train_loader,
        X_val_scaled, time_val, event_val,
        epochs=150, lr=0.001, weight_decay=1e-4
    )

    print(f"\n최고 검증 C-index: {best_val_c_index:.4f}")

    # 8. 테스트 평가
    model.eval()
    with torch.no_grad():
        test_risk = model(torch.tensor(X_test_scaled)).numpy()

    test_c_index = concordance_index_censored(
        event_test.astype(bool), time_test, test_risk
    )[0]

    print(f"\n[테스트 성능 비교]")
    print(f"- Cox PH 테스트 C-index:    {cox_test_c_index:.4f}")
    print(f"- DeepSurv 테스트 C-index:  {test_c_index:.4f}")
    print(f"- 성능 향상: {(test_c_index - cox_test_c_index)*100:.2f}%p")

    # 9. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 9-1. 학습 곡선
    ax1 = axes[0]
    epochs_range = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs_range, history['train_loss'], 'k-', linewidth=2, label='Train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cox Partial Likelihood Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs_range, history['val_c_index'], 'k--', linewidth=2, label='Validation C-index')
    ax1_twin.set_ylabel('C-index')
    ax1_twin.axhline(y=cox_val_c_index, color='gray', linestyle=':', linewidth=1.5,
                     label=f'Cox baseline ({cox_val_c_index:.3f})')
    ax1_twin.legend(loc='lower right')

    # 9-2. 위험 점수 분포
    ax2 = axes[1]
    ax2.hist(test_risk[event_test == 1], bins=30, alpha=0.8, label='Event',
             density=True, color='black', edgecolor='white')
    ax2.hist(test_risk[event_test == 0], bins=30, alpha=0.5, label='Censored',
             density=True, color='gray', edgecolor='white')
    ax2.set_xlabel('Predicted risk score')
    ax2.set_ylabel('Density')
    ax2.set_title(f'Risk Score Distribution (Test C-index: {test_c_index:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    plt.savefig(DIAGRAM_DIR / '10-5-deepsurv.png', dpi=150, bbox_inches='tight')
    plt.savefig(DATA_DIR / '10-5-deepsurv.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[저장] {DIAGRAM_DIR / '10-5-deepsurv.png'}")

    # 10. 결과 저장
    results = {
        'data_summary': {
            'total': N_SAMPLES,
            'n_features': N_FEATURES,
            'train': int(len(train_df)),
            'val': int(len(val_df)),
            'test': int(len(test_df)),
            'event_rate': float(event_rate),
            'events_test': int(event_test.sum())
        },
        'model_architecture': {
            'input_features': N_FEATURES,
            'hidden_layers': [64, 32, 16],
            'dropout': 0.2
        },
        'training': {
            'epochs_trained': len(history['train_loss']),
            'max_epochs': 150,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 128
        },
        'performance': {
            'cox_val_c_index': float(cox_val_c_index),
            'cox_test_c_index': float(cox_test_c_index),
            'deepsurv_val_c_index': float(best_val_c_index),
            'deepsurv_test_c_index': float(test_c_index),
            'improvement_pp': float((test_c_index - cox_test_c_index) * 100)
        },
        'training_history': {
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_c_index': float(history['val_c_index'][-1])
        }
    }

    with open(DATA_DIR / '10-5-deepsurv.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[저장] {DATA_DIR / '10-5-deepsurv.json'}")

    # 11. 결과 해석
    print("\n[결과 해석]")
    print(f"  비선형 관계가 포함된 합성 데이터(N={N_SAMPLES})에서")
    print(f"  DeepSurv(C-index={test_c_index:.4f})가 Cox PH(C-index={cox_test_c_index:.4f})보다")
    print(f"  {(test_c_index - cox_test_c_index)*100:.2f}%p 높은 예측력을 보였다.")
    print(f"\n  딥러닝 모델은 충분한 데이터(수천 건 이상)와 비선형 관계가")
    print(f"  존재할 때 전통적 모형 대비 이점을 보인다.")

    return results


if __name__ == '__main__':
    main()
