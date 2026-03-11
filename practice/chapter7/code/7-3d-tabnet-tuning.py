"""
7장 실습: TabNet Optuna 하이퍼파라미터 튜닝
- 고차원 + 복잡한 상호작용 합성 데이터
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ============================================================
# Optuna Objective for TabNet
# ============================================================

def create_objective(X_train, y_train, X_valid, y_valid):
    """Optuna objective function factory for TabNet"""
    from pytorch_tabnet.tab_model import TabNetClassifier

    def objective(trial):
        # 하이퍼파라미터 샘플링
        n_d = trial.suggest_categorical('n_d', [8, 16, 32, 64, 128])
        n_a = trial.suggest_categorical('n_a', [8, 16, 32, 64, 128])
        n_steps = trial.suggest_int('n_steps', 3, 10)
        gamma = trial.suggest_float('gamma', 1.0, 2.0)
        n_independent = trial.suggest_int('n_independent', 1, 5)
        n_shared = trial.suggest_int('n_shared', 1, 5)
        lambda_sparse = trial.suggest_float('lambda_sparse', 1e-6, 1e-2, log=True)
        momentum = trial.suggest_float('momentum', 0.01, 0.4)
        lr = trial.suggest_float('lr', 1e-3, 5e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])

        # TabNet 모델 생성
        tabnet = TabNetClassifier(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            lambda_sparse=lambda_sparse,
            momentum=momentum,
            verbose=0,
            seed=42,
            optimizer_params={'lr': lr}
        )

        # 학습
        try:
            tabnet.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=['auc'],
                max_epochs=100,
                patience=15,
                batch_size=batch_size
            )

            # 검증 AUC
            val_prob = tabnet.predict_proba(X_valid)[:, 1]
            val_auc = roc_auc_score(y_valid, val_prob)

        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.5

        return val_auc

    return objective


# ============================================================
# 메인 실행
# ============================================================

def main():
    # 데이터 로드
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_dir, 'data', 'synthetic_complex.csv')

    print("=" * 60)
    print("TabNet Optuna 하이퍼파라미터 튜닝")
    print("=" * 60)

    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1).values.astype(np.float32)
    y = df['target'].values.astype(np.int64)

    print(f"데이터: {X.shape[0]}건, {X.shape[1]}특성")

    # 데이터 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    print(f"학습: {len(X_train)}, 검증: {len(X_valid)}, 테스트: {len(X_test)}")

    # Optuna 튜닝
    print("\n튜닝 시작 (30 trials)...")
    start_time = time.time()

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    objective = create_objective(X_train, y_train, X_valid, y_valid)
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    tuning_time = time.time() - start_time

    # 최적 파라미터
    print("\n" + "=" * 60)
    print("튜닝 결과")
    print("=" * 60)
    print(f"최고 Validation AUC: {study.best_value:.4f}")
    print(f"튜닝 시간: {tuning_time:.1f}초 ({tuning_time/60:.1f}분)")
    print("\n최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 최적 파라미터로 최종 모델 학습 및 테스트
    print("\n최적 파라미터로 최종 평가...")
    from pytorch_tabnet.tab_model import TabNetClassifier

    best_params = study.best_params

    final_model = TabNetClassifier(
        n_d=best_params['n_d'],
        n_a=best_params['n_a'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        n_independent=best_params['n_independent'],
        n_shared=best_params['n_shared'],
        lambda_sparse=best_params['lambda_sparse'],
        momentum=best_params['momentum'],
        verbose=0,
        seed=42,
        optimizer_params={'lr': best_params['lr']}
    )

    # 학습 (더 긴 patience)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=['auc'],
        max_epochs=200,
        patience=30,
        batch_size=best_params['batch_size']
    )

    # 테스트 평가
    test_pred = final_model.predict(X_test)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_prob)

    print("\n" + "=" * 60)
    print("최종 테스트 결과 (튜닝된 TabNet)")
    print("=" * 60)
    print(f"정확도: {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")

    return study, test_acc, test_auc


if __name__ == "__main__":
    study, acc, auc = main()
