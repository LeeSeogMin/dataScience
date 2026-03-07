"""
5-4-optuna-tuning.py: Optuna를 활용한 하이퍼파라미터 최적화

XGBoost와 LightGBM의 하이퍼파라미터를 베이지안 최적화로 튜닝한다:
1. 기본 모델 성능 확인
2. Optuna로 XGBoost 튜닝
3. Optuna로 LightGBM 튜닝
4. 최적 파라미터 비교

실행 방법:
    python 5-4-optuna-tuning.py

필수 라이브러리:
    pip install optuna xgboost lightgbm scikit-learn
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("경고: optuna가 설치되지 않음.")
    print("설치: pip install optuna")

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# diagram 폴더 (문서에서 참조)
DIAGRAM_DIR = Path(__file__).parent.parent.parent.parent / "diagram"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """California Housing 데이터를 로드한다."""
    print("데이터 로드 중...")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"훈련 데이터: {X_train.shape[0]}개")
    print(f"테스트 데이터: {X_test.shape[0]}개")
    print(f"특성 수: {X_train.shape[1]}개")

    return X_train, X_test, y_train, y_test


def baseline_models(X_train, X_test, y_train, y_test):
    """기본 설정 모델의 성능을 측정한다."""
    print("\n" + "="*60)
    print("기본 설정 모델 성능")
    print("="*60)

    results = []

    # XGBoost 기본 설정
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    print(f"XGBoost 기본: RMSE={xgb_rmse:.4f}, R²={xgb_r2:.4f}")
    results.append(('XGBoost 기본', xgb_rmse, xgb_r2))

    # LightGBM 기본 설정
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
    lgb_r2 = r2_score(y_test, lgb_pred)
    print(f"LightGBM 기본: RMSE={lgb_rmse:.4f}, R²={lgb_r2:.4f}")
    results.append(('LightGBM 기본', lgb_rmse, lgb_r2))

    return results


def optimize_xgboost(X_train, X_test, y_train, y_test, n_trials=50):
    """Optuna로 XGBoost 하이퍼파라미터를 최적화한다."""
    print("\n" + "="*60)
    print(f"XGBoost 최적화 (시도 횟수: {n_trials})")
    print("="*60)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }

        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring='neg_root_mean_squared_error'
        )
        return -scores.mean()

    start_time = time.time()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    elapsed_time = time.time() - start_time

    print(f"\n최적화 완료: {elapsed_time:.1f}초")
    print(f"최적 CV RMSE: {study.best_value:.4f}")
    print("\n최적 파라미터:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {value}")

    # 최적 파라미터로 최종 모델 학습
    best_model = xgb.XGBRegressor(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"\n테스트 성능: RMSE={rmse:.4f}, R²={r2:.4f}")

    return study, best_model, rmse, r2


def optimize_lightgbm(X_train, X_test, y_train, y_test, n_trials=50):
    """Optuna로 LightGBM 하이퍼파라미터를 최적화한다."""
    print("\n" + "="*60)
    print(f"LightGBM 최적화 (시도 횟수: {n_trials})")
    print("="*60)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbose': -1
        }

        model = lgb.LGBMRegressor(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring='neg_root_mean_squared_error'
        )
        return -scores.mean()

    start_time = time.time()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    elapsed_time = time.time() - start_time

    print(f"\n최적화 완료: {elapsed_time:.1f}초")
    print(f"최적 CV RMSE: {study.best_value:.4f}")
    print("\n최적 파라미터:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.6f}")
        else:
            print(f"   {key}: {value}")

    # 최적 파라미터로 최종 모델 학습
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"\n테스트 성능: RMSE={rmse:.4f}, R²={r2:.4f}")

    return study, best_model, rmse, r2


def plot_optimization_history(xgb_study, lgb_study):
    """최적화 과정을 시각화한다 (흑백)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # XGBoost 최적화 히스토리
    ax1 = axes[0]
    trials = [t.number for t in xgb_study.trials]
    values = [t.value for t in xgb_study.trials]
    best_values = [min(values[:i+1]) for i in range(len(values))]

    ax1.scatter(trials, values, alpha=0.5, label='Trial RMSE', color='#888888', marker='o')
    ax1.plot(trials, best_values, 'k-', linewidth=2, label='Best RMSE')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('RMSE')
    ax1.set_title('XGBoost Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')

    # LightGBM 최적화 히스토리
    ax2 = axes[1]
    trials = [t.number for t in lgb_study.trials]
    values = [t.value for t in lgb_study.trials]
    best_values = [min(values[:i+1]) for i in range(len(values))]

    ax2.scatter(trials, values, alpha=0.5, label='Trial RMSE', color='#888888', marker='o')
    ax2.plot(trials, best_values, 'k-', linewidth=2, label='Best RMSE')
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('RMSE')
    ax2.set_title('LightGBM Optimization History')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "optuna_optimization_history.png", dpi=150, bbox_inches='tight')
    # diagram 폴더에도 저장
    plt.savefig(DIAGRAM_DIR / "optuna_optimization_history.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n시각화 저장: {OUTPUT_DIR}/optuna_optimization_history.png")


def plot_param_importance(xgb_study, lgb_study):
    """파라미터 중요도를 시각화한다."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # XGBoost 파라미터 중요도
    ax1 = axes[0]
    importance = optuna.importance.get_param_importances(xgb_study)
    params = list(importance.keys())[:8]
    values = [importance[p] for p in params]

    ax1.barh(params, values, color='steelblue')
    ax1.set_xlabel('Importance')
    ax1.set_title('XGBoost Parameter Importance')
    ax1.invert_yaxis()

    # LightGBM 파라미터 중요도
    ax2 = axes[1]
    importance = optuna.importance.get_param_importances(lgb_study)
    params = list(importance.keys())[:8]
    values = [importance[p] for p in params]

    ax2.barh(params, values, color='forestgreen')
    ax2.set_xlabel('Importance')
    ax2.set_title('LightGBM Parameter Importance')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "optuna_param_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"시각화 저장: {OUTPUT_DIR}/optuna_param_importance.png")


def main():
    print("="*60)
    print("Optuna를 활용한 하이퍼파라미터 최적화")
    print("="*60)

    if not OPTUNA_AVAILABLE:
        print("\nOptuna가 설치되지 않아 실행할 수 없습니다.")
        print("설치: pip install optuna")
        return

    # 1. 데이터 로드
    X_train, X_test, y_train, y_test = load_data()

    # 2. 기본 모델 성능
    baseline_results = baseline_models(X_train, X_test, y_train, y_test)

    # 3. XGBoost 최적화
    xgb_study, xgb_model, xgb_rmse, xgb_r2 = optimize_xgboost(
        X_train, X_test, y_train, y_test, n_trials=30
    )

    # 4. LightGBM 최적화
    lgb_study, lgb_model, lgb_rmse, lgb_r2 = optimize_lightgbm(
        X_train, X_test, y_train, y_test, n_trials=30
    )

    # 5. 최적화 과정 시각화
    print("\n시각화 생성 중...")
    plot_optimization_history(xgb_study, lgb_study)
    plot_param_importance(xgb_study, lgb_study)

    # 6. 결과 요약
    print("\n" + "="*60)
    print("최적화 결과 요약")
    print("="*60)

    print(f"\n{'모델':<20} {'RMSE':<12} {'R²':<12} {'개선율':<12}")
    print("-"*60)

    # 기본 모델
    xgb_base_rmse = baseline_results[0][1]
    lgb_base_rmse = baseline_results[1][1]

    print(f"{'XGBoost 기본':<20} {xgb_base_rmse:<12.4f} {baseline_results[0][2]:<12.4f} {'-':<12}")
    print(f"{'XGBoost 최적화':<20} {xgb_rmse:<12.4f} {xgb_r2:<12.4f} {(xgb_base_rmse-xgb_rmse)/xgb_base_rmse*100:>10.1f}%")
    print(f"{'LightGBM 기본':<20} {lgb_base_rmse:<12.4f} {baseline_results[1][2]:<12.4f} {'-':<12}")
    print(f"{'LightGBM 최적화':<20} {lgb_rmse:<12.4f} {lgb_r2:<12.4f} {(lgb_base_rmse-lgb_rmse)/lgb_base_rmse*100:>10.1f}%")

    # 결과 저장
    results_df = pd.DataFrame({
        'model': ['XGBoost 기본', 'XGBoost 최적화', 'LightGBM 기본', 'LightGBM 최적화'],
        'rmse': [xgb_base_rmse, xgb_rmse, lgb_base_rmse, lgb_rmse],
        'r2': [baseline_results[0][2], xgb_r2, baseline_results[1][2], lgb_r2]
    })
    results_df.to_csv(OUTPUT_DIR / "optuna_results.csv", index=False)

    print("\n" + "="*60)
    print("핵심 인사이트")
    print("="*60)
    print("""
Optuna 활용 팁:
1. 탐색 범위 설정이 중요 - 너무 넓으면 비효율, 좁으면 최적점 놓침
2. n_trials는 파라미터 수 × 10 이상 권장
3. 조기 종료(pruning) 활용으로 비효율적 시도 빠르게 중단
4. 파라미터 중요도 확인 후 중요 파라미터에 집중

실무 전략:
- 1단계: 적은 시도로 대략적 범위 파악
- 2단계: 좁은 범위에서 정밀 탐색
- 3단계: 조기 종료와 함께 최종 튜닝
""")

    print(f"\n결과 저장: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
