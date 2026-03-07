# 10-4-ml-survival.py
"""
머신러닝 생존 분석 실습
- XGBoost AFT (Accelerated Failure Time)
- Random Survival Forest (scikit-survival)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# XGBoost
import xgboost as xgb

# scikit-survival
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# lifelines 데이터셋
from lifelines.datasets import load_rossi

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리 설정
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data_xgb(data):
    """XGBoost AFT용 데이터 준비"""
    # 특성과 타겟 분리
    feature_cols = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
    X = data[feature_cols].values
    time = data['week'].values
    event = data['arrest'].values

    # XGBoost AFT용: 중도절단은 upper bound를 inf로 설정
    y_lower = time.copy().astype(float)
    y_upper = np.where(event == 1, time, np.inf).astype(float)

    return X, y_lower, y_upper, time, event, feature_cols


def prepare_data_sksurv(data):
    """scikit-survival용 데이터 준비"""
    feature_cols = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
    X = data[feature_cols].values

    # structured array로 변환
    y = np.array([(bool(e), t) for e, t in zip(data['arrest'], data['week'])],
                 dtype=[('event', bool), ('time', float)])

    return X, y, feature_cols


def train_xgboost_aft(X_train, y_lower_train, y_upper_train,
                      X_test, y_lower_test, y_upper_test):
    """XGBoost AFT 모델 학습"""
    print("\n[XGBoost AFT 모델]")

    # DMatrix 생성
    dtrain = xgb.DMatrix(X_train)
    dtrain.set_float_info('label_lower_bound', y_lower_train)
    dtrain.set_float_info('label_upper_bound', y_upper_train)

    dtest = xgb.DMatrix(X_test)
    dtest.set_float_info('label_lower_bound', y_lower_test)
    dtest.set_float_info('label_upper_bound', y_upper_test)

    # 하이퍼파라미터
    params = {
        'objective': 'survival:aft',
        'eval_metric': 'aft-nloglik',
        'aft_loss_distribution': 'normal',
        'aft_loss_distribution_scale': 1.2,
        'tree_method': 'hist',
        'learning_rate': 0.05,
        'max_depth': 3,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # 학습
    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # 예측 (생존 시간)
    pred_time = model.predict(dtest)

    print(f"- 최적 라운드: {model.best_iteration}")
    print(f"- 예측 생존 시간 범위: {pred_time.min():.1f} ~ {pred_time.max():.1f}주")

    return model, pred_time, evals_result


def train_rsf(X_train, y_train, X_test, y_test):
    """Random Survival Forest 모델 학습"""
    print("\n[Random Survival Forest 모델]")

    rsf = RandomSurvivalForest(
        n_estimators=100,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )

    rsf.fit(X_train, y_train)

    # C-index 계산
    pred_risk = rsf.predict(X_test)
    c_index = concordance_index_censored(
        y_test['event'], y_test['time'], pred_risk
    )[0]

    print(f"- 테스트 C-index: {c_index:.4f}")

    return rsf, pred_risk, c_index


def calculate_c_index_xgb(pred_time, time_test, event_test):
    """XGBoost 예측에 대한 C-index 계산"""
    # 생존 시간이 길수록 위험이 낮음 -> 역으로 변환
    risk_score = -pred_time
    c_index = concordance_index_censored(
        event_test.astype(bool), time_test, risk_score
    )[0]
    return c_index


def main():
    """머신러닝 생존 분석 메인 함수"""

    print("=" * 60)
    print("10.4 머신러닝 생존 분석 실습")
    print("=" * 60)

    # 1. 데이터 로드
    data = load_rossi()
    print(f"\n[데이터]")
    print(f"- 전체: {len(data)}명, 사건: {data['arrest'].sum()}명")

    # 2. 학습/테스트 분할
    train_idx, test_idx = train_test_split(
        range(len(data)), test_size=0.2, random_state=42,
        stratify=data['arrest']
    )

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    print(f"- 학습: {len(train_data)}명, 테스트: {len(test_data)}명")

    # 3. XGBoost AFT
    X_train, y_lower_train, y_upper_train, time_train, event_train, feature_cols = \
        prepare_data_xgb(train_data)
    X_test, y_lower_test, y_upper_test, time_test, event_test, _ = \
        prepare_data_xgb(test_data)

    xgb_model, xgb_pred_time, xgb_evals = train_xgboost_aft(
        X_train, y_lower_train, y_upper_train,
        X_test, y_lower_test, y_upper_test
    )

    xgb_c_index = calculate_c_index_xgb(xgb_pred_time, time_test, event_test)
    print(f"- 테스트 C-index: {xgb_c_index:.4f}")

    # 4. Random Survival Forest
    X_train_rsf, y_train_rsf, _ = prepare_data_sksurv(train_data)
    X_test_rsf, y_test_rsf, _ = prepare_data_sksurv(test_data)

    rsf_model, rsf_pred_risk, rsf_c_index = train_rsf(
        X_train_rsf, y_train_rsf, X_test_rsf, y_test_rsf
    )

    # 5. 변수 중요도
    print("\n[변수 중요도]")

    # XGBoost 중요도
    xgb_importance = xgb_model.get_score(importance_type='gain')
    xgb_imp_df = pd.DataFrame([
        {'변수': f'f{i}', '중요도': xgb_importance.get(f'f{i}', 0)}
        for i in range(len(feature_cols))
    ])
    xgb_imp_df['변수명'] = feature_cols
    xgb_imp_df = xgb_imp_df.sort_values('중요도', ascending=False)

    print("\nXGBoost AFT 변수 중요도:")
    for _, row in xgb_imp_df.iterrows():
        print(f"  - {row['변수명']}: {row['중요도']:.2f}")

    # RSF 중요도 (permutation importance 사용)
    try:
        rsf_importance = rsf_model.feature_importances_
        rsf_imp_df = pd.DataFrame({
            '변수': feature_cols,
            '중요도': rsf_importance
        }).sort_values('중요도', ascending=False)

        print("\nRandom Survival Forest 변수 중요도:")
        for _, row in rsf_imp_df.iterrows():
            print(f"  - {row['변수']}: {row['중요도']:.4f}")
    except NotImplementedError:
        print("\nRandom Survival Forest: feature_importances_ 미지원")
        rsf_importance = np.ones(len(feature_cols)) / len(feature_cols)  # 균등 분포
        rsf_imp_df = pd.DataFrame({
            '변수': feature_cols,
            '중요도': rsf_importance
        })

    # 6. 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 6-1. 변수 중요도 비교
    ax1 = axes[0]
    x = np.arange(len(feature_cols))
    width = 0.35

    # 정규화된 중요도
    xgb_norm = xgb_imp_df.set_index('변수명').loc[feature_cols, '중요도'].values
    xgb_norm = xgb_norm / (xgb_norm.max() + 1e-10)
    rsf_norm = rsf_imp_df.set_index('변수').loc[feature_cols, '중요도'].values
    rsf_norm = rsf_norm / (rsf_norm.max() + 1e-10)

    ax1.bar(x - width/2, xgb_norm, width, label='XGBoost AFT')
    ax1.bar(x + width/2, rsf_norm, width, label='Random Survival Forest')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_cols)
    ax1.set_ylabel('Normalized importance')
    ax1.set_title('Feature Importance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 6-2. 예측 비교
    ax2 = axes[1]
    ax2.scatter(xgb_pred_time, -rsf_pred_risk, alpha=0.5)
    ax2.set_xlabel('Predicted survival time (XGBoost AFT)')
    ax2.set_ylabel('Predicted risk score (RSF, inverted)')
    ax2.set_title('Model Prediction Comparison')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(DATA_DIR / '10-4-ml-survival.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[저장] {DATA_DIR / '10-4-ml-survival.png'}")

    # 7. 결과 요약
    print("\n[모델 성능 비교]")
    print(f"{'모델':<25} {'C-index':<10}")
    print("-" * 35)
    print(f"{'XGBoost AFT':<25} {xgb_c_index:.4f}")
    print(f"{'Random Survival Forest':<25} {rsf_c_index:.4f}")

    # 8. 결과 저장
    results = {
        'data_summary': {
            'total': int(len(data)),
            'train': int(len(train_data)),
            'test': int(len(test_data)),
            'events_train': int(train_data['arrest'].sum()),
            'events_test': int(test_data['arrest'].sum())
        },
        'xgboost_aft': {
            'c_index': float(xgb_c_index),
            'best_iteration': int(xgb_model.best_iteration),
            'feature_importance': {
                col: float(xgb_importance.get(f'f{i}', 0))
                for i, col in enumerate(feature_cols)
            }
        },
        'random_survival_forest': {
            'c_index': float(rsf_c_index),
            'feature_importance': {
                col: float(imp)
                for col, imp in zip(feature_cols, rsf_importance)
            }
        }
    }

    with open(DATA_DIR / '10-4-ml-survival.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {DATA_DIR / '10-4-ml-survival.json'}")

    return results


if __name__ == '__main__':
    main()
