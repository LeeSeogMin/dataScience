"""
2-1-missing-values.py: 결측치 처리 방법 비교

결측치 처리 전략별 성능을 비교한다:
1. 행 삭제
2. 평균/중앙값 대체
3. KNN 대체
4. 반복 대체 (MICE)

실행 방법:
    python 2-1-missing-values.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def introduce_missing_values(X: np.ndarray, missing_rate: float = 0.2, random_state: int = 42) -> np.ndarray:
    """데이터에 무작위 결측값을 도입한다."""
    np.random.seed(random_state)
    X_missing = X.copy()
    n_samples, n_features = X.shape

    # 각 위치에 missing_rate 확률로 결측 도입
    mask = np.random.random((n_samples, n_features)) < missing_rate
    X_missing[mask] = np.nan

    return X_missing


def evaluate_imputation(X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray,
                       imputer_name: str, imputer=None) -> dict:
    """결측치 처리 방법의 성능을 평가한다."""
    start_time = time.time()

    if imputer_name == "행 삭제":
        # 결측이 있는 행 제거
        train_mask = ~np.isnan(X_train).any(axis=1)
        test_mask = ~np.isnan(X_test).any(axis=1)

        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]

        # 스케일링 + 회귀
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train_clean)
        y_pred = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred))
        n_train_used = len(X_train_clean)
        n_test_used = len(X_test_clean)
    else:
        # 대체 + 스케일링 + 회귀 파이프라인
        pipeline = Pipeline([
            ('imputer', imputer),
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        n_train_used = len(X_train)
        n_test_used = len(X_test)

    elapsed_time = time.time() - start_time

    return {
        'method': imputer_name,
        'rmse': rmse,
        'time': elapsed_time,
        'n_train': n_train_used,
        'n_test': n_test_used
    }


def main():
    print("="*60)
    print("결측치 처리 방법별 성능 비교")
    print("="*60)

    # 1. 데이터 로드
    print("\n[1/3] 데이터 로드 중...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names

    print(f"   원본 데이터: {X.shape[0]}행 x {X.shape[1]}열")
    print(f"   특성: {feature_names}")

    # 2. 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. 결측값 도입
    missing_rate = 0.2
    print(f"\n[2/3] 결측값 도입 (비율: {missing_rate*100:.0f}%)...")

    X_train_missing = introduce_missing_values(X_train, missing_rate)
    X_test_missing = introduce_missing_values(X_test, missing_rate)

    train_missing_pct = np.isnan(X_train_missing).mean() * 100
    print(f"   훈련 데이터 결측 비율: {train_missing_pct:.1f}%")

    # 4. 다양한 대체 방법 비교
    print("\n[3/3] 결측치 처리 방법 비교 중...")

    methods = [
        ("행 삭제", None),
        ("평균 대체", SimpleImputer(strategy='mean')),
        ("중앙값 대체", SimpleImputer(strategy='median')),
        ("KNN 대체 (K=5)", KNNImputer(n_neighbors=5)),
        ("반복 대체 (MICE)", IterativeImputer(max_iter=10, random_state=42))
    ]

    results = []
    for method_name, imputer in methods:
        result = evaluate_imputation(
            X_train_missing, X_test_missing,
            y_train, y_test,
            method_name, imputer
        )
        results.append(result)
        print(f"   ✓ {method_name}: RMSE={result['rmse']:.3f}, 시간={result['time']:.2f}초")

    # 5. 결과 요약
    print("\n" + "="*60)
    print(f"결측치 처리 방법별 성능 비교 (California Housing)")
    print("="*60)
    print(f"결측 비율: {missing_rate*100:.0f}%")
    print()
    print(f"{'방법':<25} {'RMSE':<12} {'처리 시간':<12}")
    print("-"*60)

    for result in results:
        print(f"{result['method']:<25} {result['rmse']:<12.3f} {result['time']:.2f}초")

    # 6. 결과 저장
    df_results = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "missing_value_comparison.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n결과 저장: {output_path}")

    # 7. 분석 인사이트
    print("\n" + "="*60)
    print("분석 인사이트")
    print("="*60)

    best_method = min(results, key=lambda x: x['rmse'])
    fastest_method = min(results, key=lambda x: x['time'])

    print(f"""
- 최고 성능: {best_method['method']} (RMSE: {best_method['rmse']:.3f})
- 최고 속도: {fastest_method['method']} (시간: {fastest_method['time']:.2f}초)

권장사항:
- 빠른 프로토타입: 평균/중앙값 대체
- 높은 정확도 필요: KNN 또는 반복 대체
- 결측 비율이 낮을 때: 행 삭제도 고려 가능
""")

    return results


if __name__ == "__main__":
    main()
