"""
2-2-scaling-pipeline.py: 스케일링 방법 비교 및 파이프라인 구축

스케일링 방법별 성능을 비교한다:
1. StandardScaler (Z-score 표준화)
2. MinMaxScaler
3. RobustScaler
4. MaxAbsScaler

데이터 누수 방지를 위한 파이프라인 구축 예제 포함.

실행 방법:
    python 2-2-scaling-pipeline.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_scaling_method(X_train, X_test, y_train, y_test,
                            scaler, scaler_name, model, model_name):
    """스케일링 방법의 성능을 평가한다."""
    start_time = time.time()

    # 파이프라인 구축 (데이터 누수 방지)
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])

    # 학습 및 예측
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    elapsed_time = time.time() - start_time

    return {
        'scaler': scaler_name,
        'model': model_name,
        'rmse': rmse,
        'time': elapsed_time
    }


def demonstrate_data_leakage():
    """데이터 누수의 문제점을 보여준다."""
    print("\n" + "="*60)
    print("데이터 누수 방지 시연")
    print("="*60)

    # 데이터 로드
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 잘못된 방법: 전체 데이터로 스케일링 (데이터 누수)
    scaler_wrong = StandardScaler()
    X_all_scaled = scaler_wrong.fit_transform(np.vstack([X_train, X_test]))
    X_train_wrong = X_all_scaled[:len(X_train)]
    X_test_wrong = X_all_scaled[len(X_train):]

    model_wrong = KNeighborsRegressor(n_neighbors=5)
    model_wrong.fit(X_train_wrong, y_train)
    rmse_wrong = np.sqrt(mean_squared_error(y_test, model_wrong.predict(X_test_wrong)))

    # 올바른 방법: 훈련 데이터로만 스케일링
    scaler_correct = StandardScaler()
    X_train_correct = scaler_correct.fit_transform(X_train)
    X_test_correct = scaler_correct.transform(X_test)

    model_correct = KNeighborsRegressor(n_neighbors=5)
    model_correct.fit(X_train_correct, y_train)
    rmse_correct = np.sqrt(mean_squared_error(y_test, model_correct.predict(X_test_correct)))

    print(f"\n잘못된 방법 (전체 데이터 스케일링): RMSE = {rmse_wrong:.4f}")
    print(f"올바른 방법 (훈련 데이터만 스케일링): RMSE = {rmse_correct:.4f}")
    print(f"\n차이: {abs(rmse_wrong - rmse_correct):.4f}")
    print("\n주의: 이 예제에서는 차이가 작을 수 있지만,")
    print("      실제 운영 환경에서는 심각한 성능 저하를 야기할 수 있다.")


def main():
    print("="*60)
    print("스케일링 방법별 성능 비교")
    print("="*60)

    # 1. 데이터 로드
    print("\n[1/4] 데이터 로드 중...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names

    print(f"   데이터: {X.shape[0]}행 x {X.shape[1]}열")
    print(f"   특성: {feature_names}")

    # 2. 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   훈련: {len(X_train)}, 테스트: {len(X_test)}")

    # 3. 스케일링 방법 정의
    scalers = [
        (None, '스케일링 없음'),
        (StandardScaler(), 'StandardScaler'),
        (MinMaxScaler(), 'MinMaxScaler'),
        (RobustScaler(), 'RobustScaler'),
        (MaxAbsScaler(), 'MaxAbsScaler')
    ]

    # 4. 거리 기반 모델 (스케일링 영향 받음)
    print("\n[2/4] 거리 기반 모델 (KNN) 테스트 중...")
    knn_results = []

    for scaler, name in scalers:
        if scaler is None:
            # 스케일링 없이 직접 학습
            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            result = {'scaler': name, 'model': 'KNN', 'rmse': rmse, 'time': 0}
        else:
            result = evaluate_scaling_method(
                X_train, X_test, y_train, y_test,
                scaler, name, KNeighborsRegressor(n_neighbors=5), 'KNN'
            )
        knn_results.append(result)
        print(f"   ✓ {name}: RMSE={result['rmse']:.4f}")

    # 5. 트리 기반 모델 (스케일링 영향 없음)
    print("\n[3/4] 트리 기반 모델 (Random Forest) 테스트 중...")
    rf_results = []

    for scaler, name in scalers[:3]:  # 3개만 테스트
        if scaler is None:
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            result = {'scaler': name, 'model': 'RF', 'rmse': rmse, 'time': 0}
        else:
            result = evaluate_scaling_method(
                X_train, X_test, y_train, y_test,
                scaler, name,
                RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1), 'RF'
            )
        rf_results.append(result)
        print(f"   ✓ {name}: RMSE={result['rmse']:.4f}")

    # 6. 결과 요약
    print("\n" + "="*60)
    print("스케일링 방법별 성능 비교 (California Housing)")
    print("="*60)

    print("\n[KNN - 거리 기반 모델: 스케일링 영향 받음]")
    print(f"{'스케일링 방법':<20} {'RMSE':<12}")
    print("-"*40)
    for result in knn_results:
        print(f"{result['scaler']:<20} {result['rmse']:.4f}")

    print("\n[Random Forest - 트리 기반 모델: 스케일링 영향 없음]")
    print(f"{'스케일링 방법':<20} {'RMSE':<12}")
    print("-"*40)
    for result in rf_results:
        print(f"{result['scaler']:<20} {result['rmse']:.4f}")

    # 7. 데이터 누수 시연
    print("\n[4/4] 데이터 누수 방지 시연...")
    demonstrate_data_leakage()

    # 8. 결과 저장
    all_results = knn_results + rf_results
    df_results = pd.DataFrame(all_results)
    output_path = OUTPUT_DIR / "scaling_comparison.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n결과 저장: {output_path}")

    # 9. 분석 인사이트
    print("\n" + "="*60)
    print("분석 인사이트")
    print("="*60)

    knn_no_scale = [r for r in knn_results if r['scaler'] == '스케일링 없음'][0]['rmse']
    knn_best = min(knn_results, key=lambda x: x['rmse'])

    print(f"""
- KNN (거리 기반):
  - 스케일링 없음: RMSE = {knn_no_scale:.4f}
  - 최고 성능: {knn_best['scaler']} (RMSE = {knn_best['rmse']:.4f})
  - 개선율: {(knn_no_scale - knn_best['rmse']) / knn_no_scale * 100:.1f}%

- Random Forest (트리 기반):
  - 스케일링 유무에 관계없이 성능 동일
  - 트리 모델은 분할점 기준 분류이므로 스케일 불변

권장사항:
- KNN, SVM, 신경망: 반드시 스케일링 적용
- 트리 기반 모델: 스케일링 불필요
- 파이프라인 사용으로 데이터 누수 방지
""")

    return knn_results, rf_results


if __name__ == "__main__":
    main()
