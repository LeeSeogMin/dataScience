"""
5-2-random-forest.py: 랜덤 포레스트 학습과 OOB(Out-of-Bag) 평가

이 스크립트는 랜덤 포레스트의 핵심 메커니즘을 실습한다:
1. 기본 설정 랜덤 포레스트 학습
2. n_estimators별 OOB 오차 수렴 확인
3. 특성 중요도(MDI) 추출 및 시각화

실행 방법:
    python 5-2-random-forest.py

산출물:
    practice/chapter5/data/output/
      - rf_results.csv
      - rf_oob_convergence.png
      - rf_feature_importance.png
"""

from __future__ import annotations

import warnings
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# 경로 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """California Housing 데이터를 로드한다."""
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return X, y, list(X.columns)


def train_and_evaluate_rf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int
) -> dict:
    """주어진 n_estimators로 랜덤 포레스트를 학습하고 평가한다."""
    start_time = time.time()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # 테스트 성능
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "n_estimators": n_estimators,
        "oob_score": float(model.oob_score_),
        "oob_error": 1 - float(model.oob_score_),
        "test_rmse": rmse,
        "test_r2": r2,
        "training_time": training_time,
        "model": model
    }


def plot_oob_convergence(results: list[dict], out_path: Path) -> None:
    """n_estimators별 OOB 오차 수렴 그래프를 생성한다."""
    n_est = [r["n_estimators"] for r in results]
    oob_errors = [r["oob_error"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(n_est, oob_errors, "b-o", linewidth=2, markersize=6)
    plt.xlabel("Number of Trees (n_estimators)")
    plt.ylabel("OOB Error (1 - OOB Score)")
    plt.title("Random Forest: OOB Error Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    out_path: Path
) -> None:
    """특성 중요도(MDI) 막대그래프를 생성한다."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in indices][::-1],
        importance[indices][::-1],
        color="forestgreen"
    )
    plt.xlabel("Feature Importance (MDI)")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 60)
    print("랜덤 포레스트 학습과 OOB 평가")
    print("=" * 60)

    # 데이터 로드
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\n훈련 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")
    print(f"특성 수: {len(feature_names)}개")

    # n_estimators별 학습 및 평가
    n_estimators_range = [10, 50, 100, 200, 300, 400, 500]
    results = []

    print("\nn_estimators별 학습 중...")
    for n_est in n_estimators_range:
        result = train_and_evaluate_rf(X_train, X_test, y_train, y_test, n_est)
        results.append(result)
        print(f"  n_estimators={n_est:3d}: OOB Score={result['oob_score']:.4f}, "
              f"Test RMSE={result['test_rmse']:.4f}, Time={result['training_time']:.2f}s")

    # 결과 저장
    results_df = pd.DataFrame([{
        "n_estimators": r["n_estimators"],
        "oob_score": r["oob_score"],
        "test_rmse": r["test_rmse"],
        "test_r2": r["test_r2"],
        "training_time_sec": r["training_time"]
    } for r in results])

    csv_path = OUTPUT_DIR / "rf_results.csv"
    results_df.to_csv(csv_path, index=False)

    # 시각화
    oob_path = OUTPUT_DIR / "rf_oob_convergence.png"
    plot_oob_convergence(results, oob_path)

    # 최종 모델(n_estimators=300)의 특성 중요도
    final_model = [r for r in results if r["n_estimators"] == 300][0]["model"]
    importance_path = OUTPUT_DIR / "rf_feature_importance.png"
    plot_feature_importance(final_model, feature_names, importance_path)

    print("\n산출물 저장 완료:")
    print(f"  - {csv_path}")
    print(f"  - {oob_path}")
    print(f"  - {importance_path}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("핵심 인사이트")
    print("=" * 60)
    print("""
1. OOB 오차는 별도 검증 세트 없이 일반화 성능을 추정한다.
2. n_estimators가 증가하면 OOB 오차가 감소하다가 수렴한다.
3. 약 200-300개 이상에서는 추가 트리의 효과가 제한적이다.
4. 특성 중요도(MDI)로 예측에 중요한 변수를 빠르게 파악할 수 있다.
""")


if __name__ == "__main__":
    main()
