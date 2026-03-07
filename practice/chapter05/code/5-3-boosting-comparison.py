"""
5-3-boosting-comparison.py: 그래디언트 부스팅 3종 비교 (XGBoost, LightGBM, CatBoost)

이 스크립트는 주요 그래디언트 부스팅 라이브러리를 동일 데이터에서 비교한다:
1. XGBoost, LightGBM, CatBoost 기본 설정 학습
2. 테스트 성능(RMSE, R²) 비교
3. 학습 시간 비교

실행 방법:
    python 5-3-boosting-comparison.py

산출물:
    practice/chapter5/data/output/
      - boosting_comparison.csv
      - boosting_comparison.png
      - boosting_training_time.png
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
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# CatBoost 선택적 임포트
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("경고: catboost가 설치되지 않음. CatBoost 비교는 생략됩니다.")
    print("설치: pip install catboost")

# 경로 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# diagram 폴더 (문서에서 참조)
DIAGRAM_DIR = Path(__file__).parent.parent.parent.parent / "diagram"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# 공통 하이퍼파라미터 (공정 비교)
COMMON_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
}


def load_data():
    """California Housing 데이터를 로드한다."""
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def train_xgboost(X_train, X_test, y_train, y_test) -> dict:
    """XGBoost를 학습하고 평가한다."""
    start = time.time()
    model = xgb.XGBRegressor(
        **COMMON_PARAMS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    return {
        "model": "XGBoost",
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "training_time_sec": elapsed
    }


def train_lightgbm(X_train, X_test, y_train, y_test) -> dict:
    """LightGBM을 학습하고 평가한다."""
    start = time.time()
    model = lgb.LGBMRegressor(
        **COMMON_PARAMS,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    return {
        "model": "LightGBM",
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "training_time_sec": elapsed
    }


def train_catboost(X_train, X_test, y_train, y_test) -> dict:
    """CatBoost를 학습하고 평가한다."""
    start = time.time()
    model = CatBoostRegressor(
        iterations=COMMON_PARAMS["n_estimators"],
        depth=COMMON_PARAMS["max_depth"],
        learning_rate=COMMON_PARAMS["learning_rate"],
        random_state=RANDOM_STATE,
        verbose=0
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    return {
        "model": "CatBoost",
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "training_time_sec": elapsed
    }


def plot_comparison(results: list[dict], out_path: Path) -> None:
    """성능 비교 막대그래프를 생성한다 (흑백)."""
    models = [r["model"] for r in results]
    rmse = [r["rmse"] for r in results]
    r2 = [r["r2"] for r in results]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 흑백 스타일: 회색 계열 + 해치 패턴으로 구분
    bars1 = ax1.bar(x - width / 2, rmse, width, label="RMSE",
                    color="#333333", edgecolor="black", hatch="")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("RMSE", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, r2, width, label="R²",
                    color="#999999", edgecolor="black", hatch="///")
    ax2.set_ylabel("R²", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.set_ylim(0, 1)

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title("Gradient Boosting Libraries: Performance Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # diagram 폴더에도 저장
    plt.savefig(DIAGRAM_DIR / out_path.name, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_time(results: list[dict], out_path: Path) -> None:
    """학습 시간 비교 막대그래프를 생성한다 (흑백)."""
    models = [r["model"] for r in results]
    times = [r["training_time_sec"] for r in results]

    # 흑백 스타일: 회색 계열 + 해치 패턴으로 구분
    colors = ["#333333", "#777777", "#BBBBBB"][: len(models)]
    hatches = ["", "///", "..."][: len(models)]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, times, color=colors, edgecolor='black')
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    plt.xlabel("Model")
    plt.ylabel("Training Time (seconds)")
    plt.title("Gradient Boosting Libraries: Training Time Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # diagram 폴더에도 저장
    plt.savefig(DIAGRAM_DIR / out_path.name, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 60)
    print("그래디언트 부스팅 3종 비교 (XGBoost, LightGBM, CatBoost)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    print(f"\n공통 하이퍼파라미터:")
    for k, v in COMMON_PARAMS.items():
        print(f"  {k}: {v}")

    print(f"\n훈련 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")

    results = []

    print("\n학습 및 평가 중...")

    # XGBoost
    xgb_result = train_xgboost(X_train, X_test, y_train, y_test)
    results.append(xgb_result)
    print(f"  XGBoost:  RMSE={xgb_result['rmse']:.4f}, R²={xgb_result['r2']:.4f}, "
          f"Time={xgb_result['training_time_sec']:.2f}s")

    # LightGBM
    lgb_result = train_lightgbm(X_train, X_test, y_train, y_test)
    results.append(lgb_result)
    print(f"  LightGBM: RMSE={lgb_result['rmse']:.4f}, R²={lgb_result['r2']:.4f}, "
          f"Time={lgb_result['training_time_sec']:.2f}s")

    # CatBoost
    if CATBOOST_AVAILABLE:
        cat_result = train_catboost(X_train, X_test, y_train, y_test)
        results.append(cat_result)
        print(f"  CatBoost: RMSE={cat_result['rmse']:.4f}, R²={cat_result['r2']:.4f}, "
              f"Time={cat_result['training_time_sec']:.2f}s")

    # 결과 저장
    results_df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "boosting_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    # 시각화
    comp_path = OUTPUT_DIR / "boosting_comparison.png"
    plot_comparison(results, comp_path)

    time_path = OUTPUT_DIR / "boosting_training_time.png"
    plot_training_time(results, time_path)

    print("\n산출물 저장 완료:")
    print(f"  - {csv_path}")
    print(f"  - {comp_path}")
    print(f"  - {time_path}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("핵심 인사이트")
    print("=" * 60)
    print("""
1. 세 라이브러리 모두 유사한 예측 성능을 보인다.
2. LightGBM은 히스토그램 기반 최적화로 학습 속도가 가장 빠르다.
3. CatBoost는 대칭 트리 구조로 추론 시 효율적이다.
4. 데이터 특성과 운영 환경에 맞게 선택하면 된다.
""")


if __name__ == "__main__":
    main()
