# 10-7-model-comparison.py
"""
10.7 모형 비교 및 선택 가이드 (실제 실행용)

딥러닝(DeepSurv)에 유리한 상황을 만들기 위해 생성된 고차원/비선형 구조의 합성
생존 데이터(data/synthetic_survival_data.csv)를 불러와 아래 4개 모형을 비교한다.
- Cox 비례위험 (lifelines)
- XGBoost AFT (xgboost)
- Random Survival Forest (scikit-survival)
- DeepSurv (PyTorch, Cox partial likelihood)

데이터 생성: generate_survival_data.py 실행
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_survival_data() -> pd.DataFrame:
    """data/synthetic_survival_data.csv에서 합성 생존 데이터를 불러온다."""
    data_path = DATA_DIR / "synthetic_survival_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"데이터 파일이 존재하지 않습니다: {data_path}\n"
            "먼저 generate_survival_data.py를 실행하여 데이터를 생성하세요."
        )
    return pd.read_csv(data_path)


class DeepSurv(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def cox_partial_likelihood_loss(
    risk_scores: torch.Tensor, time_: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    # time 내림차순 정렬(리스크셋 누적합을 위해)
    order = torch.argsort(time_, descending=True)
    risk_sorted = risk_scores[order]
    event_sorted = event[order]

    log_cum_hazard = torch.log(torch.cumsum(torch.exp(risk_sorted), dim=0) + 1e-12)
    partial = (risk_sorted - log_cum_hazard) * event_sorted
    denom = torch.clamp(event_sorted.sum(), min=1.0)
    return -partial.sum() / denom


def c_index(event: np.ndarray, time_: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index_censored(event.astype(bool), time_, risk)[0])


def fit_cox(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    start = time.perf_counter()
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df[feature_cols + ["time", "event"]], duration_col="time", event_col="event")
    risk = cph.predict_partial_hazard(test_df[feature_cols]).values.reshape(-1)
    return {
        "c_index": c_index(test_df["event"].to_numpy(), test_df["time"].to_numpy(), risk),
        "train_time_seconds": time.perf_counter() - start,
    }


def fit_rsf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    event_test: np.ndarray,
    time_test: np.ndarray,
) -> dict:
    start = time.perf_counter()
    rsf = RandomSurvivalForest(
        n_estimators=50,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features="sqrt",
        max_depth=10,
        n_jobs=2,
        random_state=42,
    )
    rsf.fit(X_train, y_train)
    risk = rsf.predict(X_test)
    return {
        "c_index": c_index(event_test, time_test, risk),
        "train_time_seconds": time.perf_counter() - start,
    }


def fit_xgboost_aft(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    X_test: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
) -> dict:
    start = time.perf_counter()
    dtrain = xgb.DMatrix(X_train)
    dtrain.set_float_info("label_lower_bound", time_train.astype(float))
    dtrain.set_float_info(
        "label_upper_bound", np.where(event_train == 1, time_train, np.inf).astype(float)
    )

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.2,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 42,
    }
    model = xgb.train(params, dtrain, num_boost_round=400, verbose_eval=False)
    pred_time = model.predict(xgb.DMatrix(X_test))
    risk = -pred_time  # 더 빨리 사건이 발생할수록 위험이 크다
    return {
        "c_index": c_index(event_test, time_test, risk),
        "train_time_seconds": time.perf_counter() - start,
    }


def fit_deepsurv(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    X_test: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
) -> dict:
    start = time.perf_counter()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(X_train_s),
        torch.from_numpy(time_train.astype(np.float32)),
        torch.from_numpy(event_train.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = DeepSurv(in_features=X_train_s.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for _epoch in range(20):
        model.train()
        for batch_x, batch_t, batch_e in loader:
            optimizer.zero_grad()
            risk_scores = model(batch_x)
            loss = cox_partial_likelihood_loss(risk_scores, batch_t, batch_e)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        risk = model(torch.from_numpy(X_test_s)).numpy()

    return {
        "c_index": c_index(event_test, time_test, risk),
        "train_time_seconds": time.perf_counter() - start,
    }


def main() -> None:
    set_seed(SEED)
    print("=" * 60)
    print("10.7 모형 비교: 합성(딥러닝 유리) 생존 데이터")
    print("=" * 60)

    # 데이터 로드
    df = load_survival_data()
    feature_cols = [c for c in df.columns if c.startswith("x")]
    print(f"- N={len(df)}, p={len(feature_cols)}")
    print(f"- event_rate={df['event'].mean():.3f}, censoring_rate={1.0 - df['event'].mean():.3f}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["event"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = train_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    time_train = train_df["time"].to_numpy()
    time_test = test_df["time"].to_numpy()
    event_train = train_df["event"].to_numpy()
    event_test = test_df["event"].to_numpy()

    y_train_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(event_train, time_train)],
        dtype=[("event", bool), ("time", float)],
    )

    results = {
        "data": {
            "n_samples": len(df),
            "n_features": len(feature_cols),
            "seed": SEED,
            "event_rate": float(df["event"].mean()),
            "censoring_rate": float(1.0 - df["event"].mean()),
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
        },
        "models": {},
    }

    print("\n[Fit]")
    print("- Cox PH...")
    results["models"]["Cox PH"] = fit_cox(train_df, test_df, feature_cols)
    print(f"  c_index={results['models']['Cox PH']['c_index']:.4f}")
    print("- XGBoost AFT...")
    results["models"]["XGBoost AFT"] = fit_xgboost_aft(
        X_train, time_train, event_train, X_test, time_test, event_test
    )
    print(f"  c_index={results['models']['XGBoost AFT']['c_index']:.4f}")
    print("- Random Survival Forest...")
    results["models"]["Random Survival Forest"] = fit_rsf(
        X_train, y_train_struct, X_test, event_test, time_test
    )
    print(f"  c_index={results['models']['Random Survival Forest']['c_index']:.4f}")
    print("- DeepSurv...")
    results["models"]["DeepSurv"] = fit_deepsurv(
        X_train, time_train, event_train, X_test, time_test, event_test
    )
    print(f"  c_index={results['models']['DeepSurv']['c_index']:.4f}")

    comparison = (
        pd.DataFrame.from_dict(results["models"], orient="index")
        .reset_index(names="model")
        .sort_values("c_index", ascending=False)
        .reset_index(drop=True)
    )
    results["ranking"] = [
        {"rank": int(i + 1), "model": row["model"], "c_index": float(row["c_index"])}
        for i, row in comparison.iterrows()
    ]

    out_json = DATA_DIR / "10-7-model-comparison.json"
    out_csv = DATA_DIR / "10-7-model-comparison.csv"
    out_png = DATA_DIR / "10-7-model-comparison.png"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    comparison.to_csv(out_csv, index=False, encoding="utf-8")

    plt.figure(figsize=(10, 4))
    plt.bar(comparison["model"], comparison["c_index"], edgecolor="black")
    plt.ylim(0.5, 0.9)
    plt.ylabel("Test C-index")
    plt.title("Survival Model Comparison (Synthetic, Deep Learning Friendly)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    for i, v in enumerate(comparison["c_index"].to_numpy()):
        plt.text(i, float(v) + 0.005, f"{float(v):.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    print("[Saved]")
    print(f"- {out_json}")
    print(f"- {out_csv}")
    print(f"- {out_png}")
    print("\n[Ranking]")
    for item in results["ranking"]:
        print(f"- {item['rank']}. {item['model']}: {item['c_index']:.4f}")


if __name__ == "__main__":
    main()
