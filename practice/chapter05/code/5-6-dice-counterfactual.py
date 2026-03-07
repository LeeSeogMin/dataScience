"""
5-6-dice-counterfactual.py: DiCE로 Counterfactual Explanations 생성

이 스크립트는 DiCE(Diverse Counterfactual Explanations)를 사용해
"무엇을 바꾸면 예측이 바뀌는가?"를 데이터 기반으로 탐색한다.

데이터: scikit-learn breast cancer (malignant/benign)
모델  : XGBoost 분류기

실행 방법:
    python 5-6-dice-counterfactual.py

산출물:
    practice/chapter5/data/output/
      - dice_query_instance.csv
      - dice_counterfactuals.csv
      - dice_counterfactual_deltas.csv
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import xgboost as xgb
import dice_ml

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def sanitize_column(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={c: sanitize_column(c) for c in df.columns})

    feature_cols = [c for c in df.columns if c != "target"]
    class_names = list(data.target_names)
    return df, feature_cols, class_names


def train_model(train_df: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols], train_df["target"])
    return model


def main() -> None:
    df, feature_cols, class_names = load_data()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["target"]
    )

    model = train_model(train_df, feature_cols)

    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(test_df["target"], pred))
    auc = float(roc_auc_score(test_df["target"], proba))

    print("모델 성능 (테스트)")
    print(f"- accuracy: {acc:.4f}")
    print(f"- roc_auc : {auc:.4f}")
    print(f"- class 0: {class_names[0]} / class 1: {class_names[1]}")

    # Counterfactual은 "원하는 예측"으로 바꾸는 것이 목적이므로,
    # 테스트 세트에서 현재 예측이 0(예: malignant)인 샘플을 하나 선택한다.
    test_pred = model.predict(test_df[feature_cols])
    idx_candidates = np.where(test_pred == 0)[0]
    if len(idx_candidates) == 0:
        raise RuntimeError("예측이 0인 샘플을 찾지 못했습니다. 다른 random_state로 시도하세요.")

    query_row = test_df.iloc[[int(idx_candidates[0])]].copy()
    query_x = query_row[feature_cols]

    dice_data = dice_ml.Data(
        dataframe=train_df,
        continuous_features=feature_cols,
        outcome_name="target",
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    dice = dice_ml.Dice(dice_data, dice_model, method="random")

    print("\nDiCE counterfactual 생성")
    exp = dice.generate_counterfactuals(
        query_x,
        total_CFs=3,
        desired_class="opposite",
        random_seed=RANDOM_STATE,
    )

    cf_df = exp.cf_examples_list[0].final_cfs_df.copy()

    query_out = OUTPUT_DIR / "dice_query_instance.csv"
    cf_out = OUTPUT_DIR / "dice_counterfactuals.csv"
    delta_out = OUTPUT_DIR / "dice_counterfactual_deltas.csv"

    query_row.to_csv(query_out, index=False)
    cf_df.to_csv(cf_out, index=False)

    if set(feature_cols).issubset(cf_df.columns):
        deltas = cf_df[feature_cols] - query_x.iloc[0]
        deltas.insert(0, "cf_id", np.arange(len(cf_df)))
        deltas.to_csv(delta_out, index=False)
    else:
        delta_out.write_text("counterfactual 결과에 feature 컬럼이 포함되지 않았습니다.", encoding="utf-8")

    print("\n산출물 저장")
    print(f"- {query_out}")
    print(f"- {cf_out}")
    print(f"- {delta_out}")


if __name__ == "__main__":
    main()

