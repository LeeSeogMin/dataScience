"""
5-5-model-interpretation.py: 트리 기반 모델 심화 해석 (TreeSHAP, PDP/ICE, Permutation Importance)

이 스크립트는 트리 기반 부스팅 모델(XGBoost)을 대상으로 다음 해석 기법을 실습한다.
1) TreeSHAP으로 전역 중요도(Mean |SHAP|) 계산 및 시각화
2) SHAP dependence plot으로 변수-효과 관계 확인
3) PDP/ICE로 전역 효과와 개별 효과 비교
4) Permutation Importance로 성능 기반 중요도 계산

실행 방법:
    python 5-5-model-interpretation.py

산출물:
    practice/chapter5/data/output/
      - shap_summary_bar.png
      - shap_dependence_top_feature.png
      - pdp_ice_top_feature.png
      - permutation_importance.png
      - feature_importance_comparison.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

import shap
import xgboost as xgb

warnings.filterwarnings("ignore")


def convert_to_grayscale(image_path: Path) -> None:
    """이미지를 흑백으로 변환한다."""
    img = Image.open(image_path).convert('L')  # L = grayscale
    img.save(image_path)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# diagram 폴더 (문서에서 참조)
DIAGRAM_DIR = Path(__file__).parent.parent.parent.parent / "diagram"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# 흑백 스타일 설정
plt.rcParams['image.cmap'] = 'Greys'


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    return X, y, list(X.columns)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": rmse, "r2": r2}


def save_shap_summary_bar(
    shap_values: np.ndarray, X_sample: pd.DataFrame, out_path: Path
) -> None:
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, color='#555555')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # diagram 폴더에도 저장
    plt.savefig(DIAGRAM_DIR / out_path.name, dpi=150, bbox_inches="tight")
    plt.close()


def save_shap_dependence_plot(
    feature_name: str, shap_values: np.ndarray, X_sample: pd.DataFrame, out_path: Path
) -> None:
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(feature_name, shap_values, X_sample, show=False, cmap='Greys', dot_size=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    # diagram 폴더에도 저장
    plt.savefig(DIAGRAM_DIR / out_path.name, dpi=150, bbox_inches="tight")
    plt.close()


def save_pdp_ice_plot(model: xgb.XGBRegressor, X: pd.DataFrame, feature_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    PartialDependenceDisplay.from_estimator(
        model,
        X,
        [feature_name],
        kind="both",
        subsample=200,
        random_state=RANDOM_STATE,
        ax=ax,
    )
    ax.set_title(f"PDP/ICE: {feature_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_permutation_importance_plot(
    model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series, feature_names: list[str], out_path: Path
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=-1,
    )

    perm_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    top = perm_df.head(10).iloc[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation Importance (R² decrease)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return perm_df


def main() -> None:
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    print("모델 성능 (테스트)")
    print(f"- RMSE: {metrics['rmse']:.4f}")
    print(f"- R²  : {metrics['r2']:.4f}")

    X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    top_feature = str(shap_importance.loc[0, "feature"])
    print(f"\nSHAP 전역 중요도 Top-1: {top_feature}")


    # SHAP Summary Plot (Bar)
    shap_bar_path = OUTPUT_DIR / "shap_summary_bar.png"
    save_shap_summary_bar(shap_values, X_sample, shap_bar_path)

    # SHAP Summary Plot (Dot) - 흑백
    shap_dot_path = OUTPUT_DIR / "shap_summary_dot.png"
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False, cmap='Greys')
    plt.tight_layout()
    plt.savefig(shap_dot_path, dpi=150, bbox_inches="tight")
    plt.savefig(DIAGRAM_DIR / "shap_summary_dot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP Dependence Plot (Top Feature)
    shap_dep_path = OUTPUT_DIR / "shap_dependence_top_feature.png"
    save_shap_dependence_plot(top_feature, shap_values, X_sample, shap_dep_path)

    # SHAP Waterfall Plot (첫 번째 샘플) - PIL로 흑백 변환
    shap_waterfall_path = OUTPUT_DIR / "shap_waterfall_sample0.png"
    shap_waterfall_diagram = DIAGRAM_DIR / "shap_waterfall_sample0.png"
    base_value = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]

    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=X_sample.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(shap_waterfall_path, dpi=150, bbox_inches="tight")
    plt.savefig(shap_waterfall_diagram, dpi=150, bbox_inches="tight")
    plt.close()
    # PIL로 흑백 변환
    convert_to_grayscale(shap_waterfall_path)
    convert_to_grayscale(shap_waterfall_diagram)

    # SHAP Force Plot (첫 번째 샘플) - PIL로 흑백 변환
    shap_force_path = OUTPUT_DIR / "shap_force_sample0.png"
    shap_force_diagram = DIAGRAM_DIR / "shap_force_sample0.png"
    force_fig = shap.plots.force(
        explainer.expected_value, shap_values[0], X_sample.iloc[0], matplotlib=True, show=False
    )
    plt.tight_layout()
    plt.savefig(shap_force_path, dpi=150, bbox_inches="tight")
    plt.savefig(shap_force_diagram, dpi=150, bbox_inches="tight")
    plt.close()
    # PIL로 흑백 변환
    convert_to_grayscale(shap_force_path)
    convert_to_grayscale(shap_force_diagram)


    pdp_ice_path = OUTPUT_DIR / "pdp_ice_top_feature.png"
    save_pdp_ice_plot(model, X_test, top_feature, pdp_ice_path)

    perm_path = OUTPUT_DIR / "permutation_importance.png"
    perm_df = save_permutation_importance_plot(model, X_test, y_test, feature_names, perm_path)

    model_gain = pd.DataFrame(
        {
            "feature": feature_names,
            "xgb_gain_importance": model.feature_importances_,
        }
    )

    merged = (
        shap_importance.merge(model_gain, on="feature", how="left")
        .merge(perm_df[["feature", "importance_mean"]], on="feature", how="left")
        .rename(columns={"importance_mean": "permutation_importance_mean"})
    )

    out_csv = OUTPUT_DIR / "feature_importance_comparison.csv"
    merged.to_csv(out_csv, index=False)


    print("\n산출물 저장")
    print(f"- {shap_bar_path}")
    print(f"- {shap_dot_path}")
    print(f"- {shap_dep_path}")
    print(f"- {shap_waterfall_path}")
    print(f"- {shap_force_path}")
    print(f"- {pdp_ice_path}")
    print(f"- {perm_path}")
    print(f"- {out_csv}")


if __name__ == "__main__":
    main()

