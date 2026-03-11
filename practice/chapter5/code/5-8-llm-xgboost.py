"""
5-8-llm-xgboost.py: LLM 임베딩 + XGBoost 하이브리드 (이커머스 구매 예측)

이커머스 가상 데이터에서 텍스트 임베딩을 정형 특성과 결합해 XGBoost 입력으로 활용한다.
- 정형 특성: 나이, 구매금액, 방문횟수, 장바구니 상품수, 회원등급 등
- 텍스트 특성: 상품 리뷰, 고객 문의, 검색 키워드

1) 정형 특성만으로 XGBoost 학습
2) 텍스트 컬럼을 문장으로 결합하고 Sentence-BERT 임베딩 추출
3) PCA로 임베딩 차원 축소
4) 정형 + 임베딩 하이브리드 모델 학습 및 성능 비교

실행 방법:
    python 5-8-llm-xgboost.py

산출물:
    practice/chapter5/data/output/
      - llm_xgboost_metrics.png
      - llm_xgboost_results.json
      - llm_xgboost_feature_importance.png
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA

# 한글 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic)
import platform
if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "sentence-transformers가 필요합니다. 먼저 `pip install -r practice/chapter5/code/requirements.txt`를 실행하세요."
    ) from exc

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_ecommerce_data(n_samples: int = 2000) -> pd.DataFrame:
    """이커머스 구매 예측용 가상 데이터 생성.

    수치 특성과 텍스트 특성을 모두 포함하는 데이터를 생성한다.
    """
    # 수치 특성
    age = np.random.randint(18, 65, n_samples)
    total_purchase_amount = np.random.exponential(150, n_samples).round(2)
    visit_count = np.random.poisson(8, n_samples)
    cart_item_count = np.random.poisson(3, n_samples)
    days_since_last_visit = np.random.exponential(15, n_samples).round(0).astype(int)
    avg_session_duration = np.random.exponential(300, n_samples).round(0)  # 초 단위

    # 범주형 특성
    membership_levels = ["bronze", "silver", "gold", "platinum"]
    membership = np.random.choice(membership_levels, n_samples, p=[0.4, 0.3, 0.2, 0.1])

    device_types = ["mobile", "desktop", "tablet"]
    device = np.random.choice(device_types, n_samples, p=[0.55, 0.35, 0.10])

    # 텍스트 특성 생성
    review_templates_positive = [
        "상품 품질이 좋고 배송도 빨랐습니다. 다음에 또 구매할 의향이 있습니다.",
        "가격 대비 만족스러운 제품입니다. 포장도 꼼꼼했어요.",
        "기대했던 것보다 훨씬 좋네요. 색상도 예쁘고 재질도 좋습니다.",
        "배송이 정말 빠르고 제품 상태도 완벽했습니다. 추천합니다!",
        "여러 번 재구매하는 제품입니다. 항상 만족스럽습니다.",
        "가성비 최고! 이 가격에 이 품질이라니 놀랍습니다.",
        "선물용으로 구매했는데 반응이 아주 좋았어요.",
        "오래 사용해도 품질이 유지됩니다. 내구성이 좋아요.",
    ]

    review_templates_negative = [
        "배송이 너무 늦었고 상품도 기대 이하입니다.",
        "사진과 실물이 많이 다릅니다. 실망스럽네요.",
        "품질이 좋지 않아요. 가격만큼의 가치가 없습니다.",
        "반품 절차가 복잡하고 고객 응대가 불친절했습니다.",
        "포장이 부실해서 상품이 손상되어 왔습니다.",
        "설명과 다른 제품이 왔어요. 교환 요청 중입니다.",
        "재구매 의사 없습니다. 전체적으로 불만족스럽습니다.",
        "사이즈가 맞지 않고 교환도 어렵습니다.",
    ]

    review_templates_neutral = [
        "보통입니다. 특별히 좋지도 나쁘지도 않아요.",
        "가격에 맞는 품질입니다. 기대만큼이에요.",
        "배송은 빨랐는데 상품은 그냥 그래요.",
        "처음 사용해봤는데 아직 평가하기 어렵습니다.",
        "다른 제품과 비교해봐야 할 것 같습니다.",
        "",  # 리뷰 미작성
    ]

    inquiry_templates = [
        "재고 문의드립니다. 언제 입고되나요?",
        "배송 일정이 어떻게 되나요?",
        "반품 절차 문의합니다.",
        "사이즈 교환 가능한가요?",
        "할인 쿠폰 적용이 안 됩니다.",
        "결제 오류가 발생했습니다.",
        "포인트 적립 문의드립니다.",
        "회원 등급 관련 문의입니다.",
        "",  # 문의 없음
    ]

    search_keywords_templates = [
        "할인 특가 세일",
        "신상품 추천 인기",
        "무료배송 당일배송",
        "브랜드 정품 품질",
        "선물 세트 패키지",
        "가성비 저렴한 실속",
        "프리미엄 고급 명품",
        "여름 시즌 계절",
        "리뷰 많은 베스트셀러",
        "",  # 검색 없음
    ]

    reviews = []
    inquiries = []
    search_keywords = []

    for i in range(n_samples):
        # 리뷰 생성 (구매 이력에 따라)
        if total_purchase_amount[i] > 200:
            review = np.random.choice(review_templates_positive + review_templates_neutral, p=[0.8/8]*8 + [0.2/6]*6)
        elif total_purchase_amount[i] < 50:
            review = np.random.choice(review_templates_negative + review_templates_neutral, p=[0.6/8]*8 + [0.4/6]*6)
        else:
            all_reviews = review_templates_positive + review_templates_negative + review_templates_neutral
            review = np.random.choice(all_reviews)
        reviews.append(review)

        # 문의 생성
        inquiries.append(np.random.choice(inquiry_templates))

        # 검색 키워드 생성
        search_keywords.append(np.random.choice(search_keywords_templates))

    # 목표 변수 생성 (구매 여부)
    # 텍스트 신호가 구매 예측에 강하게 기여하도록 설계
    # 긍정 리뷰는 재구매 의도를, 부정 리뷰는 이탈 신호를 반영
    positive_review_signal = np.array([1 if r in review_templates_positive else 0 for r in reviews])
    negative_review_signal = np.array([1 if r in review_templates_negative else 0 for r in reviews])
    discount_search_signal = np.array([1 if "할인" in s or "특가" in s or "세일" in s else 0 for s in search_keywords])
    inquiry_signal = np.array([1 if "재고" in q or "배송" in q else 0 for q in inquiries])

    purchase_prob = (
        0.20 +
        0.08 * (membership == "gold").astype(float) +
        0.12 * (membership == "platinum").astype(float) +
        0.06 * (visit_count > 10).astype(float) +
        0.08 * (cart_item_count > 3).astype(float) +
        0.04 * (days_since_last_visit < 7).astype(float) +
        0.15 * positive_review_signal +  # 긍정 리뷰 작성자는 재구매 확률 높음
        -0.20 * negative_review_signal +  # 부정 리뷰 작성자는 이탈 확률 높음
        0.12 * discount_search_signal +   # 할인 검색자는 구매 의도 높음
        0.10 * inquiry_signal +           # 재고/배송 문의자는 구매 의도 높음
        np.random.normal(0, 0.08, n_samples)
    )
    purchase_prob = np.clip(purchase_prob, 0.05, 0.95)
    purchased = (np.random.random(n_samples) < purchase_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "total_purchase_amount": total_purchase_amount,
        "visit_count": visit_count,
        "cart_item_count": cart_item_count,
        "days_since_last_visit": days_since_last_visit,
        "avg_session_duration": avg_session_duration,
        "membership": membership,
        "device": device,
        "review_text": reviews,
        "inquiry_text": inquiries,
        "search_keywords": search_keywords,
        "purchased": purchased,
    })

    return df


def build_text(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """여러 텍스트 컬럼을 하나의 문장으로 결합."""
    parts: list[pd.Series] = []
    for col in cols:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    if not parts:
        raise ValueError("텍스트로 결합할 컬럼이 없습니다.")

    text = parts[0]
    for part in parts[1:]:
        text = text + " " + part

    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def build_tabular_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """정형 특성을 수치화하여 행렬로 반환."""
    train_num = train_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    medians = train_num.median()
    train_num = train_num.fillna(medians)
    test_num = test_df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(medians)

    train_cat = train_df[categorical_cols].astype(object).fillna("missing").astype(str)
    test_cat = test_df[categorical_cols].astype(object).fillna("missing").astype(str)

    train_cat_d = pd.get_dummies(train_cat, columns=categorical_cols)
    test_cat_d = pd.get_dummies(test_cat, columns=categorical_cols)

    X_train = pd.concat([train_num, train_cat_d], axis=1)
    X_test = pd.concat([test_num, test_cat_d], axis=1)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train.to_numpy(), X_test.to_numpy(), list(X_train.columns)


def train_and_eval_xgb(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> tuple[dict[str, float], xgb.XGBClassifier]:
    """XGBoost 학습 및 평가."""
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1_score": float(f1_score(y_test, pred)),
    }
    return metrics, model


def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    title: str,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """특성 중요도 시각화."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importance[indices], color="steelblue")
    plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"emb_{i}" for i in indices])
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 60)
    print("LLM 임베딩 + XGBoost 하이브리드: 이커머스 구매 예측")
    print("=" * 60)

    # 데이터 생성
    print("\n[1] 이커머스 가상 데이터 생성")
    df = generate_ecommerce_data(n_samples=2000)
    print(f"- 전체 샘플 수: {len(df)}")
    print(f"- 구매 비율: {df['purchased'].mean():.2%}")

    # 특성 정의
    cols_numeric = ["age", "total_purchase_amount", "visit_count",
                    "cart_item_count", "days_since_last_visit", "avg_session_duration"]
    cols_categorical = ["membership", "device"]
    cols_text = ["review_text", "inquiry_text", "search_keywords"]

    # 텍스트 결합
    df["combined_text"] = build_text(df, cols_text)

    # 데이터 분할
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["purchased"],
    )
    print(f"- 학습 세트: {len(train_df)}, 테스트 세트: {len(test_df)}")

    # 정형 특성 준비
    X_tab_train, X_tab_test, tab_feature_names = build_tabular_matrices(
        train_df, test_df, numeric_cols=cols_numeric, categorical_cols=cols_categorical
    )

    y_train = train_df["purchased"].to_numpy()
    y_test = test_df["purchased"].to_numpy()

    # 1) 정형 특성만 사용
    print("\n[2] 정형 특성만 사용한 XGBoost")
    tab_metrics, tab_model = train_and_eval_xgb(X_tab_train, X_tab_test, y_train, y_test)
    print(f"- accuracy: {tab_metrics['accuracy']:.4f}")
    print(f"- roc_auc : {tab_metrics['roc_auc']:.4f}")
    print(f"- f1_score: {tab_metrics['f1_score']:.4f}")

    # 2) 텍스트 임베딩 추출
    print("\n[3] 텍스트 임베딩 추출 (Sentence-BERT)")
    start = time.time()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    emb_train = embedder.encode(train_df["combined_text"].tolist(), show_progress_bar=True)
    emb_test = embedder.encode(test_df["combined_text"].tolist(), show_progress_bar=True)
    elapsed = time.time() - start
    print(f"- 소요 시간: {elapsed:.1f}초")
    print(f"- 임베딩 차원: {emb_train.shape[1]}")

    # 3) PCA 차원 축소
    n_components = min(50, emb_train.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    emb_train_reduced = pca.fit_transform(emb_train)
    emb_test_reduced = pca.transform(emb_test)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"\n[4] PCA 차원 축소: {emb_train.shape[1]} → {n_components}")
    print(f"- 분산 설명 비율 합: {explained:.4f}")

    # 4) 하이브리드 모델
    X_hybrid_train = np.hstack([X_tab_train, emb_train_reduced])
    X_hybrid_test = np.hstack([X_tab_test, emb_test_reduced])

    # 하이브리드 특성 이름 생성
    hybrid_feature_names = tab_feature_names + [f"emb_{i}" for i in range(n_components)]

    print("\n[5] 하이브리드(정형+임베딩) XGBoost")
    hybrid_metrics, hybrid_model = train_and_eval_xgb(X_hybrid_train, X_hybrid_test, y_train, y_test)
    print(f"- accuracy: {hybrid_metrics['accuracy']:.4f}")
    print(f"- roc_auc : {hybrid_metrics['roc_auc']:.4f}")
    print(f"- f1_score: {hybrid_metrics['f1_score']:.4f}")

    # 성능 변화 계산
    acc_diff = hybrid_metrics["accuracy"] - tab_metrics["accuracy"]
    auc_diff = hybrid_metrics["roc_auc"] - tab_metrics["roc_auc"]
    f1_diff = hybrid_metrics["f1_score"] - tab_metrics["f1_score"]

    print("\n[6] 성능 변화 (하이브리드 - 정형)")
    print(f"- accuracy 변화: {acc_diff:+.4f}")
    print(f"- roc_auc 변화 : {auc_diff:+.4f}")
    print(f"- f1_score 변화: {f1_diff:+.4f}")

    # 결과 저장
    results = {
        "dataset": "ecommerce_purchase_prediction",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "purchase_rate": float(df["purchased"].mean()),
        "tabular_feature_count": int(X_tab_train.shape[1]),
        "tabular_feature_names": tab_feature_names,
        "text_columns": cols_text,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": int(emb_train.shape[1]),
        "pca_components": int(n_components),
        "pca_explained_variance_ratio_sum": explained,
        "embedding_extraction_time_sec": round(elapsed, 1),
        "metrics": {
            "tabular": tab_metrics,
            "hybrid": hybrid_metrics,
            "improvement": {
                "accuracy": round(acc_diff, 4),
                "roc_auc": round(auc_diff, 4),
                "f1_score": round(f1_diff, 4),
            }
        },
    }

    out_json = OUTPUT_DIR / "llm_xgboost_results.json"
    out_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 시각화 1: 성능 비교 막대 그래프
    out_plot = OUTPUT_DIR / "llm_xgboost_metrics.png"
    plt.figure(figsize=(8, 5))
    labels = ["Tabular Only", "Hybrid\n(Tabular+Embedding)"]
    metrics_names = ["accuracy", "roc_auc", "f1_score"]
    x = np.arange(len(labels))
    width = 0.25

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, (metric, color) in enumerate(zip(metrics_names, colors)):
        offset = (i - 1) * width
        tab_val = tab_metrics[metric]
        hyb_val = hybrid_metrics[metric]
        bars = plt.bar(x + offset, [tab_val, hyb_val], width, label=metric, color=color, alpha=0.8)
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("XGBoost Performance: Tabular vs Hybrid (E-commerce Purchase)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()

    # 시각화 2: 특성 중요도
    out_importance = OUTPUT_DIR / "llm_xgboost_feature_importance.png"
    plot_feature_importance(
        hybrid_model,
        hybrid_feature_names,
        "Hybrid Model Feature Importance (Top 15)",
        out_importance,
        top_n=15
    )

    print("\n[7] 산출물 저장")
    print(f"- {out_json}")
    print(f"- {out_plot}")
    print(f"- {out_importance}")

    # 결과 테이블 출력
    print("\n" + "=" * 60)
    print("결과 요약 테이블")
    print("=" * 60)
    print(f"{'입력 구성':<30} {'accuracy':>10} {'roc_auc':>10} {'f1_score':>10}")
    print("-" * 60)
    print(f"{'정형 특성만(tabular)':<30} {tab_metrics['accuracy']:>10.4f} {tab_metrics['roc_auc']:>10.4f} {tab_metrics['f1_score']:>10.4f}")
    print(f"{'정형+임베딩(hybrid)':<30} {hybrid_metrics['accuracy']:>10.4f} {hybrid_metrics['roc_auc']:>10.4f} {hybrid_metrics['f1_score']:>10.4f}")
    print("-" * 60)
    print(f"{'변화량':<30} {acc_diff:>+10.4f} {auc_diff:>+10.4f} {f1_diff:>+10.4f}")


if __name__ == "__main__":
    main()
