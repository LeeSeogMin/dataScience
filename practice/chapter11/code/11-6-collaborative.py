"""
11-6-collaborative.py
11.6절 추천 시스템: 사용자-아이템 관계 분석

MovieLens 데이터로 협업 필터링과 행렬 분해를 비교한다.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from surprise import Dataset, Reader, SVD, KNNBasic, KNNWithMeans
    from surprise.model_selection import cross_validate, train_test_split
    from surprise import accuracy
except ImportError:
    print("scikit-surprise 설치 필요: pip install scikit-surprise")
    raise

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_movielens():
    """MovieLens 100K 데이터 로드"""
    print("MovieLens 100K 데이터 로드 중...")
    data = Dataset.load_builtin("ml-100k")

    # 전체 데이터셋 정보
    trainset = data.build_full_trainset()
    n_users = trainset.n_users
    n_items = trainset.n_items
    n_ratings = trainset.n_ratings

    print(f"사용자 수: {n_users}")
    print(f"아이템 수: {n_items}")
    print(f"평점 수: {n_ratings}")
    print(f"희소성: {1 - n_ratings / (n_users * n_items):.2%}")

    return data


def train_and_evaluate_models(data):
    """여러 모델 학습 및 평가"""
    models = {
        "SVD": SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42),
        "User-KNN": KNNWithMeans(k=40, sim_options={"name": "cosine", "user_based": True}),
        "Item-KNN": KNNWithMeans(k=40, sim_options={"name": "cosine", "user_based": False}),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{name} 모델 학습 및 교차 검증 중...")
        cv_results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5, verbose=False)

        results[name] = {
            "rmse_mean": round(cv_results["test_rmse"].mean(), 4),
            "rmse_std": round(cv_results["test_rmse"].std(), 4),
            "mae_mean": round(cv_results["test_mae"].mean(), 4),
            "mae_std": round(cv_results["test_mae"].std(), 4),
        }
        print(f"  RMSE: {results[name]['rmse_mean']:.4f} (+/- {results[name]['rmse_std']:.4f})")
        print(f"  MAE: {results[name]['mae_mean']:.4f} (+/- {results[name]['mae_std']:.4f})")

    return results


def get_top_n_recommendations(model, trainset, user_id, n=10):
    """특정 사용자에 대한 Top-N 추천"""
    # 사용자가 평가하지 않은 아이템 찾기
    user_inner_id = trainset.to_inner_uid(user_id)
    rated_items = set([i for i, _ in trainset.ur[user_inner_id]])

    predictions = []
    for item_inner_id in range(trainset.n_items):
        if item_inner_id not in rated_items:
            item_raw_id = trainset.to_raw_iid(item_inner_id)
            pred = model.predict(user_id, item_raw_id)
            predictions.append((item_raw_id, pred.est))

    # 상위 N개 선택
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


def analyze_svd_factors(model, trainset):
    """SVD 잠재 요인 분석"""
    # 사용자/아이템 잠재 요인
    user_factors = model.pu
    item_factors = model.qi

    # 잠재 요인 분포
    factor_stats = {
        "user_factors_shape": list(user_factors.shape),
        "item_factors_shape": list(item_factors.shape),
        "user_factors_mean": round(float(user_factors.mean()), 4),
        "user_factors_std": round(float(user_factors.std()), 4),
        "item_factors_mean": round(float(item_factors.mean()), 4),
        "item_factors_std": round(float(item_factors.std()), 4),
    }

    return factor_stats


def visualize_model_comparison(results):
    """모델 성능 비교 시각화 (흑백)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results.keys())
    rmse_values = [results[m]["rmse_mean"] for m in models]
    rmse_stds = [results[m]["rmse_std"] for m in models]
    mae_values = [results[m]["mae_mean"] for m in models]
    mae_stds = [results[m]["mae_std"] for m in models]

    x = np.arange(len(models))
    width = 0.5

    # 흑백 그라데이션 색상
    colors = ['#2d2d2d', '#6d6d6d', '#a0a0a0']  # 진한 회색 → 연한 회색

    # RMSE
    bars1 = axes[0].bar(x, rmse_values, width, yerr=rmse_stds, capsize=5,
                        color=colors, edgecolor='black', linewidth=1)
    axes[0].set_ylabel("RMSE", fontsize=11)
    axes[0].set_title("RMSE Comparison", fontweight="bold", fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=10)
    axes[0].set_ylim(0.85, 1.0)
    # 값 표시
    for bar, val in zip(bars1, rmse_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # MAE
    bars2 = axes[1].bar(x, mae_values, width, yerr=mae_stds, capsize=5,
                        color=colors, edgecolor='black', linewidth=1)
    axes[1].set_ylabel("MAE", fontsize=11)
    axes[1].set_title("MAE Comparison", fontweight="bold", fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=10)
    axes[1].set_ylim(0.65, 0.85)
    # 값 표시
    for bar, val in zip(bars2, mae_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # diagram 폴더에도 저장
    diagram_dir = Path(__file__).parent.parent.parent.parent / "diagram"
    diagram_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(diagram_dir / "ch11_recommendation_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "ch11_recommendation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"비교 시각화 저장: {diagram_dir / 'ch11_recommendation_comparison.png'}")


def analyze_cold_start(data):
    """Cold-start 문제 분석"""
    trainset = data.build_full_trainset()

    # 사용자별 평점 수
    user_ratings = {}
    for uid in range(trainset.n_users):
        user_ratings[uid] = len(trainset.ur[uid])

    # 아이템별 평점 수
    item_ratings = {}
    for iid in range(trainset.n_items):
        item_ratings[iid] = len(trainset.ir[iid])

    user_counts = list(user_ratings.values())
    item_counts = list(item_ratings.values())

    cold_start_analysis = {
        "users_with_less_than_5_ratings": sum(1 for c in user_counts if c < 5),
        "users_with_less_than_10_ratings": sum(1 for c in user_counts if c < 10),
        "items_with_less_than_5_ratings": sum(1 for c in item_counts if c < 5),
        "items_with_less_than_10_ratings": sum(1 for c in item_counts if c < 10),
        "avg_ratings_per_user": round(np.mean(user_counts), 2),
        "avg_ratings_per_item": round(np.mean(item_counts), 2),
        "median_ratings_per_user": int(np.median(user_counts)),
        "median_ratings_per_item": int(np.median(item_counts)),
    }

    return cold_start_analysis


def main():
    print("=" * 60)
    print("11.6 추천 시스템: 협업 필터링과 행렬 분해")
    print("=" * 60)

    # 1. 데이터 로드
    data = load_movielens()

    # 2. 모델 학습 및 평가
    results = train_and_evaluate_models(data)

    # 3. 시각화
    visualize_model_comparison(results)

    # 4. SVD 모델로 추천 생성
    print("\nSVD 모델로 Top-10 추천 생성...")
    trainset = data.build_full_trainset()
    svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)

    # 샘플 사용자에 대한 추천
    sample_user = "1"
    top_recommendations = get_top_n_recommendations(svd, trainset, sample_user, n=10)
    print(f"사용자 {sample_user}에 대한 추천:")
    for item_id, score in top_recommendations[:5]:
        print(f"  아이템 {item_id}: 예측 평점 {score:.2f}")

    # 5. SVD 잠재 요인 분석
    factor_stats = analyze_svd_factors(svd, trainset)
    print(f"\nSVD 잠재 요인 분석:")
    print(f"  사용자 요인 크기: {factor_stats['user_factors_shape']}")
    print(f"  아이템 요인 크기: {factor_stats['item_factors_shape']}")

    # 6. Cold-start 분석
    cold_start = analyze_cold_start(data)
    print(f"\nCold-start 분석:")
    print(f"  평점 5개 미만 사용자: {cold_start['users_with_less_than_5_ratings']}")
    print(f"  평점 5개 미만 아이템: {cold_start['items_with_less_than_5_ratings']}")

    # 7. 결과 저장
    summary = {
        "dataset": {
            "name": "MovieLens 100K",
            "n_users": trainset.n_users,
            "n_items": trainset.n_items,
            "n_ratings": trainset.n_ratings,
            "sparsity": round(1 - trainset.n_ratings / (trainset.n_users * trainset.n_items), 4)
        },
        "model_comparison": results,
        "best_model": min(results.items(), key=lambda x: x[1]["rmse_mean"])[0],
        "svd_factors": factor_stats,
        "cold_start_analysis": cold_start,
        "sample_recommendations": {
            "user_id": sample_user,
            "top_10": [{"item_id": item, "predicted_rating": round(score, 4)}
                       for item, score in top_recommendations]
        },
        "interpretation": {
            "svd_advantage": "희소 행렬을 저차원 잠재 공간으로 분해하여 확장성 확보",
            "knn_advantage": "직관적이고 설명 가능한 추천 (이 상품을 본 고객이...)",
            "cold_start": "신규 사용자/아이템에 대한 추천이 어려움"
        }
    }

    with open(OUTPUT_DIR / "ch11_recommendation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {OUTPUT_DIR / 'ch11_recommendation_summary.json'}")

    # 모델 비교 테이블 저장
    df = pd.DataFrame([
        {"model": model, **metrics}
        for model, metrics in results.items()
    ])
    df.to_csv(OUTPUT_DIR / "ch11_recommendation_metrics.csv", index=False)
    print(f"메트릭 저장: {OUTPUT_DIR / 'ch11_recommendation_metrics.csv'}")

    # 추천 결과 저장
    rec_df = pd.DataFrame([
        {"rank": i+1, "item_id": item, "predicted_rating": round(score, 4)}
        for i, (item, score) in enumerate(top_recommendations)
    ])
    rec_df.to_csv(OUTPUT_DIR / "ch11_recommendation_top10_user1.csv", index=False)
    print(f"추천 결과 저장: {OUTPUT_DIR / 'ch11_recommendation_top10_user1.csv'}")

    print("\n" + "=" * 60)
    print(f"최적 모델: {summary['best_model']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    summary = main()
