"""
4.3 HDBSCAN 군집 분석: 밀도 기반 비구형 군집 탐지
================================================
K-Means가 실패하는 비구형 군집 데이터에서 HDBSCAN의 우수성을 실습
노이즈 포인트 자동 탐지와 K 자동 결정의 장점 확인

실행: python 4-3-hdbscan-clustering.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan
import warnings

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def generate_complex_data(n_samples=500, random_state=42):
    """
    K-Means가 실패하는 복잡한 데이터 생성:
    - 반달 모양 군집 2개
    - 노이즈 포인트 추가
    """
    np.random.seed(random_state)

    # 반달 모양 데이터
    X_moons, y_moons = make_moons(n_samples=int(n_samples * 0.8), noise=0.08, random_state=random_state)

    # 노이즈 포인트 (균등 분포)
    n_noise = int(n_samples * 0.2)
    X_noise = np.random.uniform(-1.5, 2.5, size=(n_noise, 2))
    y_noise = np.array([-1] * n_noise)  # 노이즈 레이블

    # 합치기
    X = np.vstack([X_moons, X_noise])
    y_true = np.concatenate([y_moons, y_noise])

    return X, y_true


def generate_spherical_data(n_samples=500, random_state=42):
    """구형 군집 데이터 생성 (K-Means에 적합)"""
    X, y = make_blobs(
        n_samples=n_samples,
        centers=3,
        cluster_std=0.6,
        random_state=random_state
    )
    return X, y


def compare_clustering_spherical(X, y_true):
    """구형 데이터에서 K-Means vs HDBSCAN 비교"""
    print("\n[구형 군집 데이터]")
    print("-" * 50)

    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X)
    sil_kmeans = silhouette_score(X, labels_kmeans)
    ari_kmeans = adjusted_rand_score(y_true, labels_kmeans)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
    labels_hdbscan = clusterer.fit_predict(X)

    # 노이즈(-1) 제외하고 실루엣 계산
    mask = labels_hdbscan != -1
    if mask.sum() > 1 and len(set(labels_hdbscan[mask])) > 1:
        sil_hdbscan = silhouette_score(X[mask], labels_hdbscan[mask])
    else:
        sil_hdbscan = 0.0
    ari_hdbscan = adjusted_rand_score(y_true, labels_hdbscan)

    n_noise_hdbscan = (labels_hdbscan == -1).sum()
    n_clusters_hdbscan = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)

    print(f"{'알고리즘':<12} {'군집 수':>10} {'노이즈':>10} {'실루엣':>12} {'ARI':>10}")
    print("-" * 54)
    print(f"{'K-Means':<12} {3:>10} {0:>10} {sil_kmeans:>12.4f} {ari_kmeans:>10.4f}")
    print(f"{'HDBSCAN':<12} {n_clusters_hdbscan:>10} {n_noise_hdbscan:>10} {sil_hdbscan:>12.4f} {ari_hdbscan:>10.4f}")

    return labels_kmeans, labels_hdbscan


def compare_clustering_complex(X, y_true):
    """비구형 데이터에서 K-Means vs HDBSCAN 비교"""
    print("\n[비구형 군집 데이터 (반달 모양 + 노이즈)]")
    print("-" * 50)

    # 실제 군집 수 (노이즈 제외)
    n_true_clusters = len(set(y_true)) - (1 if -1 in y_true else 0)

    # K-Means
    kmeans = KMeans(n_clusters=n_true_clusters, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X)
    sil_kmeans = silhouette_score(X, labels_kmeans)

    # 실제 레이블에서 노이즈 제외하고 ARI 계산
    mask_true = y_true != -1
    ari_kmeans = adjusted_rand_score(y_true[mask_true], labels_kmeans[mask_true])

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
    labels_hdbscan = clusterer.fit_predict(X)

    mask = labels_hdbscan != -1
    if mask.sum() > 1 and len(set(labels_hdbscan[mask])) > 1:
        sil_hdbscan = silhouette_score(X[mask], labels_hdbscan[mask])
    else:
        sil_hdbscan = 0.0

    # 노이즈 제외하고 ARI 계산
    mask_both = (y_true != -1) & (labels_hdbscan != -1)
    if mask_both.sum() > 0:
        ari_hdbscan = adjusted_rand_score(y_true[mask_both], labels_hdbscan[mask_both])
    else:
        ari_hdbscan = 0.0

    n_noise_true = (y_true == -1).sum()
    n_noise_hdbscan = (labels_hdbscan == -1).sum()
    n_clusters_hdbscan = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)

    print(f"실제 데이터: {n_true_clusters}개 군집, {n_noise_true}개 노이즈")
    print()
    print(f"{'알고리즘':<12} {'군집 수':>10} {'노이즈 탐지':>12} {'실루엣':>12} {'ARI':>10}")
    print("-" * 56)
    print(f"{'K-Means':<12} {n_true_clusters:>10} {0:>12} {sil_kmeans:>12.4f} {ari_kmeans:>10.4f}")
    print(f"{'HDBSCAN':<12} {n_clusters_hdbscan:>10} {n_noise_hdbscan:>12} {sil_hdbscan:>12.4f} {ari_hdbscan:>10.4f}")

    return labels_kmeans, labels_hdbscan, kmeans


def analyze_hdbscan_params(X):
    """HDBSCAN 파라미터 영향 분석"""
    print("\n[HDBSCAN 파라미터 영향 분석]")
    print("-" * 60)

    params_list = [
        {'min_cluster_size': 10, 'min_samples': 3},
        {'min_cluster_size': 15, 'min_samples': 5},
        {'min_cluster_size': 20, 'min_samples': 10},
        {'min_cluster_size': 30, 'min_samples': 15},
    ]

    print(f"{'min_cluster_size':>18} {'min_samples':>14} {'군집 수':>10} {'노이즈':>10}")
    print("-" * 52)

    for params in params_list:
        clusterer = hdbscan.HDBSCAN(**params)
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"{params['min_cluster_size']:>18} {params['min_samples']:>14} {n_clusters:>10} {n_noise:>10}")

    return


def plot_clustering_comparison(X, y_true, labels_km, labels_hdb, kmeans_model, output_path):
    """K-Means vs HDBSCAN 클러스터링 결과 시각화."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 색상 설정
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    # 1. 원본 데이터
    ax0 = axes[0]
    mask_cluster = y_true != -1
    mask_noise = y_true == -1
    ax0.scatter(X[mask_cluster, 0], X[mask_cluster, 1], c='steelblue', s=30, alpha=0.7, label='데이터')
    ax0.scatter(X[mask_noise, 0], X[mask_noise, 1], c='gray', s=30, alpha=0.5, marker='x', label='노이즈')
    ax0.set_title('원본 데이터\n(반달 모양 + 노이즈)', fontsize=11, fontweight='bold')
    ax0.set_xlabel('X₁')
    ax0.set_ylabel('X₂')
    ax0.legend(loc='upper right', fontsize=9)
    ax0.set_aspect('equal')

    # 2. K-Means 결과
    ax1 = axes[1]
    for i in range(len(set(labels_km))):
        mask = labels_km == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i % len(colors)], s=30, alpha=0.7, label=f'군집 {i+1}')
    ax1.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1],
                c='black', marker='X', s=150, edgecolors='white', linewidths=2, label='중심점')
    ax1.set_title('K-Means\n구형 경계, 노이즈도 할당', fontsize=11, fontweight='bold')
    ax1.set_xlabel('X₁')
    ax1.set_ylabel('X₂')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_aspect('equal')

    # 3. HDBSCAN 결과
    ax2 = axes[2]
    unique_labels = sorted(set(labels_hdb))
    for label in unique_labels:
        mask = labels_hdb == label
        if label == -1:
            ax2.scatter(X[mask, 0], X[mask, 1], c='gray', s=30, alpha=0.5, marker='x', label='노이즈 (-1)')
        else:
            ax2.scatter(X[mask, 0], X[mask, 1], c=colors[label % len(colors)], s=30, alpha=0.7, label=f'군집 {label+1}')
    ax2.set_title('HDBSCAN\n밀도 기반, 노이즈 분리', fontsize=11, fontweight='bold')
    ax2.set_xlabel('X₁')
    ax2.set_ylabel('X₂')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n시각화 저장: {output_path}")


def main():
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("HDBSCAN 군집 분석: 밀도 기반 비구형 군집 탐지")
    print("=" * 60)

    # 1. 구형 군집 데이터
    print("\n" + "=" * 60)
    print("실험 1: 구형 군집 (K-Means에 유리)")
    print("=" * 60)

    X_spherical, y_spherical = generate_spherical_data(n_samples=500)
    print(f"- 데이터 포인트: {len(X_spherical)}개")
    print(f"- 군집 수: {len(set(y_spherical))}개")

    labels_km_sph, labels_hdb_sph = compare_clustering_spherical(X_spherical, y_spherical)

    print("\n결론: 구형 군집에서는 두 알고리즘 모두 잘 작동")

    # 2. 비구형 군집 데이터
    print("\n" + "=" * 60)
    print("실험 2: 비구형 군집 + 노이즈 (K-Means 실패 예상)")
    print("=" * 60)

    X_complex, y_complex = generate_complex_data(n_samples=500)
    n_noise_true = (y_complex == -1).sum()
    print(f"- 데이터 포인트: {len(X_complex)}개")
    print(f"- 실제 군집: 2개 (반달 모양)")
    print(f"- 실제 노이즈: {n_noise_true}개")

    labels_km_cplx, labels_hdb_cplx, kmeans_model = compare_clustering_complex(X_complex, y_complex)

    print("\n결론: 비구형 군집에서 HDBSCAN이 우수, 노이즈도 자동 탐지")

    # 시각화 저장
    plot_clustering_comparison(
        X_complex, y_complex, labels_km_cplx, labels_hdb_cplx,
        kmeans_model, os.path.join(OUTPUT_DIR, 'kmeans_vs_hdbscan.png')
    )

    # 3. HDBSCAN 파라미터 분석
    print("\n" + "=" * 60)
    print("실험 3: HDBSCAN 파라미터 영향")
    print("=" * 60)

    analyze_hdbscan_params(X_complex)

    print("\n파라미터 가이드:")
    print("- min_cluster_size: 최소 군집 크기 (크게 → 군집 수 감소, 노이즈 증가)")
    print("- min_samples: 코어 포인트 기준 (크게 → 더 보수적, 노이즈 증가)")

    # 4. 최종 비교 요약
    print("\n" + "=" * 60)
    print("K-Means vs HDBSCAN 비교 요약")
    print("=" * 60)

    comparison = """
| 특성            | K-Means           | HDBSCAN           |
|-----------------|-------------------|-------------------|
| 군집 형태       | 구형만            | 임의 형태         |
| K 결정          | 사전 지정 필요    | 자동 결정         |
| 노이즈 처리     | 불가              | 자동 탐지 (-1)    |
| 계산 복잡도     | O(nKT)            | O(n log n)        |
| 적합한 상황     | 구형 군집, K 알 때 | 비구형, 노이즈 있을 때 |
"""
    print(comparison)

    # 5. 결과 저장
    output_path = os.path.join(OUTPUT_DIR, 'hdbscan_comparison.csv')
    result_df = pd.DataFrame({
        'x': X_complex[:, 0],
        'y': X_complex[:, 1],
        'true_label': y_complex,
        'kmeans_label': labels_km_cplx,
        'hdbscan_label': labels_hdb_cplx
    })
    result_df.to_csv(output_path, index=False)
    print(f"\n결과 저장: {output_path}")

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
