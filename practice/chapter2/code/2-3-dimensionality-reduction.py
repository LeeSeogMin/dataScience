"""
2-3-dimensionality-reduction.py: 차원 축소 방법 비교

PCA, t-SNE, UMAP을 비교한다:
1. 처리 속도
2. 군집 분리도 (Silhouette Score)
3. 시각화 품질

실행 방법:
    python 2-3-dimensionality-reduction.py

필수 라이브러리:
    pip install umap-learn
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("경고: umap-learn이 설치되지 않음. UMAP 비교 생략.")
    print("설치: pip install umap-learn")

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def reduce_with_pca(X: np.ndarray, n_components: int = 2) -> tuple:
    """PCA로 차원 축소한다."""
    start_time = time.time()
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    elapsed_time = time.time() - start_time

    return X_reduced, elapsed_time, pca.explained_variance_ratio_


def reduce_with_tsne(X: np.ndarray, n_components: int = 2, perplexity: int = 30) -> tuple:
    """t-SNE로 차원 축소한다."""
    start_time = time.time()
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=42, n_iter=1000)
    X_reduced = tsne.fit_transform(X)
    elapsed_time = time.time() - start_time

    return X_reduced, elapsed_time


def reduce_with_umap(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15) -> tuple:
    """UMAP으로 차원 축소한다."""
    if not UMAP_AVAILABLE:
        return None, 0

    start_time = time.time()
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=0.1, random_state=42)
    X_reduced = reducer.fit_transform(X)
    elapsed_time = time.time() - start_time

    return X_reduced, elapsed_time


def plot_reduction(X_reduced: np.ndarray, y: np.ndarray, title: str, ax):
    """차원 축소 결과를 시각화한다."""
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                        c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    return scatter


def main():
    print("="*60)
    print("차원 축소 방법 비교 (PCA vs t-SNE vs UMAP)")
    print("="*60)

    # 1. 데이터 로드
    print("\n[1/4] 데이터 로드 중...")
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"   데이터: {X.shape[0]}개 샘플, {X.shape[1]}차원")
    print(f"   클래스: {len(np.unique(y))}개 (숫자 0-9)")

    # 2. 스케일링
    print("\n[2/4] 데이터 스케일링...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 차원 축소 수행
    print("\n[3/4] 차원 축소 수행...")
    results = []

    # PCA
    print("   PCA 수행 중...")
    X_pca, time_pca, explained_var = reduce_with_pca(X_scaled)
    sil_pca = silhouette_score(X_pca, y)
    results.append({
        'method': 'PCA',
        'time': time_pca,
        'silhouette': sil_pca,
        'X_reduced': X_pca
    })
    print(f"   ✓ PCA 완료: {time_pca:.2f}초, Silhouette: {sil_pca:.3f}")
    print(f"     설명 분산: PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%")

    # t-SNE
    print("   t-SNE 수행 중...")
    X_tsne, time_tsne = reduce_with_tsne(X_scaled)
    sil_tsne = silhouette_score(X_tsne, y)
    results.append({
        'method': 't-SNE',
        'time': time_tsne,
        'silhouette': sil_tsne,
        'X_reduced': X_tsne
    })
    print(f"   ✓ t-SNE 완료: {time_tsne:.2f}초, Silhouette: {sil_tsne:.3f}")

    # UMAP
    if UMAP_AVAILABLE:
        print("   UMAP 수행 중...")
        X_umap, time_umap = reduce_with_umap(X_scaled)
        sil_umap = silhouette_score(X_umap, y)
        results.append({
            'method': 'UMAP',
            'time': time_umap,
            'silhouette': sil_umap,
            'X_reduced': X_umap
        })
        print(f"   ✓ UMAP 완료: {time_umap:.2f}초, Silhouette: {sil_umap:.3f}")

    # 4. 시각화
    print("\n[4/4] 시각화 생성...")

    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        scatter = plot_reduction(result['X_reduced'], y,
                                f"{result['method']} (Sil: {result['silhouette']:.3f})", ax)

    plt.colorbar(scatter, ax=axes[-1], label='Digit')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "dimensionality_reduction_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   시각화 저장: {output_path}")

    # 5. 결과 요약
    print("\n" + "="*60)
    print(f"차원 축소 방법 비교 (Digits, n={X.shape[0]}, {X.shape[1]}차원→2차원)")
    print("="*60)
    print()
    print(f"{'방법':<12} {'처리 시간':<15} {'군집 분리도(Silhouette)':<20}")
    print("-"*60)

    for result in results:
        print(f"{result['method']:<12} {result['time']:.2f}초{'':<10} {result['silhouette']:.3f}")

    # 6. PCA 추가 정보
    print("\n" + "-"*60)
    print("PCA 주성분 분석:")
    pca_full = PCA().fit(X_scaled)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_90 = np.argmax(cumsum >= 0.9) + 1
    n_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"   - 90% 분산 설명에 필요한 주성분 수: {n_90}개")
    print(f"   - 95% 분산 설명에 필요한 주성분 수: {n_95}개")
    print(f"   - 상위 10개 PC 누적 설명: {cumsum[9]*100:.1f}%")

    # 7. 결과 저장
    df_results = pd.DataFrame([
        {'method': r['method'], 'time': r['time'], 'silhouette': r['silhouette']}
        for r in results
    ])
    csv_path = OUTPUT_DIR / "dimensionality_reduction_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n결과 저장: {csv_path}")

    print("\n" + "="*60)
    print("분석 인사이트")
    print("="*60)
    print("""
- PCA: 가장 빠르지만 비선형 구조 포착에 한계
- t-SNE: 지역 구조를 잘 보존하지만 느림
- UMAP: t-SNE와 유사한 품질, 더 빠른 속도

선택 가이드:
- 빠른 탐색/전처리: PCA
- 시각화 (소규모): t-SNE 또는 UMAP
- 시각화 (대규모): UMAP (속도 우위)
""")

    return results


if __name__ == "__main__":
    main()
