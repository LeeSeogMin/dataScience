"""K-Means vs HDBSCAN 클러스터링 비교 시각화."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import hdbscan

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 반달 모양 데이터 생성 (비구형)
np.random.seed(42)
X, y_true = make_moons(n_samples=300, noise=0.08, random_state=42)

# 노이즈 추가
noise_points = np.random.uniform(-1.5, 2.5, size=(20, 2))
X_with_noise = np.vstack([X, noise_points])

# K-Means 클러스터링
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_with_noise)

# HDBSCAN 클러스터링
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
hdbscan_labels = clusterer.fit_predict(X_with_noise)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 원본 데이터
ax0 = axes[0]
ax0.scatter(X[:, 0], X[:, 1], c='steelblue', s=30, alpha=0.7, label='데이터')
ax0.scatter(noise_points[:, 0], noise_points[:, 1], c='gray', s=30, alpha=0.5, marker='x', label='노이즈')
ax0.set_title('원본 데이터 (반달 + 노이즈)', fontsize=12, fontweight='bold')
ax0.set_xlabel('X₁')
ax0.set_ylabel('X₂')
ax0.legend(loc='upper right', fontsize=9)
ax0.set_aspect('equal')

# K-Means 결과
ax1 = axes[1]
colors_km = ['#e74c3c', '#3498db']
for i in range(2):
    mask = kmeans_labels == i
    ax1.scatter(X_with_noise[mask, 0], X_with_noise[mask, 1],
                c=colors_km[i], s=30, alpha=0.7, label=f'군집 {i+1}')
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', marker='X', s=150, edgecolors='white', linewidths=2, label='중심점')
ax1.set_title('K-Means (K=2)\n→ 구형 경계, 노이즈도 할당', fontsize=12, fontweight='bold')
ax1.set_xlabel('X₁')
ax1.set_ylabel('X₂')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_aspect('equal')

# HDBSCAN 결과
ax2 = axes[2]
colors_hd = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
unique_labels = set(hdbscan_labels)
for label in unique_labels:
    mask = hdbscan_labels == label
    if label == -1:
        ax2.scatter(X_with_noise[mask, 0], X_with_noise[mask, 1],
                    c='gray', s=30, alpha=0.5, marker='x', label='노이즈 (-1)')
    else:
        ax2.scatter(X_with_noise[mask, 0], X_with_noise[mask, 1],
                    c=colors_hd[label % len(colors_hd)], s=30, alpha=0.7, label=f'군집 {label+1}')
ax2.set_title('HDBSCAN\n→ 밀도 기반, 노이즈 분리', fontsize=12, fontweight='bold')
ax2.set_xlabel('X₁')
ax2.set_ylabel('X₂')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('C:/Dev/book-datascience/diagram/4-3-clustering-comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("저장 완료: diagram/4-3-clustering-comparison.png")
print(f"K-Means: 모든 {len(X_with_noise)}개 점을 2개 군집에 할당")
print(f"HDBSCAN: 노이즈 {sum(hdbscan_labels == -1)}개 분리, {len(set(hdbscan_labels)) - 1}개 군집 탐지")
