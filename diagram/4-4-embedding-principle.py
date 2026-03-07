"""임베딩이 의미를 포착하는 원리 시각화."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 1. 텍스트 입력 (의미적 그룹)
ax0 = axes[0]
ax0.set_xlim(0, 10)
ax0.set_ylim(0, 10)
ax0.axis('off')
ax0.set_title('① 텍스트 입력', fontsize=12, fontweight='bold', pad=10)

# 배송 관련 텍스트 (파란색)
texts_delivery = ['"배송이 빨랐어요"', '"빠른 배달 감사"', '"택배가 빨리 왔어요"']
# 색상 관련 텍스트 (빨간색)
texts_color = ['"색상이 예뻐요"', '"컬러가 마음에 들어요"']

y_pos = 8
for t in texts_delivery:
    ax0.text(5, y_pos, t, ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#bbdefb', edgecolor='#1976d2'))
    y_pos -= 1.5

y_pos = 3
for t in texts_color:
    ax0.text(5, y_pos, t, ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffcdd2', edgecolor='#d32f2f'))
    y_pos -= 1.5

# 2. 임베딩 모델 (변환 과정)
ax1 = axes[1]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('② 임베딩 모델', fontsize=12, fontweight='bold', pad=10)

# 모델 박스
model_box = FancyBboxPatch((1.5, 2), 7, 6, boxstyle="round,pad=0.1",
                            facecolor='#fff3e0', edgecolor='#f57c00', linewidth=2)
ax1.add_patch(model_box)

ax1.text(5, 7, '문맥 학습', ha='center', va='center', fontsize=11, fontweight='bold')
ax1.text(5, 5.5, '"함께 등장하는 단어는', ha='center', va='center', fontsize=9)
ax1.text(5, 4.5, '비슷한 의미"', ha='center', va='center', fontsize=9)
ax1.text(5, 3, '→ 벡터로 변환', ha='center', va='center', fontsize=10, fontweight='bold')

# 모델 종류
ax1.text(5, 1, 'Word2Vec · BERT · Sentence-BERT', ha='center', va='center',
         fontsize=9, style='italic', color='#666666')

# 3. 벡터 공간 (유사 의미 = 가까운 거리)
ax2 = axes[2]
ax2.set_xlim(-2, 3)
ax2.set_ylim(-2, 3)
ax2.set_title('③ 벡터 공간', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('차원 1', fontsize=9)
ax2.set_ylabel('차원 2', fontsize=9)

# 배송 관련 벡터 (가깝게 배치)
np.random.seed(42)
delivery_x = np.random.normal(1.5, 0.3, 3)
delivery_y = np.random.normal(1.5, 0.3, 3)
ax2.scatter(delivery_x, delivery_y, c='#1976d2', s=150, alpha=0.8, edgecolors='white', linewidths=2)

# 색상 관련 벡터 (다른 위치에 가깝게 배치)
color_x = np.random.normal(-0.5, 0.25, 2)
color_y = np.random.normal(-0.5, 0.25, 2)
ax2.scatter(color_x, color_y, c='#d32f2f', s=150, alpha=0.8, edgecolors='white', linewidths=2)

# 군집 표시 (점선 원)
circle1 = plt.Circle((1.5, 1.5), 0.7, fill=False, linestyle='--', color='#1976d2', linewidth=1.5)
circle2 = plt.Circle((-0.5, -0.5), 0.6, fill=False, linestyle='--', color='#d32f2f', linewidth=1.5)
ax2.add_patch(circle1)
ax2.add_patch(circle2)

# 레이블
ax2.annotate('배송 관련', xy=(1.5, 2.3), ha='center', fontsize=9, color='#1976d2', fontweight='bold')
ax2.annotate('색상 관련', xy=(-0.5, -1.3), ha='center', fontsize=9, color='#d32f2f', fontweight='bold')

# 거리 표시
ax2.annotate('', xy=(0.3, 0.3), xytext=(1.0, 1.0),
             arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.5))
ax2.text(0.9, 0.3, '먼 거리\n(다른 의미)', fontsize=8, color='#666666', ha='center')

ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

# 범례
legend_elements = [
    mpatches.Patch(facecolor='#bbdefb', edgecolor='#1976d2', label='배송 관련'),
    mpatches.Patch(facecolor='#ffcdd2', edgecolor='#d32f2f', label='색상 관련')
]
ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('C:/Dev/book-datascience/diagram/4-4-embedding-principle.png',
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("저장 완료: diagram/4-4-embedding-principle.png")
