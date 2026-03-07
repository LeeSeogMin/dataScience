"""
오토인코더 신경망 구조 시각화
- 입력층 → 인코더 → 잠재공간(병목) → 디코더 → 출력층
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def draw_neural_network():
    fig, ax = plt.subplots(figsize=(14, 8))

    # 레이어 구성: [입력, 인코더1, 인코더2, 잠재공간, 디코더1, 디코더2, 출력]
    layer_sizes = [8, 6, 4, 2, 4, 6, 8]
    layer_names = ['입력층\n(Input)', '인코더\n(Encoder)', '', '잠재공간\n(Latent)', '', '디코더\n(Decoder)', '출력층\n(Output)']

    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    # 레이어 위치
    layer_positions = np.linspace(0.1, 0.9, n_layers)

    # 노드 그리기
    node_positions = {}

    for i, (n_neurons, x_pos) in enumerate(zip(layer_sizes, layer_positions)):
        y_positions = np.linspace(0.5 - (n_neurons-1)*0.05, 0.5 + (n_neurons-1)*0.05, n_neurons)

        for j, y_pos in enumerate(y_positions):
            # 색상: 잠재공간은 진하게
            if i == 3:  # 잠재공간
                color = '#333333'
                edgecolor = 'black'
            elif i < 3:  # 인코더
                color = '#888888'
                edgecolor = '#444444'
            else:  # 디코더
                color = '#AAAAAA'
                edgecolor = '#666666'

            circle = plt.Circle((x_pos, y_pos), 0.018, color=color, ec=edgecolor, linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            node_positions[(i, j)] = (x_pos, y_pos)

    # 연결선 그리기
    for i in range(n_layers - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                start = node_positions[(i, j)]
                end = node_positions[(i + 1, k)]

                # 선 스타일
                alpha = 0.3
                linewidth = 0.5

                ax.plot([start[0], end[0]], [start[1], end[1]],
                       'k-', alpha=alpha, linewidth=linewidth, zorder=1)

    # 레이어 이름
    for i, (x_pos, name) in enumerate(zip(layer_positions, layer_names)):
        if name:
            ax.text(x_pos, 0.15, name, ha='center', va='top', fontsize=11, fontweight='bold')

    # 화살표와 레이블
    # 압축 화살표
    ax.annotate('', xy=(0.42, 0.85), xytext=(0.22, 0.85),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=2))
    ax.text(0.32, 0.88, '압축 (Encoding)', ha='center', fontsize=10, color='#444444')

    # 복원 화살표
    ax.annotate('', xy=(0.78, 0.85), xytext=(0.58, 0.85),
                arrowprops=dict(arrowstyle='->', color='#444444', lw=2))
    ax.text(0.68, 0.88, '복원 (Decoding)', ha='center', fontsize=10, color='#444444')

    # 병목 표시
    ax.annotate('병목\n(Bottleneck)', xy=(0.5, 0.62), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#666666'))

    # 차원 표시
    ax.text(0.1, 0.02, 'D차원', ha='center', fontsize=9, color='#666666')
    ax.text(0.5, 0.02, 'd차원 (d << D)', ha='center', fontsize=9, color='#666666')
    ax.text(0.9, 0.02, 'D차원', ha='center', fontsize=9, color='#666666')

    # 재구성 오류 표시
    ax.text(0.5, -0.02, '재구성 오류 = ||입력 - 출력||²', ha='center', fontsize=10,
            style='italic', color='#333333')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    fig = draw_neural_network()

    # 저장
    output_path = "C:/Dev/book-datascience/content/graphics/ch07/7-3-autoencoder-nn.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"저장 완료: {output_path}")

    plt.close()
