"""
FT-Transformer 아키텍처 다이어그램
- Feature Tokenizer + Transformer 구조
- 수치형/범주형 특성 처리
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


def draw_ft_transformer():
    fig, ax = plt.subplots(figsize=(14, 10))

    # 색상 정의 (흑백 계열)
    colors = {
        'input': '#E0E0E0',
        'tokenizer': '#AAAAAA',
        'embedding': '#888888',
        'transformer': '#555555',
        'output': '#333333',
        'cls': '#222222'
    }

    def draw_block(x, y, w, h, text, color, fontsize=10):
        """블록 그리기"""
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.02",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        text_color = 'white' if color in ['#888888', '#666666', '#555555', '#444444', '#333333', '#222222'] else 'black'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color)

    def draw_arrow(x1, y1, x2, y2, style='->', color='black', lw=1.5):
        """화살표 그리기"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    # === 입력 특성 ===
    # 수치형 특성
    draw_block(0.12, 0.85, 0.14, 0.07, '수치형 특성\nx₁, x₂, ...', colors['input'], 9)
    # 범주형 특성
    draw_block(0.12, 0.70, 0.14, 0.07, '범주형 특성\nc₁, c₂, ...', colors['input'], 9)

    # === Feature Tokenizer ===
    # 수치형 처리
    draw_block(0.32, 0.85, 0.14, 0.06, 'Linear + Bias', colors['tokenizer'], 9)
    # 범주형 처리
    draw_block(0.32, 0.70, 0.14, 0.06, 'Embedding\nLookup', colors['tokenizer'], 9)

    # 화살표: 입력 → Tokenizer
    draw_arrow(0.19, 0.85, 0.25, 0.85)
    draw_arrow(0.19, 0.70, 0.25, 0.70)

    # === 임베딩 시퀀스 ===
    # 토큰 박스들
    token_y = 0.775
    token_xs = [0.48, 0.54, 0.60, 0.66, 0.72]
    token_labels = ['e₁', 'e₂', 'e₃', '...', 'eN']

    for tx, label in zip(token_xs, token_labels):
        draw_block(tx, token_y, 0.045, 0.06, label, colors['embedding'], 9)

    # [CLS] 토큰
    draw_block(0.42, token_y, 0.045, 0.06, '[CLS]', colors['cls'], 8)

    # 화살표: Tokenizer → 임베딩
    draw_arrow(0.39, 0.85, 0.455, 0.81)
    draw_arrow(0.39, 0.70, 0.455, 0.745)

    # 임베딩 시퀀스 레이블
    ax.text(0.57, 0.86, '임베딩 시퀀스', fontsize=11, ha='center', fontweight='bold', color='black')

    # === Transformer Block ===
    transformer_y = 0.50
    draw_block(0.57, transformer_y + 0.12, 0.28, 0.08, 'Multi-Head Self-Attention', colors['transformer'], 10)
    draw_block(0.57, transformer_y, 0.28, 0.06, 'Add & Norm', colors['tokenizer'], 9)
    draw_block(0.57, transformer_y - 0.10, 0.28, 0.06, 'Feed Forward', colors['transformer'], 10)
    draw_block(0.57, transformer_y - 0.20, 0.28, 0.06, 'Add & Norm', colors['tokenizer'], 9)

    # Transformer Block 테두리
    block_rect = mpatches.FancyBboxPatch((0.40, 0.24), 0.34, 0.46,
                                          boxstyle="round,pad=0.02",
                                          fill=False, edgecolor='#666666',
                                          linewidth=2, linestyle='--')
    ax.add_patch(block_rect)
    ax.text(0.76, 0.68, 'Transformer\nBlock\n(× L층)', fontsize=10, ha='left',
            fontweight='bold', color='#444444')

    # 화살표: 임베딩 → Transformer
    draw_arrow(0.57, 0.745, 0.57, 0.66)

    # Transformer 내부 화살표
    draw_arrow(0.57, 0.58, 0.57, 0.53)
    draw_arrow(0.57, 0.47, 0.57, 0.43)
    draw_arrow(0.57, 0.37, 0.57, 0.33)

    # === 출력 ===
    draw_block(0.57, 0.15, 0.12, 0.06, '[CLS] 출력', colors['cls'], 9)
    draw_block(0.57, 0.05, 0.12, 0.05, '예측', colors['output'], 10)

    # 화살표: Transformer → 출력
    draw_arrow(0.57, 0.27, 0.57, 0.18)
    draw_arrow(0.57, 0.12, 0.57, 0.08)

    # === 설명 텍스트 ===
    ax.text(0.88, 0.85, '• Feature Tokenizer:\n  모든 특성을 동일 차원\n  벡터로 변환',
            fontsize=9, va='top', fontweight='bold', color='#222222')
    ax.text(0.88, 0.70, '• [CLS] 토큰:\n  전체 샘플 대표 벡터\n  (BERT에서 차용)',
            fontsize=9, va='top', fontweight='bold', color='#222222')
    ax.text(0.88, 0.52, '• Self-Attention:\n  모든 특성 쌍의\n  상호작용 학습',
            fontsize=9, va='top', fontweight='bold', color='#222222')
    ax.text(0.88, 0.35, '• 최종 예측:\n  [CLS] 출력만 사용\n  (분류/회귀 헤드)',
            fontsize=9, va='top', fontweight='bold', color='#222222')

    # 제목
    ax.text(0.50, 0.96, 'FT-Transformer 아키텍처', fontsize=14, ha='center',
            fontweight='bold', color='black')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw_ft_transformer()

    output_path = "C:/Dev/book-datascience/content/graphics/ch07/7-5a-ft-transformer-arch.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"저장 완료: {output_path}")

    plt.close()
