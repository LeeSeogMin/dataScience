"""
TabNet 아키텍처 다이어그램
- Sequential Attention 기반 특성 선택 구조
- Attentive Transformer + Feature Transformer
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


def draw_tabnet_architecture():
    fig, ax = plt.subplots(figsize=(14, 10))

    # 색상 정의 (흑백 계열)
    colors = {
        'input': '#E8E8E8',
        'attention': '#888888',
        'feature': '#AAAAAA',
        'split': '#CCCCCC',
        'output': '#666666',
        'mask': '#444444'
    }

    def draw_block(x, y, w, h, text, color, fontsize=9):
        """블록 그리기"""
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.02",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        # 어두운 배경에는 흰색 글자
        text_color = 'white' if color in ['#888888', '#666666', '#444444'] else 'black'
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize+1,
                fontweight='bold', color=text_color)

    def draw_arrow(x1, y1, x2, y2, style='->', color='black'):
        """화살표 그리기"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=color, lw=1.5))

    # === 입력 영역 ===
    draw_block(0.15, 0.85, 0.18, 0.08, '입력 특성\n(D차원)', colors['input'], 10)
    draw_block(0.15, 0.72, 0.14, 0.06, 'BN', colors['input'], 9)
    draw_arrow(0.15, 0.81, 0.15, 0.75)

    # === Step 1 ===
    step1_x = 0.40

    # Attentive Transformer
    draw_block(step1_x, 0.85, 0.16, 0.06, 'Attentive\nTransformer', colors['attention'], 8)

    # Sparsemax & Mask
    draw_block(step1_x, 0.72, 0.12, 0.05, 'Sparsemax', colors['mask'], 8)
    draw_block(step1_x, 0.62, 0.10, 0.04, 'Mask', colors['mask'], 8)

    # Feature Transformer
    draw_block(step1_x, 0.48, 0.16, 0.08, 'Feature\nTransformer', colors['feature'], 9)

    # Split
    draw_block(step1_x, 0.32, 0.10, 0.05, 'Split', colors['split'], 9)

    # 화살표들 - Step 1
    draw_arrow(0.24, 0.85, 0.32, 0.85)  # 입력 → Attentive
    draw_arrow(step1_x, 0.82, step1_x, 0.75)  # Attentive → Sparsemax
    draw_arrow(step1_x, 0.695, step1_x, 0.64)  # Sparsemax → Mask
    draw_arrow(0.22, 0.72, 0.32, 0.62)  # BN → Mask (마스킹)
    draw_arrow(step1_x, 0.60, step1_x, 0.52)  # Mask → Feature Transformer
    draw_arrow(step1_x, 0.44, step1_x, 0.35)  # Feature → Split

    # === Step 2 ===
    step2_x = 0.65

    draw_block(step2_x, 0.85, 0.16, 0.06, 'Attentive\nTransformer', colors['attention'], 8)
    draw_block(step2_x, 0.72, 0.12, 0.05, 'Sparsemax', colors['mask'], 8)
    draw_block(step2_x, 0.62, 0.10, 0.04, 'Mask', colors['mask'], 8)
    draw_block(step2_x, 0.48, 0.16, 0.08, 'Feature\nTransformer', colors['feature'], 9)
    draw_block(step2_x, 0.32, 0.10, 0.05, 'Split', colors['split'], 9)

    # 화살표들 - Step 2
    draw_arrow(step1_x + 0.05, 0.32, step2_x - 0.08, 0.85)  # Step1 Split → Step2 Attentive
    draw_arrow(step2_x, 0.82, step2_x, 0.75)
    draw_arrow(step2_x, 0.695, step2_x, 0.64)
    draw_arrow(0.22, 0.70, step2_x - 0.08, 0.62)  # BN → Mask2
    draw_arrow(step2_x, 0.60, step2_x, 0.52)
    draw_arrow(step2_x, 0.44, step2_x, 0.35)

    # === Step N (점선 표시) ===
    ax.text(0.78, 0.55, '...', fontsize=24, ha='center', va='center', fontweight='bold', color='#333333')
    ax.text(0.78, 0.48, 'Step N', fontsize=11, ha='center', va='center', fontweight='bold', color='#333333')

    # === 출력 집계 ===
    draw_block(0.52, 0.15, 0.12, 0.06, '∑ 집계', colors['output'], 10)
    draw_block(0.52, 0.05, 0.14, 0.05, '최종 예측', colors['output'], 9)

    # Split → 집계 화살표
    draw_arrow(step1_x, 0.295, 0.46, 0.18)
    draw_arrow(step2_x, 0.295, 0.58, 0.18)
    draw_arrow(0.52, 0.12, 0.52, 0.08)

    # === 레이블 및 주석 ===
    # Step 레이블
    ax.text(step1_x, 0.93, 'Step 1', fontsize=13, ha='center', fontweight='bold', color='black')
    ax.text(step2_x, 0.93, 'Step 2', fontsize=13, ha='center', fontweight='bold', color='black')

    # 설명 박스
    ax.text(0.88, 0.80, '• Attentive Transformer:\n  어떤 특성을 선택할지 결정',
            fontsize=9, va='top', fontweight='bold', color='#222222')
    ax.text(0.88, 0.65, '• Sparsemax:\n  희소 마스크 생성\n  (대부분 0, 일부만 선택)',
            fontsize=9, va='top', fontweight='bold', color='#222222')
    ax.text(0.88, 0.48, '• Feature Transformer:\n  선택된 특성 처리',
            fontsize=9, va='top', fontweight='bold', color='#222222')
    ax.text(0.88, 0.35, '• Split:\n  다음 단계 입력 /\n  출력 기여 분리',
            fontsize=9, va='top', fontweight='bold', color='#222222')

    # 특성 재사용 표시 (점선 곡선)
    ax.annotate('', xy=(step2_x - 0.08, 0.72), xytext=(step1_x + 0.08, 0.72),
               arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5,
                              connectionstyle='arc3,rad=-0.3', linestyle='--'))
    ax.text(0.52, 0.78, '이전 마스크\n정보 전달', fontsize=9, ha='center', fontweight='bold', color='#333333')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    fig = draw_tabnet_architecture()

    output_path = "C:/Dev/book-datascience/content/graphics/ch07/7-5-tabnet-arch.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"저장 완료: {output_path}")

    plt.close()
