"""
12-5-kge-viz-only.py
KGE 성능 비교 시각화 (문서 결과값 기반)

PyKEEN 실행 없이 문서의 결과값을 사용하여 시각화만 생성
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'

# 결과 저장 경로
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("12.5 지식 그래프 임베딩: TransE vs RotatE (시각화)")
    print("=" * 60)

    # 문서에 있는 결과값 사용
    results_summary = {
        "dataset": {
            "name": "FB15k-237",
            "num_entities": 14541,
            "num_relations": 237,
            "train_triples": 272115,
            "valid_triples": 17535,
            "test_triples": 20466
        },
        "models": {
            "TransE": {
                "MRR": 0.164,
                "Hits@1": 0.094,
                "Hits@3": 0.176,
                "Hits@10": 0.304,
                "embedding_dim": 100,
                "epochs": 100
            },
            "RotatE": {
                "MRR": 0.238,
                "Hits@1": 0.155,
                "Hits@3": 0.263,
                "Hits@10": 0.407,
                "embedding_dim": 100,
                "epochs": 100
            }
        },
        "comparison": {
            "mrr_diff": 0.074,
            "better_model": "RotatE",
            "interpretation": {
                "TransE": "간단하고 효율적, 1:1 관계에 적합",
                "RotatE": "복소 공간 회전으로 대칭/반대칭/역/합성 관계 모델링 가능"
            }
        },
        "relation_patterns": {
            "TransE": {
                "symmetric": False,
                "antisymmetric": True,
                "inverse": False,
                "composition": True
            },
            "RotatE": {
                "symmetric": True,
                "antisymmetric": True,
                "inverse": True,
                "composition": True
            }
        }
    }

    # 시각화: TransE vs RotatE 성능 비교
    print("\n[1] 시각화 생성")
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
    transe_scores = [
        results_summary['models']['TransE']['MRR'],
        results_summary['models']['TransE']['Hits@1'],
        results_summary['models']['TransE']['Hits@3'],
        results_summary['models']['TransE']['Hits@10']
    ]
    rotate_scores = [
        results_summary['models']['RotatE']['MRR'],
        results_summary['models']['RotatE']['Hits@1'],
        results_summary['models']['RotatE']['Hits@3'],
        results_summary['models']['RotatE']['Hits@10']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, transe_scores, width, label='TransE', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, rotate_scores, width, label='RotatE', color='#3498db', edgecolor='black')

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Knowledge Graph Embedding Performance\n(FB15k-237 Dataset)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y', alpha=0.3)

    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'ch12_kge_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"그래프 저장: {fig_path}")

    # 결과 저장
    output_path = os.path.join(OUTPUT_DIR, 'ch12_kge_summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"결과 저장: {output_path}")

    print("\n" + "=" * 60)
    print("시각화 완료!")
    print("=" * 60)

    return results_summary


if __name__ == "__main__":
    main()
