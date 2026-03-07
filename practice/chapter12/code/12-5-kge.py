"""
12-5-kge.py
지식 그래프 임베딩 실습

PyKEEN을 이용한 TransE와 RotatE 비교
- FB15k-237 데이터셋
- 링크 예측 성능 평가
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

# PyKEEN 임포트
try:
    from pykeen.pipeline import pipeline
    from pykeen.datasets import FB15k237
    from pykeen.models import TransE, RotatE
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    print("PyKEEN이 설치되지 않았습니다. pip install pykeen으로 설치해주세요.")

# 결과 저장 경로
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 재현성
torch.manual_seed(42)
np.random.seed(42)


def main():
    print("=" * 60)
    print("12.5 지식 그래프 임베딩: TransE vs RotatE")
    print("=" * 60)

    if not PYKEEN_AVAILABLE:
        print("\nPyKEEN이 필요합니다. 설치 후 다시 실행해주세요.")
        return None

    # 데이터셋 로드
    print("\n[1] FB15k-237 데이터셋 로드")
    dataset = FB15k237()

    print(f"엔티티 수: {dataset.num_entities:,}")
    print(f"관계 수: {dataset.num_relations:,}")
    print(f"학습 트리플: {dataset.training.num_triples:,}")
    print(f"검증 트리플: {dataset.validation.num_triples:,}")
    print(f"테스트 트리플: {dataset.testing.num_triples:,}")

    results_summary = {
        "dataset": {
            "name": "FB15k-237",
            "num_entities": dataset.num_entities,
            "num_relations": dataset.num_relations,
            "train_triples": dataset.training.num_triples,
            "valid_triples": dataset.validation.num_triples,
            "test_triples": dataset.testing.num_triples
        },
        "models": {}
    }

    # TransE 학습
    print("\n" + "=" * 60)
    print("[2] TransE 학습")
    print("=" * 60)
    print("TransE: h + r ≈ t (관계를 벡터 이동으로 모델링)")

    transe_result = pipeline(
        dataset=dataset,
        model='TransE',
        model_kwargs={'embedding_dim': 100},
        training_kwargs={
            'num_epochs': 100,
            'batch_size': 256
        },
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.001},
        negative_sampler='basic',
        negative_sampler_kwargs={'num_negs_per_pos': 1},
        evaluator_kwargs={'batch_size': 256},
        random_seed=42
    )

    transe_metrics = transe_result.metric_results.to_dict()

    print(f"\nTransE 테스트 성능:")
    print(f"  MRR: {transe_metrics['both']['realistic']['inverse_harmonic_mean_rank']:.4f}")
    print(f"  Hits@1: {transe_metrics['both']['realistic']['hits_at_1']:.4f}")
    print(f"  Hits@3: {transe_metrics['both']['realistic']['hits_at_3']:.4f}")
    print(f"  Hits@10: {transe_metrics['both']['realistic']['hits_at_10']:.4f}")

    results_summary["models"]["TransE"] = {
        "MRR": round(transe_metrics['both']['realistic']['inverse_harmonic_mean_rank'], 4),
        "Hits@1": round(transe_metrics['both']['realistic']['hits_at_1'], 4),
        "Hits@3": round(transe_metrics['both']['realistic']['hits_at_3'], 4),
        "Hits@10": round(transe_metrics['both']['realistic']['hits_at_10'], 4),
        "embedding_dim": 100,
        "epochs": 100
    }

    # RotatE 학습
    print("\n" + "=" * 60)
    print("[3] RotatE 학습")
    print("=" * 60)
    print("RotatE: t = h ∘ r (관계를 복소 공간에서 회전으로 모델링)")

    rotate_result = pipeline(
        dataset=dataset,
        model='RotatE',
        model_kwargs={'embedding_dim': 100},
        training_kwargs={
            'num_epochs': 100,
            'batch_size': 256
        },
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.001},
        negative_sampler='basic',
        negative_sampler_kwargs={'num_negs_per_pos': 1},
        evaluator_kwargs={'batch_size': 256},
        random_seed=42
    )

    rotate_metrics = rotate_result.metric_results.to_dict()

    print(f"\nRotatE 테스트 성능:")
    print(f"  MRR: {rotate_metrics['both']['realistic']['inverse_harmonic_mean_rank']:.4f}")
    print(f"  Hits@1: {rotate_metrics['both']['realistic']['hits_at_1']:.4f}")
    print(f"  Hits@3: {rotate_metrics['both']['realistic']['hits_at_3']:.4f}")
    print(f"  Hits@10: {rotate_metrics['both']['realistic']['hits_at_10']:.4f}")

    results_summary["models"]["RotatE"] = {
        "MRR": round(rotate_metrics['both']['realistic']['inverse_harmonic_mean_rank'], 4),
        "Hits@1": round(rotate_metrics['both']['realistic']['hits_at_1'], 4),
        "Hits@3": round(rotate_metrics['both']['realistic']['hits_at_3'], 4),
        "Hits@10": round(rotate_metrics['both']['realistic']['hits_at_10'], 4),
        "embedding_dim": 100,
        "epochs": 100
    }

    # 모델 비교
    print("\n" + "=" * 60)
    print("[4] TransE vs RotatE 비교")
    print("=" * 60)

    print(f"\n{'모델':<10} {'MRR':<10} {'Hits@1':<10} {'Hits@3':<10} {'Hits@10':<10}")
    print("-" * 50)
    for model_name in ['TransE', 'RotatE']:
        m = results_summary['models'][model_name]
        print(f"{model_name:<10} {m['MRR']:<10.4f} {m['Hits@1']:<10.4f} {m['Hits@3']:<10.4f} {m['Hits@10']:<10.4f}")

    # 성능 차이 분석
    mrr_diff = results_summary['models']['RotatE']['MRR'] - results_summary['models']['TransE']['MRR']
    better_model = 'RotatE' if mrr_diff > 0 else 'TransE'

    results_summary["comparison"] = {
        "mrr_diff": round(mrr_diff, 4),
        "better_model": better_model,
        "interpretation": {
            "TransE": "간단하고 효율적, 1:1 관계에 적합",
            "RotatE": "복소 공간 회전으로 대칭/반대칭/역/합성 관계 모델링 가능"
        }
    }

    print(f"\nMRR 차이: {mrr_diff:+.4f}")
    print(f"더 나은 모델: {better_model}")

    # 관계 패턴 분석
    print("\n" + "=" * 60)
    print("[5] 관계 패턴 모델링 능력")
    print("=" * 60)

    print("\n관계 패턴별 모델링 능력:")
    print(f"{'패턴':<15} {'TransE':<10} {'RotatE':<10}")
    print("-" * 35)
    print(f"{'대칭 (A↔B)':<15} {'X':<10} {'O':<10}")
    print(f"{'반대칭 (A→B)':<15} {'O':<10} {'O':<10}")
    print(f"{'역 (A→B ⟹ B→A)':<15} {'X':<10} {'O':<10}")
    print(f"{'합성':<15} {'O':<10} {'O':<10}")

    results_summary["relation_patterns"] = {
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

    # 결과 저장
    output_path = os.path.join(OUTPUT_DIR, 'ch12_kge_summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")

    # 시각화: TransE vs RotatE 성능 비교
    print("\n[6] 시각화 생성")
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

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)

    return results_summary


if __name__ == "__main__":
    main()
