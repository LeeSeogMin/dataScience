"""
11-3-community.py
11.3절 커뮤니티 탐지: 숨겨진 그룹 발견

Louvain 알고리즘으로 커뮤니티를 탐지하고 모듈성을 계산한다.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    import community as community_louvain
except ImportError:
    print("python-louvain 설치 필요: pip install python-louvain")
    raise

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_communities_louvain(G):
    """Louvain 알고리즘으로 커뮤니티 탐지"""
    partition = community_louvain.best_partition(G, random_state=42)
    modularity = community_louvain.modularity(partition, G)
    return partition, modularity


def analyze_communities(G, partition):
    """커뮤니티별 통계 분석"""
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    stats = []
    for comm_id, nodes in sorted(communities.items()):
        subgraph = G.subgraph(nodes)
        stats.append({
            "community_id": comm_id,
            "size": len(nodes),
            "nodes": sorted(nodes),
            "internal_edges": subgraph.number_of_edges(),
            "density": round(nx.density(subgraph), 4) if len(nodes) > 1 else 0,
            "avg_degree": round(sum(dict(subgraph.degree()).values()) / len(nodes), 2)
        })

    return stats


def visualize_communities(G, partition):
    """커뮤니티별 흑백 패턴으로 네트워크 시각화"""
    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, seed=42)

    # 커뮤니티별 그레이스케일 색상 지정
    n_communities = len(set(partition.values()))
    gray_colors = ['white', 'lightgray', 'darkgray', 'black']
    node_colors = [gray_colors[partition[node] % len(gray_colors)] for node in G.nodes()]

    # 노드 크기는 연결 중심성에 비례
    degree_cent = nx.degree_centrality(G)
    node_sizes = [degree_cent[node] * 1500 + 200 for node in G.nodes()]

    # 노드 그리기
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',
        linewidths=1.5
    )

    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.6)

    # 레이블 그리기 (검정/흰색 배경에 따라 색상 조절)
    for node in G.nodes():
        color = partition[node]
        font_color = 'white' if color == 3 else 'black'  # 검정 노드만 흰색 글씨
        nx.draw_networkx_labels(
            G, pos, labels={node: node}, ax=ax,
            font_size=9, font_color=font_color
        )

    ax.set_title("Louvain Community Detection - Karate Club", fontsize=14, fontweight="bold")
    ax.axis("off")

    # 범례 추가
    legend_elements = [
        plt.scatter([], [], c='white', edgecolors='black', s=100, label='Community 0'),
        plt.scatter([], [], c='lightgray', edgecolors='black', s=100, label='Community 1'),
        plt.scatter([], [], c='darkgray', edgecolors='black', s=100, label='Community 2'),
        plt.scatter([], [], c='black', edgecolors='black', s=100, label='Community 3'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ch11_community_detection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"시각화 저장: {OUTPUT_DIR / 'ch11_community_detection.png'}")


def compare_with_ground_truth(G, partition):
    """실제 분열 결과와 비교 (Karate Club의 실제 분열)"""
    # Karate Club의 실제 분열: club 속성
    ground_truth = nx.get_node_attributes(G, "club")

    # 커뮤니티를 두 그룹으로 매핑
    comm_0_nodes = [n for n, c in partition.items() if c == 0]
    comm_1_nodes = [n for n, c in partition.items() if c != 0]

    # Mr. Hi의 실제 그룹과 비교
    mr_hi_group = set(n for n, club in ground_truth.items() if club == "Mr. Hi")
    officer_group = set(n for n, club in ground_truth.items() if club == "Officer")

    # 정확도 계산 (두 가지 매핑 중 더 나은 것 선택)
    mapping1_correct = len(set(comm_0_nodes) & mr_hi_group) + len(set(comm_1_nodes) & officer_group)
    mapping2_correct = len(set(comm_0_nodes) & officer_group) + len(set(comm_1_nodes) & mr_hi_group)

    accuracy = max(mapping1_correct, mapping2_correct) / G.number_of_nodes()

    return {
        "detected_communities": len(set(partition.values())),
        "accuracy_vs_ground_truth": round(accuracy, 4),
        "mr_hi_group_size": len(mr_hi_group),
        "officer_group_size": len(officer_group)
    }


def resolution_analysis(G):
    """해상도 파라미터에 따른 커뮤니티 수 변화"""
    resolutions = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = []

    for res in resolutions:
        partition = community_louvain.best_partition(G, resolution=res, random_state=42)
        modularity = community_louvain.modularity(partition, G)
        n_communities = len(set(partition.values()))
        results.append({
            "resolution": res,
            "n_communities": n_communities,
            "modularity": round(modularity, 4)
        })

    return results


def main():
    print("=" * 60)
    print("11.3 커뮤니티 탐지: Louvain 알고리즘")
    print("=" * 60)

    # 1. 그래프 로드
    G = nx.karate_club_graph()
    print(f"노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")

    # 2. 커뮤니티 탐지
    print("\nLouvain 알고리즘 실행 중...")
    partition, modularity = detect_communities_louvain(G)
    n_communities = len(set(partition.values()))
    print(f"탐지된 커뮤니티 수: {n_communities}")
    print(f"모듈성 (Modularity): {modularity:.4f}")

    # 3. 커뮤니티 분석
    community_stats = analyze_communities(G, partition)
    print("\n커뮤니티별 통계:")
    for stat in community_stats:
        print(f"  커뮤니티 {stat['community_id']}: {stat['size']}개 노드, 밀도={stat['density']}")

    # 4. 시각화
    visualize_communities(G, partition)

    # 5. 실제 분열과 비교
    ground_truth_comparison = compare_with_ground_truth(G, partition)
    print(f"\n실제 분열 대비 정확도: {ground_truth_comparison['accuracy_vs_ground_truth']:.1%}")

    # 6. 해상도 분석
    resolution_results = resolution_analysis(G)
    print("\n해상도별 커뮤니티 수:")
    for r in resolution_results:
        print(f"  resolution={r['resolution']}: {r['n_communities']}개 커뮤니티, modularity={r['modularity']}")

    # 7. 결과 저장
    summary = {
        "algorithm": "Louvain",
        "graph_info": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges()
        },
        "result": {
            "n_communities": n_communities,
            "modularity": round(modularity, 4)
        },
        "community_stats": community_stats,
        "ground_truth_comparison": ground_truth_comparison,
        "resolution_analysis": resolution_results,
        "interpretation": {
            "modularity_meaning": "0.3 이상이면 유의미한 커뮤니티 구조",
            "karate_club_context": "두 리더(Mr. Hi, Officer) 간 갈등으로 실제 분열된 사례"
        }
    }

    with open(OUTPUT_DIR / "ch11_community_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {OUTPUT_DIR / 'ch11_community_summary.json'}")

    # 커뮤니티 할당 저장
    df = pd.DataFrame([
        {"node": node, "community": comm}
        for node, comm in sorted(partition.items())
    ])
    df.to_csv(OUTPUT_DIR / "ch11_community_assignments.csv", index=False)
    print(f"커뮤니티 할당 저장: {OUTPUT_DIR / 'ch11_community_assignments.csv'}")

    return summary


if __name__ == "__main__":
    summary = main()
