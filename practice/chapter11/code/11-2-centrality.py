"""
11-2-centrality.py
11.2절 중심성 분석: 영향력 있는 노드 찾기

Zachary's Karate Club 데이터로 4가지 중심성 지표를 계산하고 비교한다.
- 연결 중심성 (Degree Centrality)
- 매개 중심성 (Betweenness Centrality)
- 근접 중심성 (Closeness Centrality)
- PageRank
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_karate_club():
    """Zachary's Karate Club 그래프 로드"""
    G = nx.karate_club_graph()
    print(f"노드 수: {G.number_of_nodes()}")
    print(f"엣지 수: {G.number_of_edges()}")
    return G


def compute_centralities(G):
    """4가지 중심성 지표 계산"""
    centralities = {
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G),
        "closeness": nx.closeness_centrality(G),
        "pagerank": nx.pagerank(G, alpha=0.85),
    }
    return centralities


def get_top_nodes(centralities, top_k=5):
    """각 중심성 지표별 상위 노드 추출"""
    results = {}
    for name, cent_dict in centralities.items():
        sorted_nodes = sorted(cent_dict.items(), key=lambda x: x[1], reverse=True)
        results[name] = [
            {"node": node, "score": round(score, 4)}
            for node, score in sorted_nodes[:top_k]
        ]
    return results


def create_comparison_table(centralities):
    """중심성 지표 비교 테이블 생성"""
    df = pd.DataFrame(centralities)
    df.index.name = "node"
    df = df.round(4)
    return df


def visualize_centralities(G, centralities):
    """중심성별 네트워크 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    pos = nx.spring_layout(G, seed=42)
    titles = ["Degree Centrality", "Betweenness Centrality",
              "Closeness Centrality", "PageRank"]
    cent_names = ["degree", "betweenness", "closeness", "pagerank"]

    for ax, title, cent_name in zip(axes.flatten(), titles, cent_names):
        cent_values = centralities[cent_name]
        node_sizes = [cent_values[node] * 2000 + 100 for node in G.nodes()]
        node_colors = [cent_values[node] for node in G.nodes()]

        nx.draw_networkx(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.Reds,
            with_labels=True,
            font_size=8,
            font_color="black",
            edge_color="gray",
            alpha=0.9
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ch11_centrality_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"시각화 저장: {OUTPUT_DIR / 'ch11_centrality_comparison.png'}")


def analyze_leader_nodes(G, centralities):
    """핵심 리더 노드 분석 (Karate Club의 두 리더: 0, 33)"""
    leaders = [0, 33]
    analysis = {}

    for leader in leaders:
        analysis[f"node_{leader}"] = {
            "degree": round(centralities["degree"][leader], 4),
            "betweenness": round(centralities["betweenness"][leader], 4),
            "closeness": round(centralities["closeness"][leader], 4),
            "pagerank": round(centralities["pagerank"][leader], 4),
            "neighbors": list(G.neighbors(leader)),
            "neighbor_count": G.degree(leader)
        }

    return analysis


def main():
    print("=" * 60)
    print("11.2 중심성 분석: Zachary's Karate Club")
    print("=" * 60)

    # 1. 그래프 로드
    G = load_karate_club()

    # 2. 중심성 계산
    print("\n중심성 지표 계산 중...")
    centralities = compute_centralities(G)

    # 3. 상위 노드 추출
    top_nodes = get_top_nodes(centralities, top_k=5)
    print("\n각 중심성별 상위 5개 노드:")
    for name, nodes in top_nodes.items():
        print(f"  {name}: {[n['node'] for n in nodes]}")

    # 4. 비교 테이블 생성
    df = create_comparison_table(centralities)
    df.to_csv(OUTPUT_DIR / "ch11_centrality_all_nodes.csv")
    print(f"\n전체 노드 중심성 저장: {OUTPUT_DIR / 'ch11_centrality_all_nodes.csv'}")

    # 5. 시각화
    visualize_centralities(G, centralities)

    # 6. 리더 노드 분석
    leader_analysis = analyze_leader_nodes(G, centralities)

    # 7. 요약 결과 저장
    summary = {
        "graph_info": {
            "name": "Zachary's Karate Club",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": round(nx.density(G), 4)
        },
        "top_nodes_by_centrality": top_nodes,
        "leader_analysis": leader_analysis,
        "interpretation": {
            "node_0": "Mr. Hi (강사) - 연결 중심성과 매개 중심성 모두 높음",
            "node_33": "John A (회장) - 연결 중심성이 높고 별도 커뮤니티의 허브"
        }
    }

    with open(OUTPUT_DIR / "ch11_centrality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"요약 저장: {OUTPUT_DIR / 'ch11_centrality_summary.json'}")

    # 8. 핵심 결과 출력
    print("\n" + "=" * 60)
    print("분석 결과 요약")
    print("=" * 60)
    print(f"그래프 밀도: {summary['graph_info']['density']}")
    print(f"\n연결 중심성 1위: 노드 {top_nodes['degree'][0]['node']} (점수: {top_nodes['degree'][0]['score']})")
    print(f"매개 중심성 1위: 노드 {top_nodes['betweenness'][0]['node']} (점수: {top_nodes['betweenness'][0]['score']})")
    print(f"PageRank 1위: 노드 {top_nodes['pagerank'][0]['node']} (점수: {top_nodes['pagerank'][0]['score']})")

    return summary


if __name__ == "__main__":
    summary = main()
