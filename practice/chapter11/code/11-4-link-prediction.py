"""
11-4-link-prediction.py
11.4절 링크 예측: 미래 관계 예측

공통 이웃, Jaccard, Adamic-Adar 지표로 링크 예측 성능을 평가한다.
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_train_test_edges(G, test_ratio=0.2, seed=42):
    """엣지를 학습/테스트로 분할"""
    random.seed(seed)
    np.random.seed(seed)

    edges = list(G.edges())
    n_test = int(len(edges) * test_ratio)

    # 테스트 엣지 선택 (그래프 연결성 유지를 위해 bridge가 아닌 엣지만)
    test_edges = []
    G_temp = G.copy()

    random.shuffle(edges)
    for edge in edges:
        if len(test_edges) >= n_test:
            break
        G_temp.remove_edge(*edge)
        if nx.is_connected(G_temp):
            test_edges.append(edge)
        else:
            G_temp.add_edge(*edge)

    train_edges = [e for e in edges if e not in test_edges]

    # 학습 그래프 생성
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)

    return G_train, test_edges, train_edges


def generate_negative_samples(G, n_samples, seed=42):
    """존재하지 않는 엣지 (네거티브 샘플) 생성"""
    random.seed(seed)
    nodes = list(G.nodes())
    non_edges = []

    existing_edges = set(G.edges())
    existing_edges.update([(v, u) for u, v in existing_edges])

    while len(non_edges) < n_samples:
        u, v = random.sample(nodes, 2)
        if (u, v) not in existing_edges and (v, u) not in existing_edges:
            non_edges.append((u, v))
            existing_edges.add((u, v))
            existing_edges.add((v, u))

    return non_edges


def compute_link_scores(G, edges, method="common_neighbors"):
    """링크 예측 점수 계산"""
    scores = []

    if method == "common_neighbors":
        preds = nx.common_neighbor_centrality(G, edges)
    elif method == "jaccard":
        preds = nx.jaccard_coefficient(G, edges)
    elif method == "adamic_adar":
        preds = nx.adamic_adar_index(G, edges)
    elif method == "preferential_attachment":
        preds = nx.preferential_attachment(G, edges)
    else:
        raise ValueError(f"Unknown method: {method}")

    for u, v, score in preds:
        scores.append({"u": u, "v": v, "score": score})

    return scores


def evaluate_link_prediction(G_train, test_edges, neg_edges):
    """여러 방법의 링크 예측 성능 평가"""
    methods = ["common_neighbors", "jaccard", "adamic_adar", "preferential_attachment"]

    # 포지티브/네거티브 레이블
    all_edges = test_edges + neg_edges
    labels = [1] * len(test_edges) + [0] * len(neg_edges)

    results = {}

    for method in methods:
        try:
            scores = compute_link_scores(G_train, all_edges, method)
            score_values = [s["score"] for s in scores]

            # AUC-ROC 계산
            auc = roc_auc_score(labels, score_values)

            # Average Precision 계산
            ap = average_precision_score(labels, score_values)

            results[method] = {
                "auc_roc": round(auc, 4),
                "avg_precision": round(ap, 4)
            }
            print(f"{method}: AUC={auc:.4f}, AP={ap:.4f}")

        except Exception as e:
            print(f"{method}: 오류 발생 - {e}")
            results[method] = {"auc_roc": None, "avg_precision": None, "error": str(e)}

    return results


def visualize_link_prediction_example(G, node_pair):
    """링크 예측 예시 시각화 (흑백)"""
    u, v = node_pair

    # 공통 이웃 찾기
    common = list(nx.common_neighbors(G, u, v))

    fig, ax = plt.subplots(figsize=(10, 8))

    # 관련 노드만 추출
    relevant_nodes = set([u, v] + common + list(G.neighbors(u)) + list(G.neighbors(v)))
    subgraph = G.subgraph(relevant_nodes)

    pos = nx.spring_layout(subgraph, seed=42)

    # 노드 색상 및 레이블 색상 설정 (흑백)
    node_colors = []
    font_colors = {}
    for node in subgraph.nodes():
        if node in [u, v]:
            node_colors.append("black")  # 대상 노드: 검정
            font_colors[node] = "white"
        elif node in common:
            node_colors.append("darkgray")  # 공통 이웃: 진한 회색
            font_colors[node] = "white"
        else:
            node_colors.append("white")  # 일반 이웃: 흰색
            font_colors[node] = "black"

    # 노드 그리기
    nx.draw_networkx_nodes(
        subgraph, pos, ax=ax,
        node_size=500,
        node_color=node_colors,
        edgecolors='black',
        linewidths=1.5
    )

    # 엣지 그리기
    nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color='gray', alpha=0.6)

    # 레이블 그리기 (배경색에 따라 글씨 색상 조절)
    for node in subgraph.nodes():
        nx.draw_networkx_labels(
            subgraph, pos, labels={node: node}, ax=ax,
            font_size=10, font_color=font_colors[node]
        )

    ax.set_title(f"Link Prediction: Node {u} - Node {v}\nCommon Neighbors: {common if common else 'None'}",
                 fontsize=12, fontweight="bold")
    ax.axis("off")

    # 범례 추가
    legend_elements = [
        plt.scatter([], [], c='black', edgecolors='black', s=100, label='Target nodes'),
        plt.scatter([], [], c='darkgray', edgecolors='black', s=100, label='Common neighbors'),
        plt.scatter([], [], c='white', edgecolors='black', s=100, label='Other neighbors'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ch11_link_prediction_example.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"예시 시각화 저장: {OUTPUT_DIR / 'ch11_link_prediction_example.png'}")

    return common


def main():
    print("=" * 60)
    print("11.4 링크 예측: 미래 관계 예측")
    print("=" * 60)

    # 1. 그래프 로드
    G = nx.karate_club_graph()
    print(f"원본 그래프 - 노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()}")

    # 2. 학습/테스트 분할
    print("\n엣지 분할 중...")
    G_train, test_edges, train_edges = prepare_train_test_edges(G, test_ratio=0.2)
    print(f"학습 엣지: {len(train_edges)}, 테스트 엣지: {len(test_edges)}")

    # 3. 네거티브 샘플 생성
    neg_edges = generate_negative_samples(G_train, n_samples=len(test_edges))
    print(f"네거티브 샘플: {len(neg_edges)}")

    # 4. 링크 예측 평가
    print("\n링크 예측 성능 평가:")
    results = evaluate_link_prediction(G_train, test_edges, neg_edges)

    # 5. 예시 시각화 (연결되지 않은 노드 쌍)
    example_pair = neg_edges[0]
    common_neighbors = visualize_link_prediction_example(G_train, example_pair)

    # 6. 상위 예측 결과 저장
    all_non_edges = list(nx.non_edges(G_train))[:100]  # 상위 100개만
    top_predictions = []

    for method in ["common_neighbors", "jaccard", "adamic_adar"]:
        scores = compute_link_scores(G_train, all_non_edges, method)
        sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)[:10]
        top_predictions.append({
            "method": method,
            "top_10": sorted_scores
        })

    # 7. 결과 저장
    summary = {
        "graph_info": {
            "original_edges": G.number_of_edges(),
            "train_edges": len(train_edges),
            "test_edges": len(test_edges),
            "negative_samples": len(neg_edges)
        },
        "evaluation_results": results,
        "best_method": max(results.items(), key=lambda x: x[1].get("auc_roc", 0) or 0)[0],
        "example": {
            "node_pair": list(example_pair),
            "common_neighbors": common_neighbors
        },
        "interpretation": {
            "auc_meaning": "0.5=랜덤, 1.0=완벽한 예측",
            "common_neighbors": "공통 이웃 수가 많을수록 연결 가능성 높음",
            "jaccard": "공통 이웃 비율로 정규화",
            "adamic_adar": "희귀한 공통 이웃에 더 높은 가중치"
        }
    }

    with open(OUTPUT_DIR / "ch11_link_prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {OUTPUT_DIR / 'ch11_link_prediction_summary.json'}")

    # 성능 비교 테이블 저장
    df = pd.DataFrame([
        {"method": method, **metrics}
        for method, metrics in results.items()
    ])
    df.to_csv(OUTPUT_DIR / "ch11_link_prediction_metrics.csv", index=False)
    print(f"메트릭 저장: {OUTPUT_DIR / 'ch11_link_prediction_metrics.csv'}")

    print("\n" + "=" * 60)
    print("최적 방법:", summary["best_method"])
    print("=" * 60)

    return summary


if __name__ == "__main__":
    summary = main()
