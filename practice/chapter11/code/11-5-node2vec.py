"""
11-5-node2vec.py
11.5절 그래프 임베딩: 노드를 벡터로 표현

Node2Vec을 사용하여 노드 임베딩을 학습하고 유사 노드를 검색한다.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

try:
    from node2vec import Node2Vec
except ImportError:
    print("node2vec 설치 필요: pip install node2vec")
    raise

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def train_node2vec(G, dimensions=64, walk_length=30, num_walks=200, p=1, q=1, workers=1, seed=42):
    """Node2Vec 모델 학습"""
    print(f"Node2Vec 학습 중... (p={p}, q={q})")

    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
        quiet=True
    )

    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return model


def get_embeddings(model, nodes):
    """모든 노드의 임베딩 추출"""
    embeddings = {}
    for node in nodes:
        embeddings[node] = model.wv[str(node)]
    return embeddings


def find_similar_nodes(model, node, topn=5):
    """특정 노드와 유사한 노드 찾기"""
    similar = model.wv.most_similar(str(node), topn=topn)
    return [(int(n), round(score, 4)) for n, score in similar]


def visualize_embeddings(embeddings, partition=None):
    """t-SNE로 임베딩 시각화 (흑백)"""
    nodes = list(embeddings.keys())
    vectors = np.array([embeddings[n] for n in nodes])

    # t-SNE 차원 축소
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(nodes)-1))
    coords = tsne.fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 그레이스케일 색상 및 마커 설정
    gray_colors = ['white', 'lightgray', 'darkgray', 'black']
    markers = ['o', 's', '^', 'D']  # 원, 사각형, 삼각형, 다이아몬드

    if partition:
        # 커뮤니티별로 다른 색상과 마커 사용
        unique_communities = sorted(set(partition.values()))
        for comm_id in unique_communities:
            comm_nodes = [n for n in nodes if partition.get(n, 0) == comm_id]
            comm_indices = [nodes.index(n) for n in comm_nodes]
            color = gray_colors[comm_id % len(gray_colors)]
            marker = markers[comm_id % len(markers)]
            ax.scatter(coords[comm_indices, 0], coords[comm_indices, 1],
                      c=color, marker=marker, s=200, edgecolors='black',
                      linewidths=1.5, label=f'Community {comm_id}')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], c='white', s=200,
                  edgecolors='black', linewidths=1.5)

    # 노드 레이블 추가 (배경색에 따라 글씨 색상 조절)
    for i, node in enumerate(nodes):
        if partition:
            comm = partition.get(node, 0)
            font_color = 'white' if comm == 3 else 'black'
        else:
            font_color = 'black'
        ax.annotate(str(node), (coords[i, 0], coords[i, 1]),
                   fontsize=9, ha='center', va='center', color=font_color)

    ax.set_title("Node2Vec Embeddings (t-SNE)", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    if partition:
        ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ch11_node2vec_tsne.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE 시각화 저장: {OUTPUT_DIR / 'ch11_node2vec_tsne.png'}")

    return coords


def compare_pq_parameters(G):
    """p, q 파라미터에 따른 임베딩 비교"""
    params = [
        {"p": 1, "q": 1, "name": "DeepWalk (p=1, q=1)"},
        {"p": 0.5, "q": 2, "name": "BFS-like (p=0.5, q=2)"},
        {"p": 2, "q": 0.5, "name": "DFS-like (p=2, q=0.5)"},
    ]

    results = []
    for param in params:
        model = train_node2vec(G, dimensions=32, walk_length=20, num_walks=100,
                               p=param["p"], q=param["q"])
        embeddings = get_embeddings(model, G.nodes())

        # 클러스터링으로 구조 학습 능력 평가
        vectors = np.array([embeddings[n] for n in G.nodes()])
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vectors)

        # 실제 커뮤니티와 비교
        try:
            import community as community_louvain
            true_partition = community_louvain.best_partition(G, random_state=42)
            true_labels = [true_partition[n] for n in G.nodes()]

            # 일치율 계산 (두 매핑 중 더 나은 것)
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(true_labels, labels)
        except ImportError:
            ari = None

        results.append({
            "name": param["name"],
            "p": param["p"],
            "q": param["q"],
            "ari_score": round(ari, 4) if ari is not None else None
        })

    return results


def main():
    print("=" * 60)
    print("11.5 그래프 임베딩: Node2Vec")
    print("=" * 60)

    # 1. 그래프 로드
    G = nx.karate_club_graph()
    print(f"노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")

    # 2. Node2Vec 학습 (기본 설정)
    model = train_node2vec(G, dimensions=64, walk_length=30, num_walks=200, p=1, q=1)

    # 3. 임베딩 추출
    embeddings = get_embeddings(model, G.nodes())
    print(f"임베딩 차원: {len(list(embeddings.values())[0])}")

    # 4. 유사 노드 검색
    print("\n유사 노드 검색:")
    target_nodes = [0, 33]  # 두 리더
    similar_results = {}
    for node in target_nodes:
        similar = find_similar_nodes(model, node, topn=5)
        similar_results[node] = similar
        print(f"  노드 {node}와 유사한 노드: {[s[0] for s in similar]}")

    # 5. t-SNE 시각화
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, random_state=42)
    except ImportError:
        partition = None

    coords = visualize_embeddings(embeddings, partition)

    # 6. p, q 파라미터 비교
    print("\np, q 파라미터 비교:")
    pq_comparison = compare_pq_parameters(G)
    for result in pq_comparison:
        print(f"  {result['name']}: ARI={result['ari_score']}")

    # 7. 임베딩 저장
    embedding_df = pd.DataFrame([
        {"node": node, **{f"dim_{i}": v for i, v in enumerate(emb)}}
        for node, emb in embeddings.items()
    ])
    embedding_df.to_csv(OUTPUT_DIR / "ch11_node2vec_embeddings.csv", index=False)
    print(f"\n임베딩 저장: {OUTPUT_DIR / 'ch11_node2vec_embeddings.csv'}")

    # 8. 요약 결과 저장
    summary = {
        "model_config": {
            "dimensions": 64,
            "walk_length": 30,
            "num_walks": 200,
            "p": 1,
            "q": 1
        },
        "graph_info": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges()
        },
        "similar_nodes": {str(k): v for k, v in similar_results.items()},
        "pq_comparison": pq_comparison,
        "interpretation": {
            "p_meaning": "p가 작으면 출발 노드 근처 탐색 (BFS-like)",
            "q_meaning": "q가 작으면 멀리 탐색 (DFS-like)",
            "deepwalk": "p=q=1은 균형 잡힌 랜덤 워크 (DeepWalk)"
        }
    }

    with open(OUTPUT_DIR / "ch11_node2vec_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"요약 저장: {OUTPUT_DIR / 'ch11_node2vec_summary.json'}")

    return summary


if __name__ == "__main__":
    summary = main()
