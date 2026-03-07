"""
12-6-graphrag.py
GraphRAG 개념 실습

간단한 지식 그래프 + LLM 연동 데모
- 엔티티/관계 추출
- 그래프 기반 질의응답
- Vector RAG vs GraphRAG 비교

참고: 전체 GraphRAG 파이프라인은 Microsoft GraphRAG 라이브러리 참조
https://github.com/microsoft/graphrag
"""

import os
import json
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

# 결과 저장 경로
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 샘플 지식 그래프 (수동 구축)
SAMPLE_KNOWLEDGE_GRAPH = {
    "entities": [
        {"id": "e1", "name": "Albert Einstein", "type": "Person"},
        {"id": "e2", "name": "Theory of Relativity", "type": "Theory"},
        {"id": "e3", "name": "Niels Bohr", "type": "Person"},
        {"id": "e4", "name": "Quantum Mechanics", "type": "Theory"},
        {"id": "e5", "name": "Max Planck", "type": "Person"},
        {"id": "e6", "name": "Planck Constant", "type": "Concept"},
        {"id": "e7", "name": "Princeton University", "type": "Organization"},
        {"id": "e8", "name": "Copenhagen Interpretation", "type": "Theory"},
        {"id": "e9", "name": "Nobel Prize in Physics", "type": "Award"},
        {"id": "e10", "name": "Photoelectric Effect", "type": "Phenomenon"}
    ],
    "relations": [
        {"source": "e1", "target": "e2", "type": "developed"},
        {"source": "e1", "target": "e10", "type": "explained"},
        {"source": "e1", "target": "e9", "type": "received"},
        {"source": "e1", "target": "e7", "type": "worked_at"},
        {"source": "e1", "target": "e3", "type": "debated_with"},
        {"source": "e3", "target": "e4", "type": "contributed_to"},
        {"source": "e3", "target": "e8", "type": "developed"},
        {"source": "e3", "target": "e9", "type": "received"},
        {"source": "e5", "target": "e6", "type": "discovered"},
        {"source": "e5", "target": "e4", "type": "founded"},
        {"source": "e5", "target": "e9", "type": "received"},
        {"source": "e10", "target": "e4", "type": "evidence_for"},
        {"source": "e6", "target": "e4", "type": "fundamental_to"}
    ]
}


def build_knowledge_graph(kg_data):
    """NetworkX 그래프 구축"""
    G = nx.DiGraph()

    # 노드 추가
    for entity in kg_data["entities"]:
        G.add_node(entity["id"], name=entity["name"], type=entity["type"])

    # 엣지 추가
    for rel in kg_data["relations"]:
        G.add_edge(rel["source"], rel["target"], relation=rel["type"])

    return G


def get_entity_by_name(G, name):
    """이름으로 엔티티 검색"""
    for node, attrs in G.nodes(data=True):
        if attrs.get("name", "").lower() == name.lower():
            return node
    return None


def single_hop_query(G, entity_id, relation_type=None):
    """단일 홉 질의: 직접 연결된 엔티티 검색"""
    results = []
    for neighbor in G.neighbors(entity_id):
        edge_data = G.get_edge_data(entity_id, neighbor)
        if relation_type is None or edge_data.get("relation") == relation_type:
            results.append({
                "entity": G.nodes[neighbor]["name"],
                "relation": edge_data.get("relation"),
                "type": G.nodes[neighbor]["type"]
            })
    return results


def multi_hop_query(G, start_entity_id, max_hops=2):
    """다중 홉 질의: 경로 탐색"""
    paths = []

    def dfs(current, path, visited, depth):
        if depth > max_hops:
            return
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                edge_data = G.get_edge_data(current, neighbor)
                new_path = path + [(G.nodes[neighbor]["name"], edge_data.get("relation"))]
                paths.append(new_path)
                dfs(neighbor, new_path, visited | {neighbor}, depth + 1)

    dfs(start_entity_id, [], {start_entity_id}, 0)
    return paths


def community_detection(G):
    """커뮤니티 탐지 (전역 요약용)"""
    # 방향 그래프를 무방향으로 변환
    G_undirected = G.to_undirected()

    # Louvain 커뮤니티 탐지
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_undirected)
    except ImportError:
        # 대체: 연결 컴포넌트
        partition = {}
        for i, comp in enumerate(nx.connected_components(G_undirected)):
            for node in comp:
                partition[node] = i

    # 커뮤니티별 엔티티 그룹화
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(G.nodes[node]["name"])

    return dict(communities)


def vector_rag_simulation(query, documents):
    """Vector RAG 시뮬레이션 (단순 키워드 매칭)"""
    query_terms = set(query.lower().split())
    scored_docs = []
    for doc in documents:
        doc_terms = set(doc.lower().split())
        score = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
        scored_docs.append((doc, score))
    return sorted(scored_docs, key=lambda x: -x[1])[:3]


def graph_rag_query(G, query):
    """GraphRAG 질의 처리"""
    # 질의에서 엔티티 추출 (단순화: 알려진 엔티티 매칭)
    query_lower = query.lower()
    matched_entities = []

    for node, attrs in G.nodes(data=True):
        if attrs.get("name", "").lower() in query_lower:
            matched_entities.append(node)

    if not matched_entities:
        return {"type": "no_match", "results": []}

    # 다중 홉 탐색
    all_paths = []
    for entity_id in matched_entities:
        paths = multi_hop_query(G, entity_id, max_hops=2)
        entity_name = G.nodes[entity_id]["name"]
        for path in paths:
            all_paths.append({"start": entity_name, "path": path})

    return {"type": "graph_traversal", "results": all_paths}


def main():
    print("=" * 60)
    print("12.6 GraphRAG: 지식 그래프 + LLM 융합")
    print("=" * 60)

    # 지식 그래프 구축
    print("\n[1] 샘플 지식 그래프 구축")
    G = build_knowledge_graph(SAMPLE_KNOWLEDGE_GRAPH)

    print(f"엔티티 수: {G.number_of_nodes()}")
    print(f"관계 수: {G.number_of_edges()}")

    print("\n엔티티 목록:")
    for node, attrs in G.nodes(data=True):
        print(f"  {attrs['name']} ({attrs['type']})")

    results_summary = {
        "knowledge_graph": {
            "num_entities": G.number_of_nodes(),
            "num_relations": G.number_of_edges(),
            "entity_types": list(set(attrs["type"] for _, attrs in G.nodes(data=True))),
            "relation_types": list(set(data["relation"] for _, _, data in G.edges(data=True)))
        }
    }

    # 단일 홉 질의
    print("\n" + "=" * 60)
    print("[2] 단일 홉 질의 (로컬 검색)")
    print("=" * 60)

    einstein_id = get_entity_by_name(G, "Albert Einstein")
    single_hop_results = single_hop_query(G, einstein_id)

    print(f"\n질의: Albert Einstein과 직접 연결된 엔티티는?")
    print(f"결과:")
    for r in single_hop_results:
        print(f"  - {r['entity']} (관계: {r['relation']})")

    results_summary["single_hop_query"] = {
        "query": "Albert Einstein 직접 연결",
        "results": single_hop_results
    }

    # 다중 홉 질의
    print("\n" + "=" * 60)
    print("[3] 다중 홉 질의 (글로벌 검색)")
    print("=" * 60)

    multi_hop_results = multi_hop_query(G, einstein_id, max_hops=2)

    print(f"\n질의: Albert Einstein에서 2홉 이내 도달 가능한 경로는?")
    print(f"결과 (상위 5개):")
    for i, path in enumerate(multi_hop_results[:5], 1):
        path_str = " → ".join([f"{name} ({rel})" for name, rel in path])
        print(f"  {i}. Albert Einstein → {path_str}")

    results_summary["multi_hop_query"] = {
        "query": "Einstein 2홉 경로",
        "num_paths": len(multi_hop_results),
        "sample_paths": multi_hop_results[:5]
    }

    # 커뮤니티 탐지
    print("\n" + "=" * 60)
    print("[4] 커뮤니티 탐지 (전역 요약)")
    print("=" * 60)

    communities = community_detection(G)

    print(f"\n발견된 커뮤니티: {len(communities)}개")
    for comm_id, members in communities.items():
        print(f"  커뮤니티 {comm_id}: {', '.join(members)}")

    results_summary["communities"] = communities

    # Vector RAG vs GraphRAG 비교
    print("\n" + "=" * 60)
    print("[5] Vector RAG vs GraphRAG 비교")
    print("=" * 60)

    # 샘플 문서 (지식 그래프에서 생성)
    documents = [
        "Albert Einstein developed the Theory of Relativity.",
        "Niels Bohr contributed to Quantum Mechanics.",
        "Max Planck discovered the Planck Constant.",
        "Einstein and Bohr debated about quantum mechanics.",
        "The photoelectric effect provided evidence for quantum mechanics."
    ]

    # 테스트 질의
    test_queries = [
        "What is the connection between Einstein and quantum mechanics?",
        "Who received the Nobel Prize?",
        "What theories did Einstein develop?"
    ]

    comparison_results = []

    for query in test_queries:
        print(f"\n질의: {query}")

        # Vector RAG (키워드 매칭 시뮬레이션)
        vector_results = vector_rag_simulation(query, documents)
        print(f"\n  Vector RAG 결과:")
        for doc, score in vector_results[:2]:
            print(f"    - {doc[:50]}... (score: {score:.2f})")

        # GraphRAG
        graph_results = graph_rag_query(G, query)
        print(f"\n  GraphRAG 결과:")
        if graph_results["type"] == "graph_traversal":
            for r in graph_results["results"][:3]:
                path_str = " → ".join([f"{name}" for name, rel in r["path"]])
                print(f"    - {r['start']} → {path_str}")
        else:
            print("    - 매칭되는 엔티티 없음")

        comparison_results.append({
            "query": query,
            "vector_rag": [{"doc": doc[:50], "score": score} for doc, score in vector_results[:2]],
            "graph_rag": graph_results
        })

    results_summary["rag_comparison"] = comparison_results

    # 비교 분석
    print("\n" + "=" * 60)
    print("[6] GraphRAG vs Vector RAG 분석")
    print("=" * 60)

    print("\n비교표:")
    print(f"{'특성':<20} {'Vector RAG':<25} {'GraphRAG':<25}")
    print("-" * 70)
    print(f"{'인덱싱':<20} {'문서 → 벡터 임베딩':<25} {'문서 → 엔티티/관계 그래프':<25}")
    print(f"{'검색':<20} {'유사도 기반':<25} {'그래프 탐색 + 유사도':<25}")
    print(f"{'추론':<20} {'단일 홉':<25} {'다중 홉':<25}")
    print(f"{'전역 요약':<20} {'어려움':<25} {'커뮤니티 요약 활용':<25}")
    print(f"{'구축 비용':<20} {'낮음':<25} {'높음 (LLM 호출 필요)':<25}")

    results_summary["analysis"] = {
        "vector_rag_strengths": [
            "빠른 구축",
            "낮은 비용",
            "특정 문서 검색에 효과적"
        ],
        "graph_rag_strengths": [
            "다중 홉 추론",
            "관계 기반 질의",
            "전역 요약 질의",
            "Hallucination 감소"
        ],
        "recommendation": "단순 검색은 Vector RAG, 복잡한 관계 추론은 GraphRAG 권장"
    }

    # 결과 저장
    output_path = os.path.join(OUTPUT_DIR, 'ch12_graphrag_summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")

    # 시각화: 지식 그래프 네트워크 시각화
    print("\n[7] 지식 그래프 시각화 생성")
    fig, ax = plt.subplots(figsize=(14, 10))

    # 노드 타입별 색상
    color_map = {
        'Person': '#e74c3c',
        'Theory': '#3498db',
        'Concept': '#2ecc71',
        'Organization': '#9b59b6',
        'Award': '#f39c12',
        'Phenomenon': '#1abc9c'
    }

    node_colors = [color_map.get(G.nodes[node]['type'], '#95a5a6') for node in G.nodes()]
    node_labels = {node: G.nodes[node]['name'].replace(' ', '\n') for node in G.nodes()}

    # 레이아웃 설정
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold', ax=ax)

    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                           arrowsize=15, alpha=0.6, width=1.5,
                           connectionstyle='arc3,rad=0.1', ax=ax)

    # 엣지 레이블
    edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7,
                                  font_color='#2c3e50', ax=ax)

    # 범례
    legend_elements = [plt.scatter([], [], c=color, s=100, label=entity_type)
                       for entity_type, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, title='Entity Types')

    ax.set_title('GraphRAG Knowledge Graph: Physics Pioneers\n(Entity-Relation Network)', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'ch12_graphrag_network.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"그래프 저장: {fig_path}")

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)

    return results_summary


if __name__ == "__main__":
    main()
