"""
12-3-gnn-link-prediction.py
GraphSAGE를 이용한 링크 예측 실습

Cora 데이터셋에서 링크 예측 수행
- GraphSAGE 임베딩 학습
- 전통적 방법(Common Neighbors, Jaccard)과 비교
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

# 결과 저장 경로
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 재현성
torch.manual_seed(42)
np.random.seed(42)


class GraphSAGE(torch.nn.Module):
    """GraphSAGE 인코더"""
    def __init__(self, num_features, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(torch.nn.Module):
    """링크 예측 디코더"""
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return torch.sigmoid(x).squeeze()


def train_epoch(encoder, predictor, data, optimizer):
    """한 에폭 학습"""
    encoder.train()
    predictor.train()

    optimizer.zero_grad()

    # 노드 임베딩
    z = encoder(data.x, data.edge_index)

    # Positive edges
    pos_edge = data.edge_label_index[:, data.edge_label == 1]
    pos_pred = predictor(z[pos_edge[0]], z[pos_edge[1]])

    # Negative edges
    neg_edge = data.edge_label_index[:, data.edge_label == 0]
    neg_pred = predictor(z[neg_edge[0]], z[neg_edge[1]])

    # Binary cross entropy loss
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(encoder, predictor, data):
    """평가"""
    encoder.eval()
    predictor.eval()

    z = encoder(data.x, data.edge_index)

    # Positive edges
    pos_edge = data.edge_label_index[:, data.edge_label == 1]
    pos_pred = predictor(z[pos_edge[0]], z[pos_edge[1]]).cpu().numpy()

    # Negative edges
    neg_edge = data.edge_label_index[:, data.edge_label == 0]
    neg_pred = predictor(z[neg_edge[0]], z[neg_edge[1]]).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    y_score = np.concatenate([pos_pred, neg_pred])

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return auc, ap


def traditional_link_prediction(G, test_edges, test_non_edges):
    """전통적 링크 예측 방법"""
    results = {}

    # Common Neighbors
    cn_scores = []
    for u, v in test_edges:
        cn = len(list(nx.common_neighbors(G, u, v)))
        cn_scores.append(cn)
    for u, v in test_non_edges:
        cn = len(list(nx.common_neighbors(G, u, v)))
        cn_scores.append(cn)

    y_true = [1] * len(test_edges) + [0] * len(test_non_edges)
    results['Common_Neighbors'] = {
        'AUC': round(roc_auc_score(y_true, cn_scores), 4),
        'AP': round(average_precision_score(y_true, cn_scores), 4)
    }

    # Jaccard Coefficient
    jaccard_scores = []
    for u, v in test_edges:
        preds = list(nx.jaccard_coefficient(G, [(u, v)]))
        jaccard_scores.append(preds[0][2] if preds else 0)
    for u, v in test_non_edges:
        preds = list(nx.jaccard_coefficient(G, [(u, v)]))
        jaccard_scores.append(preds[0][2] if preds else 0)

    results['Jaccard'] = {
        'AUC': round(roc_auc_score(y_true, jaccard_scores), 4),
        'AP': round(average_precision_score(y_true, jaccard_scores), 4)
    }

    # Adamic-Adar
    aa_scores = []
    for u, v in test_edges:
        preds = list(nx.adamic_adar_index(G, [(u, v)]))
        aa_scores.append(preds[0][2] if preds else 0)
    for u, v in test_non_edges:
        preds = list(nx.adamic_adar_index(G, [(u, v)]))
        aa_scores.append(preds[0][2] if preds else 0)

    results['Adamic_Adar'] = {
        'AUC': round(roc_auc_score(y_true, aa_scores), 4),
        'AP': round(average_precision_score(y_true, aa_scores), 4)
    }

    return results


def main():
    print("=" * 60)
    print("12.3 GNN 링크 예측: GraphSAGE vs 전통 방법")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] Cora 데이터셋 로드 및 분할")
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]

    # 엣지 분할: 85% train, 5% val, 10% test
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0
    )
    train_data, val_data, test_data = transform(data)

    print(f"노드 수: {data.num_nodes:,}")
    print(f"원본 엣지 수: {data.num_edges:,}")
    print(f"학습 엣지: {train_data.edge_label.sum().item():.0f}")
    print(f"검증 엣지: {val_data.edge_label.sum().item():.0f}")
    print(f"테스트 엣지: {test_data.edge_label.sum().item():.0f}")

    results_summary = {
        "dataset": {
            "name": "Cora",
            "nodes": data.num_nodes,
            "edges": data.num_edges,
            "test_pos_edges": int(test_data.edge_label.sum().item()),
            "test_neg_edges": int((test_data.edge_label == 0).sum().item())
        }
    }

    # GraphSAGE 학습
    print("\n" + "=" * 60)
    print("[2] GraphSAGE 링크 예측 학습")
    print("=" * 60)

    encoder = GraphSAGE(data.num_features)
    predictor = LinkPredictor(32)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=0.01
    )

    best_val_auc = 0
    best_state = None

    for epoch in range(100):
        loss = train_epoch(encoder, predictor, train_data, optimizer)
        val_auc, val_ap = evaluate(encoder, predictor, val_data)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                'encoder': encoder.state_dict().copy(),
                'predictor': predictor.state_dict().copy()
            }

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}")

    # 최적 모델로 테스트
    encoder.load_state_dict(best_state['encoder'])
    predictor.load_state_dict(best_state['predictor'])
    test_auc, test_ap = evaluate(encoder, predictor, test_data)

    print(f"\nGraphSAGE 테스트 성능:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  AP:  {test_ap:.4f}")

    results_summary["GraphSAGE"] = {
        "AUC": round(test_auc, 4),
        "AP": round(test_ap, 4)
    }

    # 전통적 방법과 비교
    print("\n" + "=" * 60)
    print("[3] 전통적 링크 예측 방법")
    print("=" * 60)

    # NetworkX 그래프 생성 (학습 엣지만 사용)
    edge_index = train_data.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))

    # 테스트 엣지 준비
    test_edge_index = test_data.edge_label_index.numpy()
    test_labels = test_data.edge_label.numpy()
    test_edges = [(test_edge_index[0, i], test_edge_index[1, i])
                  for i in range(len(test_labels)) if test_labels[i] == 1]
    test_non_edges = [(test_edge_index[0, i], test_edge_index[1, i])
                      for i in range(len(test_labels)) if test_labels[i] == 0]

    traditional_results = traditional_link_prediction(G, test_edges, test_non_edges)

    for method, metrics in traditional_results.items():
        print(f"{method}: AUC={metrics['AUC']:.4f}, AP={metrics['AP']:.4f}")
        results_summary[method] = metrics

    # 비교 요약
    print("\n" + "=" * 60)
    print("[4] 방법별 성능 비교")
    print("=" * 60)

    print(f"\n{'방법':<18} {'AUC':<10} {'AP':<10}")
    print("-" * 38)
    print(f"{'GraphSAGE':<18} {test_auc:<10.4f} {test_ap:<10.4f}")
    for method, metrics in traditional_results.items():
        print(f"{method:<18} {metrics['AUC']:<10.4f} {metrics['AP']:<10.4f}")

    # 성능 향상 분석
    best_traditional_auc = max(m['AUC'] for m in traditional_results.values())
    improvement = ((test_auc - best_traditional_auc) / best_traditional_auc) * 100

    results_summary["comparison"] = {
        "graphsage_auc": round(test_auc, 4),
        "best_traditional_auc": round(best_traditional_auc, 4),
        "improvement_pct": round(improvement, 2),
        "interpretation": "GraphSAGE가 다중 홉 이웃 정보를 학습하여 더 정확한 링크 예측 수행" if improvement > 0 else "데이터 특성에 따라 전통적 방법도 효과적"
    }

    print(f"\nGraphSAGE vs 최고 전통 방법:")
    print(f"  AUC 향상: {improvement:+.2f}%")
    print(f"  해석: {results_summary['comparison']['interpretation']}")

    # 결과 저장
    output_path = os.path.join(OUTPUT_DIR, 'ch12_link_prediction_summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")

    # 시각화: 방법별 AUC/AP 비교 막대 그래프 (흑백)
    print("\n[5] 시각화 생성")
    methods = ['GraphSAGE', 'Common\nNeighbors', 'Jaccard', 'Adamic-Adar']
    auc_scores = [
        results_summary['GraphSAGE']['AUC'],
        results_summary['Common_Neighbors']['AUC'],
        results_summary['Jaccard']['AUC'],
        results_summary['Adamic_Adar']['AUC']
    ]
    ap_scores = [
        results_summary['GraphSAGE']['AP'],
        results_summary['Common_Neighbors']['AP'],
        results_summary['Jaccard']['AP'],
        results_summary['Adamic_Adar']['AP']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35

    # 흑백 스타일: 회색 계열 + 해치 패턴으로 구분
    bars1 = ax.bar(x - width/2, auc_scores, width, label='AUC',
                   color='#333333', edgecolor='black', hatch='')
    bars2 = ax.bar(x + width/2, ap_scores, width, label='AP',
                   color='#999999', edgecolor='black', hatch='///')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Link Prediction Performance: GraphSAGE vs Traditional Methods\n(Cora Dataset)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0.65, 0.78)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

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

    # data/output과 diagram 폴더 모두에 저장
    fig_path = os.path.join(OUTPUT_DIR, 'ch12_link_prediction_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')

    # diagram 폴더에도 저장 (문서에서 참조)
    diagram_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'diagram')
    os.makedirs(diagram_dir, exist_ok=True)
    diagram_path = os.path.join(diagram_dir, 'ch12_link_prediction_comparison.png')
    plt.savefig(diagram_path, dpi=150, bbox_inches='tight')

    plt.close()
    print(f"그래프 저장: {fig_path}")
    print(f"그래프 저장: {diagram_path}")

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)

    return results_summary


if __name__ == "__main__":
    main()
