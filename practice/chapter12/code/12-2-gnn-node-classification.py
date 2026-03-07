"""
12-2-gnn-node-classification.py
GCN, GAT, GraphSAGE를 이용한 노드 분류 실습

Cora 데이터셋에서 논문 분류 수행
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE (Graph Sample and Aggregate)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 결과 저장 경로
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 재현성
torch.manual_seed(42)
np.random.seed(42)


class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 1층: 입력 -> 은닉층
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 2층: 은닉층 -> 출력
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), x  # 임베딩 반환 추가


class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, num_features, num_classes, hidden_dim=8, heads=8, dropout=0.6):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 1층: Multi-head attention
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # 2층: Single-head attention
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), x


class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""
    def __init__(self, num_features, num_classes, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), x


def train(model, data, optimizer):
    """한 에폭 학습"""
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    """평가"""
    model.eval()
    out, embedding = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    results = {}
    for split, mask in [('train', data.train_mask),
                        ('val', data.val_mask),
                        ('test', data.test_mask)]:
        acc = accuracy_score(data.y[mask].cpu(), pred[mask].cpu())
        results[split] = acc
    return results, pred, embedding


def visualize_embeddings(embedding, labels, title, filename):
    """t-SNE로 임베딩 시각화"""
    z = TSNE(n_components=2, random_state=42).fit_transform(embedding.cpu().numpy())
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z[:, 0], z[:, 1], s=20, c=labels.cpu().numpy(), cmap="Set2")
    plt.colorbar(scatter)
    plt.title(title)
    plt.axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"시각화 저장: {save_path}")


def main():
    print("=" * 60)
    print("12.2 GNN 노드 분류: Cora 논문 분류 (GCN, GAT, GraphSAGE)")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] Cora 데이터셋 로드")
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]

    print(f"노드 수: {data.num_nodes:,}")
    print(f"엣지 수: {data.num_edges:,}")
    print(f"특성 차원: {data.num_features}")
    print(f"클래스 수: {dataset.num_classes}")

    results_summary = {
        "dataset": "Cora",
        "models": {}
    }

    models = [
        ("GCN", GCN(data.num_features, dataset.num_classes)),
        ("GAT", GAT(data.num_features, dataset.num_classes)),
        ("GraphSAGE", GraphSAGE(data.num_features, dataset.num_classes))
    ]

    for name, model in models:
        print(f"\n[Model] {name} 학습 시작")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        best_val_acc = 0
        best_state = None
        
        for epoch in range(200):
            loss = train(model, data, optimizer)
            accs, _, _ = evaluate(model, data)
            
            if accs['val'] > best_val_acc:
                best_val_acc = accs['val']
                best_state = model.state_dict().copy()
                
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d}: Loss={loss:.4f}, Val={accs['val']:.4f}")

        # 최적 모델 로드 및 평가
        model.load_state_dict(best_state)
        accs, pred, embedding = evaluate(model, data)
        
        print(f"  {name} 최종 Test Acc: {accs['test']:.4f}")
        
        results_summary["models"][name] = {
            "train_acc": round(accs['train'], 4),
            "val_acc": round(accs['val'], 4),
            "test_acc": round(accs['test'], 4)
        }
        
        # t-SNE 시각화
        visualize_embeddings(embedding, data.y, f"{name} Embeddings (t-SNE)", f"ch12_{name.lower()}_tsne.png")

    # 결과 비교 출력
    print("\n" + "=" * 60)
    print("[결과 비교]")
    print(f"{'Model':<12} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 42)
    for name, metrics in results_summary["models"].items():
        print(f"{name:<12} {metrics['train_acc']:<10.4f} {metrics['val_acc']:<10.4f} {metrics['test_acc']:<10.4f}")

    # 결과 저장
    output_path = os.path.join(OUTPUT_DIR, 'ch12_node_classification_summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장: {output_path}")

    return results_summary


if __name__ == "__main__":
    main()
