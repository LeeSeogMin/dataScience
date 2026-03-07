"""
12-4-lightgcn.py
LightGCN을 이용한 추천 시스템 실습

MovieLens 100K 데이터셋에서 영화 추천 수행
- LightGCN: 단순화된 GCN 기반 협업 필터링
- 11장 SVD 결과와 비교
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split
from collections import defaultdict
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

# 결과 저장 경로
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'input')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

# 재현성
torch.manual_seed(42)
np.random.seed(42)


def download_movielens():
    """MovieLens 100K 데이터셋 다운로드"""
    data_file = os.path.join(INPUT_DIR, 'ml-100k', 'u.data')
    if os.path.exists(data_file):
        return data_file

    print("MovieLens 100K 다운로드 중...")
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(INPUT_DIR, 'ml-100k.zip')
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(INPUT_DIR)
    os.remove(zip_path)
    print("다운로드 완료!")

    return data_file


class LightGCN(nn.Module):
    """LightGCN 모델"""
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        # 사용자/아이템 임베딩
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # 초기화
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, edge_index):
        """LightGCN forward pass"""
        # 초기 임베딩
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # 모든 임베딩 concat (user: 0~num_users-1, item: num_users~)
        all_emb = torch.cat([user_emb, item_emb], dim=0)

        # 정규화 계수 계산
        row, col = edge_index
        deg = degree(col, all_emb.size(0), dtype=all_emb.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 레이어별 임베딩 저장 (layer combination)
        embs = [all_emb]

        for _ in range(self.num_layers):
            # 메시지 패싱: 정규화된 이웃 집계
            out = torch.zeros_like(all_emb)
            out.index_add_(0, col, all_emb[row] * norm.unsqueeze(1))
            all_emb = out
            embs.append(all_emb)

        # Layer combination: 모든 레이어의 평균
        embs = torch.stack(embs, dim=0)
        final_emb = embs.mean(dim=0)

        user_final = final_emb[:self.num_users]
        item_final = final_emb[self.num_users:]

        return user_final, item_final

    def predict(self, user_ids, item_ids, user_emb, item_emb):
        """예측 점수 계산"""
        user_e = user_emb[user_ids]
        item_e = item_emb[item_ids]
        return (user_e * item_e).sum(dim=1)


def bpr_loss(pos_scores, neg_scores):
    """BPR 손실 함수"""
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def ndcg_at_k(actual, predicted, k=10):
    """NDCG@K 계산"""
    if len(predicted) > k:
        predicted = predicted[:k]

    dcg = 0.0
    for i, item in enumerate(predicted):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(actual, predicted, k=10):
    """Recall@K 계산"""
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(actual) & set(predicted)) / len(actual) if actual else 0.0


def main():
    print("=" * 60)
    print("12.4 LightGCN 추천 시스템: MovieLens 100K")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] MovieLens 100K 데이터셋 로드")
    data_file = download_movielens()

    df = pd.read_csv(data_file, sep='\t', header=None,
                     names=['user_id', 'item_id', 'rating', 'timestamp'])

    # ID 재매핑 (0부터 시작)
    user_mapping = {u: i for i, u in enumerate(df['user_id'].unique())}
    item_mapping = {i: j for j, i in enumerate(df['item_id'].unique())}

    df['user'] = df['user_id'].map(user_mapping)
    df['item'] = df['item_id'].map(item_mapping)

    num_users = len(user_mapping)
    num_items = len(item_mapping)

    print(f"사용자 수: {num_users:,}")
    print(f"아이템 수: {num_items:,}")
    print(f"평점 수: {len(df):,}")
    print(f"희소성: {1 - len(df) / (num_users * num_items):.3f}")

    # 암묵적 피드백으로 변환 (rating >= 4를 positive로)
    df_pos = df[df['rating'] >= 4].copy()
    print(f"Positive 상호작용 수 (rating >= 4): {len(df_pos):,}")

    # Train/Test 분할 (시간순)
    df_pos = df_pos.sort_values('timestamp')
    train_df, test_df = train_test_split(df_pos, test_size=0.2, shuffle=False)

    print(f"학습 데이터: {len(train_df):,}")
    print(f"테스트 데이터: {len(test_df):,}")

    # 그래프 구성
    # 이분 그래프: user(0~num_users-1) <-> item(num_users~num_users+num_items-1)
    train_user = torch.LongTensor(train_df['user'].values)
    train_item = torch.LongTensor(train_df['item'].values) + num_users

    # 양방향 엣지
    edge_index = torch.stack([
        torch.cat([train_user, train_item]),
        torch.cat([train_item, train_user])
    ])

    # 사용자별 학습/테스트 아이템
    train_user_items = defaultdict(set)
    for _, row in train_df.iterrows():
        train_user_items[row['user']].add(row['item'])

    test_user_items = defaultdict(set)
    for _, row in test_df.iterrows():
        test_user_items[row['user']].add(row['item'])

    results_summary = {
        "dataset": {
            "name": "MovieLens 100K",
            "num_users": num_users,
            "num_items": num_items,
            "train_interactions": len(train_df),
            "test_interactions": len(test_df)
        }
    }

    # LightGCN 학습
    print("\n" + "=" * 60)
    print("[2] LightGCN 학습")
    print("=" * 60)

    model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    epochs = 100
    batch_size = 1024

    training_log = []
    all_items = set(range(num_items))

    for epoch in range(epochs):
        model.train()

        # 미니배치 샘플링
        perm = torch.randperm(len(train_df))[:batch_size]
        batch_users = torch.LongTensor(train_df.iloc[perm]['user'].values)
        batch_pos_items = torch.LongTensor(train_df.iloc[perm]['item'].values)

        # Negative 샘플링
        batch_neg_items = []
        for u in batch_users.numpy():
            neg_pool = list(all_items - train_user_items[u])
            neg_item = np.random.choice(neg_pool)
            batch_neg_items.append(neg_item)
        batch_neg_items = torch.LongTensor(batch_neg_items)

        # Forward
        user_emb, item_emb = model(edge_index)

        pos_scores = model.predict(batch_users, batch_pos_items, user_emb, item_emb)
        neg_scores = model.predict(batch_users, batch_neg_items, user_emb, item_emb)

        # BPR Loss
        loss = bpr_loss(pos_scores, neg_scores)

        # L2 정규화
        reg_loss = 0.0001 * (
            model.user_embedding.weight[batch_users].norm(2).pow(2) +
            model.item_embedding.weight[batch_pos_items].norm(2).pow(2) +
            model.item_embedding.weight[batch_neg_items].norm(2).pow(2)
        ) / batch_size
        total_loss = loss + reg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: BPR Loss={loss.item():.4f}")
            training_log.append({'epoch': epoch+1, 'loss': round(loss.item(), 4)})

    # 평가
    print("\n" + "=" * 60)
    print("[3] 평가")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)

    # 테스트 사용자에 대해 평가
    recalls = []
    ndcgs = []
    k = 10

    test_users = list(test_user_items.keys())
    sample_users = test_users[:min(500, len(test_users))]  # 샘플링

    for user in sample_users:
        if user not in train_user_items:
            continue

        # 모든 아이템에 대한 점수 계산
        scores = (user_emb[user] * item_emb).sum(dim=1).cpu().numpy()

        # 학습 데이터에서 본 아이템 제외
        for item in train_user_items[user]:
            scores[item] = -np.inf

        # Top-K 추천
        top_k_items = np.argsort(scores)[-k:][::-1].tolist()

        # 메트릭 계산
        actual = list(test_user_items[user])
        recalls.append(recall_at_k(actual, top_k_items, k))
        ndcgs.append(ndcg_at_k(actual, top_k_items, k))

    avg_recall = np.mean(recalls)
    avg_ndcg = np.mean(ndcgs)

    print(f"\nLightGCN 성능 (K={k}):")
    print(f"  Recall@{k}: {avg_recall:.4f}")
    print(f"  NDCG@{k}:   {avg_ndcg:.4f}")

    results_summary["LightGCN"] = {
        "Recall@10": round(avg_recall, 4),
        "NDCG@10": round(avg_ndcg, 4),
        "embedding_dim": 64,
        "num_layers": 3
    }

    # 11장 SVD 결과와 비교
    print("\n" + "=" * 60)
    print("[4] 11장 SVD 결과와 비교")
    print("=" * 60)

    # 11장 SVD 결과 (ch11_recommendation_summary.json에서)
    svd_results = {
        "RMSE": 0.9347,  # rating 예측 기준
        "method": "SVD (Surprise)"
    }

    print("\n비교 (서로 다른 평가 방식 주의):")
    print(f"  11장 SVD: RMSE = {svd_results['RMSE']:.4f} (평점 예측)")
    print(f"  12장 LightGCN: Recall@10 = {avg_recall:.4f}, NDCG@10 = {avg_ndcg:.4f} (Top-N 추천)")

    results_summary["comparison"] = {
        "svd_rmse": svd_results['RMSE'],
        "lightgcn_recall": round(avg_recall, 4),
        "lightgcn_ndcg": round(avg_ndcg, 4),
        "interpretation": "SVD는 평점 예측(RMSE), LightGCN은 랭킹 품질(Recall/NDCG) 최적화. 목적에 따라 선택."
    }

    # 샘플 추천
    print("\n" + "=" * 60)
    print("[5] 샘플 추천 (사용자 1)")
    print("=" * 60)

    sample_user = 0
    with torch.no_grad():
        scores = (user_emb[sample_user] * item_emb).sum(dim=1).cpu().numpy()

    for item in train_user_items[sample_user]:
        scores[item] = -np.inf

    top_10_items = np.argsort(scores)[-10:][::-1]

    print(f"\n사용자 {sample_user}의 Top-10 추천:")
    sample_recs = []
    for rank, item in enumerate(top_10_items, 1):
        original_item_id = list(item_mapping.keys())[list(item_mapping.values()).index(item)]
        print(f"  {rank}. Item {original_item_id} (Score: {scores[item]:.4f})")
        sample_recs.append({"rank": rank, "item_id": int(original_item_id), "score": round(float(scores[item]), 4)})

    results_summary["sample_recommendations"] = {
        "user_id": sample_user,
        "top_10": sample_recs
    }

    # 결과 저장
    output_path = os.path.join(OUTPUT_DIR, 'ch12_lightgcn_summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {output_path}")

    # 시각화: LightGCN vs SVD 비교
    print("\n[6] 시각화 생성")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 왼쪽: Top-N 추천 성능 (LightGCN)
    metrics = ['Recall@10', 'NDCG@10']
    values = [avg_recall, avg_ndcg]
    colors = ['#3498db', '#2ecc71']
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', width=0.5)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('LightGCN Top-N Recommendation\n(MovieLens 100K)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 0.2)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 오른쪽: MF vs LightGCN 개념 비교
    methods = ['Matrix\nFactorization', 'LightGCN']
    hop_info = [1, 3]  # MF는 1-hop, LightGCN은 multi-hop
    ax2.bar(methods, hop_info, color=['#e74c3c', '#9b59b6'], edgecolor='black', width=0.5)
    ax2.set_ylabel('Neighbor Hop Range', fontsize=12)
    ax2.set_title('Information Aggregation Range\n(MF vs LightGCN)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 4)
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0, 1.3, '1-hop\n(Direct)', ha='center', fontsize=10)
    ax2.text(1, 3.3, '3-hop\n(Multi-hop)', ha='center', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'ch12_lightgcn_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"그래프 저장: {fig_path}")

    print("\n" + "=" * 60)
    print("실습 완료!")
    print("=" * 60)

    return results_summary


if __name__ == "__main__":
    main()
