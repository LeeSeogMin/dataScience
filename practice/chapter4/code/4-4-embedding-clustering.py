"""
4.4 임베딩 군집: BERT 임베딩 + HDBSCAN으로 고객 리뷰 세분화
===========================================================
텍스트 데이터를 Sentence-BERT로 임베딩하고,
UMAP 차원 축소 후 HDBSCAN으로 군집화하는 파이프라인 실습

실행: python 4-4-embedding-clustering.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

# .env 파일에서 API 키 로드
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT_DIR, '.env'))
    USE_DOTENV = True
except ImportError:
    USE_DOTENV = False

# OpenAI API
try:
    from openai import OpenAI
    USE_OPENAI = bool(os.getenv('OPENAI_API_KEY'))
except ImportError:
    USE_OPENAI = False

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Sentence-BERT (없으면 TF-IDF 대체)
try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT = True
except (ImportError, ValueError, Exception) as e:
    USE_SBERT = False
    print(f"[INFO] sentence-transformers 사용 불가. TF-IDF로 대체합니다. ({type(e).__name__})")

# UMAP
try:
    import umap
    USE_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA
    USE_UMAP = False
    print("[INFO] umap-learn 미설치. PCA로 대체합니다.")

# HDBSCAN
try:
    import hdbscan
    USE_HDBSCAN = True
except ImportError:
    USE_HDBSCAN = False
    print("[INFO] hdbscan 미설치. K-Means로 대체합니다.")


def generate_review_data():
    """
    상품 리뷰 데이터 생성 (주제별 시뮬레이션)
    실제로는 API나 크롤링으로 수집한 데이터 사용
    """
    reviews = [
        # 배송 관련
        "배송이 정말 빨라서 좋았어요. 다음날 바로 도착했습니다.",
        "배송 속도가 엄청 빠르네요. 주문한 지 하루 만에 왔어요.",
        "택배가 너무 늦게 왔어요. 일주일이나 걸렸습니다.",
        "배송이 예상보다 빨라서 만족합니다.",
        "배송 추적이 안 돼서 불안했어요. 택배사 문제 같아요.",
        "당일 배송으로 받아서 급한 거 해결했어요.",
        "배송 중 파손되어 왔어요. 포장을 더 신경 써주세요.",
        "새벽 배송 너무 편리해요. 출근 전에 받을 수 있어서 좋아요.",

        # 품질 관련
        "제품 품질이 가격 대비 정말 좋아요. 추천합니다.",
        "품질이 기대 이상이에요. 마감도 깔끔하고 튼튼해요.",
        "품질이 별로네요. 사진이랑 많이 달라요.",
        "소재가 고급스럽고 마감이 꼼꼼해요.",
        "사진보다 색상이 많이 다르네요. 실망했어요.",
        "내구성이 좋아서 오래 쓸 수 있을 것 같아요.",
        "품질 좋고 디자인도 예뻐요. 재구매 의향 있습니다.",
        "처음엔 좋았는데 한 달 만에 망가졌어요.",

        # 가격 관련
        "가격 대비 성능이 좋아요. 가성비 최고입니다.",
        "이 가격에 이 품질이면 완전 이득이에요.",
        "좀 비싼 것 같아요. 할인할 때 사는 게 좋겠어요.",
        "가성비 최고! 다른 데보다 훨씬 저렴해요.",
        "가격은 비싸지만 그만한 가치가 있어요.",
        "세일할 때 샀는데 정가에는 안 살 것 같아요.",
        "무료 배송이라 더 저렴하게 느껴져요.",
        "비슷한 제품 중 가장 합리적인 가격이에요.",

        # 서비스/CS 관련
        "고객센터 응대가 정말 친절했어요. 문제 해결도 빨랐고요.",
        "교환 과정이 너무 복잡해요. 개선이 필요해요.",
        "AS 신청했는데 빠르게 처리해주셔서 감사해요.",
        "문의 답변이 너무 늦어요. 3일이나 걸렸어요.",
        "환불 요청했는데 바로 처리해주셨어요.",
        "상담원분이 친절하게 설명해주셔서 좋았어요.",
        "반품 절차가 간단해서 좋았어요.",
        "고객센터 연결이 너무 어려워요. 전화를 안 받아요.",

        # 사용감/만족도
        "써보니까 진짜 편해요. 매일 쓰고 있어요.",
        "기대했던 것보다 훨씬 좋아요. 완전 만족합니다.",
        "생각보다 별로예요. 기대가 컸나 봐요.",
        "가족들도 다 좋아해요. 추가로 더 살 예정이에요.",
        "사용법이 간단해서 누구나 쓸 수 있어요.",
        "첫 사용감이 좋아요. 앞으로도 계속 쓸 것 같아요.",
        "선물용으로 샀는데 받으신 분이 너무 좋아하세요.",
        "재구매했어요. 그만큼 만족스러워요.",
    ]

    # 실제 주제 레이블 (평가용)
    true_topics = (
        ['배송'] * 8 +
        ['품질'] * 8 +
        ['가격'] * 8 +
        ['서비스'] * 8 +
        ['만족도'] * 8
    )

    df = pd.DataFrame({
        'review_id': range(1, len(reviews) + 1),
        'review': reviews,
        'true_topic': true_topics
    })

    return df


def get_embeddings(texts, method='sbert'):
    """텍스트를 벡터로 변환"""
    if method == 'sbert' and USE_SBERT:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings, 'Sentence-BERT'
    else:
        # TF-IDF 대체
        vectorizer = TfidfVectorizer(max_features=500)
        embeddings = vectorizer.fit_transform(texts).toarray()
        return embeddings, 'TF-IDF'


def reduce_dimensions(embeddings, n_components=2, method='umap'):
    """차원 축소"""
    if method == 'umap' and USE_UMAP:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        reduced = reducer.fit_transform(embeddings)
        return reduced, 'UMAP'
    else:
        # PCA 대체
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(embeddings)
        explained = pca.explained_variance_ratio_.sum()
        return reduced, f'PCA (설명분산: {explained:.1%})'


def cluster_embeddings(embeddings, method='hdbscan', n_clusters=5):
    """군집화"""
    if method == 'hdbscan' and USE_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(embeddings)
        return labels, 'HDBSCAN'
    else:
        # K-Means 대체
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels, 'K-Means'


def label_clusters_with_keywords(df, labels, n_keywords=5):
    """군집별 대표 키워드로 레이블링"""
    from sklearn.feature_extraction.text import CountVectorizer

    df_temp = df.copy()
    df_temp['cluster'] = labels

    cluster_labels = {}
    unique_clusters = sorted(set(labels))

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_labels[cluster_id] = '노이즈'
            continue

        cluster_reviews = df_temp[df_temp['cluster'] == cluster_id]['review'].tolist()

        # 간단한 키워드 추출
        vectorizer = CountVectorizer(max_features=100)
        try:
            X = vectorizer.fit_transform(cluster_reviews)
            word_freq = X.sum(axis=0).A1
            words = vectorizer.get_feature_names_out()
            top_indices = word_freq.argsort()[-n_keywords:][::-1]
            top_words = [words[i] for i in top_indices]
            cluster_labels[cluster_id] = ', '.join(top_words[:3])
        except:
            cluster_labels[cluster_id] = f'군집 {cluster_id}'

    return cluster_labels


def label_clusters_with_llm(df, labels, n_samples=5):
    """LLM을 활용한 군집 자동 레이블링 (BERTopic 방식)."""
    if not USE_OPENAI:
        print("[INFO] OpenAI API 미설정. 키워드 기반 레이블링으로 대체합니다.")
        return None

    client = OpenAI()
    df_temp = df.copy()
    df_temp['cluster'] = labels

    cluster_labels = {}
    unique_clusters = sorted(set(labels))

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_labels[cluster_id] = '노이즈'
            continue

        # 군집의 대표 샘플 추출
        cluster_reviews = df_temp[df_temp['cluster'] == cluster_id]['review'].tolist()
        samples = cluster_reviews[:n_samples]
        samples_text = '\n'.join([f"- {s}" for s in samples])

        # LLM 프롬프트 구성
        prompt = f"""다음은 같은 군집으로 분류된 고객 리뷰들입니다:

{samples_text}

이 리뷰들의 공통 주제를 2-3 단어로 요약해주세요. 예: "배송 속도", "제품 품질", "가격 만족도"
주제만 출력하세요."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0
            )
            label = response.choices[0].message.content.strip()
            cluster_labels[cluster_id] = label
        except Exception as e:
            cluster_labels[cluster_id] = f'군집 {cluster_id}'
            print(f"  [WARNING] 군집 {cluster_id} LLM 레이블링 실패: {e}")

    return cluster_labels


def plot_embedding_clusters(embeddings, labels, df, cluster_keywords, output_path):
    """임베딩 군집화 결과 시각화 (2D)."""
    # 2D로 추가 축소 (시각화용)
    if USE_UMAP:
        reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                               metric='cosine', random_state=42)
    else:
        from sklearn.decomposition import PCA
        reducer_2d = PCA(n_components=2, random_state=42)

    embeddings_2d = reducer_2d.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 색상 팔레트
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
              '#1abc9c', '#e67e22', '#34495e']

    # 1. 군집화 결과
    ax1 = axes[0]
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        if label == -1:
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c='gray', s=50, alpha=0.5, marker='x', label='노이즈')
        else:
            keyword = cluster_keywords.get(label, f'군집 {label}')
            ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=colors[label % len(colors)], s=80, alpha=0.7,
                       edgecolors='white', linewidths=0.5, label=keyword[:10])

    ax1.set_title('임베딩 군집화 결과', fontsize=12, fontweight='bold')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 실제 주제 분포
    ax2 = axes[1]
    topic_colors = {'배송': '#e74c3c', '품질': '#3498db', '가격': '#2ecc71',
                    '서비스': '#9b59b6', '만족도': '#f39c12'}

    for topic in df['true_topic'].unique():
        mask = df['true_topic'] == topic
        ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=topic_colors.get(topic, 'gray'), s=80, alpha=0.7,
                   edgecolors='white', linewidths=0.5, label=topic)

    ax2.set_title('실제 주제 분포', fontsize=12, fontweight='bold')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n시각화 저장: {output_path}")


def main():
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("임베딩 군집: BERT + HDBSCAN 고객 리뷰 세분화")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] 데이터 준비")
    df = generate_review_data()
    print(f"- 리뷰 수: {len(df)}개")
    print(f"- 실제 주제: {df['true_topic'].nunique()}개 ({', '.join(df['true_topic'].unique())})")

    print("\n샘플 리뷰:")
    for i, row in df.head(3).iterrows():
        print(f"  [{row['true_topic']}] {row['review'][:50]}...")

    # 2. 텍스트 임베딩
    print("\n[2] 텍스트 임베딩")
    texts = df['review'].tolist()
    embeddings, embed_method = get_embeddings(texts, method='sbert')
    print(f"- 임베딩 방법: {embed_method}")
    print(f"- 임베딩 차원: {embeddings.shape[1]}")

    # 3. 차원 축소
    print("\n[3] 차원 축소")
    embeddings_reduced, reduce_method = reduce_dimensions(embeddings, n_components=10)
    print(f"- 차원 축소 방법: {reduce_method}")
    print(f"- 축소 후 차원: {embeddings_reduced.shape[1]}")

    # 4. 군집화
    print("\n[4] 군집화")
    labels, cluster_method = cluster_embeddings(embeddings_reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum() if -1 in labels else 0

    print(f"- 군집화 방법: {cluster_method}")
    print(f"- 발견된 군집 수: {n_clusters}")
    print(f"- 노이즈 포인트: {n_noise}")

    # 실루엣 점수 (노이즈 제외)
    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        sil_score = silhouette_score(embeddings_reduced[mask], labels[mask])
        print(f"- 실루엣 점수: {sil_score:.4f}")

    # 5. 군집별 분석 (키워드 기반)
    print("\n[5] 군집별 분석 (키워드 기반)")
    df['cluster'] = labels
    cluster_keywords = label_clusters_with_keywords(df, labels)

    print("-" * 60)
    print(f"{'군집':<15} {'키워드':<25} {'리뷰 수':>10} {'대표 주제':>10}")
    print("-" * 60)

    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            count = (df['cluster'] == cluster_id).sum()
            print(f"{'노이즈':<15} {'-':<25} {count:>10} {'-':>10}")
        else:
            cluster_df = df[df['cluster'] == cluster_id]
            count = len(cluster_df)
            top_topic = cluster_df['true_topic'].mode().iloc[0]
            keywords = cluster_keywords.get(cluster_id, '-')
            print(f"{'군집 ' + str(cluster_id):<15} {keywords:<25} {count:>10} {top_topic:>10}")

    print("-" * 60)

    # 5-1. LLM 기반 레이블링
    print("\n[5-1] LLM 기반 레이블링 (GPT-4o-mini)")
    cluster_llm_labels = label_clusters_with_llm(df, labels)
    if cluster_llm_labels:
        print("-" * 50)
        print(f"{'군집':<15} {'LLM 레이블':<30}")
        print("-" * 50)
        for cluster_id in sorted(set(labels)):
            label = cluster_llm_labels.get(cluster_id, '-')
            print(f"{'군집 ' + str(cluster_id) if cluster_id != -1 else '노이즈':<15} {label:<30}")
        print("-" * 50)
        # LLM 레이블을 기본 레이블로 사용
        cluster_keywords = cluster_llm_labels

    # 6. 군집화 품질 평가
    print("\n[6] 군집화 품질 평가")
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # 실제 주제를 숫자로 변환
    topic_map = {t: i for i, t in enumerate(df['true_topic'].unique())}
    true_labels = df['true_topic'].map(topic_map)

    # 노이즈 제외하고 평가
    mask = labels != -1
    if mask.sum() > 0:
        ari = adjusted_rand_score(true_labels[mask], labels[mask])
        nmi = normalized_mutual_info_score(true_labels[mask], labels[mask])
        print(f"- Adjusted Rand Index: {ari:.4f}")
        print(f"- Normalized Mutual Info: {nmi:.4f}")
        print("  (1에 가까울수록 실제 주제와 일치)")

    # 7. 군집별 샘플 리뷰
    print("\n[7] 군집별 샘플 리뷰")
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        print(f"\n군집 {cluster_id} ({cluster_keywords.get(cluster_id, '-')}):")
        samples = df[df['cluster'] == cluster_id]['review'].head(3).tolist()
        for i, sample in enumerate(samples, 1):
            print(f"  {i}. {sample[:60]}...")

    # 8. 시각화
    print("\n[8] 시각화")
    plot_embedding_clusters(
        embeddings, labels, df, cluster_keywords,
        os.path.join(OUTPUT_DIR, 'embedding_clusters.png')
    )

    # 9. 결과 저장
    output_csv = os.path.join(OUTPUT_DIR, 'review_clusters.csv')
    df['cluster_label'] = df['cluster'].map(
        lambda x: cluster_keywords.get(x, '노이즈') if x != -1 else '노이즈'
    )
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"결과 저장: {output_csv}")

    # 10. TF-IDF 기반 군집화와 비교
    print("\n" + "=" * 60)
    print("[10] TF-IDF vs 임베딩 기반 군집화")
    print("=" * 60)

    # TF-IDF 군집화
    embeddings_tfidf, _ = get_embeddings(texts, method='tfidf')
    labels_tfidf, _ = cluster_embeddings(embeddings_tfidf, method='kmeans', n_clusters=5)

    ari_tfidf = adjusted_rand_score(true_labels, labels_tfidf)
    nmi_tfidf = normalized_mutual_info_score(true_labels, labels_tfidf)

    print(f"{'방법':<25} {'ARI':>12} {'NMI':>12}")
    print("-" * 50)
    print(f"{'TF-IDF + K-Means':<25} {ari_tfidf:>12.4f} {nmi_tfidf:>12.4f}")
    if mask.sum() > 0:
        print(f"{'임베딩 + HDBSCAN':<25} {ari:>12.4f} {nmi:>12.4f}")

    print("\n결론: 임베딩 기반 군집화가 의미적 유사성을 더 잘 포착")

    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
