"""
6장 6.3절 실습: BERTopic 기본 사용법
- BERTopic 5단계 파이프라인 실습
- LDA vs BERTopic 비교
- 다양한 토픽 시각화 (Plotly 인터랙티브)
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# 출력 경로 설정
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_newsgroups_data(n_samples=2000):
    """20 Newsgroups 데이터 로드 (4개 카테고리)"""
    categories = ['rec.sport.baseball', 'sci.med', 'comp.graphics', 'talk.politics.misc']
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )

    # 샘플링
    indices = np.random.RandomState(42).choice(len(newsgroups.data), min(n_samples, len(newsgroups.data)), replace=False)
    texts = [newsgroups.data[i] for i in indices]
    labels = [newsgroups.target_names[newsgroups.target[i]] for i in indices]

    # 빈 문서 제거
    valid_idx = [i for i, t in enumerate(texts) if len(t.strip()) > 50]
    texts = [texts[i] for i in valid_idx]
    labels = [labels[i] for i in valid_idx]

    print(f"로드된 문서 수: {len(texts)}")
    print(f"카테고리: {categories}")
    return texts, labels

def create_bertopic_model():
    """BERTopic 모델 생성 (5단계 파이프라인 명시)"""

    # 1단계: 임베딩 모델
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2단계: UMAP 차원 축소
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # 3단계: HDBSCAN 클러스터링
    hdbscan_model = HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # 4단계: 토큰화 (불용어 제거)
    vectorizer_model = CountVectorizer(
        stop_words='english',
        min_df=2,
        ngram_range=(1, 1)
    )

    # BERTopic 모델 생성
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
        calculate_probabilities=True
    )

    return topic_model

def train_bertopic(texts, topic_model):
    """BERTopic 학습"""
    print("\nBERTopic 학습 중...")
    topics, probs = topic_model.fit_transform(texts)

    # 토픽 정보 출력
    topic_info = topic_model.get_topic_info()
    print(f"\n발견된 토픽 수 (축소 전): {len(topic_info) - 1}")  # -1 제외
    print(f"노이즈(-1)로 분류된 문서: {sum(1 for t in topics if t == -1)}")

    # 5개 토픽으로 축소 (카테고리 4개 + 노이즈 1개)
    topic_model.reduce_topics(texts, nr_topics=5)
    topics = topic_model.topics_
    topic_info = topic_model.get_topic_info()
    print(f"발견된 토픽 수 (축소 후): {len(topic_info) - 1}")
    print(f"노이즈(-1)로 분류된 문서: {sum(1 for t in topics if t == -1)}")

    return topics, probs, topic_info

def get_topic_details(topic_model, num_words=10):
    """토픽별 상세 정보 추출"""
    topics = []
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            continue

        topic_words = topic_model.get_topic(topic_id)
        keywords = [word for word, _ in topic_words[:num_words]]
        keyword_probs = [(word, float(prob)) for word, prob in topic_words[:num_words]]

        topics.append({
            'topic_id': topic_id,
            'count': int(row['Count']),
            'name': row.get('Name', f'Topic_{topic_id}'),
            'keywords': keywords,
            'keyword_probs': keyword_probs
        })

    return topics

def compare_with_ground_truth(topics, labels, topic_model):
    """실제 레이블과 토픽 매핑 비교"""
    df = pd.DataFrame({'topic': topics, 'label': labels})

    # 토픽별 레이블 분포
    topic_label_dist = {}
    for topic_id in df['topic'].unique():
        if topic_id == -1:
            continue
        topic_docs = df[df['topic'] == topic_id]
        label_counts = topic_docs['label'].value_counts().to_dict()
        topic_label_dist[int(topic_id)] = label_counts

    return topic_label_dist

def visualize_topics_2d(topic_model, texts, topics, output_path):
    """토픽 2D 시각화 (matplotlib 기반)"""
    try:
        # 임베딩 생성
        embeddings = topic_model._extract_embeddings(texts)

        # UMAP으로 2D 투영
        umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
        coords = umap_2d.fit_transform(embeddings)

        # 시각화
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=topics, cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Topic')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('BERTopic: Document Topics in 2D Space')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"저장됨: {output_path}")
    except Exception as e:
        print(f"시각화 생성 실패: {e}")


def generate_bertopic_visualizations(topic_model, texts, topics, probs, labels, output_dir):
    """BERTopic 내장 시각화 메서드들을 사용하여 다양한 시각화 생성

    BERTopic은 Plotly 기반의 인터랙티브 시각화를 제공한다.
    - visualize_topics(): 토픽 간 거리 맵 (Intertopic Distance Map)
    - visualize_documents(): 문서 임베딩 2D 시각화
    - visualize_hierarchy(): 토픽 계층 구조 (덴드로그램)
    - visualize_heatmap(): 토픽 간 유사도 히트맵
    - visualize_barchart(): 토픽별 키워드 막대 차트
    - visualize_distribution(): 단일 문서의 토픽 분포
    - visualize_topics_per_class(): 클래스별 토픽 분포
    """
    print("\n" + "=" * 60)
    print("BERTopic 시각화 생성")
    print("=" * 60)

    # 1. 토픽 간 거리 맵 (Intertopic Distance Map)
    # 2D 공간에서 토픽의 크기(문서 수)와 토픽 간 거리를 시각화
    try:
        fig = topic_model.visualize_topics()
        fig.write_html(os.path.join(output_dir, '6-3-viz-topics-distance-map.html'))
        print("✓ 토픽 거리 맵 저장: 6-3-viz-topics-distance-map.html")
    except Exception as e:
        print(f"✗ 토픽 거리 맵 생성 실패: {e}")

    # 2. 문서 임베딩 시각화
    # 각 문서를 2D 공간에 투영하여 토픽별로 색상 구분
    try:
        embeddings = topic_model._extract_embeddings(texts)
        fig = topic_model.visualize_documents(texts, embeddings=embeddings)
        fig.write_html(os.path.join(output_dir, '6-3-viz-documents.html'))
        print("✓ 문서 임베딩 시각화 저장: 6-3-viz-documents.html")
    except Exception as e:
        print(f"✗ 문서 임베딩 시각화 생성 실패: {e}")

    # 3. 토픽 계층 구조 (덴드로그램)
    # 토픽 간 유사성을 계층적으로 표현, 토픽 병합 시 참고
    try:
        fig = topic_model.visualize_hierarchy()
        fig.write_html(os.path.join(output_dir, '6-3-viz-hierarchy.html'))
        print("✓ 토픽 계층 구조 저장: 6-3-viz-hierarchy.html")
    except Exception as e:
        print(f"✗ 토픽 계층 구조 생성 실패: {e}")

    # 4. 토픽 유사도 히트맵
    # 토픽 간 코사인 유사도를 히트맵으로 표현
    try:
        fig = topic_model.visualize_heatmap()
        fig.write_html(os.path.join(output_dir, '6-3-viz-heatmap.html'))
        print("✓ 토픽 유사도 히트맵 저장: 6-3-viz-heatmap.html")
    except Exception as e:
        print(f"✗ 토픽 유사도 히트맵 생성 실패: {e}")

    # 5. 토픽별 키워드 막대 차트
    # 각 토픽의 상위 키워드와 c-TF-IDF 점수
    try:
        fig = topic_model.visualize_barchart(top_n_topics=8)
        fig.write_html(os.path.join(output_dir, '6-3-viz-barchart.html'))
        print("✓ 토픽별 키워드 차트 저장: 6-3-viz-barchart.html")
    except Exception as e:
        print(f"✗ 토픽별 키워드 차트 생성 실패: {e}")

    # 6. 단일 문서의 토픽 분포 (probs 필요)
    # 특정 문서가 각 토픽에 속할 확률 분포
    try:
        if probs is not None and len(probs) > 0:
            # 첫 번째 문서의 토픽 분포 시각화
            fig = topic_model.visualize_distribution(probs[0])
            fig.write_html(os.path.join(output_dir, '6-3-viz-distribution-doc0.html'))
            print("✓ 문서 토픽 분포 저장: 6-3-viz-distribution-doc0.html")
    except Exception as e:
        print(f"✗ 문서 토픽 분포 생성 실패: {e}")

    # 7. 클래스별 토픽 분포
    # 실제 레이블(클래스)별로 토픽이 어떻게 분포하는지 시각화
    try:
        if labels is not None:
            # 먼저 topics_per_class DataFrame 생성
            topics_per_class = topic_model.topics_per_class(texts, classes=labels)
            fig = topic_model.visualize_topics_per_class(topics_per_class)
            fig.write_html(os.path.join(output_dir, '6-3-viz-topics-per-class.html'))
            print("✓ 클래스별 토픽 분포 저장: 6-3-viz-topics-per-class.html")
    except Exception as e:
        print(f"✗ 클래스별 토픽 분포 생성 실패: {e}")

    print("\n모든 시각화 파일은 HTML 형식으로 저장되었습니다.")
    print("브라우저에서 열어 인터랙티브하게 탐색할 수 있습니다.")

def compare_lda_bertopic(lda_results_path, bertopic_topics):
    """LDA와 BERTopic 결과 비교"""
    comparison = {}

    # LDA 결과 로드
    try:
        with open(lda_results_path, 'r') as f:
            lda_results = json.load(f)

        comparison['lda'] = {
            'num_topics': lda_results['num_topics'],
            'coherence_score': lda_results['coherence_score'],
            'topics': lda_results['topics']
        }
    except FileNotFoundError:
        print("LDA 결과 파일 없음. LDA 실습을 먼저 실행하세요.")
        comparison['lda'] = None

    # BERTopic 결과
    comparison['bertopic'] = {
        'num_topics': len(bertopic_topics),
        'topics': bertopic_topics
    }

    return comparison

def main():
    print("=" * 60)
    print("6.3절 실습: BERTopic 토픽 모델링")
    print("=" * 60)

    # 1. 데이터 로드
    texts, labels = load_newsgroups_data(n_samples=2000)

    # 2. BERTopic 모델 생성 (5단계 파이프라인)
    print("\n[BERTopic 5단계 파이프라인]")
    print("1. 임베딩: SentenceTransformer (all-MiniLM-L6-v2)")
    print("2. 차원 축소: UMAP (384D -> 5D)")
    print("3. 클러스터링: HDBSCAN")
    print("4. 토큰화: CountVectorizer")
    print("5. 토픽 표현: c-TF-IDF")

    topic_model = create_bertopic_model()

    # 3. 학습
    topics, probs, topic_info = train_bertopic(texts, topic_model)

    # 4. 토픽 상세 정보
    topic_details = get_topic_details(topic_model, num_words=10)

    print("\n" + "=" * 60)
    print("발견된 토픽 및 키워드")
    print("=" * 60)
    for topic in topic_details:
        print(f"\n[토픽 {topic['topic_id']}] ({topic['count']}개 문서)")
        print(f"  키워드: {', '.join(topic['keywords'][:7])}")

    # 5. 실제 레이블과 비교
    topic_label_dist = compare_with_ground_truth(topics, labels, topic_model)

    print("\n" + "=" * 60)
    print("토픽-레이블 매핑")
    print("=" * 60)
    for topic_id, dist in topic_label_dist.items():
        main_label = max(dist, key=dist.get)
        print(f"토픽 {topic_id}: {main_label} ({dist[main_label]}건)")

    # 5-1. reduce_outliers 적용
    print("\n" + "=" * 60)
    print("reduce_outliers 적용 전후 비교")
    print("=" * 60)

    # 적용 전 토픽별 문서 수
    topic_counts_before = {}
    for t in topics:
        topic_counts_before[t] = topic_counts_before.get(t, 0) + 1

    print("적용 전:")
    for topic_id in sorted(topic_counts_before.keys()):
        print(f"  토픽 {topic_id}: {topic_counts_before[topic_id]}개")

    # reduce_outliers 적용
    new_topics = topic_model.reduce_outliers(texts, topics, strategy="distributions")

    # 적용 후 토픽별 문서 수
    topic_counts_after = {}
    for t in new_topics:
        topic_counts_after[t] = topic_counts_after.get(t, 0) + 1

    print("\n적용 후:")
    for topic_id in sorted(topic_counts_after.keys()):
        print(f"  토픽 {topic_id}: {topic_counts_after[topic_id]}개")

    # 6. 시각화
    # 6-1. matplotlib 기반 2D 시각화
    vis_path = os.path.join(OUTPUT_DIR, '6-3-bertopic-2d.png')
    visualize_topics_2d(topic_model, texts, topics, vis_path)

    # 6-2. BERTopic 내장 시각화 (Plotly 기반, HTML 출력)
    generate_bertopic_visualizations(topic_model, texts, topics, probs, labels, OUTPUT_DIR)

    # 7. LDA와 비교
    lda_path = os.path.join(OUTPUT_DIR, '6-2-lda-results.json')
    comparison = compare_lda_bertopic(lda_path, topic_details)

    # 8. 결과 저장
    results = {
        'model': 'BERTopic',
        'num_topics': len(topic_details),
        'num_documents': len(texts),
        'noise_documents': sum(1 for t in topics if t == -1),
        'topics': topic_details,
        'topic_label_mapping': topic_label_dist,
        'comparison': comparison
    }

    results_path = os.path.join(OUTPUT_DIR, '6-3-bertopic-results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장됨: {results_path}")

    # 9. 요약 출력
    print("\n" + "=" * 60)
    print("BERTopic 실습 결과 요약")
    print("=" * 60)
    print(f"- 문서 수: {len(texts)}")
    print(f"- 발견된 토픽 수: {len(topic_details)}")
    print(f"- 노이즈 문서: {sum(1 for t in topics if t == -1)}")
    print(f"- 출력 파일: {results_path}")

    # LDA vs BERTopic 비교 요약
    if comparison.get('lda'):
        print("\n[LDA vs BERTopic 비교]")
        print(f"  LDA: {comparison['lda']['num_topics']}개 토픽, Coherence={comparison['lda']['coherence_score']:.4f}")
        print(f"  BERTopic: {comparison['bertopic']['num_topics']}개 토픽 (자동 결정)")

    return results

if __name__ == "__main__":
    results = main()
