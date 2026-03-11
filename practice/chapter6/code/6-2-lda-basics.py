"""
9장 9.2절 실습: LDA 기초
- 20 Newsgroups 데이터셋으로 LDA 토픽 모델링
- 토픽 일관성(Coherence) 평가
- 토픽별 키워드 추출
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
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

    print(f"로드된 문서 수: {len(texts)}")
    print(f"카테고리: {categories}")
    return texts, labels

def preprocess_for_lda(texts):
    """LDA를 위한 전처리 (BOW 변환)"""
    # CountVectorizer로 토큰화 및 빈도 계산
    vectorizer = CountVectorizer(
        max_df=0.95,      # 95% 이상 문서에 등장하는 단어 제외
        min_df=2,         # 최소 2개 문서에 등장해야 함
        stop_words='english',
        max_features=5000
    )

    # 문서-단어 행렬
    doc_term_matrix = vectorizer.fit_transform(texts)

    # Gensim 형식으로 변환
    vocab = vectorizer.get_feature_names_out()
    id2word = {i: word for i, word in enumerate(vocab)}

    # 각 문서를 (단어 ID, 빈도) 형태로 변환
    corpus = []
    for i in range(doc_term_matrix.shape[0]):
        row = doc_term_matrix[i].toarray().flatten()
        doc_bow = [(j, int(row[j])) for j in range(len(row)) if row[j] > 0]
        corpus.append(doc_bow)

    # Gensim Dictionary 생성
    dictionary = corpora.Dictionary()
    dictionary.id2token = id2word
    dictionary.token2id = {v: k for k, v in id2word.items()}

    print(f"어휘 크기: {len(vocab)}")
    return corpus, dictionary, vocab

def train_lda(corpus, dictionary, num_topics=4, passes=15):
    """LDA 모델 학습"""
    print(f"\nLDA 모델 학습 중... (토픽 수: {num_topics})")

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary.id2token,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha='auto',
        eta='auto'
    )

    return lda_model

def evaluate_coherence(lda_model, corpus, dictionary, texts):
    """토픽 일관성 평가 (C_v만 반환 - 하위 호환성)"""
    metrics = evaluate_all_metrics(lda_model, corpus, dictionary, texts)
    return metrics['c_v']

def evaluate_all_metrics(lda_model, corpus, dictionary, texts):
    """다양한 평가지표 계산"""
    # 토큰화된 텍스트 (Coherence 계산용)
    tokenized_texts = [text.lower().split() for text in texts]

    metrics = {}

    # C_v Coherence
    coherence_cv = CoherenceModel(
        model=lda_model,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )
    metrics['c_v'] = coherence_cv.get_coherence()

    # UMass Coherence
    coherence_umass = CoherenceModel(
        model=lda_model,
        corpus=corpus,
        dictionary=dictionary,
        coherence='u_mass'
    )
    metrics['u_mass'] = coherence_umass.get_coherence()

    # NPMI Coherence
    coherence_npmi = CoherenceModel(
        model=lda_model,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_npmi'
    )
    metrics['npmi'] = coherence_npmi.get_coherence()

    # Topic Diversity (상위 10개 단어 기준 고유 단어 비율)
    num_words = 10
    all_words = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=num_words)
        all_words.extend([word for word, prob in topic_words])

    unique_words = len(set(all_words))
    total_words = len(all_words)
    metrics['topic_diversity'] = unique_words / total_words if total_words > 0 else 0

    return metrics

def get_topic_keywords(lda_model, num_words=10):
    """토픽별 상위 키워드 추출"""
    topics = []
    for topic_id in range(lda_model.num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=num_words)
        keywords = [word for word, prob in topic_words]
        topics.append({
            'topic_id': topic_id,
            'keywords': keywords,
            'keyword_probs': [(word, float(prob)) for word, prob in topic_words]
        })
    return topics

def find_optimal_topics(corpus, dictionary, texts, min_topics=2, max_topics=8):
    """최적 토픽 수 찾기 (Coherence Score 기반)"""
    print("\n최적 토픽 수 탐색 중...")
    coherence_scores = []

    for k in range(min_topics, max_topics + 1):
        lda = train_lda(corpus, dictionary, num_topics=k, passes=10)
        score = evaluate_coherence(lda, corpus, dictionary, texts)
        coherence_scores.append((k, score))
        print(f"  K={k}: Coherence = {score:.4f}")

    return coherence_scores

def visualize_coherence(coherence_scores, output_path):
    """Coherence Score 시각화"""
    ks = [x[0] for x in coherence_scores]
    scores = [x[1] for x in coherence_scores]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Topics (K)', fontsize=12)
    plt.ylabel('Coherence Score (C_v)', fontsize=12)
    plt.title('LDA Topic Coherence by Number of Topics', fontsize=14)
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)

    # 최적 K 표시
    best_idx = np.argmax(scores)
    plt.axvline(x=ks[best_idx], color='r', linestyle='--', label=f'Best K={ks[best_idx]}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"저장됨: {output_path}")

def main():
    print("=" * 60)
    print("9.2절 실습: LDA 토픽 모델링")
    print("=" * 60)

    # 1. 데이터 로드
    texts, labels = load_newsgroups_data(n_samples=2000)

    # 2. 전처리
    corpus, dictionary, vocab = preprocess_for_lda(texts)

    # 3. 최적 토픽 수 탐색
    coherence_scores = find_optimal_topics(corpus, dictionary, texts, min_topics=2, max_topics=8)

    # Coherence 시각화
    coherence_path = os.path.join(OUTPUT_DIR, '9-2-lda-coherence.png')
    visualize_coherence(coherence_scores, coherence_path)

    # 최적 K 선택
    best_k = max(coherence_scores, key=lambda x: x[1])[0]
    print(f"\n최적 토픽 수: K={best_k}")

    # 4. 최적 K로 LDA 학습
    lda_model = train_lda(corpus, dictionary, num_topics=best_k, passes=20)

    # 5. 모든 평가지표 계산
    print("\n평가지표 계산 중...")
    all_metrics = evaluate_all_metrics(lda_model, corpus, dictionary, texts)
    coherence = all_metrics['c_v']

    print("\n" + "=" * 60)
    print("평가지표 결과")
    print("=" * 60)
    print(f"| 지표             | 값      | 해석 기준              |")
    print(f"|------------------|---------|------------------------|")
    print(f"| C_v (일관성)     | {all_metrics['c_v']:.4f}  | ≥0.5 양호              |")
    print(f"| UMass (일관성)   | {all_metrics['u_mass']:.4f} | 0에 가까울수록 좋음    |")
    print(f"| NPMI (일관성)    | {all_metrics['npmi']:.4f}  | -1~1, 높을수록 좋음    |")
    print(f"| 토픽 다양성      | {all_metrics['topic_diversity']:.4f}  | 1에 가까울수록 좋음    |")

    # 6. 토픽별 키워드 출력
    topics = get_topic_keywords(lda_model, num_words=10)

    print("\n" + "=" * 60)
    print("발견된 토픽 및 키워드")
    print("=" * 60)
    for topic in topics:
        print(f"\n[토픽 {topic['topic_id']}]")
        print(f"  키워드: {', '.join(topic['keywords'])}")

    # 7. 결과 저장
    results = {
        'model': 'LDA',
        'num_topics': best_k,
        'coherence_score': float(coherence),
        'all_metrics': {k: float(v) for k, v in all_metrics.items()},
        'coherence_scores_by_k': coherence_scores,
        'topics': topics,
        'vocab_size': len(vocab),
        'num_documents': len(texts)
    }

    results_path = os.path.join(OUTPUT_DIR, '6-2-lda-results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장됨: {results_path}")

    # 8. 요약 출력
    print("\n" + "=" * 60)
    print("LDA 실습 결과 요약")
    print("=" * 60)
    print(f"- 문서 수: {len(texts)}")
    print(f"- 어휘 크기: {len(vocab)}")
    print(f"- 최적 토픽 수: {best_k}")
    print(f"- C_v Coherence: {all_metrics['c_v']:.4f}")
    print(f"- UMass Coherence: {all_metrics['u_mass']:.4f}")
    print(f"- NPMI Coherence: {all_metrics['npmi']:.4f}")
    print(f"- Topic Diversity: {all_metrics['topic_diversity']:.4f}")
    print(f"- 출력 파일: {results_path}")

    return results

if __name__ == "__main__":
    results = main()
