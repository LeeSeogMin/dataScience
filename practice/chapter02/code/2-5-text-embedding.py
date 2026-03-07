"""
2-5-text-embedding.py: 텍스트 임베딩과 분류 비교

TF-IDF와 Sentence-BERT 임베딩을 비교한다:
1. 텍스트 벡터화 성능
2. 분류 정확도
3. 벡터 차원

실행 방법:
    python 2-5-text-embedding.py

필수 라이브러리:
    pip install sentence-transformers
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("경고: sentence-transformers가 설치되지 않음.")
    print("설치: pip install sentence-transformers")

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_subset_data(n_samples: int = 2000, categories: list = None):
    """뉴스그룹 데이터의 부분집합을 로드한다."""
    if categories is None:
        categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball', 'talk.politics.misc']

    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )

    # 데이터 샘플링
    if len(newsgroups.data) > n_samples:
        indices = np.random.RandomState(42).choice(
            len(newsgroups.data), n_samples, replace=False
        )
        texts = [newsgroups.data[i] for i in indices]
        labels = newsgroups.target[indices]
    else:
        texts = newsgroups.data
        labels = newsgroups.target

    return texts, labels, newsgroups.target_names


def vectorize_tfidf(texts_train: list, texts_test: list, max_features: int = 5000):
    """TF-IDF로 텍스트를 벡터화한다."""
    start_time = time.time()

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train = vectorizer.fit_transform(texts_train).toarray()
    X_test = vectorizer.transform(texts_test).toarray()

    elapsed_time = time.time() - start_time

    return X_train, X_test, elapsed_time


def vectorize_sbert(texts_train: list, texts_test: list, model_name: str = 'all-MiniLM-L6-v2'):
    """Sentence-BERT로 텍스트를 임베딩한다."""
    if not SBERT_AVAILABLE:
        return None, None, 0

    start_time = time.time()

    model = SentenceTransformer(model_name)
    X_train = model.encode(texts_train, show_progress_bar=True)
    X_test = model.encode(texts_test, show_progress_bar=False)

    elapsed_time = time.time() - start_time

    return X_train, X_test, elapsed_time


def train_and_evaluate(X_train, X_test, y_train, y_test, method_name: str):
    """로지스틱 회귀로 분류하고 평가한다."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return {
        'method': method_name,
        'accuracy': accuracy,
        'dimensions': X_train.shape[1],
        'y_pred': y_pred
    }


def main():
    print("="*60)
    print("텍스트 분류 성능 비교: TF-IDF vs Sentence-BERT")
    print("="*60)

    # 1. 데이터 로드
    print("\n[1/4] 데이터 로드 중...")
    n_samples = 2000
    texts, labels, target_names = load_subset_data(n_samples=n_samples)

    print(f"   샘플 수: {len(texts)}")
    print(f"   카테고리: {target_names}")

    # 2. 훈련/테스트 분할
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   훈련: {len(texts_train)}, 테스트: {len(texts_test)}")

    # 3. 벡터화 및 분류
    print("\n[2/4] TF-IDF 벡터화 중...")
    X_train_tfidf, X_test_tfidf, time_tfidf = vectorize_tfidf(texts_train, texts_test)
    print(f"   ✓ 완료: {X_train_tfidf.shape[1]}차원, {time_tfidf:.2f}초")

    results = []

    # TF-IDF 분류
    print("\n[3/4] 분류 모델 훈련 및 평가...")
    result_tfidf = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF")
    result_tfidf['time'] = time_tfidf
    results.append(result_tfidf)
    print(f"   ✓ TF-IDF: 정확도={result_tfidf['accuracy']:.3f}")

    # SBERT 벡터화 및 분류
    if SBERT_AVAILABLE:
        print("\n   Sentence-BERT 임베딩 중...")
        X_train_sbert, X_test_sbert, time_sbert = vectorize_sbert(texts_train, texts_test)

        if X_train_sbert is not None:
            result_sbert = train_and_evaluate(X_train_sbert, X_test_sbert, y_train, y_test, "SBERT")
            result_sbert['time'] = time_sbert
            results.append(result_sbert)
            print(f"   ✓ SBERT: 정확도={result_sbert['accuracy']:.3f}, {time_sbert:.2f}초")

    # 4. 결과 요약
    print("\n" + "="*60)
    print(f"텍스트 분류 성능 비교 (20 Newsgroups, n={n_samples})")
    print("="*60)
    print()
    print(f"{'특성 추출 방법':<25} {'정확도':<12} {'벡터 차원':<12} {'처리 시간':<12}")
    print("-"*60)

    for result in results:
        print(f"{result['method']:<25} {result['accuracy']:<12.3f} {result['dimensions']:<12} {result['time']:.2f}초")

    # 성능 비교
    if len(results) > 1:
        tfidf_acc = results[0]['accuracy']
        sbert_acc = results[1]['accuracy']
        tfidf_dim = results[0]['dimensions']
        sbert_dim = results[1]['dimensions']

        print("\n" + "-"*60)
        print(f"SBERT가 TF-IDF보다 정확도가 {(sbert_acc - tfidf_acc)*100:.1f}%p 높으면서")
        print(f"차원은 {tfidf_dim/sbert_dim:.0f}배 작음 ({sbert_dim} vs {tfidf_dim})")

    # 5. 상세 분류 리포트
    print("\n" + "="*60)
    print("상세 분류 리포트 (TF-IDF)")
    print("="*60)
    print(classification_report(y_test, results[0]['y_pred'], target_names=target_names))

    if len(results) > 1 and SBERT_AVAILABLE:
        print("\n" + "="*60)
        print("상세 분류 리포트 (SBERT)")
        print("="*60)
        print(classification_report(y_test, results[1]['y_pred'], target_names=target_names))

    # 6. 결과 저장
    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'y_pred'}
        for r in results
    ])
    output_path = OUTPUT_DIR / "text_embedding_comparison.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n결과 저장: {output_path}")

    print("\n" + "="*60)
    print("분석 인사이트")
    print("="*60)
    print("""
- SBERT는 TF-IDF보다 낮은 차원으로 더 높은 정확도 달성
- 임베딩은 단어의 의미적 유사성을 벡터 거리로 표현
- SBERT는 처리 시간이 더 걸리지만 품질이 우수

활용 사례:
- 유사 문서 검색: 코사인 유사도 기반
- 군집화: HDBSCAN과 결합 (4장 참조)
- 분류: 적은 라벨 데이터로도 높은 성능
""")

    return results


if __name__ == "__main__":
    main()
