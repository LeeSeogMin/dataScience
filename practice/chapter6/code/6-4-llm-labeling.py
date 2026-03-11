"""
9장 9.4절 실습: LLM 기반 토픽 레이블링
- BERTopic + OpenAI GPT 연동
- 프롬프트 엔지니어링으로 토픽 자동 명명
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from bertopic.representation import OpenAI as BertopicOpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()

# 출력 경로 설정
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_newsgroups_data(n_samples=1500):
    """20 Newsgroups 데이터 로드"""
    categories = ['rec.sport.baseball', 'sci.med', 'comp.graphics', 'talk.politics.misc']
    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )

    indices = np.random.RandomState(42).choice(len(newsgroups.data), min(n_samples, len(newsgroups.data)), replace=False)
    texts = [newsgroups.data[i] for i in indices]
    labels = [newsgroups.target_names[newsgroups.target[i]] for i in indices]

    # 빈 문서 제거
    valid_idx = [i for i, t in enumerate(texts) if len(t.strip()) > 50]
    texts = [texts[i] for i in valid_idx]
    labels = [labels[i] for i in valid_idx]

    print(f"로드된 문서 수: {len(texts)}")
    return texts, labels

def create_llm_representation():
    """LLM 기반 토픽 표현 모델 생성"""

    # OpenAI API 키 확인
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("LLM 레이블링을 건너뛰고 기본 키워드만 사용합니다.")
        return None

    # 프롬프트 템플릿
    prompt = """
다음은 하나의 토픽을 대표하는 키워드 목록입니다.
이 키워드들을 바탕으로 토픽의 이름을 5단어 이내로 간결하게 지어주세요.

키워드: [KEYWORDS]

토픽 이름:
"""

    # BERTopic용 OpenAI 표현 모델
    representation_model = BertopicOpenAI(
        model="gpt-3.5-turbo",
        chat=True,
        prompt=prompt,
        nr_docs=5,
        delay_in_seconds=0.5
    )

    return representation_model

def create_bertopic_with_llm(representation_model):
    """LLM 표현 모델을 포함한 BERTopic 생성"""

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if representation_model:
        topic_model = BERTopic(
            embedding_model=embedding_model,
            representation_model=representation_model,
            verbose=True,
            calculate_probabilities=True
        )
    else:
        topic_model = BERTopic(
            embedding_model=embedding_model,
            verbose=True,
            calculate_probabilities=True
        )

    return topic_model

def manual_labeling_simulation(topics_info):
    """LLM 없이 키워드 기반 수동 레이블링 시뮬레이션"""

    # 키워드 패턴 기반 레이블 매핑 (예시)
    keyword_patterns = {
        ('game', 'team', 'player', 'baseball', 'hit'): '스포츠/야구',
        ('medical', 'doctor', 'patient', 'health', 'disease'): '의료/건강',
        ('graphics', 'image', 'computer', 'software', 'file'): '컴퓨터 그래픽스',
        ('government', 'president', 'law', 'state', 'political'): '정치/정부'
    }

    labeled_topics = []
    for topic in topics_info:
        keywords = set(topic['keywords'][:5])
        matched_label = None

        for pattern_keywords, label in keyword_patterns.items():
            if len(keywords.intersection(pattern_keywords)) >= 2:
                matched_label = label
                break

        if not matched_label:
            matched_label = f"토픽_{topic['topic_id']}"

        labeled_topics.append({
            **topic,
            'auto_label': matched_label
        })

    return labeled_topics

def get_topic_details(topic_model, use_llm=True):
    """토픽 상세 정보 추출"""
    topics = []
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:
            continue

        topic_words = topic_model.get_topic(topic_id)
        keywords = [word for word, _ in topic_words[:10]]

        topic_data = {
            'topic_id': topic_id,
            'count': int(row['Count']),
            'name': row.get('Name', f'Topic_{topic_id}'),
            'keywords': keywords,
            'llm_label': row.get('CustomName', row.get('Name', None))
        }
        topics.append(topic_data)

    return topics

def compare_labeling_methods(topics_with_labels):
    """레이블링 방법 비교"""
    print("\n" + "=" * 60)
    print("토픽 레이블링 비교")
    print("=" * 60)

    for topic in topics_with_labels:
        print(f"\n[토픽 {topic['topic_id']}]")
        print(f"  키워드: {', '.join(topic['keywords'][:5])}")
        print(f"  자동 레이블: {topic.get('auto_label', 'N/A')}")
        if topic.get('llm_label'):
            print(f"  LLM 레이블: {topic['llm_label']}")

def main():
    print("=" * 60)
    print("9.4절 실습: LLM 기반 토픽 레이블링")
    print("=" * 60)

    # 1. 데이터 로드
    texts, labels = load_newsgroups_data(n_samples=1500)

    # 2. LLM 표현 모델 생성
    representation_model = create_llm_representation()

    # 3. BERTopic 모델 생성 및 학습
    topic_model = create_bertopic_with_llm(representation_model)
    print("\nBERTopic 학습 중...")
    topics, probs = topic_model.fit_transform(texts)

    # 4. 토픽 정보 추출
    topic_details = get_topic_details(topic_model, use_llm=representation_model is not None)

    print("\n" + "=" * 60)
    print("발견된 토픽")
    print("=" * 60)
    for topic in topic_details:
        print(f"\n[토픽 {topic['topic_id']}] ({topic['count']}개 문서)")
        print(f"  키워드: {', '.join(topic['keywords'][:7])}")
        print(f"  이름: {topic['name']}")

    # 5. 수동 레이블링 시뮬레이션 (LLM 없을 때)
    topics_with_labels = manual_labeling_simulation(topic_details)

    # 6. 레이블링 방법 비교
    compare_labeling_methods(topics_with_labels)

    # 7. 결과 저장
    results = {
        'model': 'BERTopic + LLM Labeling',
        'llm_used': representation_model is not None,
        'num_topics': len(topic_details),
        'num_documents': len(texts),
        'topics': topics_with_labels
    }

    results_path = os.path.join(OUTPUT_DIR, '9-4-llm-labeling-results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장됨: {results_path}")

    # 8. 프롬프트 엔지니어링 팁 출력
    print("\n" + "=" * 60)
    print("프롬프트 엔지니어링 팁")
    print("=" * 60)
    print("""
1. 도메인 컨텍스트 제공:
   "당신은 이커머스 고객 리뷰 분석 전문가입니다..."

2. 출력 형식 지정:
   "JSON 형식으로 {'label': '...', 'confidence': 0.9} 반환"

3. Few-shot 예시 제공:
   "예: ['배송', '택배', '도착'] -> '배송 서비스'"

4. 제약 조건 명시:
   "5단어 이내로, 한국어로 작성"
""")

    return results

if __name__ == "__main__":
    results = main()
