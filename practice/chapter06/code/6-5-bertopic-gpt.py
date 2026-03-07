"""
6장 통합 실습: BERTopic + GPT로 뉴스 토픽 자동 분석
- 동일한 데이터(한국어 뉴스)를 사용하여 정적/동적 분석 통합 수행
- 명사 추출 전처리 적용
- BERTopic 5단계 파이프라인 + 동적 분석
- 토픽 생명 주기 분석 (출현/최고점/소멸)
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

# 형태소 분석기
from kiwipiepy import Kiwi

# .env 파일에서 API 키 로드
load_dotenv()

# 경로 설정
CURRENT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================ 
# 0. 전처리 (명사 추출)
# ============================================================ 

def extract_nouns_korean(texts):
    """국문 텍스트에서 명사 추출 중 (Kiwi 사용)"""
    print("[전처리] 국문 텍스트에서 명사 추출 중...")
    kiwi = Kiwi()
    
    extracted_texts = []
    # 명사 태그: NNG(일반), NNP(고유), SL(알파벳)
    target_tags = {'NNG', 'NNP', 'SL'}
    
    for text in texts:
        if not isinstance(text, str):
            extracted_texts.append("")
            continue
            
        try:
            result = kiwi.analyze(text)
            nouns = [token.form for token in result[0][0] if token.tag in target_tags and len(token.form) > 1]
            extracted_texts.append(" ".join(nouns))
        except Exception:
            extracted_texts.append(text)
            
    return extracted_texts

# ============================================================ 
# 1. 데이터 준비 (CSV 로드)
# ============================================================ 

def load_data():
    """저장된 한국어 뉴스 시뮬레이션 데이터 로드 및 전처리"""
    csv_path = os.path.join(DATA_DIR, 'korean_news.csv')
    if not os.path.exists(csv_path):
        # 데이터가 없으면 새로 생성 (최초 1회)
        print("데이터 파일이 없어 새로 생성합니다...")
        from datetime import timedelta
        np.random.seed(42)
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2021, 12, 31)
        topic_templates = {
            '경제': ["주식 시장이 상승세를 보이며 투자자들의 관심이 높아지고 있다", "중앙은행이 기준금리를 동결하며 경제 안정을 도모했다", "수출이 증가하며 무역 흑자가 확대되었다"],
            '코로나': ["코로나19 확진자가 급증하며 방역 조치가 강화되었다", "백신 접종이 시작되며 집단 면역에 대한 기대가 높아졌다", "사회적 거리두기로 인해 자영업자들이 어려움을 겪고 있다"],
            '기술': ["인공지능 기술이 다양한 산업에 적용되고 있다", "스마트폰 신제품이 출시되며 소비자 관심이 집중되었다"],
            '재택근무': ["재택근무가 확산되며 업무 환경이 변화하고 있다", "화상회의 플랫폼 사용이 급증했다"]
        }
        def get_topic_weights(date):
            if date < datetime(2020, 1, 1): return {'경제': 0.4, '코로나': 0.0, '기술': 0.4, '재택근무': 0.2}
            elif date < datetime(2020, 6, 1): return {'경제': 0.15, '코로나': 0.55, '기술': 0.15, '재택근무': 0.15}
            elif date < datetime(2021, 1, 1): return {'경제': 0.2, '코로나': 0.35, '기술': 0.2, '재택근무': 0.25}
            else: return {'경제': 0.3, '코로나': 0.25, '기술': 0.25, '재택근무': 0.2}
        
        news_data = []
        total_days = (end_date - start_date).days
        for _ in range(1000):
            random_days = np.random.randint(0, total_days)
            date = start_date + timedelta(days=random_days)
            weights = get_topic_weights(date)
            selected_topic = np.random.choice(list(weights.keys()), p=list(weights.values()))
            news_data.append({'text': np.random.choice(topic_templates[selected_topic]), 'date': date, 'true_topic': selected_topic})
        df = pd.DataFrame(news_data).sort_values('date').reset_index(drop=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding='utf-8')

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    texts = df['text'].tolist()
    timestamps = df['date'].tolist()
    
    # 명사 추출 전처리
    texts_nouns = extract_nouns_korean(texts)
    
    print(f"[데이터] 한국 뉴스 시뮬레이션: {len(texts)}개 문서 로드 및 명사 추출 완료")
    return texts_nouns, timestamps

# ============================================================
# 2. 토픽 생명 주기 분석
# ============================================================

def analyze_topic_lifecycle(topics_over_time, topic_model, threshold=0.01):
    """토픽의 출현, 최고점, 소멸 시점을 분석"""
    topic_lifecycle = {}

    unique_topics = [t for t in topics_over_time["Topic"].unique() if t != -1]

    for topic in unique_topics:
        topic_data = topics_over_time[topics_over_time["Topic"] == topic].copy()
        topic_data = topic_data.sort_values("Timestamp")

        # 출현: 빈도가 임계값을 초과하는 첫 시점
        above_threshold = topic_data[topic_data["Frequency"] > threshold]
        emergence = above_threshold.iloc[0]["Timestamp"] if len(above_threshold) > 0 else None

        # 최고점: 빈도가 최대인 시점
        peak_idx = topic_data["Frequency"].idxmax()
        peak = topic_data.loc[peak_idx, "Timestamp"]
        peak_freq = topic_data.loc[peak_idx, "Frequency"]

        # 소멸: 최고점 이후 빈도가 임계값 아래로 떨어지는 첫 시점
        after_peak = topic_data[topic_data["Timestamp"] >= peak]
        below_threshold = after_peak[after_peak["Frequency"] < threshold]
        decline = below_threshold.iloc[0]["Timestamp"] if len(below_threshold) > 0 else None

        # 키워드 추출
        topic_words = topic_model.get_topic(topic)
        keywords = [w for w, _ in topic_words[:5]] if topic_words else []

        topic_lifecycle[topic] = {
            "emergence": emergence,
            "peak": peak,
            "peak_frequency": peak_freq,
            "decline": decline,
            "keywords": keywords
        }

    return topic_lifecycle


def visualize_topic_lifecycle(topics_over_time, topic_lifecycle, topic_model, output_path):
    """토픽 생명 주기 시각화 (선 그래프 + 마커) - 흑백 버전"""
    # 상위 토픽만 선택 (문서 수 기준)
    topic_info = topic_model.get_topic_info()
    top_topics = [t for t in topic_info["Topic"].tolist() if t != -1][:5]

    plt.figure(figsize=(14, 8))

    # 흑백 스타일: 선 스타일과 마커로 구분
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['black'] * 5  # 모두 검정색

    for i, topic in enumerate(top_topics):
        if topic not in topic_lifecycle:
            continue

        topic_data = topics_over_time[topics_over_time["Topic"] == topic].copy()
        topic_data = topic_data.sort_values("Timestamp")

        lifecycle = topic_lifecycle[topic]
        label = f"토픽 {topic}: {', '.join(lifecycle['keywords'][:3])}"

        # 선 그래프
        plt.plot(topic_data["Timestamp"], topic_data["Frequency"],
                 label=label, color=colors[i], linestyle=line_styles[i],
                 linewidth=2, marker=markers[i], markersize=4, markevery=2)

        # 최고점 마커
        if lifecycle["peak"] is not None:
            peak_data = topic_data[topic_data["Timestamp"] == lifecycle["peak"]]
            if len(peak_data) > 0:
                plt.scatter(lifecycle["peak"], lifecycle["peak_frequency"],
                           color='black', s=200, marker='*',
                           edgecolors='black', linewidths=2, zorder=5)

        # 출현 시점 마커
        if lifecycle["emergence"] is not None:
            emergence_data = topic_data[topic_data["Timestamp"] == lifecycle["emergence"]]
            if len(emergence_data) > 0:
                plt.scatter(lifecycle["emergence"], emergence_data["Frequency"].values[0],
                           color='white', s=120, marker='o',
                           edgecolors='black', linewidths=2, zorder=5)

    plt.xlabel('시간', fontsize=12)
    plt.ylabel('토픽 빈도', fontsize=12)
    plt.title('토픽 생명 주기 분석: 출현(○) → 최고점(★) → 소멸 추이', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, color='gray')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[시각화] 토픽 생명 주기 그래프 저장: {output_path}")


# ============================================================
# 3. BERTopic 모델 구축 및 실행
# ============================================================

def main():
    print("=" * 60)
    print("통합 실습: 동일 데이터를 활용한 정적/동적 토픽 분석")
    print("=" * 60)

    # 1. 데이터 로드
    texts, timestamps = load_data()

    # 2. BERTopic 파이프라인 구성
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
        calculate_probabilities=True
    )

    # 3. 모델 학습 (정적 분석 결과 도출)
    print("\n[단계 1] 정적 토픽 분석 실행 중...")
    topics, probs = topic_model.fit_transform(texts)
    
    # 정적 결과 출력 및 저장
    topic_info = topic_model.get_topic_info()
    print("\n발견된 주요 토픽 (정적):")
    print(topic_info.head(10))
    
    # 4. 동적 토픽 분석 실행
    print("\n[단계 2] 동적 토픽 분석 실행 중 (시간에 따른 변화 추적)...")
    topics_over_time = topic_model.topics_over_time(texts, timestamps, nr_bins=12)

    # 5. 토픽 생명 주기 분석
    print("\n[단계 3] 토픽 생명 주기 분석 실행 중...")
    topic_lifecycle = analyze_topic_lifecycle(topics_over_time, topic_model, threshold=0.02)

    # 생명 주기 결과 출력
    print("\n" + "-" * 40)
    print("토픽 생명 주기 분석 결과:")
    print("-" * 40)
    for topic_id, lifecycle in topic_lifecycle.items():
        keywords = ', '.join(lifecycle['keywords'][:3])
        emergence = lifecycle['emergence'].strftime('%Y-%m') if lifecycle['emergence'] else '-'
        peak = lifecycle['peak'].strftime('%Y-%m') if lifecycle['peak'] else '-'
        decline = lifecycle['decline'].strftime('%Y-%m') if lifecycle['decline'] else '-'
        print(f"토픽 {topic_id} ({keywords})")
        print(f"  출현: {emergence} | 최고점: {peak} (빈도: {lifecycle['peak_frequency']:.3f}) | 소멸: {decline}")

    # 6. 토픽 생명 주기 시각화
    lifecycle_plot_path = os.path.join(OUTPUT_DIR, '6-5-topic-lifecycle.png')
    visualize_topic_lifecycle(topics_over_time, topic_lifecycle, topic_model, lifecycle_plot_path)

    # 7. 결과 저장
    # lifecycle을 JSON 직렬화 가능하게 변환
    lifecycle_serializable = {}
    for topic_id, lc in topic_lifecycle.items():
        lifecycle_serializable[str(topic_id)] = {
            'emergence': lc['emergence'].isoformat() if lc['emergence'] else None,
            'peak': lc['peak'].isoformat() if lc['peak'] else None,
            'peak_frequency': lc['peak_frequency'],
            'decline': lc['decline'].isoformat() if lc['decline'] else None,
            'keywords': lc['keywords']
        }

    results = {
        'static_topics': topic_info.to_dict(orient='records'),
        'dynamic_topics': topics_over_time.to_dict(orient='records'),
        'topic_lifecycle': lifecycle_serializable
    }

    results_path = os.path.join(OUTPUT_DIR, '6-5-integrated-results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[완료] 모든 분석 결과가 저장되었습니다: {results_path}")

    # 간단한 결과 요약 출력
    print("\n" + "-" * 40)
    print("토픽별 대표 키워드 (정적):")
    for _, row in topic_info.iterrows():
        if row['Topic'] == -1: continue
        words = [w for w, _ in topic_model.get_topic(row['Topic'])[:5]]
        print(f"토픽 {row['Topic']}: {', '.join(words)} ({row['Count']}건)")

if __name__ == "__main__":
    main()
