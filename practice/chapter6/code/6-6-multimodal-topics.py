import os
import pandas as pd
from bertopic import BERTopic
from bertopic.backend import MultiModalBackend
from bertopic.representation import VisualRepresentation
import matplotlib.pyplot as plt

# 1. 데이터 로드
# 챕터 6 데이터 경로 설정
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, "data")
OUTPUT_PATH = os.path.join(DATA_PATH, "output")

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# 데이터 불러오기
df = pd.read_csv(os.path.join(DATA_PATH, "multimodal_data.csv"))
captions = df['caption'].tolist()
# 이미지 절대 경로로 변환
image_paths = [os.path.join(DATA_PATH, path) for path in df['image_path']]

print(f"로드된 문서 수: {len(captions)}")
print(f"로드된 이미지 수: {len(image_paths)}")

# 2. 멀티모달 임베딩 모델 (CLIP 기반)
# 실습 환경에서 CLIP 모델을 로드합니다.
print("멀티모달 백엔드 로드 중 (CLIP ViT-B-32)...")
embedding_model = MultiModalBackend('clip-ViT-B-32', batch_size=32)

# 3. 시각적 토픽 표현 모델
visual_model = VisualRepresentation()
representation_model = {
    "Visual_Aspect": visual_model
}

# 4. BERTopic 생성 및 학습
# 소규모 데이터셋이므로 n_neighbors 등을 조정할 수 있으나 기본값으로 진행
print("BERTopic 학습 중...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    representation_model=representation_model,
    verbose=True,
    min_topic_size=3  # 데이터가 적으므로 최소 토픽 크기를 줄임
)

topics, probs = topic_model.fit_transform(
    documents=captions,
    images=image_paths
)

# 5. 결과 확인 및 저장
topic_info = topic_model.get_topic_info()
print("\n[토픽 분석 결과]")
print(topic_info[['Topic', 'Count', 'Name', 'Representation']])

# 결과 저장
result_file = os.path.join(OUTPUT_PATH, "6-6-multimodal-results.csv")
topic_info.to_csv(result_file, index=False)
print(f"\n결과가 {result_file}에 저장되었습니다.")

# 토픽별 대표 키워드 출력
print("\n[토픽별 키워드]")
for i in range(len(topic_info)):
    topic_id = topic_info.iloc[i]['Topic']
    if topic_id == -1:
        continue
    words = topic_model.get_topic(topic_id)
    print(f"Topic {topic_id}: {[w[0] for w in words[:5]]}")
