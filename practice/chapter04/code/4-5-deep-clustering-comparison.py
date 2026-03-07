"""
4.5.4 딥 클러스터링 비교 실습: KMeans, DEC, IDEC, VaDE
=========================================================
두 가지 텍스트 데이터셋으로 딥 클러스터링 알고리즘 성능 비교

실행: python 4-5-deep-clustering-comparison.py

핵심 실험:
- 실험 A: 동일 도메인 내 세부 주제 (쇼핑 리뷰: 배송/품질/가격/서비스/불만)
- 실험 B: 완전히 다른 도메인 (음식/기술/여행/건강/엔터테인먼트)

목적: "임베딩이 포착하는 유사성"과 "군집화 목적의 유사성"이
      일치하느냐에 따라 딥 클러스터링 효과가 달라짐을 보여줌
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 시드 고정
np.random.seed(42)


# ============================================================
# 1. 텍스트 데이터 생성 (두 가지 유형)
# ============================================================
def generate_review_topics(n_samples=15000):
    """실험 A: 동일 도메인(쇼핑) 내 세부 주제 - 임베딩 공간에서 구분 어려움"""

    templates = {
        'delivery': [
            "배송이 정말 빨라서 좋았어요",
            "택배가 생각보다 빨리 도착했습니다",
            "배달이 신속해서 만족합니다",
            "배송 속도가 빨라서 좋아요",
            "주문하고 다음날 바로 받았어요",
            "빠른 배송 감사합니다",
            "배송이 하루만에 왔어요",
            "택배 기사님이 친절하셨어요",
            "배송 추적이 잘 되어서 좋았습니다",
            "안전하게 포장되어 배송왔어요",
            "새벽배송으로 받아서 편했어요",
            "배송 예정일보다 일찍 도착했습니다",
            "당일배송 너무 좋아요",
            "배송 상태가 실시간으로 확인되어 좋았어요",
            "포장이 꼼꼼하게 되어 왔습니다",
        ],
        'quality': [
            "품질이 정말 좋아요",
            "제품 퀄리티가 기대 이상입니다",
            "마감이 깔끔하고 튼튼해요",
            "소재가 고급스러워요",
            "만듦새가 정교합니다",
            "내구성이 좋아 보여요",
            "품질 대비 가격이 합리적이에요",
            "기대했던 것보다 품질이 좋네요",
            "재질이 부드럽고 좋아요",
            "견고하게 잘 만들어졌어요",
            "디자인도 예쁘고 품질도 좋아요",
            "오래 사용할 수 있을 것 같아요",
            "꼼꼼하게 제작된 느낌이에요",
            "고급 제품 느낌이 납니다",
            "세부 마감까지 신경 쓴 제품이에요",
        ],
        'price': [
            "가격 대비 정말 좋아요",
            "가성비 최고입니다",
            "이 가격에 이 품질이라니 놀랍네요",
            "합리적인 가격이에요",
            "저렴하게 잘 샀어요",
            "세일할 때 사서 더 좋았어요",
            "가격이 착해요",
            "이 가격에 이 정도면 만족이에요",
            "돈값 하는 제품입니다",
            "가격 대비 성능이 좋아요",
            "할인받아서 득템했어요",
            "비싸지 않아서 좋았어요",
            "적당한 가격에 좋은 품질이에요",
            "쿠폰 적용하니 더 저렴하게 샀어요",
            "경쟁 제품보다 가격이 좋아요",
        ],
        'service': [
            "고객 서비스가 훌륭해요",
            "문의에 빠르게 답변해주셨어요",
            "친절한 상담 감사합니다",
            "교환 절차가 간편했어요",
            "A/S가 잘 되어 있어요",
            "반품 처리가 빨랐습니다",
            "고객센터 응대가 좋았어요",
            "문제 해결을 신속하게 해주셨어요",
            "직원분이 정말 친절하셨어요",
            "환불 절차가 간단했습니다",
            "카톡 상담이 편리해요",
            "24시간 문의 가능해서 좋아요",
            "상담원이 전문적이었어요",
            "불편사항을 바로 처리해주셨어요",
            "사후관리가 잘 되는 업체에요",
        ],
        'complaint': [
            "제품에 하자가 있었어요",
            "설명과 다른 제품이 왔어요",
            "포장이 망가져서 왔습니다",
            "배송이 너무 늦게 왔어요",
            "사이즈가 맞지 않아요",
            "색상이 사진과 달라요",
            "기능이 제대로 작동하지 않아요",
            "품질이 기대에 못 미쳐요",
            "가격 대비 실망스러워요",
            "교환하려니 절차가 복잡해요",
            "연락이 잘 안 돼요",
            "환불이 늦어지고 있어요",
            "재구매 의사 없습니다",
            "다른 제품으로 교환 요청했어요",
            "부품이 누락되어 왔어요",
        ],
    }

    variations = [
        "{}", "{}.", "{} 추천해요", "{} 재구매 의사 있어요",
        "솔직히 {}", "{} 다음에도 이용할게요", "처음 샀는데 {}",
    ]

    reviews, labels = [], []
    label_names = list(templates.keys())
    samples_per_topic = n_samples // len(label_names)

    for label_idx, (topic, topic_templates) in enumerate(templates.items()):
        for _ in range(samples_per_topic):
            template = np.random.choice(topic_templates)
            variation = np.random.choice(variations)
            reviews.append(variation.format(template))
            labels.append(label_idx)

    indices = np.random.permutation(len(reviews))
    return [reviews[i] for i in indices], np.array([labels[i] for i in indices]), label_names


def generate_different_domains(n_samples=15000):
    """실험 B: 완전히 다른 도메인 - 임베딩 공간에서 잘 분리됨"""

    templates = {
        'food': [
            "오늘 저녁은 김치찌개를 끓여 먹었습니다",
            "새로 생긴 이탈리안 레스토랑 파스타가 맛있어요",
            "집에서 만든 케이크가 카페 못지않게 맛있네요",
            "삼겹살 구워 먹으니 소주가 생각나요",
            "엄마가 해주신 된장국이 최고입니다",
            "매운 떡볶이에 치즈 토핑 추가하면 맛있어요",
            "프렌치 프라이에 케첩 찍어 먹는 게 제일 좋아요",
            "스테이크는 미디엄 레어가 가장 맛있습니다",
            "아침에 토스트와 커피로 간단히 식사했어요",
            "라면에 계란 넣어 끓이면 더 맛있어요",
            "초밥은 신선한 생선이 생명이에요",
            "베이커리에서 갓 구운 빵 냄새가 좋아요",
            "비빔밥에 고추장 넣고 쓱쓱 비벼 먹었습니다",
            "치킨은 양념이냐 후라이드냐 고민되네요",
            "디저트로 티라미수 먹으니 하루가 달콤해요",
            "고등어 구이에 무조림 반찬이 어울려요",
            "우동 국물이 진하고 시원합니다",
            "샐러드에 발사믹 드레싱 뿌려 먹었어요",
            "불고기 양념이 달콤짭짤해서 밥도둑이에요",
            "브런치로 에그 베네딕트 먹었습니다",
        ],
        'tech': [
            "새 노트북 CPU 성능이 정말 빠릅니다",
            "파이썬으로 데이터 분석 프로젝트 진행 중이에요",
            "클라우드 서버 마이그레이션 작업을 완료했습니다",
            "스마트폰 배터리가 하루 종일 가네요",
            "인공지능 모델 학습에 GPU가 필수입니다",
            "와이파이 공유기 속도가 기가비트급이에요",
            "SSD로 바꾸니 부팅이 10초 만에 됩니다",
            "블루투스 이어폰 노이즈 캔슬링이 좋아요",
            "코딩할 때 듀얼 모니터가 생산성을 높여줘요",
            "앱 개발에 리액트 네이티브를 사용합니다",
            "데이터베이스 쿼리 최적화로 속도가 향상됐어요",
            "VPN 연결하니 보안이 강화된 느낌입니다",
            "딥러닝 프레임워크로 텐서플로우 사용해요",
            "메모리 16기가로 업그레이드했습니다",
            "API 연동 테스트가 성공적으로 완료됐어요",
            "도커 컨테이너로 배포 환경 구성했습니다",
            "깃허브에 코드 푸시하고 PR 올렸어요",
            "키보드 기계식으로 바꾸니 타이핑이 좋아요",
            "소프트웨어 업데이트로 버그가 수정됐습니다",
            "모바일 앱 UI/UX 개선 작업 중이에요",
        ],
        'travel': [
            "제주도 한라산 등반이 힘들지만 보람있었어요",
            "파리 에펠탑 야경이 정말 아름다웠습니다",
            "방콕 길거리 음식 투어가 재미있었어요",
            "일본 온천 료칸에서 힐링했습니다",
            "하와이 해변에서 서핑 배우는 중이에요",
            "스위스 알프스 설경이 환상적이었습니다",
            "베트남 하롱베이 크루즈 여행 추천해요",
            "뉴욕 타임스퀘어 야간 산책이 좋았어요",
            "발리 리조트에서 스파 받으니 천국이에요",
            "싱가포르 마리나베이샌즈 전망이 대단합니다",
            "그리스 산토리니 석양이 잊을 수 없어요",
            "런던 대영박물관 관람은 하루가 부족해요",
            "부산 해운대 해수욕장에서 휴가 보냈습니다",
            "캐나다 록키산맥 드라이브 코스가 멋졌어요",
            "이탈리아 베니스 곤돌라 타보세요",
            "호주 시드니 오페라하우스를 봤습니다",
            "두바이 부르즈칼리파 전망대 올라갔어요",
            "스페인 바르셀로나 가우디 건축물이 독특해요",
            "터키 이스탄불 그랜드 바자르에서 쇼핑했습니다",
            "프라하 구시가지 광장 야경이 아름다웠어요",
        ],
        'health': [
            "매일 아침 조깅으로 체력이 좋아졌어요",
            "헬스장에서 웨이트 트레이닝 시작했습니다",
            "요가 수업 듣고 나면 마음이 편안해져요",
            "단백질 쉐이크로 근육 회복 중이에요",
            "수영이 전신 운동으로 좋다고 합니다",
            "필라테스로 자세 교정 효과를 봤어요",
            "만보기 목표 달성하려고 걷기 운동해요",
            "스트레칭으로 유연성이 향상됐습니다",
            "자전거 타기가 무릎에 부담이 적어요",
            "플랭크 1분 버티기가 목표입니다",
            "홈트레이닝으로 집에서 운동하고 있어요",
            "등산하면 심폐 기능이 좋아진대요",
            "건강검진 결과 콜레스테롤 수치가 좋아졌어요",
            "명상으로 스트레스 관리를 합니다",
            "점심 후 산책이 소화에 도움이 돼요",
            "스쿼트 100개 챌린지 도전 중입니다",
            "비타민 영양제 꾸준히 먹고 있어요",
            "규칙적인 수면이 건강의 기본이에요",
            "크로스핏 수업이 강도가 높지만 재미있어요",
            "복근 운동으로 뱃살 빼는 중이에요",
        ],
        'entertainment': [
            "넷플릭스 신작 드라마 정주행했습니다",
            "최근 개봉한 마블 영화가 재미있었어요",
            "BTS 콘서트 티켓팅 성공했습니다",
            "무한도전 다시보기 하면서 웃었어요",
            "이번 시즌 오징어게임 기대됩니다",
            "클래식 음악회 가서 힐링했어요",
            "뮤지컬 오페라의 유령 공연 보고 왔습니다",
            "인디밴드 라이브 공연 분위기가 좋았어요",
            "요즘 K-POP 아이돌 노래가 중독성 있어요",
            "스포티파이로 플레이리스트 만들었습니다",
            "호러 영화 밤에 보니 무서웠어요",
            "유튜브에서 먹방 보는 게 취미에요",
            "로맨스 소설 읽으며 감성에 젖었습니다",
            "온라인 게임에서 레벨업 했어요",
            "웹툰 완결 보고 아쉬웠습니다",
            "팟캐스트 들으면서 출퇴근해요",
            "예능 프로그램 리얼 버라이어티가 재밌어요",
            "SF 영화 특수효과가 대단합니다",
            "아이유 새 앨범 음원 차트 1위래요",
            "코미디 영화 보면서 스트레스 풀었어요",
        ],
    }

    # 변형 패턴
    variations = [
        "{}", "{}.", "오늘 {}", "요즘 {}",
        "{} 좋네요", "{} 추천합니다", "역시 {}",
        "다들 {}", "{} 완전", "{} 진짜",
    ]

    reviews = []
    labels = []
    label_names = list(templates.keys())
    samples_per_topic = n_samples // len(label_names)

    for label_idx, (topic, topic_templates) in enumerate(templates.items()):
        for _ in range(samples_per_topic):
            template = np.random.choice(topic_templates)
            variation = np.random.choice(variations)
            review = variation.format(template)
            reviews.append(review)
            labels.append(label_idx)

    # 셔플
    indices = np.random.permutation(len(reviews))
    reviews = [reviews[i] for i in indices]
    labels = [labels[i] for i in indices]

    return reviews, np.array(labels), label_names


# ============================================================
# 2. 임베딩 생성
# ============================================================
def get_embeddings(texts, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
    """Sentence-BERT 임베딩 생성"""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  - 모델 로드: {model_name}")
        model = SentenceTransformer(model_name)
        print(f"  - {len(texts)}개 텍스트 임베딩 생성 중...")
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        return embeddings
    except ImportError:
        print("[ERROR] sentence-transformers가 설치되지 않았습니다.")
        print("pip install sentence-transformers")
        sys.exit(1)


# ============================================================
# 3. 딥 클러스터링 모델 정의
# ============================================================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    USE_TORCH = True
except ImportError:
    USE_TORCH = False
    print("[WARNING] PyTorch 미설치. K-Means 결과만 출력합니다.")


class Autoencoder(nn.Module):
    """딥 클러스터링용 Autoencoder"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], latent_dim=32):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class ClusteringLayer(nn.Module):
    """DEC/IDEC용 클러스터링 레이어"""
    def __init__(self, n_clusters, latent_dim, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.1)

    def forward(self, z):
        # Student's t-distribution
        dist = torch.cdist(z, self.cluster_centers, p=2) ** 2
        q = (1 + dist / self.alpha) ** (-(self.alpha + 1) / 2)
        q = q / q.sum(dim=1, keepdim=True)
        return q


def target_distribution(q):
    """타깃 분포 P 계산"""
    f = q.sum(dim=0)
    p = (q ** 2) / f
    p = p / p.sum(dim=1, keepdim=True)
    return p


class VaDE(nn.Module):
    """Variational Deep Embedding (VaDE)"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], latent_dim=32, n_clusters=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # GMM parameters
        self.pi = nn.Parameter(torch.ones(n_clusters) / n_clusters)  # 군집 사전 확률
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.1)  # 군집 평균
        self.logvar_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))  # 군집 분산

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def get_gamma(self, z):
        """군집 할당 확률 계산"""
        pi = torch.softmax(self.pi, dim=0)

        # 각 군집에 대한 로그 확률 계산
        log_p = []
        for c in range(self.n_clusters):
            mu_c = self.mu_c[c]
            logvar_c = self.logvar_c[c]

            # 가우시안 로그 확률
            log_p_c = -0.5 * (
                self.latent_dim * np.log(2 * np.pi) +
                logvar_c.sum() +
                ((z - mu_c) ** 2 / torch.exp(logvar_c)).sum(dim=1)
            )
            log_p.append(log_p_c + torch.log(pi[c] + 1e-10))

        log_p = torch.stack(log_p, dim=1)
        gamma = torch.softmax(log_p, dim=1)
        return gamma

    def loss_function(self, x, x_recon, mu, logvar, z):
        # 재구성 손실
        recon_loss = nn.MSELoss()(x_recon, x)

        # KL divergence
        gamma = self.get_gamma(z)
        pi = torch.softmax(self.pi, dim=0)

        # 간소화된 ELBO
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 군집 손실
        cluster_loss = -torch.mean(torch.sum(gamma * torch.log(pi + 1e-10), dim=1))

        return recon_loss + 0.1 * kl_loss + 0.1 * cluster_loss


# ============================================================
# 4. 학습 함수
# ============================================================
def pretrain_autoencoder(model, dataloader, epochs=30, lr=0.001, device='cpu'):
    """Autoencoder 사전학습"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    return model


def train_dec(autoencoder, cluster_layer, dataloader, epochs=50, lr=0.0001, device='cpu'):
    """DEC 학습"""
    autoencoder.to(device)
    cluster_layer.to(device)

    optimizer = optim.Adam(
        list(autoencoder.parameters()) + list(cluster_layer.parameters()),
        lr=lr
    )

    autoencoder.train()
    cluster_layer.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            _, z = autoencoder(x)
            q = cluster_layer(z)
            p = target_distribution(q.detach())

            loss = (p * (p.log() - q.log())).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, KL Loss: {total_loss/len(dataloader):.6f}")

    return autoencoder, cluster_layer


def train_idec(autoencoder, cluster_layer, dataloader, epochs=50, lr=0.0001, gamma=0.1, device='cpu'):
    """IDEC 학습 (재구성 손실 추가)"""
    autoencoder.to(device)
    cluster_layer.to(device)

    optimizer = optim.Adam(
        list(autoencoder.parameters()) + list(cluster_layer.parameters()),
        lr=lr
    )
    recon_criterion = nn.MSELoss()

    autoencoder.train()
    cluster_layer.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            x_recon, z = autoencoder(x)
            q = cluster_layer(z)
            p = target_distribution(q.detach())

            # KL + 재구성 손실
            kl_loss = (p * (p.log() - q.log())).sum(dim=1).mean()
            recon_loss = recon_criterion(x_recon, x)
            loss = kl_loss + gamma * recon_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    return autoencoder, cluster_layer


def train_vade(model, dataloader, epochs=50, lr=0.001, device='cpu'):
    """VaDE 학습"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            x_recon, mu, logvar, z = model(x)
            loss = model.loss_function(x, x_recon, mu, logvar, z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    return model


# ============================================================
# 5. 평가 함수
# ============================================================
def get_cluster_labels(model, cluster_layer, dataloader, method='dec', device='cpu'):
    """군집 레이블 추출"""
    model.eval()
    if cluster_layer:
        cluster_layer.eval()

    all_labels = []
    all_z = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)

            if method == 'vade':
                mu, _ = model.encode(x)
                gamma = model.get_gamma(mu)
                labels = gamma.argmax(dim=1)
                all_z.append(mu.cpu().numpy())
            else:
                _, z = model(x)
                q = cluster_layer(z)
                labels = q.argmax(dim=1)
                all_z.append(z.cpu().numpy())

            all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_z)


def evaluate_clustering(y_true, y_pred):
    """클러스터링 평가"""
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        silhouette_score, homogeneity_score, completeness_score
    )

    metrics = {
        'ARI': adjusted_rand_score(y_true, y_pred),
        'NMI': normalized_mutual_info_score(y_true, y_pred),
        'Homogeneity': homogeneity_score(y_true, y_pred),
        'Completeness': completeness_score(y_true, y_pred),
    }
    return metrics


# ============================================================
# 6. 시각화
# ============================================================
def plot_comparison_results(results_df, z_dict, labels_dict, y_true, label_names, output_path):
    """딥 클러스터링 비교 결과 시각화"""
    from sklearn.manifold import TSNE

    fig = plt.figure(figsize=(16, 10))

    # 상단: t-SNE 시각화
    methods = list(z_dict.keys())
    n_methods = len(methods)

    for idx, method in enumerate(methods):
        ax = fig.add_subplot(2, n_methods, idx + 1)

        z = z_dict[method]
        labels = labels_dict[method]

        # t-SNE (샘플링하여 속도 향상)
        sample_size = min(3000, len(z))
        sample_idx = np.random.choice(len(z), sample_size, replace=False)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        z_2d = tsne.fit_transform(z[sample_idx])

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i in range(labels[sample_idx].max() + 1):
            mask = labels[sample_idx] == i
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[colors[i]], s=5, alpha=0.5, label=f'군집 {i}')

        ax.set_title(f'{method}', fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)

    # 하단: 성능 비교 바 차트
    ax_bar = fig.add_subplot(2, 1, 2)

    metrics = ['ARI', 'NMI', 'Homogeneity', 'Completeness']
    x = np.arange(len(metrics))
    width = 0.15

    for idx, method in enumerate(results_df['Method']):
        values = [results_df[results_df['Method'] == method][m].values[0] for m in metrics]
        ax_bar.bar(x + idx * width, values, width, label=method)

    ax_bar.set_xlabel('평가 지표')
    ax_bar.set_ylabel('점수')
    ax_bar.set_title('딥 클러스터링 알고리즘 성능 비교', fontsize=12, fontweight='bold')
    ax_bar.set_xticks(x + width * (len(results_df) - 1) / 2)
    ax_bar.set_xticklabels(metrics)
    ax_bar.legend(loc='upper right')
    ax_bar.set_ylim(0, 1)
    ax_bar.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n시각화 저장: {output_path}")


# ============================================================
# 단일 실험 수행 함수
# ============================================================
def run_experiment(reviews, y_true, label_names, experiment_name, output_suffix):
    """하나의 데이터셋에 대해 모든 알고리즘 실험 수행"""
    from sklearn.cluster import KMeans

    print(f"\n{'='*70}")
    print(f"실험: {experiment_name}")
    print(f"{'='*70}")

    print(f"\n[1] 데이터 정보")
    print(f"  - 샘플 수: {len(reviews)}")
    print(f"  - 주제: {label_names}")
    print(f"  - 주제별 분포: {np.bincount(y_true)}")

    # 임베딩 생성
    print("\n[2] Sentence-BERT 임베딩 생성")
    embeddings = get_embeddings(reviews)
    print(f"  - 임베딩 차원: {embeddings.shape}")

    # 결과 저장용
    results = []
    z_dict = {}
    labels_dict = {}

    # K-Means (베이스라인)
    print("\n[3] K-Means 클러스터링 (베이스라인)")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(embeddings)

    metrics_kmeans = evaluate_clustering(y_true, labels_kmeans)
    metrics_kmeans['Method'] = 'KMeans'
    results.append(metrics_kmeans)
    z_dict['KMeans'] = embeddings
    labels_dict['KMeans'] = labels_kmeans

    print(f"  - ARI: {metrics_kmeans['ARI']:.4f}")
    print(f"  - NMI: {metrics_kmeans['NMI']:.4f}")

    if not USE_TORCH:
        print("\n[INFO] PyTorch 미설치로 딥 클러스터링 실험을 건너뜁니다.")
        results_df = pd.DataFrame(results)
        return results_df, z_dict, labels_dict

    # PyTorch 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  - 디바이스: {device}")

    # 데이터 준비
    X_tensor = torch.FloatTensor(embeddings)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    dataloader_eval = DataLoader(dataset, batch_size=512, shuffle=False)

    input_dim = embeddings.shape[1]
    n_clusters = 5
    latent_dim = 32

    # DEC
    print("\n[4] DEC (Deep Embedded Clustering)")
    print("  - Autoencoder 사전학습...")
    ae_dec = Autoencoder(input_dim, hidden_dims=[256, 128, 64], latent_dim=latent_dim)
    ae_dec = pretrain_autoencoder(ae_dec, dataloader, epochs=30, device=device)

    ae_dec.eval()
    with torch.no_grad():
        _, z_init = ae_dec(X_tensor.to(device))
        z_init = z_init.cpu().numpy()

    kmeans_init = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_init.fit(z_init)

    cluster_layer_dec = ClusteringLayer(n_clusters, latent_dim)
    cluster_layer_dec.cluster_centers.data = torch.FloatTensor(kmeans_init.cluster_centers_)

    print("  - DEC 학습...")
    ae_dec, cluster_layer_dec = train_dec(ae_dec, cluster_layer_dec, dataloader, epochs=50, device=device)

    labels_dec, z_dec = get_cluster_labels(ae_dec, cluster_layer_dec, dataloader_eval, method='dec', device=device)
    metrics_dec = evaluate_clustering(y_true, labels_dec)
    metrics_dec['Method'] = 'DEC'
    results.append(metrics_dec)
    z_dict['DEC'] = z_dec
    labels_dict['DEC'] = labels_dec

    print(f"  - ARI: {metrics_dec['ARI']:.4f}")
    print(f"  - NMI: {metrics_dec['NMI']:.4f}")

    # IDEC
    print("\n[5] IDEC (Improved DEC)")
    print("  - Autoencoder 사전학습...")
    ae_idec = Autoencoder(input_dim, hidden_dims=[256, 128, 64], latent_dim=latent_dim)
    ae_idec = pretrain_autoencoder(ae_idec, dataloader, epochs=30, device=device)

    ae_idec.eval()
    with torch.no_grad():
        _, z_init = ae_idec(X_tensor.to(device))
        z_init = z_init.cpu().numpy()

    kmeans_init = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_init.fit(z_init)

    cluster_layer_idec = ClusteringLayer(n_clusters, latent_dim)
    cluster_layer_idec.cluster_centers.data = torch.FloatTensor(kmeans_init.cluster_centers_)

    print("  - IDEC 학습...")
    ae_idec, cluster_layer_idec = train_idec(ae_idec, cluster_layer_idec, dataloader, epochs=50, gamma=0.1, device=device)

    labels_idec, z_idec = get_cluster_labels(ae_idec, cluster_layer_idec, dataloader_eval, method='idec', device=device)
    metrics_idec = evaluate_clustering(y_true, labels_idec)
    metrics_idec['Method'] = 'IDEC'
    results.append(metrics_idec)
    z_dict['IDEC'] = z_idec
    labels_dict['IDEC'] = labels_idec

    print(f"  - ARI: {metrics_idec['ARI']:.4f}")
    print(f"  - NMI: {metrics_idec['NMI']:.4f}")

    # VaDE
    print("\n[6] VaDE (Variational Deep Embedding)")
    print("  - VaDE 학습...")
    vade = VaDE(input_dim, hidden_dims=[256, 128, 64], latent_dim=latent_dim, n_clusters=n_clusters)

    ae_temp = Autoencoder(input_dim, hidden_dims=[256, 128, 64], latent_dim=latent_dim)
    ae_temp = pretrain_autoencoder(ae_temp, dataloader, epochs=20, device=device)
    ae_temp.eval()
    with torch.no_grad():
        _, z_init = ae_temp(X_tensor.to(device))
        z_init = z_init.cpu().numpy()

    kmeans_init = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_init.fit(z_init)
    vade.mu_c.data = torch.FloatTensor(kmeans_init.cluster_centers_)

    vade = train_vade(vade, dataloader, epochs=50, device=device)

    labels_vade, z_vade = get_cluster_labels(vade, None, dataloader_eval, method='vade', device=device)
    metrics_vade = evaluate_clustering(y_true, labels_vade)
    metrics_vade['Method'] = 'VaDE'
    results.append(metrics_vade)
    z_dict['VaDE'] = z_vade
    labels_dict['VaDE'] = labels_vade

    print(f"  - ARI: {metrics_vade['ARI']:.4f}")
    print(f"  - NMI: {metrics_vade['NMI']:.4f}")

    # 결과 정리
    results_df = pd.DataFrame(results)
    results_df = results_df[['Method', 'ARI', 'NMI', 'Homogeneity', 'Completeness']]
    results_df = results_df.sort_values('ARI', ascending=False).reset_index(drop=True)

    print(f"\n[결과 요약] {experiment_name}")
    print("-" * 60)
    print(results_df.to_string(index=False))

    # CSV 저장
    csv_path = os.path.join(OUTPUT_DIR, f'deep_clustering_{output_suffix}.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n결과 저장: {csv_path}")

    # 시각화
    plot_path = os.path.join(OUTPUT_DIR, f'deep_clustering_{output_suffix}.png')
    plot_comparison_results(results_df, z_dict, labels_dict, y_true, label_names, plot_path)

    return results_df, z_dict, labels_dict


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("딥 클러스터링 비교 실습: 데이터 특성에 따른 성능 차이 분석")
    print("=" * 70)
    print("""
핵심 질문: 딥 클러스터링은 항상 K-Means보다 좋은가?
→ 데이터 특성(임베딩 유사성 vs 군집화 목적 유사성)에 따라 다르다!

실험 A: 동일 도메인 내 세부 주제 (쇼핑 리뷰)
  - 임베딩이 포착하는 유사성: '쇼핑/구매' 경험
  - 군집화 목적의 유사성: 배송/품질/가격/서비스/불만
  - 예상: 임베딩 공간에서 군집이 잘 분리되지 않음 → 딥 클러스터링 효과 제한적

실험 B: 완전히 다른 도메인
  - 임베딩이 포착하는 유사성: 각 도메인의 고유한 어휘와 맥락
  - 군집화 목적의 유사성: 음식/기술/여행/건강/엔터테인먼트
  - 예상: 임베딩 공간에서 군집이 잘 분리됨 → K-Means도 충분히 효과적
""")

    # ================================================================
    # 실험 A: 동일 도메인 내 세부 주제
    # ================================================================
    print("\n" + "#" * 70)
    print("# 실험 A: 동일 도메인(쇼핑) 내 세부 주제")
    print("#" * 70)

    reviews_A, y_true_A, labels_A = generate_review_topics(n_samples=15000)
    results_A, z_dict_A, labels_dict_A = run_experiment(
        reviews_A, y_true_A, labels_A,
        experiment_name="실험 A: 쇼핑 리뷰 주제 분류",
        output_suffix="exp_A_review_topics"
    )

    # ================================================================
    # 실험 B: 완전히 다른 도메인
    # ================================================================
    print("\n" + "#" * 70)
    print("# 실험 B: 완전히 다른 도메인")
    print("#" * 70)

    reviews_B, y_true_B, labels_B = generate_different_domains(n_samples=15000)
    results_B, z_dict_B, labels_dict_B = run_experiment(
        reviews_B, y_true_B, labels_B,
        experiment_name="실험 B: 다른 도메인 분류",
        output_suffix="exp_B_different_domains"
    )

    # ================================================================
    # 두 실험 비교 분석
    # ================================================================
    print("\n" + "=" * 70)
    print("두 실험 비교 분석: 데이터 특성이 딥 클러스터링 효과에 미치는 영향")
    print("=" * 70)

    # 비교 테이블 생성
    comparison_data = []
    for method in ['KMeans', 'DEC', 'IDEC', 'VaDE']:
        row_A = results_A[results_A['Method'] == method]
        row_B = results_B[results_B['Method'] == method]

        if len(row_A) > 0 and len(row_B) > 0:
            comparison_data.append({
                'Method': method,
                'ARI_A (리뷰 주제)': row_A['ARI'].values[0],
                'ARI_B (다른 도메인)': row_B['ARI'].values[0],
                'NMI_A': row_A['NMI'].values[0],
                'NMI_B': row_B['NMI'].values[0],
            })

    comparison_df = pd.DataFrame(comparison_data)

    print("\n[ARI 비교]")
    print("-" * 70)
    print(f"{'Method':<10} {'실험A (리뷰 주제)':>18} {'실험B (다른 도메인)':>20} {'차이':>12}")
    print("-" * 70)
    for _, row in comparison_df.iterrows():
        diff = row['ARI_B (다른 도메인)'] - row['ARI_A (리뷰 주제)']
        print(f"{row['Method']:<10} {row['ARI_A (리뷰 주제)']:>18.4f} {row['ARI_B (다른 도메인)']:>20.4f} {diff:>+12.4f}")
    print("-" * 70)

    # 비교 결과 저장
    comparison_path = os.path.join(OUTPUT_DIR, 'deep_clustering_comparison_both.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n비교 결과 저장: {comparison_path}")

    # ================================================================
    # 비교 시각화
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = comparison_df['Method'].tolist()
    x = np.arange(len(methods))
    width = 0.35

    # ARI 비교
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, comparison_df['ARI_A (리뷰 주제)'], width, label='실험 A: 리뷰 주제', color='steelblue')
    bars2 = ax1.bar(x + width/2, comparison_df['ARI_B (다른 도메인)'], width, label='실험 B: 다른 도메인', color='coral')
    ax1.set_xlabel('알고리즘')
    ax1.set_ylabel('ARI (Adjusted Rand Index)')
    ax1.set_title('데이터 특성에 따른 군집화 성능 비교', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.set_ylim(0, max(comparison_df['ARI_B (다른 도메인)'].max() * 1.2, 0.2))
    ax1.grid(True, alpha=0.3, axis='y')

    # 값 표시
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # NMI 비교
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, comparison_df['NMI_A'], width, label='실험 A: 리뷰 주제', color='steelblue')
    bars4 = ax2.bar(x + width/2, comparison_df['NMI_B'], width, label='실험 B: 다른 도메인', color='coral')
    ax2.set_xlabel('알고리즘')
    ax2.set_ylabel('NMI (Normalized Mutual Information)')
    ax2.set_title('NMI 비교', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.set_ylim(0, max(comparison_df['NMI_B'].max() * 1.2, 0.3))
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    comparison_plot_path = os.path.join(OUTPUT_DIR, 'deep_clustering_comparison_both.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"비교 시각화 저장: {comparison_plot_path}")

    # ================================================================
    # 핵심 통찰 정리
    # ================================================================
    print("\n" + "=" * 70)
    print("핵심 통찰: 딥 클러스터링은 언제 효과적인가?")
    print("=" * 70)

    # 결과에서 통찰 도출
    kmeans_A = comparison_df[comparison_df['Method'] == 'KMeans']['ARI_A (리뷰 주제)'].values[0]
    kmeans_B = comparison_df[comparison_df['Method'] == 'KMeans']['ARI_B (다른 도메인)'].values[0]
    best_deep_A = comparison_df[comparison_df['Method'].isin(['DEC', 'IDEC'])]['ARI_A (리뷰 주제)'].max()
    best_deep_B = comparison_df[comparison_df['Method'].isin(['DEC', 'IDEC'])]['ARI_B (다른 도메인)'].max()

    summary = f"""
[실험 결과 요약]

┌─────────────────────────────────────────────────────────────────────┐
│ 실험 A: 동일 도메인 내 세부 주제 (쇼핑 리뷰)                          │
├─────────────────────────────────────────────────────────────────────┤
│ - K-Means ARI: {kmeans_A:.4f}                                          │
│ - 최고 딥 클러스터링 ARI: {best_deep_A:.4f}                             │
│ - 임베딩이 '쇼핑 경험'이라는 공통 도메인에 집중                         │
│ - 세부 주제(배송/품질/가격)는 임베딩 공간에서 잘 분리되지 않음          │
│ - 딥 클러스터링으로도 한계가 있음                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 실험 B: 완전히 다른 도메인                                            │
├─────────────────────────────────────────────────────────────────────┤
│ - K-Means ARI: {kmeans_B:.4f}                                          │
│ - 최고 딥 클러스터링 ARI: {best_deep_B:.4f}                             │
│ - 각 도메인(음식/기술/여행 등)은 고유한 어휘와 맥락 보유                │
│ - 임베딩 공간에서 이미 잘 분리되어 있음                                │
│ - K-Means만으로도 효과적 → 딥 클러스터링의 추가 이점 제한적            │
└─────────────────────────────────────────────────────────────────────┘

[핵심 통찰]

"임베딩이 포착하는 유사성"과 "군집화 목적의 유사성"이 일치하느냐가 핵심이다.

1. 두 유사성이 일치할 때 (실험 B):
   - 임베딩 공간에서 군집이 자연스럽게 형성됨
   - K-Means만으로도 충분히 좋은 결과
   - 딥 클러스터링의 추가 복잡성이 불필요

2. 두 유사성이 불일치할 때 (실험 A):
   - 임베딩 공간에서 군집 구분이 어려움
   - 딥 클러스터링이 잠재 표현을 재학습하여 도움될 수 있음
   - 그러나 근본적으로 임베딩이 군집 구조를 포착하지 못하면 한계 존재

[실무 적용 지침]

✓ 먼저 간단한 K-Means로 시도하라
✓ 임베딩이 군집 목적과 잘 맞는지 확인하라 (t-SNE 시각화 활용)
✓ K-Means 결과가 불만족스러울 때만 딥 클러스터링 고려
✓ 도메인 특화 임베딩 모델 사용 검토 (Fine-tuned BERT 등)
✓ VaDE는 학습 불안정성으로 실무 적용 시 주의 필요
"""
    print(summary)

    print("=" * 70)
    print("실습 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
