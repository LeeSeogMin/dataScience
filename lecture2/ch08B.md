# 8장 B: 생성 모델 기반 데이터 분석 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: 간단한 GAN으로 합성 데이터 생성

### 제공 코드 실행 결과 해설

Adult Income 데이터(1,000명, 4개 연속형 변수 + 소득 타겟)에 간단한 GAN을 200 에포크 학습한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| D Loss | 1.19 전후 | 이론적 최적값(ln4 ≈ 1.39)에 가까워지지만 완전 수렴하지 않음 |
| G Loss | 1.08 전후 | Generator가 Discriminator를 어느 정도 속이는 수준 |
| KS p-value | 전부 0.05 미만 | 간단한 GAN은 복잡한 분포(편포, 다봉)를 충분히 학습하지 못함 |
| TSTR Ratio | 81% 전후 | 합성 데이터의 분포 왜곡으로 ML 효용성이 제한적 |

핵심 코드 구조:

```python
# Generator: 10차원 노이즈 → 64 → 128 → 5차원 데이터 (4특성 + 1타겟)
generator = Generator(latent_dim=10, output_dim=5)
# Discriminator: 5차원 데이터 → 128 → 64 → 1 (진짜/가짜 확률)
discriminator = Discriminator(input_dim=5)
# BCELoss: 이진 교차 엔트로피로 진짜/가짜 판별
criterion = nn.BCELoss()
```

간단한 GAN으로는 분포 보존과 ML 효용성 모두 제한적인 이유: (1) 정형 데이터의 혼합 분포를 다루는 특별한 기법이 없고, (2) Generator 구조가 단순하여 복잡한 조건부 의존성을 학습하기 어렵다. CTGAN은 Mode-specific Normalization, Conditional Vector 등으로 이를 해결한다.

### 프롬프트 1 모범 구현: GAN 에포크 수 변경

```python
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"

df = pd.read_csv(INPUT_DIR / "adult_income.csv")
continuous_cols = ['age', 'education_num', 'hours_per_week', 'capital_gain']
target_col = 'income'

X = df[continuous_cols].values.astype(float)
y = df[target_col].values.astype(int)
train_data = df[continuous_cols + [target_col]].values.astype(float)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

latent_dim = 10
epoch_list = [50, 100, 200, 500]

print(f"{'epochs':<10} {'D Loss':<12} {'G Loss':<12} {'TSTR Ratio':<12}")
print("-" * 46)

for epochs in epoch_list:
    set_seed(42)
    input_dim = train_scaled.shape[1]
    generator = Generator(latent_dim, input_dim)
    discriminator = Discriminator(input_dim)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    X_tensor = torch.FloatTensor(train_scaled)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for real_data, in loader:
            batch_size = real_data.size(0)

            d_optimizer.zero_grad()
            d_real_loss = criterion(discriminator(real_data), torch.ones(batch_size, 1))
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            d_fake_loss = criterion(discriminator(fake_data.detach()), torch.zeros(batch_size, 1))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            g_loss = criterion(discriminator(fake_data), torch.ones(batch_size, 1))
            g_loss.backward()
            g_optimizer.step()

    # 합성 데이터 생성
    generator.eval()
    with torch.no_grad():
        z = torch.randn(len(train_scaled), latent_dim)
        synthetic_scaled = generator(z).numpy()
    synthetic_data = scaler.inverse_transform(synthetic_scaled)

    X_syn = synthetic_data[:, :len(continuous_cols)]
    y_syn = (synthetic_data[:, -1] >= 0.5).astype(int)

    # TSTR 평가
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.3, random_state=42)
    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(X_train_r, y_train_r)
    acc_trtr = accuracy_score(y_test_r, clf_trtr.predict(X_test_r))

    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_tstr.fit(X_syn, y_syn)
    acc_tstr = accuracy_score(y_test_r, clf_tstr.predict(X_test_r))

    tstr_ratio = acc_tstr / acc_trtr
    print(f"{epochs:<10} {d_loss.item():<12.4f} {g_loss.item():<12.4f} {tstr_ratio:<12.2%}")
```

기대 결과 해석:
- epochs=50: 학습 부족으로 합성 데이터 품질이 낮고, TSTR Ratio가 가장 낮음
- epochs=200: 기본 설정. 적당한 수준의 학습
- epochs=500: 더 학습하지만 간단한 GAN의 구조적 한계로 대폭 개선되지 않을 수 있음. D Loss와 G Loss 균형이 무너질 수도 있음
- 핵심: 에포크만 늘린다고 해서 간단한 GAN의 품질이 무한히 개선되지 않는다. 아키텍처 개선(CTGAN)이 더 근본적인 해결책

### 프롬프트 2 모범 구현: 합성 데이터 분포 비교 시각화

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy import stats

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "8-2-gan-practice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

set_seed(42)
df = pd.read_csv(INPUT_DIR / "adult_income.csv")
continuous_cols = ['age', 'education_num', 'hours_per_week', 'capital_gain']
target_col = 'income'

X = df[continuous_cols].values.astype(float)
train_data = df[continuous_cols + [target_col]].values.astype(float)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

# GAN 학습
latent_dim = 10
input_dim = train_scaled.shape[1]
generator = Generator(latent_dim, input_dim)
discriminator = Discriminator(input_dim)
g_opt = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_opt = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

loader = DataLoader(TensorDataset(torch.FloatTensor(train_scaled)), batch_size=64, shuffle=True)
for epoch in range(200):
    for real_data, in loader:
        bs = real_data.size(0)
        d_opt.zero_grad()
        d_loss = criterion(discriminator(real_data), torch.ones(bs, 1)) + \
                 criterion(discriminator(generator(torch.randn(bs, latent_dim)).detach()), torch.zeros(bs, 1))
        d_loss.backward(); d_opt.step()

        g_opt.zero_grad()
        g_loss = criterion(discriminator(generator(torch.randn(bs, latent_dim))), torch.ones(bs, 1))
        g_loss.backward(); g_opt.step()

# 합성 데이터 생성
generator.eval()
with torch.no_grad():
    synthetic_scaled = generator(torch.randn(len(train_scaled), latent_dim)).numpy()
synthetic_data = scaler.inverse_transform(synthetic_scaled)
X_syn = synthetic_data[:, :len(continuous_cols)]

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, col in enumerate(continuous_cols):
    ax = axes[idx]
    ax.hist(X[:, idx], bins=30, alpha=0.5, label='Real', density=True, color='steelblue')
    ax.hist(X_syn[:, idx], bins=30, alpha=0.5, label='Synthetic', density=True, color='coral')

    ks_stat, ks_pval = stats.ks_2samp(X[:, idx], X_syn[:, idx])
    ax.set_title(f'{col}\n(KS p-value: {ks_pval:.4f})')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Real vs Synthetic Feature Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distribution_comparison_with_ks.png", dpi=150)
plt.show()
print("시각화 저장 완료")
```

기대 결과: 모든 변수에서 KS p-value < 0.05. 특히 hours_per_week와 capital_gain에서 분포 차이가 크게 나타남. 간단한 GAN은 편포와 제한된 범위를 가진 분포를 재현하기 어렵다.

### 프롬프트 3 모범 구현: latent_dim 변경

```python
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"

df = pd.read_csv(INPUT_DIR / "adult_income.csv")
continuous_cols = ['age', 'education_num', 'hours_per_week', 'capital_gain']
target_col = 'income'

X = df[continuous_cols].values.astype(float)
y = df[target_col].values.astype(int)
train_data = df[continuous_cols + [target_col]].values.astype(float)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

latent_dims = [5, 10, 20, 50]

print(f"{'latent_dim':<14} {'TSTR Ratio':<14} {'KS p-value 평균':<18}")
print("-" * 46)

for latent_dim in latent_dims:
    set_seed(42)
    input_dim = train_scaled.shape[1]
    generator = Generator(latent_dim, input_dim)
    discriminator = Discriminator(input_dim)
    g_opt = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    loader = DataLoader(TensorDataset(torch.FloatTensor(train_scaled)), batch_size=64, shuffle=True)
    for epoch in range(200):
        for real_data, in loader:
            bs = real_data.size(0)
            d_opt.zero_grad()
            d_loss = criterion(discriminator(real_data), torch.ones(bs, 1)) + \
                     criterion(discriminator(generator(torch.randn(bs, latent_dim)).detach()), torch.zeros(bs, 1))
            d_loss.backward(); d_opt.step()

            g_opt.zero_grad()
            g_loss = criterion(discriminator(generator(torch.randn(bs, latent_dim))), torch.ones(bs, 1))
            g_loss.backward(); g_opt.step()

    generator.eval()
    with torch.no_grad():
        synthetic_scaled = generator(torch.randn(len(train_scaled), latent_dim)).numpy()
    synthetic_data = scaler.inverse_transform(synthetic_scaled)
    X_syn = synthetic_data[:, :len(continuous_cols)]
    y_syn = (synthetic_data[:, -1] >= 0.5).astype(int)

    # KS p-value 평균
    ks_pvals = []
    for i in range(len(continuous_cols)):
        _, pval = stats.ks_2samp(X[:, i], X_syn[:, i])
        ks_pvals.append(pval)
    avg_ks = np.mean(ks_pvals)

    # TSTR
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.3, random_state=42)
    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(X_train_r, y_train_r)
    acc_trtr = accuracy_score(y_test_r, clf_trtr.predict(X_test_r))

    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_tstr.fit(X_syn, y_syn)
    acc_tstr = accuracy_score(y_test_r, clf_tstr.predict(X_test_r))

    tstr_ratio = acc_tstr / acc_trtr
    print(f"{latent_dim:<14} {tstr_ratio:<14.2%} {avg_ks:<18.6f}")
```

기대 결과 해석:
- latent_dim=5: 잠재공간이 너무 좁아 데이터 다양성이 제한될 수 있음
- latent_dim=10: 기본값. 적절한 복잡도
- latent_dim=20, 50: 잠재공간이 넓어지면 Generator가 더 다양한 데이터를 생성할 수 있지만, 학습이 어려워질 수 있음
- 핵심: 지나치게 큰 잠재 차원은 노이즈에 민감해지고 학습 불안정성을 유발한다. 데이터 차원(5)보다 과도하게 큰 잠재 차원은 비효율적

---

## 실습 2 해설: VAE 데이터 증강과 이상치 탐지

### 제공 코드 실행 결과 해설

신용카드 사기 거래 데이터(정상 900, 사기 100)에서 VAE를 활용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| Baseline Recall | 0.67 전후 | 사기 30개로는 패턴 학습이 부족 |
| Augmented Recall | 0.83 전후 | VAE가 200개 합성 사기를 추가하여 학습 데이터 보강 |
| F1 개선도 | +11~12% | Recall 개선이 F1 향상으로 이어짐 |
| 오탐률 (FPR) | 5% 전후 | 95% 백분위수 임계값의 설계상 정상 5%가 이상치로 판정 |
| 미탐률 (FNR) | 49% 전후 | 사기의 절반이 정상 범위 내 재구성 오차를 보임 |

핵심 코드 구조:

```python
# VAE 증강: 원본 사기 → 잠재공간 인코딩 → 분포 파악 → 새 잠재벡터 샘플링 → 디코딩
mu, logvar = vae.encode(X_fraud_tensor)
z_mean = mu.mean(dim=0)
z_std = mu.std(dim=0) + 0.1  # 약간의 여유로 다양성 확보
z = z_mean + z_std * torch.randn(n_samples, vae.latent_dim)
synthetic = vae.decode(z).numpy()
```

VAE 이상치 탐지의 미탐률이 49%인 이유: 사기 거래 중 일부는 정상과 유사한 패턴을 보여 재구성 오차가 낮다. VAE는 "형태가 다른" 이상치는 잘 탐지하지만, "값만 다른" 이상치는 놓칠 수 있다. 따라서 VAE 단독이 아닌 규칙 기반 + 지도학습 + VAE를 결합한 다층 방어가 실무에서 필수적이다.

### 프롬프트 4 모범 구현: VAE 합성 샘플 수 변경

```python
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    return F.mse_loss(recon_x, x, reduction='sum') - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"

set_seed(42)
df = pd.read_csv(INPUT_DIR / "credit_fraud.csv")
feature_cols = [c for c in df.columns if c.startswith("feature_")]
X = df[feature_cols].values.astype(np.float32)
y = df["fraud"].values.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

X_normal = X_train_full[y_train_full == 0]
X_fraud_full = X_train_full[y_train_full == 1]

n_fraud_limited = 30
np.random.seed(42)
fraud_idx = np.random.choice(len(X_fraud_full), n_fraud_limited, replace=False)
X_fraud_limited = X_fraud_full[fraud_idx]

X_train = np.vstack([X_normal, X_fraud_limited])
y_train = np.hstack([np.zeros(len(X_normal)), np.ones(n_fraud_limited)])

# VAE 학습 (사기 샘플만)
set_seed(42)
vae = VAE(X_scaled.shape[1], latent_dim=2)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
X_fraud_tensor = torch.FloatTensor(X_fraud_limited)
loader = DataLoader(TensorDataset(X_fraud_tensor, X_fraud_tensor), batch_size=32, shuffle=True)

for epoch in range(100):
    vae.train()
    for bx, _ in loader:
        optimizer.zero_grad()
        recon, mu, logvar = vae(bx)
        loss = vae_loss(recon, bx, mu, logvar)
        loss.backward()
        optimizer.step()

# 합성 샘플 수 변경 실험
sample_counts = [50, 100, 200, 500]

print(f"{'합성 샘플 수':<14} {'Accuracy':<12} {'F1-Score':<12} {'Recall':<12}")
print("-" * 50)

for n_samples in sample_counts:
    vae.eval()
    with torch.no_grad():
        mu, logvar = vae.encode(X_fraud_tensor)
        z_mean = mu.mean(dim=0)
        z_std = mu.std(dim=0) + 0.1
        z = z_mean + z_std * torch.randn(n_samples, vae.latent_dim)
        synthetic_fraud = vae.decode(z).numpy()

    X_aug = np.vstack([X_train, synthetic_fraud])
    y_aug = np.hstack([y_train, np.ones(n_samples)])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_aug, y_aug)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"{n_samples:<14} {acc:<12.4f} {f1:<12.4f} {rec:<12.4f}")
```

기대 결과 해석:
- n_samples=50: 소량의 증강으로 약간의 개선
- n_samples=200: 기본 설정. 원본의 약 7배로 좋은 균형
- n_samples=500: 합성 데이터가 지나치게 많으면 합성 데이터의 노이즈가 학습을 방해할 수 있음
- 핵심: 합성 샘플 수에는 최적 구간이 있다. 원본의 3-10배 정도가 일반적으로 적절하며, 너무 많으면 오히려 성능이 저하될 수 있다

### 프롬프트 5 모범 구현: 이상치 탐지 임계값 변경

```python
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    return F.mse_loss(recon_x, x, reduction='sum') - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"

set_seed(42)
df = pd.read_csv(INPUT_DIR / "credit_fraud.csv")
feature_cols = [c for c in df.columns if c.startswith("feature_")]
X = df[feature_cols].values.astype(np.float32)
y = df["fraud"].values.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 정상 데이터만으로 VAE 학습
X_normal = X_scaled[y == 0]
vae = VAE(X_scaled.shape[1], latent_dim=2)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
loader = DataLoader(TensorDataset(torch.FloatTensor(X_normal), torch.FloatTensor(X_normal)),
                    batch_size=32, shuffle=True)

for epoch in range(50):
    vae.train()
    for bx, _ in loader:
        optimizer.zero_grad()
        recon, mu, logvar = vae(bx)
        loss = vae_loss(recon, bx, mu, logvar)
        loss.backward()
        optimizer.step()

# 재구성 오차 계산
vae.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    recon, _, _ = vae(X_tensor)
    recon_errors = ((X_tensor - recon) ** 2).mean(dim=1).numpy()

# 임계값 변경 실험
percentiles = [90, 95, 99]

print(f"{'임계값 (백분위수)':<20} {'오탐률 (FPR)':<16} {'미탐률 (FNR)':<16}")
print("-" * 52)

for pct in percentiles:
    threshold = np.percentile(recon_errors[y == 0], pct)
    predictions = (recon_errors > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print(f"{pct}%{'':<17} {fpr:<16.2%} {fnr:<16.2%}")
```

기대 결과 해석:
- 90% 백분위수: 오탐률 약 10%, 미탐률 더 낮음. 더 많이 잡지만 거짓 경보도 많음
- 95% 백분위수: 기본 설정. 오탐률 5%, 미탐률 49% 전후
- 99% 백분위수: 오탐률 1%로 매우 낮지만, 미탐률이 크게 올라감
- 핵심: 임계값을 낮추면 더 많이 탐지(미탐률 감소)하지만 오탐도 증가. 실무에서는 검토 인력 용량과 사기 한 건당 손실을 고려하여 결정

### 프롬프트 6 모범 구현: 잠재공간에서 합성 사기 시각화

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    return F.mse_loss(recon_x, x, reduction='sum') - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "8-3-vae-practice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

set_seed(42)
df = pd.read_csv(INPUT_DIR / "credit_fraud.csv")
feature_cols = [c for c in df.columns if c.startswith("feature_")]
X = df[feature_cols].values.astype(np.float32)
y = df["fraud"].values.astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 사기 샘플만으로 VAE 학습
X_fraud = X_scaled[y == 1][:30]
vae = VAE(X_scaled.shape[1], latent_dim=2)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
loader = DataLoader(TensorDataset(torch.FloatTensor(X_fraud), torch.FloatTensor(X_fraud)),
                    batch_size=32, shuffle=True)

for epoch in range(100):
    vae.train()
    for bx, _ in loader:
        optimizer.zero_grad()
        recon, mu, logvar = vae(bx)
        loss = vae_loss(recon, bx, mu, logvar)
        loss.backward()
        optimizer.step()

# 합성 사기 생성
vae.eval()
with torch.no_grad():
    mu_fraud, _ = vae.encode(torch.FloatTensor(X_fraud))
    z_mean = mu_fraud.mean(dim=0)
    z_std = mu_fraud.std(dim=0) + 0.1
    z_syn = z_mean + z_std * torch.randn(200, vae.latent_dim)

# 잠재공간 좌표 추출
with torch.no_grad():
    # 원본 전체
    mu_all, _ = vae.encode(torch.FloatTensor(X_scaled))
    z_all = mu_all.numpy()
    # 합성 사기
    z_synthetic = z_syn.numpy()

# 시각화
fig, ax = plt.subplots(figsize=(10, 8))

# 원본 정상 (회색 원)
mask_normal = y == 0
ax.scatter(z_all[mask_normal, 0], z_all[mask_normal, 1],
           c='gray', marker='o', alpha=0.3, s=20, label='Original Normal')

# 원본 사기 (검정 X)
mask_fraud = y == 1
ax.scatter(z_all[mask_fraud, 0], z_all[mask_fraud, 1],
           c='black', marker='x', alpha=0.8, s=50, label='Original Fraud')

# 합성 사기 (빨간 삼각형)
ax.scatter(z_synthetic[:, 0], z_synthetic[:, 1],
           c='red', marker='^', alpha=0.6, s=40, label='Synthetic Fraud')

ax.set_xlabel('Latent Dimension 1')
ax.set_ylabel('Latent Dimension 2')
ax.set_title('VAE Latent Space: Original vs Synthetic Fraud')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "latent_space_with_synthetic.png", dpi=150)
plt.show()
print("시각화 저장 완료")
```

기대 결과:
- 합성 사기(빨간 삼각형)가 원본 사기(검정 X) 주변에 분포해야 좋은 증강
- 합성 사기가 정상 영역(회색 원)에 침범하면, 분류기가 정상을 사기로 오인(FP 증가)
- z_std에 +0.1 여유를 준 이유: 원본과 완전 동일하면 다양성이 없고, 너무 크면 정상 영역 침범

---

## 실습 3 해설: Diffusion Model 불확실성 정량화

### 제공 코드 실행 결과 해설

월별 매출 시계열(60개월)에 Diffusion Model을 적용하여 불확실성을 정량화한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 중앙값 | 110 전후 | 과거 12개월 패턴을 조건으로 한 중립 예측 |
| 불확실성 범위 | 약 50 (95%ile - 5%ile) | Diffusion의 확률적 샘플링이 다양한 미래를 생성 |
| 상대적 불확실성 | 약 45% | 불확실성이 높은 편 → 리스크 관리 강화 필요 |

핵심 코드 구조:

```python
# SimpleDiffusion: 조건부 Diffusion
# condition(과거 12개월) → noise prediction network → reverse diffusion
class SimpleDiffusion(nn.Module):
    def __init__(self, input_dim=1, condition_dim=12, hidden_dim=64):
        self.T = 50  # 50 단계의 노이즈 추가/제거
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim + 1, hidden_dim),  # +1은 timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
```

상대적 불확실성이 45%인 의미: 예측값이 중앙값 기준 약 +-45% 범위에서 변동할 수 있다. CFO는 중앙값만 보고 예산을 편성하면 안 되며, 5%ile(최악 시나리오)을 고려하여 비상 자금을 마련해야 한다.

### 프롬프트 7 모범 구현: Diffusion Steps 변경

```python
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class SimpleDiffusion(nn.Module):
    def __init__(self, input_dim=1, condition_dim=12, hidden_dim=64, T=50):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def add_noise(self, x, t):
        beta_t = torch.linspace(1e-4, 0.02, self.T)[t]
        alpha_t = 1 - beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_bar_t.unsqueeze(-1)) * x + torch.sqrt(1 - alpha_bar_t.unsqueeze(-1)) * noise
        return noisy_x, noise

    def predict_noise(self, x_t, t, condition):
        t_embed = t.float().unsqueeze(-1) / self.T
        return self.net(torch.cat([x_t, condition, t_embed], dim=-1))

    def reverse_diffusion(self, condition, n_samples=100):
        x_t = torch.randn(n_samples, 1)
        condition = condition.unsqueeze(0).repeat(n_samples, 1)
        for t in reversed(range(self.T)):
            t_tensor = torch.tensor([t] * n_samples)
            noise_pred = self.predict_noise(x_t, t_tensor, condition)
            beta_t = torch.linspace(1e-4, 0.02, self.T)[t]
            x_t = x_t - beta_t * noise_pred
            if t > 0:
                x_t = x_t + torch.sqrt(beta_t) * torch.randn_like(x_t)
        return x_t.detach().numpy().flatten()

def create_sequences(data, lookback=12):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"

df = pd.read_csv(INPUT_DIR / "monthly_sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

scaler = StandardScaler()
sales_scaled = scaler.fit_transform(df[['sales']].values).flatten()

X, y = create_sequences(sales_scaled, lookback=12)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

last_sequence = X_test[-1]

steps_list = [10, 25, 50, 100]

print(f"{'Diffusion Steps':<18} {'중앙값':<12} {'불확실성 범위':<16} {'상대적 불확실성':<18}")
print("-" * 64)

for T in steps_list:
    set_seed(42)
    model = SimpleDiffusion(input_dim=1, condition_dim=12, T=T)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    for epoch in range(100):
        model.train()
        for batch_cond, batch_y in loader:
            optimizer.zero_grad()
            t = torch.randint(0, model.T, (len(batch_y),))
            noisy_y, true_noise = model.add_noise(batch_y, t)
            pred_noise = model.predict_noise(noisy_y, t, batch_cond)
            loss = nn.MSELoss()(pred_noise, true_noise)
            loss.backward()
            optimizer.step()

    model.eval()
    condition = torch.FloatTensor(last_sequence)
    samples = model.reverse_diffusion(condition, n_samples=100)
    samples_units = scaler.inverse_transform(samples.reshape(-1, 1)).flatten()

    p5 = np.percentile(samples_units, 5)
    p50 = np.percentile(samples_units, 50)
    p95 = np.percentile(samples_units, 95)

    uncertainty = p95 - p5
    relative = (uncertainty / p50) * 100 if p50 != 0 else float('inf')

    print(f"{T:<18} {p50:<12.2f} {uncertainty:<16.2f} {relative:<18.1f}%")
```

기대 결과 해석:
- T=10: 노이즈 제거 단계가 너무 적어 예측이 부정확하고 불확실성이 클 수 있음
- T=25: 최소한의 단계. 어느 정도 안정적
- T=50: 기본 설정. 적절한 균형
- T=100: 더 정밀한 노이즈 제거. 학습 시간이 2배이지만, 품질 개선은 완만
- 핵심: Steps가 너무 적으면 노이즈가 충분히 제거되지 않아 예측이 "흐려지고", 너무 많으면 계산 비용이 증가하면서 개선은 체감됨

### 프롬프트 8 모범 구현: 샘플 수 변경

```python
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class SimpleDiffusion(nn.Module):
    def __init__(self, input_dim=1, condition_dim=12, hidden_dim=64):
        super().__init__()
        self.T = 50
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def add_noise(self, x, t):
        beta_t = torch.linspace(1e-4, 0.02, self.T)[t]
        alpha_t = 1 - beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_bar_t.unsqueeze(-1)) * x + torch.sqrt(1 - alpha_bar_t.unsqueeze(-1)) * noise
        return noisy_x, noise

    def predict_noise(self, x_t, t, condition):
        t_embed = t.float().unsqueeze(-1) / self.T
        return self.net(torch.cat([x_t, condition, t_embed], dim=-1))

    def reverse_diffusion(self, condition, n_samples=100):
        x_t = torch.randn(n_samples, 1)
        condition = condition.unsqueeze(0).repeat(n_samples, 1)
        for t in reversed(range(self.T)):
            t_tensor = torch.tensor([t] * n_samples)
            noise_pred = self.predict_noise(x_t, t_tensor, condition)
            beta_t = torch.linspace(1e-4, 0.02, self.T)[t]
            x_t = x_t - beta_t * noise_pred
            if t > 0:
                x_t = x_t + torch.sqrt(beta_t) * torch.randn_like(x_t)
        return x_t.detach().numpy().flatten()

def create_sequences(data, lookback=12):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"

df = pd.read_csv(INPUT_DIR / "monthly_sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

scaler = StandardScaler()
sales_scaled = scaler.fit_transform(df[['sales']].values).flatten()

X, y = create_sequences(sales_scaled, lookback=12)
train_size = int(0.8 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]

last_sequence = X_test[-1]

# 모델 학습 (한 번만)
set_seed(42)
model = SimpleDiffusion()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
X_tensor = torch.FloatTensor(X_train)
y_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

for epoch in range(100):
    model.train()
    for batch_cond, batch_y in loader:
        optimizer.zero_grad()
        t = torch.randint(0, model.T, (len(batch_y),))
        noisy_y, true_noise = model.add_noise(batch_y, t)
        pred_noise = model.predict_noise(noisy_y, t, batch_cond)
        loss = nn.MSELoss()(pred_noise, true_noise)
        loss.backward()
        optimizer.step()

# 샘플 수 변경 실험
sample_counts = [10, 50, 100, 500]

print(f"{'n_samples':<14} {'5%ile':<12} {'중앙값':<12} {'95%ile':<12}")
print("-" * 50)

for n_samples in sample_counts:
    set_seed(42)
    model.eval()
    condition = torch.FloatTensor(last_sequence)
    samples = model.reverse_diffusion(condition, n_samples=n_samples)
    samples_units = scaler.inverse_transform(samples.reshape(-1, 1)).flatten()

    p5 = np.percentile(samples_units, 5)
    p50 = np.percentile(samples_units, 50)
    p95 = np.percentile(samples_units, 95)

    print(f"{n_samples:<14} {p5:<12.2f} {p50:<12.2f} {p95:<12.2f}")
```

기대 결과 해석:
- n_samples=10: 분위수 추정이 매우 불안정. 같은 모델이라도 실행마다 결과가 크게 달라짐
- n_samples=50: 어느 정도 안정적이지만 꼬리 분포(5%ile, 95%ile)는 여전히 흔들림
- n_samples=100: 기본 설정. 분위수 추정이 안정적
- n_samples=500: 가장 안정적이지만 계산 시간이 5배
- 핵심: 분위수 추정의 안정성은 샘플 수에 비례한다. 특히 극단 분위수(5%, 95%)는 꼬리 부분이므로 더 많은 샘플이 필요하다. 실무에서는 100-500개가 적절한 균형점

---

## 8장 전체 핵심 정리

```text
1. GAN: Generator와 Discriminator의 경쟁. 간단한 GAN은 정형 데이터의 복잡한 분포를
   재현하지 못한다. CTGAN은 Mode-specific Normalization과 Conditional Vector로 해결.
2. 합성 데이터 품질: TSTR Ratio가 90% 이상이면 ML 효용성 우수. 하지만 분포 보존과
   프라이버시도 함께 평가해야 한다. 지표를 계산하는 것보다 어떤 의사결정에 연결하는지가 중요.
3. VAE 데이터 증강: 30개의 소수 클래스 샘플만으로도 합성 데이터를 생성하여 Recall을 25%
   개선할 수 있다. 합성 샘플 수는 원본의 3-10배가 적절하며, 너무 많으면 오히려 성능 저하.
4. VAE 이상치 탐지: 재구성 오차가 큰 샘플이 이상치. 임계값에 따라 오탐률/미탐률이 달라지므로,
   도메인 전문가와 함께 설정한다. 단독 사용보다 다층 방어 시스템의 일부로 활용.
5. Diffusion Model: 100개 샘플링으로 점 추정 대신 분포 전체를 제공. Steps가 적으면
   노이즈 제거 불충분, 많으면 계산 비용 증가. 50-100 Steps가 실습에서 적절한 균형.
6. 모든 생성 모델은 과거 데이터에 기반하므로, 구조적 변화(팬데믹 등)는 반영하지 못한다.
   AI 도구로 코드를 생성하되, 결과의 의미를 반드시 해석하는 습관이 중요하다.
```
