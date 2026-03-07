# generate_survival_data.py
"""
10.7 모형 비교용 합성 생존 데이터 생성 스크립트

딥러닝(DeepSurv)에 유리한 상황을 만들기 위해, 고차원/비선형 구조를 가진
합성 생존 데이터를 생성하여 CSV로 저장한다.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataConfig:
    n_samples: int = 8000
    n_features: int = 100
    n_latent: int = 12
    weibull_shape: float = 1.6
    weibull_scale: float = 12.0
    censoring_target: float = 0.35
    seed: int = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_deeplearning_friendly_survival(config: DataConfig) -> pd.DataFrame:
    """
    Cox 형태(비례위험)로 데이터 생성하되, log-risk가 "다층 비선형"이 되도록 설계한다.
    - 관측 특성 X는 잠재변수 z의 선형 혼합(차원 확장 + 잡음)
    - 진짜 위험도는 z에 대한 2층 MLP 형태(사인/제곱/상호작용 포함)
    """
    set_seed(config.seed)

    z = np.random.normal(size=(config.n_samples, config.n_latent)).astype(np.float32)

    # 고차원 관측 특성 (z의 선형 혼합 + 약간의 비선형 왜곡 + 잡음)
    mix = np.random.normal(scale=0.7, size=(config.n_latent, config.n_features)).astype(
        np.float32
    )
    x_linear = z @ mix
    x_nonlinear = np.concatenate(
        [
            np.tanh(x_linear[:, : config.n_features // 3]),
            np.sin(x_linear[:, config.n_features // 3 : 2 * config.n_features // 3]),
            np.square(x_linear[:, 2 * config.n_features // 3 :]),
        ],
        axis=1,
    )
    x = (0.65 * x_linear + 0.35 * x_nonlinear) + np.random.normal(
        scale=0.25, size=x_linear.shape
    ).astype(np.float32)

    # 진짜 log-risk: z에 대한 2층 비선형 + 상호작용
    w1 = np.random.normal(scale=1.0, size=(config.n_latent, 32)).astype(np.float32)
    b1 = np.random.normal(scale=0.2, size=(32,)).astype(np.float32)
    h1 = np.maximum(0.0, z @ w1 + b1)  # ReLU
    w2 = np.random.normal(scale=0.6, size=(32, 1)).astype(np.float32)
    log_risk = (h1 @ w2).squeeze(-1)
    log_risk += 0.8 * np.sin(z[:, 0] * z[:, 1]) + 0.6 * np.square(z[:, 2]) - 0.5 * _sigmoid(
        z[:, 3] * 2.0
    )
    log_risk += 0.3 * (z[:, 4] * z[:, 5]) - 0.2 * np.abs(z[:, 6])
    log_risk = (log_risk - log_risk.mean()) / (log_risk.std() + 1e-8)

    # Cox-위험: H(t|x)=exp(log_risk)*H0(t), H0(t)=(t/scale)^shape
    u = np.random.uniform(size=config.n_samples).astype(np.float32)
    t_event = config.weibull_scale * np.power(
        (-np.log(u) / np.exp(log_risk)).astype(np.float32), 1.0 / config.weibull_shape
    )

    # 목표 중도절단률 근사(간단한 스케일 조정)
    censor_scale = np.quantile(t_event, 1.0 - config.censoring_target)
    t_censor = np.random.exponential(scale=float(censor_scale), size=config.n_samples).astype(
        np.float32
    )

    time_observed = np.minimum(t_event, t_censor)
    event_observed = (t_event <= t_censor).astype(np.int32)

    df = pd.DataFrame(x, columns=[f"x{i:03d}" for i in range(config.n_features)])
    df["time"] = time_observed.astype(np.float32)
    df["event"] = event_observed.astype(np.int32)
    return df


def main() -> None:
    config = DataConfig()
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_survival_data.csv"

    print("=" * 60)
    print("합성 생존 데이터 생성")
    print("=" * 60)
    print(f"- N={config.n_samples}, p={config.n_features}, latent={config.n_latent}")

    df = generate_deeplearning_friendly_survival(config)

    print(f"- event_rate={df['event'].mean():.3f}, censoring_rate={1.0 - df['event'].mean():.3f}")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[Saved] {output_path}")
    print(f"- Shape: {df.shape}")


if __name__ == "__main__":
    main()
