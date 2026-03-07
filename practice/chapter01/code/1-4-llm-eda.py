"""
1-4-llm-eda.py: LLM 기반 탐색적 데이터 분석 (EDA)

이 스크립트는 LLM을 활용하여 데이터셋의 탐색적 분석을 자동화하는 방법을 보여준다.
전통적인 수동 EDA와 달리, LLM이 데이터 패턴을 해석하고 인사이트를 제안한다.

실행 방법:
    python 1-4-llm-eda.py

필수 환경 변수:
    OPENAI_API_KEY: OpenAI API 키
"""

import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_california_housing

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# 출력 디렉토리 설정
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_data_profile(df: pd.DataFrame, dataset_name: str) -> dict:
    """데이터셋의 기본 프로파일을 생성한다."""

    profile = {
        "dataset_name": dataset_name,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": {},
        "missing_values": {},
        "correlations": {}
    }

    # 컬럼별 통계
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "unique_values": int(df[col].nunique())
        }

        if df[col].dtype in ['int64', 'float64']:
            col_info.update({
                "mean": float(round(df[col].mean(), 4)),
                "std": float(round(df[col].std(), 4)),
                "min": float(round(float(df[col].min()), 4)),
                "max": float(round(float(df[col].max()), 4)),
                "median": float(round(df[col].median(), 4))
            })

        profile["columns"][col] = col_info

    # 결측치 요약
    missing = df.isnull().sum()
    profile["missing_values"] = {
        "total_missing": int(missing.sum()),
        "columns_with_missing": [col for col in missing.index if missing[col] > 0]
    }

    # 수치형 컬럼 간 상관관계 (상위 5개)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()

        # 상삼각 행렬에서 상관계수 추출
        correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                correlations.append({
                    "pair": f"{numeric_cols[i]} - {numeric_cols[j]}",
                    "correlation": float(round(corr_matrix.iloc[i, j], 4))
                })

        # 절대값 기준 상위 5개
        correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        profile["correlations"]["top_5"] = correlations[:5]

    return profile


def create_eda_visualizations(df: pd.DataFrame, dataset_name: str) -> list:
    """EDA 시각화를 생성하고 저장한다."""

    saved_files = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 1. 분포 히스토그램
    if len(numeric_cols) > 0:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = np.array(axes).flatten() if n_rows > 1 or n_cols > 1 else [axes]

        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'{col}', fontsize=10)
                axes[idx].set_xlabel('')

        # 빈 subplot 제거
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'{dataset_name} - Distribution', fontsize=12)
        plt.tight_layout()

        hist_path = OUTPUT_DIR / f"{dataset_name}_distributions.png"
        plt.savefig(hist_path, dpi=100, bbox_inches='tight')
        plt.close()
        saved_files.append(str(hist_path))

    # 2. 상관관계 히트맵
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0,
                    fmt='.2f', square=True, linewidths=0.5)
        plt.title(f'{dataset_name} - Correlation Matrix')
        plt.tight_layout()

        corr_path = OUTPUT_DIR / f"{dataset_name}_correlation.png"
        plt.savefig(corr_path, dpi=100, bbox_inches='tight')
        plt.close()
        saved_files.append(str(corr_path))

    return saved_files


def generate_llm_insights(profile: dict, use_api: bool = True) -> str:
    """LLM을 사용하여 데이터 인사이트를 생성한다."""

    prompt = f"""당신은 데이터 분석 전문가입니다. 다음 데이터 프로파일을 분석하고 핵심 인사이트를 제공하세요.

## 데이터셋 정보
- 이름: {profile['dataset_name']}
- 크기: {profile['shape']['rows']}행 x {profile['shape']['columns']}열

## 컬럼별 통계
{json.dumps(profile['columns'], indent=2, ensure_ascii=False)}

## 결측치 현황
{json.dumps(profile['missing_values'], indent=2, ensure_ascii=False)}

## 주요 상관관계
{json.dumps(profile.get('correlations', {}), indent=2, ensure_ascii=False)}

다음 형식으로 분석 결과를 제공하세요:

### 1. 데이터 품질 평가
(결측치, 이상치 가능성, 데이터 타입 적절성)

### 2. 주요 패턴 및 인사이트
(변수 간 관계, 분포 특성, 주목할 만한 점)

### 3. 추가 분석 권장사항
(더 깊이 탐색할 영역, 필요한 전처리)

### 4. 모델링 시 고려사항
(적합한 알고리즘 유형, 주의점)
"""

    if use_api:
        try:
            from openai import OpenAI

            client = OpenAI()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1500,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 데이터 분석 전문가다. 제공된 데이터 프로파일만 바탕으로 신중하고 구조화된 EDA 인사이트를 작성한다."
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"API 호출 실패: {e}")
            return generate_fallback_insights(profile)
    else:
        return generate_fallback_insights(profile)


def generate_fallback_insights(profile: dict) -> str:
    """API 없이 규칙 기반 인사이트를 생성한다."""

    insights = []

    # 데이터 품질 평가
    insights.append("### 1. 데이터 품질 평가")

    if profile['missing_values']['total_missing'] == 0:
        insights.append("- 결측치 없음: 데이터 품질이 양호합니다.")
    else:
        missing_cols = profile['missing_values']['columns_with_missing']
        insights.append(f"- 결측치 발견: {len(missing_cols)}개 컬럼에서 결측치 확인됨")

    # 주요 패턴
    insights.append("\n### 2. 주요 패턴 및 인사이트")

    if 'top_5' in profile.get('correlations', {}):
        top_corr = profile['correlations']['top_5'][0]
        insights.append(f"- 가장 강한 상관관계: {top_corr['pair']} (r={top_corr['correlation']})")

    # 분포 특성
    for col, stats in profile['columns'].items():
        if 'mean' in stats and 'std' in stats:
            cv = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else 0
            if cv > 1:
                insights.append(f"- {col}: 높은 변동성 (CV={cv:.2f})")

    # 권장사항
    insights.append("\n### 3. 추가 분석 권장사항")
    insights.append("- 이상치 탐지를 위한 박스플롯 분석 권장")
    insights.append("- 범주형 변수가 있다면 그룹별 분석 수행")

    # 모델링 고려사항
    insights.append("\n### 4. 모델링 시 고려사항")
    n_features = len(profile['columns'])
    n_samples = profile['shape']['rows']

    if n_samples / n_features < 10:
        insights.append("- 샘플 대비 특성 비율이 낮음: 정규화 또는 차원 축소 고려")
    else:
        insights.append("- 샘플 수가 충분함: 다양한 알고리즘 적용 가능")

    return "\n".join(insights)


def run_llm_eda(dataset_name: str = "iris", use_api: bool = True):
    """LLM 기반 EDA를 실행한다."""

    print(f"\n{'='*60}")
    print(f"LLM 기반 탐색적 데이터 분석 (EDA)")
    print(f"{'='*60}")

    # 1. 데이터 로드
    print(f"\n[1/4] 데이터 로드: {dataset_name}")

    if dataset_name == "iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == "california_housing":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"   - 크기: {df.shape[0]}행 x {df.shape[1]}열")

    # 2. 데이터 프로파일 생성
    print("\n[2/4] 데이터 프로파일 생성 중...")
    profile = generate_data_profile(df, dataset_name)

    # 프로파일 저장
    profile_path = OUTPUT_DIR / f"{dataset_name}_profile.json"
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"   - 프로파일 저장: {profile_path}")

    # 3. 시각화 생성
    print("\n[3/4] 시각화 생성 중...")
    viz_files = create_eda_visualizations(df, dataset_name)
    for viz in viz_files:
        print(f"   - 저장됨: {viz}")

    # 4. LLM 인사이트 생성
    print("\n[4/4] LLM 인사이트 생성 중...")
    insights = generate_llm_insights(profile, use_api=use_api)

    # 인사이트 저장
    insights_path = OUTPUT_DIR / f"{dataset_name}_insights.md"
    with open(insights_path, 'w', encoding='utf-8') as f:
        f.write(f"# {dataset_name} 데이터셋 자동 분석 리포트\n\n")
        f.write(f"**생성일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**데이터 크기**: {profile['shape']['rows']}행 x {profile['shape']['columns']}열\n\n")
        f.write("---\n\n")
        f.write(insights)
    print(f"   - 인사이트 저장: {insights_path}")

    # 결과 출력
    print(f"\n{'='*60}")
    print("LLM 분석 결과")
    print(f"{'='*60}")
    print(insights)

    # 결과 요약 반환
    return {
        "dataset": dataset_name,
        "shape": profile['shape'],
        "profile_path": str(profile_path),
        "insights_path": str(insights_path),
        "visualizations": viz_files
    }


def compare_traditional_vs_llm_eda():
    """전통적 EDA와 LLM EDA 비교 분석을 수행한다."""

    print("\n" + "="*70)
    print("전통적 EDA vs LLM 기반 EDA 비교")
    print("="*70)

    comparison = {
        "전통적 EDA": {
            "process": [
                "1. 데이터 로드 및 기본 통계 확인 (df.describe())",
                "2. 결측치/이상치 수동 탐색",
                "3. 시각화 직접 생성 및 해석",
                "4. 상관관계 분석",
                "5. 인사이트 수동 도출"
            ],
            "장점": [
                "분석가의 도메인 지식 활용",
                "세밀한 가설 검증 가능",
                "분석 과정 완전 통제"
            ],
            "단점": [
                "시간 소요 (수 시간~수일)",
                "분석가 역량에 의존",
                "반복 작업 많음"
            ]
        },
        "LLM 기반 EDA": {
            "process": [
                "1. 데이터 프로파일 자동 생성",
                "2. 시각화 자동 생성",
                "3. LLM에 컨텍스트 전달",
                "4. 인사이트 자동 도출",
                "5. 리포트 자동 생성"
            ],
            "장점": [
                "빠른 초기 분석 (수 분)",
                "일관된 분석 품질",
                "놓칠 수 있는 패턴 발견"
            ],
            "단점": [
                "도메인 맥락 부족 가능",
                "Hallucination 위험",
                "API 비용 발생"
            ]
        }
    }

    print("\n### 비교 요약\n")
    print(f"{'항목':<20} {'전통적 EDA':<30} {'LLM 기반 EDA':<30}")
    print("-" * 80)
    print(f"{'소요 시간':<20} {'수 시간 ~ 수일':<30} {'수 분':<30}")
    print(f"{'초기 인사이트':<20} {'분석가 역량 의존':<30} {'자동 생성':<30}")
    print(f"{'도메인 지식':<20} {'분석가가 제공':<30} {'프롬프트로 주입':<30}")
    print(f"{'비용':<20} {'인건비':<30} {'API 비용':<30}")
    print(f"{'확장성':<20} {'선형 증가':<30} {'병렬 처리 가능':<30}")

    print("\n### 권장 활용 전략")
    print("""
1. **LLM을 첫 단계 도구로 활용**: 빠른 데이터 파악용
2. **전통적 EDA로 심화 분석**: 도메인 특화 가설 검증
3. **하이브리드 접근**: LLM 인사이트 + 분석가 검증
4. **반복 작업 자동화**: 유사 데이터셋 분석 시 LLM 템플릿 활용
""")

    return comparison


if __name__ == "__main__":
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    use_api = api_key is not None and len(api_key) > 0

    if not use_api:
        print("경고: OPENAI_API_KEY가 설정되지 않음. 규칙 기반 분석 사용.")

    # 1. Iris 데이터셋으로 LLM EDA 실행
    result = run_llm_eda("iris", use_api=use_api)

    print(f"\n\n{'='*60}")
    print("분석 완료 요약")
    print(f"{'='*60}")
    print(f"데이터셋: {result['dataset']}")
    print(f"크기: {result['shape']}")
    print(f"프로파일: {result['profile_path']}")
    print(f"인사이트: {result['insights_path']}")

    # 2. 전통 EDA vs LLM EDA 비교
    compare_traditional_vs_llm_eda()
