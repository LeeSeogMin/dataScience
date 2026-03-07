# 10-3-cox.py
"""
Cox 비례위험 모형 실습
- Rossi 재범 데이터셋을 사용한 Cox 회귀 분석
- 위험비(Hazard Ratio) 추정 및 해석
- 비례위험 가정 검정 (Schoenfeld 잔차)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

# 폰트 설정 (크로스 플랫폼)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리 설정
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Cox 비례위험 모형 메인 함수"""

    # 1. 데이터 로드
    print("=" * 60)
    print("10.3 Cox 비례위험 모형 실습")
    print("=" * 60)

    data = load_rossi()

    # 변수 설명
    var_desc = {
        'week': '관측 기간 (주)',
        'arrest': '재범 여부 (1=재범, 0=중도절단)',
        'fin': '재정 지원 여부 (1=예, 0=아니오)',
        'age': '나이 (년)',
        'race': '인종 (1=흑인, 0=기타)',
        'wexp': '전과 취업 경험 (1=예, 0=아니오)',
        'mar': '결혼 여부 (1=기혼, 0=미혼)',
        'paro': '가석방 여부 (1=예, 0=아니오)',
        'prio': '전과 횟수'
    }

    print(f"\n[변수 설명]")
    for var, desc in var_desc.items():
        print(f"  - {var}: {desc}")

    # 2. Cox 비례위험 모형 적합
    print("\n[Cox 비례위험 모형 적합]")
    cph = CoxPHFitter()
    cph.fit(data, duration_col='week', event_col='arrest')

    # 모형 요약 출력
    print("\n[모형 적합 결과]")
    print(cph.summary.to_string())

    # 3. 위험비 해석
    print("\n[위험비(Hazard Ratio) 해석]")
    summary_df = cph.summary[['coef', 'exp(coef)', 'se(coef)',
                              'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
    summary_df.columns = ['계수', '위험비', '표준오차', 'HR 하한 95%', 'HR 상한 95%', 'p-value']

    # 유의한 변수 (p < 0.05)
    significant = summary_df[summary_df['p-value'] < 0.05]
    print("\n유의한 변수 (p < 0.05):")
    if len(significant) > 0:
        for var, row in significant.iterrows():
            hr = row['위험비']
            direction = "증가" if hr > 1 else "감소"
            effect = abs(hr - 1) * 100
            print(f"  - {var}: HR={hr:.3f} ({effect:.1f}% 위험 {direction}), p={row['p-value']:.4f}")
    else:
        print("  유의한 변수 없음")

    # 4. 비례위험 가정 검정
    print("\n[비례위험 가정 검정 (Schoenfeld 잔차)]")
    try:
        ph_test = cph.check_assumptions(data, p_value_threshold=0.05, show_plots=False)
        print("비례위험 가정 검정 완료")
    except Exception as e:
        print(f"검정 중 경고: {e}")
        ph_test = None

    # 5. 시각화 (흑백)

    # 5-1. 위험비 Forest Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    cph.plot(ax=ax1)
    # 흑백으로 변환
    for line in ax1.get_lines():
        line.set_color('black')
    ax1.set_title('Cox Model Forest Plot (Hazard Ratio)')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('log(Hazard Ratio)')
    ax1.grid(True, alpha=0.3)

    # 5-2. 각 변수별 생존 곡선 (age 예시) - 흑백
    ax2 = axes[1]
    # 나이 그룹별 생존 곡선 (흑백: 선 스타일로 구분)
    cph.plot_partial_effects_on_outcome(
        covariates='age',
        values=[20, 30, 40],
        ax=ax2,
        cmap='gray'
    )
    ax2.set_title('Predicted Survival Curves by Age')
    ax2.set_xlabel('Time (weeks)')
    ax2.set_ylabel('Survival probability')
    ax2.legend(title='Age', labels=['20', '30', '40'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(DATA_DIR / '10-3-cox-forest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[저장] {DATA_DIR / '10-3-cox-forest.png'}")

    # 6. 적합도 지표
    print("\n[모형 적합도]")
    print(f"- Concordance Index (C-index): {cph.concordance_index_:.4f}")
    print(f"- Log-Likelihood: {cph.log_likelihood_:.2f}")
    print(f"- AIC (부분 우도 기반): {cph.AIC_partial_:.2f}")

    # 7. 결과 저장 (JSON)
    results = {
        'model_summary': {
            'n_observations': int(len(data)),
            'n_events': int(data['arrest'].sum()),
            'concordance_index': float(cph.concordance_index_),
            'log_likelihood': float(cph.log_likelihood_),
            'AIC_partial': float(cph.AIC_partial_)
        },
        'coefficients': {}
    }

    for var in cph.summary.index:
        row = cph.summary.loc[var]
        results['coefficients'][var] = {
            'coef': float(row['coef']),
            'hazard_ratio': float(row['exp(coef)']),
            'se': float(row['se(coef)']),
            'hr_lower_95': float(row['exp(coef) lower 95%']),
            'hr_upper_95': float(row['exp(coef) upper 95%']),
            'p_value': float(row['p']),
            'significant': bool(row['p'] < 0.05)
        }

    with open(DATA_DIR / '10-3-cox.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[저장] {DATA_DIR / '10-3-cox.json'}")

    # 8. 결과 표
    print("\n[Cox 비례위험 모형 결과 요약]")
    result_table = []
    for var in cph.summary.index:
        row = cph.summary.loc[var]
        result_table.append({
            '변수': var,
            '계수': f"{row['coef']:.4f}",
            '위험비': f"{row['exp(coef)']:.3f}",
            '95% CI': f"({row['exp(coef) lower 95%']:.3f}, {row['exp(coef) upper 95%']:.3f})",
            'p-value': f"{row['p']:.4f}" if row['p'] >= 0.0001 else "<0.0001"
        })

    df_result = pd.DataFrame(result_table)
    print(df_result.to_string(index=False))

    return results


if __name__ == '__main__':
    main()
