# 10-2-kaplan-meier.py
"""
Kaplan-Meier 추정 및 로그순위 검정 실습
- Rossi 재범 데이터셋을 사용한 생존 분석
- 그룹 간 생존 곡선 비교
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from lifelines import KaplanMeierFitter
from lifelines.datasets import load_rossi
from lifelines.statistics import logrank_test

# 폰트 설정 (크로스 플랫폼)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리 설정
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Kaplan-Meier 분석 메인 함수"""

    # 1. 데이터 로드
    print("=" * 60)
    print("10.2 Kaplan-Meier 추정 실습")
    print("=" * 60)

    data = load_rossi()
    print(f"\n[데이터 개요]")
    print(f"- 전체 관측치: {len(data)}명")
    print(f"- 사건 발생(재범): {data['arrest'].sum()}명")
    print(f"- 중도절단: {len(data) - data['arrest'].sum()}명")
    print(f"- 관측 기간: {data['week'].min()} ~ {data['week'].max()}주")

    # 2. 전체 생존 곡선 추정
    print("\n[전체 생존 곡선 추정]")
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data['week'], event_observed=data['arrest'], label='Overall')

    # 주요 시점 생존 확률
    print(f"- 12주 시점 생존 확률: {kmf.survival_function_at_times(12).values[0]:.3f}")
    print(f"- 26주 시점 생존 확률: {kmf.survival_function_at_times(26).values[0]:.3f}")
    print(f"- 52주 시점 생존 확률: {kmf.survival_function_at_times(52).values[0]:.3f}")
    print(f"- 중앙 생존 시간: {kmf.median_survival_time_}주")

    # 3. 재정 지원(fin) 여부에 따른 그룹 비교
    print("\n[재정 지원 여부에 따른 생존 비교]")

    # 그룹 분리
    fin_yes = data[data['fin'] == 1]  # 재정 지원 O
    fin_no = data[data['fin'] == 0]   # 재정 지원 X

    print(f"- 재정 지원 O: {len(fin_yes)}명 (재범 {fin_yes['arrest'].sum()}명)")
    print(f"- 재정 지원 X: {len(fin_no)}명 (재범 {fin_no['arrest'].sum()}명)")

    # 각 그룹별 KM 추정
    kmf_fin_yes = KaplanMeierFitter()
    kmf_fin_yes.fit(fin_yes['week'], fin_yes['arrest'], label='Financial Aid: Yes')

    kmf_fin_no = KaplanMeierFitter()
    kmf_fin_no.fit(fin_no['week'], fin_no['arrest'], label='Financial Aid: No')

    # 4. 로그순위 검정
    print("\n[로그순위 검정 결과]")
    results = logrank_test(
        fin_yes['week'], fin_no['week'],
        fin_yes['arrest'], fin_no['arrest']
    )

    print(f"- 검정 통계량: {results.test_statistic:.3f}")
    print(f"- p-value: {results.p_value:.4f}")

    if results.p_value < 0.05:
        print("- 결론: 두 그룹의 생존 함수가 통계적으로 유의하게 다름 (p < 0.05)")
    else:
        print("- 결론: 두 그룹의 생존 함수 차이가 통계적으로 유의하지 않음 (p >= 0.05)")

    # 5. 시각화 (흑백)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5-1. 전체 생존 곡선 (검정색)
    ax1 = axes[0]
    kmf.plot_survival_function(ax=ax1, ci_show=True, color='black')
    ax1.set_xlabel('Time (weeks)')
    ax1.set_ylabel('Survival probability')
    ax1.set_title('Kaplan-Meier Survival Curve (Overall)')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # 5-2. 그룹별 생존 곡선 비교 (흑백: 실선 vs 점선)
    ax2 = axes[1]
    kmf_fin_yes.plot_survival_function(ax=ax2, ci_show=True, color='black', linestyle='-', label='Financial Aid: Yes')
    kmf_fin_no.plot_survival_function(ax=ax2, ci_show=True, color='gray', linestyle='--', label='Financial Aid: No')
    ax2.set_xlabel('Time (weeks)')
    ax2.set_ylabel('Survival probability')
    ax2.set_title(f'Kaplan-Meier Survival Curves by Financial Aid\n(Log-rank p = {results.p_value:.4f})')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(DATA_DIR / '10-2-kaplan-meier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[저장] {DATA_DIR / '10-2-kaplan-meier.png'}")

    # 6. 주요 시점별 생존 확률 표
    print("\n[주요 시점별 생존 확률]")
    time_points = [12, 26, 39, 52]
    survival_table = []

    for t in time_points:
        s_overall = kmf.survival_function_at_times(t).values[0]
        s_fin_yes = kmf_fin_yes.survival_function_at_times(t).values[0]
        s_fin_no = kmf_fin_no.survival_function_at_times(t).values[0]

        survival_table.append({
            '시점(주)': t,
            '전체': f'{s_overall:.3f}',
            '재정 지원 O': f'{s_fin_yes:.3f}',
            '재정 지원 X': f'{s_fin_no:.3f}'
        })

    df_table = pd.DataFrame(survival_table)
    print(df_table.to_string(index=False))

    # 7. 결과 저장 (JSON)
    summary = {
        'data_summary': {
            'total_observations': int(len(data)),
            'events': int(data['arrest'].sum()),
            'censored': int(len(data) - data['arrest'].sum()),
            'observation_period_weeks': int(data['week'].max())
        },
        'overall_survival': {
            'median_survival_time': float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None,
            'survival_at_12_weeks': float(kmf.survival_function_at_times(12).values[0]),
            'survival_at_26_weeks': float(kmf.survival_function_at_times(26).values[0]),
            'survival_at_52_weeks': float(kmf.survival_function_at_times(52).values[0])
        },
        'group_comparison': {
            'fin_yes': {
                'n': int(len(fin_yes)),
                'events': int(fin_yes['arrest'].sum()),
                'survival_at_52_weeks': float(kmf_fin_yes.survival_function_at_times(52).values[0])
            },
            'fin_no': {
                'n': int(len(fin_no)),
                'events': int(fin_no['arrest'].sum()),
                'survival_at_52_weeks': float(kmf_fin_no.survival_function_at_times(52).values[0])
            }
        },
        'logrank_test': {
            'test_statistic': float(results.test_statistic),
            'p_value': float(results.p_value),
            'significant': bool(results.p_value < 0.05)
        }
    }

    with open(DATA_DIR / '10-2-kaplan-meier.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[저장] {DATA_DIR / '10-2-kaplan-meier.json'}")

    return summary


if __name__ == '__main__':
    main()
