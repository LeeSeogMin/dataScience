# 9-4-prophet.py
# Prophet을 이용한 CO2 농도 시계열 예측
# 추세, 계절성, 휴일 효과의 가법/승법 분해

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# 출력 디렉토리 설정
output_dir = Path(__file__).parent.parent / "data"
diagram_dir = Path(__file__).parent.parent.parent.parent / "diagram"

# 한글 폰트 설정 (크로스 플랫폼)
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

def load_and_preprocess_data():
    """CO2 데이터 로드 및 Prophet 형식으로 변환"""
    data_path = Path(__file__).parent.parent / "data" / "co2.csv"
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    y = df['co2']

    # Prophet 형식으로 변환 (ds, y 컬럼 필요)
    prophet_df = pd.DataFrame({
        'ds': y.index,
        'y': y.values
    })
    return prophet_df, y

def evaluate_forecast(y_test, forecast):
    """예측 성능 평가"""
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    mape = mean_absolute_percentage_error(y_test, forecast) * 100
    return rmse, mape

def main():
    print("=" * 60)
    print("9-4 Prophet 시계열 예측 실습")
    print("=" * 60)

    # 1. 데이터 로드
    df, y = load_and_preprocess_data()
    print(f"\n데이터 기간: {df['ds'].min()} ~ {df['ds'].max()}")
    print(f"총 관측치 수: {len(df)}")

    # 2. 학습/테스트 분할
    train_size = int(len(df) * 0.8)
    df_train = df[:train_size]
    df_test = df[train_size:]

    print(f"\n학습 데이터: {len(df_train)}개 ({df_train['ds'].iloc[0]} ~ {df_train['ds'].iloc[-1]})")
    print(f"테스트 데이터: {len(df_test)}개 ({df_test['ds'].iloc[0]} ~ {df_test['ds'].iloc[-1]})")

    # 3. Prophet 모형 적합 (가법 모형)
    print("\n" + "=" * 40)
    print("Prophet 모형 적합 (가법 계절성)")
    print("=" * 40)

    model_add = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,  # 월별 데이터이므로 주간 계절성 불필요
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05  # 변화점 탐지 민감도
    )
    model_add.fit(df_train)

    # 4. 예측
    future = model_add.make_future_dataframe(periods=len(df_test), freq='MS')
    forecast_add = model_add.predict(future)

    # 테스트 구간만 추출
    forecast_test_add = forecast_add[forecast_add['ds'] >= df_test['ds'].iloc[0]]

    # 5. 성능 평가 (가법)
    rmse_add, mape_add = evaluate_forecast(
        df_test['y'].values,
        forecast_test_add['yhat'].values
    )
    print(f"\n=== 가법 모형 예측 성능 ===")
    print(f"RMSE: {rmse_add:.4f} ppm")
    print(f"MAPE: {mape_add:.2f}%")

    # 6. Prophet 모형 적합 (승법 모형)
    print("\n" + "=" * 40)
    print("Prophet 모형 적합 (승법 계절성)")
    print("=" * 40)

    model_mult = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    model_mult.fit(df_train)

    forecast_mult = model_mult.predict(future)
    forecast_test_mult = forecast_mult[forecast_mult['ds'] >= df_test['ds'].iloc[0]]

    rmse_mult, mape_mult = evaluate_forecast(
        df_test['y'].values,
        forecast_test_mult['yhat'].values
    )
    print(f"\n=== 승법 모형 예측 성능 ===")
    print(f"RMSE: {rmse_mult:.4f} ppm")
    print(f"MAPE: {mape_mult:.2f}%")

    # 7. 모형 비교 결과
    print("\n" + "=" * 40)
    print("가법 vs 승법 비교")
    print("=" * 40)

    comparison = pd.DataFrame({
        '모형': ['가법(Additive)', '승법(Multiplicative)'],
        'RMSE': [rmse_add, rmse_mult],
        'MAPE(%)': [mape_add, mape_mult]
    })
    print(comparison.to_string(index=False))

    # 더 나은 모형 선택
    if rmse_add < rmse_mult:
        best_model = model_add
        best_forecast = forecast_add
        best_forecast_test = forecast_test_add
        best_mode = 'additive'
        best_rmse, best_mape = rmse_add, mape_add
    else:
        best_model = model_mult
        best_forecast = forecast_mult
        best_forecast_test = forecast_test_mult
        best_mode = 'multiplicative'
        best_rmse, best_mape = rmse_mult, mape_mult

    print(f"\n선택된 모형: {best_mode}")

    # 8. 구성 요소 분해 시각화
    print("\n" + "=" * 40)
    print("구성 요소 분해")
    print("=" * 40)

    fig = best_model.plot_components(best_forecast)
    fig.savefig(output_dir / "9-4-prophet-components.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"구성 요소 그래프 저장: {output_dir / '9-4-prophet-components.png'}")

    # 9. 예측 결과 시각화
    fig, ax = plt.subplots(figsize=(14, 6))

    # 학습 데이터
    ax.plot(df_train['ds'], df_train['y'], 'b-', label='학습 데이터', linewidth=1)
    # 테스트 데이터
    ax.plot(df_test['ds'], df_test['y'], 'g-', label='실제값 (테스트)', linewidth=1.5)
    # Prophet 예측
    ax.plot(best_forecast_test['ds'], best_forecast_test['yhat'], 'r--',
            label='Prophet 예측', linewidth=1.5)
    # 신뢰구간
    ax.fill_between(best_forecast_test['ds'],
                    best_forecast_test['yhat_lower'],
                    best_forecast_test['yhat_upper'],
                    color='red', alpha=0.2, label='95% 신뢰구간')

    ax.set_xlabel('날짜')
    ax.set_ylabel('CO2 농도 (ppm)')
    ax.set_title(f'Prophet ({best_mode}) 예측 결과 (RMSE={best_rmse:.2f}, MAPE={best_mape:.2f}%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "9-4-prophet-forecast.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"예측 그래프 저장: {output_dir / '9-4-prophet-forecast.png'}")

    # 10. 변화점 시각화
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['ds'], df['y'], 'b-', linewidth=0.8, alpha=0.7)

    # 변화점 표시
    changepoints = best_model.changepoints
    for cp in changepoints:
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    ax.set_xlabel('날짜')
    ax.set_ylabel('CO2 농도 (ppm)')
    ax.set_title(f'Prophet 변화점 탐지 ({len(changepoints)}개)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "9-4-prophet-changepoints.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"변화점 그래프 저장: {output_dir / '9-4-prophet-changepoints.png'}")

    # 11. 결과 저장
    results = {
        "model": "Prophet",
        "seasonality_mode": best_mode,
        "train_size": len(df_train),
        "test_size": len(df_test),
        "rmse_additive": round(rmse_add, 4),
        "mape_additive": round(mape_add, 2),
        "rmse_multiplicative": round(rmse_mult, 4),
        "mape_multiplicative": round(mape_mult, 2),
        "best_rmse": round(best_rmse, 4),
        "best_mape": round(best_mape, 2),
        "num_changepoints": len(changepoints)
    }

    with open(output_dir / "9-4-prophet-results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {output_dir / '9-4-prophet-results.json'}")

    # 예측값 CSV 저장
    forecast_df = pd.DataFrame({
        'date': df_test['ds'].values,
        'actual': df_test['y'].values,
        'forecast': best_forecast_test['yhat'].values,
        'lower_ci': best_forecast_test['yhat_lower'].values,
        'upper_ci': best_forecast_test['yhat_upper'].values,
        'trend': best_forecast_test['trend'].values,
        'yearly': best_forecast_test['yearly'].values
    })
    forecast_df.to_csv(output_dir / "9-4-prophet-forecast.csv", index=False)
    print(f"예측값 저장: {output_dir / '9-4-prophet-forecast.csv'}")

    return results

if __name__ == "__main__":
    main()
