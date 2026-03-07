# 9-3-arima.py
# SARIMA 모형을 이용한 CO2 농도 시계열 예측
# Mauna Loa CO2 데이터를 월별로 리샘플링하여 분석

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
    """CO2 데이터 로드 및 전처리"""
    data_path = Path(__file__).parent.parent / "data" / "co2.csv"
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    return df['co2']

def test_stationarity(series, name="시계열"):
    """ADF 검정으로 정상성 테스트"""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n=== {name} ADF 검정 결과 ===")
    print(f"검정 통계량: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"사용된 시차: {result[2]}")
    print(f"관측치 수: {result[3]}")
    for key, value in result[4].items():
        print(f"임계값 ({key}): {value:.4f}")

    if result[1] < 0.05:
        print("결론: 귀무가설 기각 - 시계열이 정상(stationary)")
    else:
        print("결론: 귀무가설 채택 불가 - 시계열이 비정상(non-stationary)")

    return result[1]

def plot_decomposition(y, output_dir):
    """시계열 분해 시각화"""
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(y, model='additive', period=12)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(y.index, y.values, 'b-', linewidth=0.8)
    axes[0].set_title('원본 시계열 (CO2 농도)', fontsize=12)
    axes[0].set_ylabel('CO2 (ppm)')

    axes[1].plot(decomposition.trend.index, decomposition.trend.values, 'g-', linewidth=0.8)
    axes[1].set_title('추세 (Trend)', fontsize=12)
    axes[1].set_ylabel('CO2 (ppm)')

    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 'r-', linewidth=0.8)
    axes[2].set_title('계절성 (Seasonality)', fontsize=12)
    axes[2].set_ylabel('CO2 (ppm)')

    axes[3].plot(decomposition.resid.index, decomposition.resid.values, 'purple', linewidth=0.8)
    axes[3].set_title('잔차 (Residual)', fontsize=12)
    axes[3].set_ylabel('CO2 (ppm)')

    plt.tight_layout()
    plt.savefig(output_dir / "9-3-decomposition.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"분해 그래프 저장: {output_dir / '9-3-decomposition.png'}")

def plot_acf_pacf(y, output_dir):
    """ACF/PACF 플롯"""
    # 1차 차분 + 계절 차분 적용
    y_diff = y.diff(1).diff(12).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 원본 시계열의 ACF/PACF
    plot_acf(y.dropna(), ax=axes[0, 0], lags=40, title='원본 시계열 ACF')
    plot_pacf(y.dropna(), ax=axes[0, 1], lags=40, title='원본 시계열 PACF')

    # 차분된 시계열의 ACF/PACF
    plot_acf(y_diff, ax=axes[1, 0], lags=40, title='차분 후 ACF (d=1, D=1)')
    plot_pacf(y_diff, ax=axes[1, 1], lags=40, title='차분 후 PACF (d=1, D=1)')

    plt.tight_layout()
    plt.savefig(output_dir / "9-3-acf-pacf.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ACF/PACF 그래프 저장: {output_dir / '9-3-acf-pacf.png'}")

def fit_sarima(y_train, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)):
    """SARIMA 모형 적합"""
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    return result

def evaluate_forecast(y_test, forecast):
    """예측 성능 평가"""
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    mape = mean_absolute_percentage_error(y_test, forecast) * 100
    return rmse, mape

def main():
    print("=" * 60)
    print("9-3 SARIMA 시계열 예측 실습")
    print("=" * 60)

    # 1. 데이터 로드
    y = load_and_preprocess_data()
    print(f"\n데이터 기간: {y.index[0]} ~ {y.index[-1]}")
    print(f"총 관측치 수: {len(y)}")

    # 2. 정상성 검정
    print("\n" + "=" * 40)
    print("정상성 검정")
    print("=" * 40)
    test_stationarity(y, "원본 시계열")

    # 1차 차분 후 검정
    y_diff1 = y.diff(1).dropna()
    test_stationarity(y_diff1, "1차 차분 후")

    # 계절 차분 후 검정 (12개월)
    y_diff_seasonal = y.diff(12).dropna()
    test_stationarity(y_diff_seasonal, "계절 차분 후 (D=1)")

    # 1차 + 계절 차분 후 검정
    y_diff_both = y.diff(1).diff(12).dropna()
    test_stationarity(y_diff_both, "1차 + 계절 차분 후 (d=1, D=1)")

    # 3. 시계열 분해 시각화
    print("\n" + "=" * 40)
    print("시계열 분해")
    print("=" * 40)
    plot_decomposition(y, output_dir)

    # 4. ACF/PACF 시각화
    plot_acf_pacf(y, output_dir)

    # 5. 학습/테스트 분할
    train_size = int(len(y) * 0.8)
    y_train = y[:train_size]
    y_test = y[train_size:]
    print(f"\n학습 데이터: {len(y_train)}개 ({y_train.index[0]} ~ {y_train.index[-1]})")
    print(f"테스트 데이터: {len(y_test)}개 ({y_test.index[0]} ~ {y_test.index[-1]})")

    # 6. SARIMA 모형 적합 및 예측
    print("\n" + "=" * 40)
    print("SARIMA 모형 적합")
    print("=" * 40)

    # SARIMA(0,1,1)(0,1,1)_12 모형 사용
    order = (0, 1, 1)
    seasonal_order = (0, 1, 1, 12)
    print(f"모형: SARIMA{order}x{seasonal_order}")

    result = fit_sarima(y_train, order, seasonal_order)

    print("\n=== 모형 요약 ===")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")
    print(f"Log-Likelihood: {result.llf:.2f}")

    print("\n=== 모형 계수 ===")
    print(result.summary().tables[1])

    # 7. 예측
    forecast_result = result.get_forecast(steps=len(y_test))
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    # 8. 성능 평가
    rmse, mape = evaluate_forecast(y_test, forecast)
    print(f"\n=== 예측 성능 ===")
    print(f"RMSE: {rmse:.4f} ppm")
    print(f"MAPE: {mape:.2f}%")

    # 9. 예측 결과 시각화
    fig, ax = plt.subplots(figsize=(14, 6))

    # 학습 데이터
    ax.plot(y_train.index, y_train.values, 'b-', label='학습 데이터', linewidth=1)
    # 테스트 데이터
    ax.plot(y_test.index, y_test.values, 'g-', label='실제값 (테스트)', linewidth=1.5)
    # 예측값
    ax.plot(forecast.index, forecast.values, 'r--', label='SARIMA 예측', linewidth=1.5)
    # 신뢰구간
    ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    color='red', alpha=0.2, label='95% 신뢰구간')

    ax.set_xlabel('날짜')
    ax.set_ylabel('CO2 농도 (ppm)')
    ax.set_title(f'SARIMA{order}x{seasonal_order} 예측 결과 (RMSE={rmse:.2f}, MAPE={mape:.2f}%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "9-3-sarima-forecast.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n예측 그래프 저장: {output_dir / '9-3-sarima-forecast.png'}")

    # 10. 결과 저장
    results = {
        "model": f"SARIMA{order}x{seasonal_order}",
        "train_size": len(y_train),
        "test_size": len(y_test),
        "aic": round(result.aic, 2),
        "bic": round(result.bic, 2),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2)
    }

    import json
    with open(output_dir / "9-3-arima-results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {output_dir / '9-3-arima-results.json'}")

    # 예측값 CSV 저장
    forecast_df = pd.DataFrame({
        'date': y_test.index,
        'actual': y_test.values,
        'forecast': forecast.values,
        'lower_ci': conf_int.iloc[:, 0].values,
        'upper_ci': conf_int.iloc[:, 1].values
    })
    forecast_df.to_csv(output_dir / "9-3-arima-forecast.csv", index=False)
    print(f"예측값 저장: {output_dir / '9-3-arima-forecast.csv'}")

    return results

if __name__ == "__main__":
    main()
