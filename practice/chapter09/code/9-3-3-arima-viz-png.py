import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os

# Set aesthetic parameters for black and white
plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "font.family": "serif"
})

def generate_arima_data(n=100):
    # Generate ARIMA(1, 1, 1) process
    # First, generate ARMA(1, 1)
    np.random.seed(42)
    ar = np.array([0.7])
    ma = np.array([0.4])
    
    # Simulate ARMA(1, 1) errors
    errors = np.random.normal(0, 1, n)
    arma_values = np.zeros(n)
    for t in range(1, n):
        arma_values[t] = ar[0] * arma_values[t-1] + errors[t] + ma[0] * errors[t-1]
    
    # Integrate to get ARIMA(1, 1, 1)
    # Start with a trend
    trend = 0.5 * np.arange(n)
    arima_values = np.cumsum(arma_values) + trend + 50
    return arima_values

def save_arima_viz():
    n = 100
    data = generate_arima_data(n)
    train_size = int(n * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # Fit ARIMA(1, 1, 1)
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Forecast
    forecast_steps = 20
    forecast_res = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    
    t = np.arange(n)
    t_forecast = np.arange(train_size, train_size + forecast_steps)
    
    plt.figure(figsize=(10, 5), facecolor='white')
    
    # Observation
    plt.plot(t[:train_size], train, color='black', linewidth=1.5, label='Actual (Train)')
    plt.plot(t[train_size:], test[:len(t[train_size:])], color='#bbbbbb', linewidth=1, label='Actual (Test)')
    
    # Forecast
    plt.plot(t_forecast, forecast_mean, color='black', linestyle='--', linewidth=2, label='ARIMA(1,1,1) Forecast')
    
    # Confidence Interval
    plt.fill_between(t_forecast, conf_int[:, 0], conf_int[:, 1], color='black', alpha=0.1, label='95% Confidence Interval')
    
    plt.title('ARIMA(1,1,1) Model: Observation and Forecast', fontsize=14, fontweight='bold')
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(frameon=False, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    # Clean up spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = r'c:\Dev\book-datascience\content\graphics\ch09\9-3-3-arima-viz.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    save_arima_viz()
