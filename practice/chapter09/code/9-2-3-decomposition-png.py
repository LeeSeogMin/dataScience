import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
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

def generate_data(n=120, noise_level=5):
    t = np.arange(n)
    # Trend: slightly curved
    trend = 20 + 0.1 * t + 0.001 * t**2
    # Seasonality: 12-day cycle
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    # Residual
    np.random.seed(42)
    resid = (np.random.rand(n) - 0.5) * noise_level * 2
    
    raw = trend + seasonal + resid
    return t, raw, trend, seasonal, resid

def save_moving_average_plot(t, raw, k=7):
    # Calculate Moving Average
    window = 2*k + 1
    ma = pd.Series(raw).rolling(window=window, center=True).mean()
    
    plt.figure(figsize=(10, 5), facecolor='white')
    plt.plot(t, raw, color='#bbbbbb', linewidth=1, label='Original (Y_t)')
    plt.plot(t, ma, color='black', linewidth=2, label=f'Moving Average (Trend, k={k})')
    
    plt.title('Moving Average Smoothing', fontsize=14, fontweight='bold')
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    
    output_path = r'c:\Dev\book-datascience\content\graphics\ch09\9-2-3-moving-average.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def save_stl_decomposition_plot(t, raw):
    # STL Decomposition
    series = pd.Series(raw, index=pd.date_range(start='2020-01-01', periods=len(raw), freq='D'))
    res = STL(series, period=12).fit()
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), facecolor='white', sharex=True)
    
    # Original
    axes[0].plot(t, raw, color='black', linewidth=1)
    axes[0].set_ylabel('Original', fontsize=10)
    axes[0].set_title('STL Decomposition', fontsize=14, fontweight='bold')
    
    # Trend
    axes[1].plot(t, res.trend, color='black', linewidth=1.5)
    axes[1].set_ylabel('Trend', fontsize=10)
    
    # Seasonality
    axes[2].plot(t, res.seasonal, color='black', linewidth=1)
    axes[2].set_ylabel('Seasonal', fontsize=10)
    
    # Residuals
    axes[3].scatter(t, res.resid, color='black', s=5)
    axes[3].vlines(t, 0, res.resid, color='black', linewidth=0.5)
    axes[3].set_ylabel('Residual', fontsize=10)
    axes[3].set_xlabel('Time (t)', fontsize=12)
    
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    output_path = r'c:\Dev\book-datascience\content\graphics\ch09\9-2-3-stl-decomposition.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    t, raw, trend, seasonal, resid = generate_data()
    save_moving_average_plot(t, raw)
    save_stl_decomposition_plot(t, raw)
