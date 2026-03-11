import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

def generate_data(n=120):
    t = np.arange(n)
    # Trend: Linear
    trend = 0.5 * t
    # Seasonality: 12-day cycle
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    # Noise
    np.random.seed(42)
    noise = np.random.normal(0, 1, n)
    
    raw = trend + seasonal + noise
    return t, raw

def save_stationarity_plots():
    t, raw = generate_data()
    series = pd.Series(raw)
    
    # 1st Difference
    diff1 = series.diff().dropna()
    
    # 1st + Seasonal Difference (period 12)
    diff1_seasonal = diff1.diff(12).dropna()
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor='white', sharex=True)
    
    # 1. Original
    axes[0].plot(t, raw, color='black', linewidth=1)
    axes[0].set_title('Original Series (Non-stationary: Trend + Seasonality)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Value')
    
    # 2. 1st Difference
    axes[1].plot(t[1:], diff1, color='black', linewidth=1)
    axes[1].set_title('1st Differencing (Trend Removed, Seasonality Remains)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('ΔY_t')
    
    # 3. 1st + Seasonal Difference
    # Note: index adjustment for double diff
    t_final = t[1+12:]
    axes[2].plot(t_final, diff1_seasonal, color='black', linewidth=1)
    axes[2].set_title('1st + Seasonal Differencing (Stationary)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Δ_12 ΔY_t')
    axes[2].set_xlabel('Time (t)')
    
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    output_path = r'c:\Dev\book-datascience\content\graphics\ch09\9-2-4-stationarity-diff.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    save_stationarity_plots()
