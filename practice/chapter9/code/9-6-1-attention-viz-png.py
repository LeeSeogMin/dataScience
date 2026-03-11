import numpy as np
import matplotlib.pyplot as plt
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

def save_attention_viz():
    n = 60
    t = np.arange(n)
    
    # Generate a synthetic time series with seasonality
    np.random.seed(42)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    trend = 0.2 * t
    noise = np.random.normal(0, 1, n)
    y = seasonal + trend + noise + 20
    
    # Target index (current time step)
    target_idx = 58
    
    # Simulate attention weights
    # High attention to same seasonal phase (t-12, t-24, t-36...)
    attention_weights = np.zeros(n)
    for i in range(n):
        dist = abs(i - target_idx)
        # Seasonal similarity
        seasonal_sim = np.exp(-0.5 * ((i % 12 - target_idx % 12) / 1.0)**2)
        # Recency bias
        recency = np.exp(-0.05 * dist)
        attention_weights[i] = seasonal_sim * recency
    
    # Normalize weights
    attention_weights /= np.sum(attention_weights)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='white', gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Time Series Plot with Attention Highlights
    ax1.plot(t, y, color='#bbbbbb', linewidth=1, label='Time Series')
    ax1.scatter(t, y, c=attention_weights, cmap='Greys', s=attention_weights*2000, edgecolors='black', alpha=0.8, label='Attention Intensity')
    ax1.annotate('Target (Query)', xy=(target_idx, y[target_idx]), xytext=(target_idx-10, y[target_idx]+10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=10, fontweight='bold')
    
    # Mark seasonal peaks that get high attention
    for lag in [12, 24, 36, 48]:
        idx = target_idx - lag
        if idx >= 0:
            ax1.annotate('', xy=(idx, y[idx]), xytext=(target_idx, y[target_idx]),
                         arrowprops=dict(arrowstyle='<->', color='black', alpha=0.3, linestyle='--'))

    ax1.set_title('Transformer Self-Attention in Time Series', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Attention Weights Heatmap/Bar
    ax2.bar(t, attention_weights, color='black', alpha=0.7)
    ax2.set_ylabel('Weight')
    ax2.set_xlabel('Time (t)')
    ax2.set_title('Attention Score Distribution', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_path = r'c:\Dev\book-datascience\content\graphics\ch09\9-6-1-attention-viz.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    save_attention_viz()
