#!/usr/bin/env python3
"""
7-4c-latent-umap.py
------------------
Autoencoder latent space를 2차원으로 UMAP 시각화하여 원본 특성과 비교함으로써
잠재 벡터가 제공하는 대안적인 시그널을 살펴보는 실습입니다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import umap
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_autoencoder(model: Autoencoder, loader: DataLoader, device: torch.device, epochs: int = 40):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(loader.dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[AE] Epoch {epoch:02d} | Train Loss: {avg_loss:.4f}")
    return model


def build_loader(X: np.ndarray, batch_size: int = 512) -> DataLoader:
    tensor = torch.from_numpy(X)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_synthetic_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; generate it via practice/chapter7/code/7-3b-synthetic-comparison.py")
    df = pd.read_csv(path)
    X = df.drop(columns="target").values.astype(np.float32)
    y = df["target"].values
    return X, y


def compute_latent(model: Autoencoder, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(X).to(device)
        z = model.get_latent(tensor)
    return z.cpu().numpy()


def visualize(original_2d, latent_2d, labels, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    palette = ["#1f77b4", "#ff7f0e"]
    for ax, values, title in zip(
        axes,
        (original_2d, latent_2d),
        ("Original Feature UMAP", "Latent Representation UMAP")
    ):
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(values[mask, 0], values[mask, 1], s=9, alpha=0.7, label=f"class {label}",
                       c=palette[label])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(frameon=False, title="Type")
    plt.suptitle("Original vs Autoencoder Latent Space UMAP", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Figure saved to {output_path}")


def main():
    set_seed(42)
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "data" / "synthetic_complex.csv"
    X, y = load_synthetic_data(data_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(X_scaled)
    autoencoder = Autoencoder(input_dim=X_scaled.shape[1], latent_dim=32)
    train_autoencoder(autoencoder, loader, device, epochs=40)

    sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=5000, replace=False)
    sample_X = X_scaled[sample_idx]
    sample_y = y[sample_idx]

    latent = compute_latent(autoencoder, sample_X, device)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    original_umap = reducer.fit_transform(sample_X)
    latent_umap = reducer.fit_transform(latent)

    root = Path(__file__).resolve().parents[3]
    fig_path = root / "content" / "graphics" / "ch07" / "7-4c-latent-umap.png"
    visualize(original_umap, latent_umap, sample_y, fig_path)


if __name__ == "__main__":
    main()
