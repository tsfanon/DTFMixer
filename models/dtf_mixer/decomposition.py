import torch
import torch.nn as nn


class DynamicFrequencyDecomposition(nn.Module):
    """Frequency-domain seasonal/trend split with adaptive top-k selection."""

    def __init__(self, d_model, top_k=5, d_ff=64):
        super().__init__()
        self.top_k = top_k
        freq_dim = d_model // 2 + 1
        self.frequency_weights = nn.Parameter(torch.ones(1, 1, freq_dim))
        self.score_mlp = nn.Sequential(
            nn.Linear(freq_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, freq_dim),
        )

    def forward(self, x):
        # x: [B, T, C]
        _, _, channels = x.shape
        xf = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(xf) * self.frequency_weights
        mean_amp = torch.mean(amp, dim=(0, 1))
        scores = torch.sigmoid(self.score_mlp(mean_amp))
        _, top_indices = torch.topk(scores, self.top_k)
        mask = torch.zeros_like(scores, device=x.device)
        mask.scatter_(0, top_indices, 1.0)
        xf_masked = xf * mask.view(1, 1, -1)
        season = torch.fft.irfft(xf_masked, n=channels, dim=-1)
        trend = x - season
        return season, trend
