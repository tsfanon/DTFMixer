import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp
from .decomposition import DynamicFrequencyDecomposition
from .mixing import CrossScaleSeasonFusion, CrossScaleTrendFusion


class TimeFrequencyDistillation(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channel_independence = configs.channel_independence

        fusion_hidden = max(configs.d_model, configs.d_ff)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(configs.d_model * 2, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, configs.d_model),
            nn.Sigmoid(),
        )

        if configs.decomp_method == "moving_avg":
            self.decomposition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            channels = configs.d_model if not self.channel_independence else 1
            self.decomposition = DynamicFrequencyDecomposition(
                d_model=channels,
                top_k=configs.top_k,
                d_ff=configs.d_ff,
            )
        else:
            raise ValueError("Unknown decomposition method")

        if not self.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
            )

        self.mixing_multi_scale_season = CrossScaleSeasonFusion(configs)
        self.mixing_multi_scale_trend = CrossScaleTrendFusion(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]
        season_list, trend_list = [], []

        for x in x_list:
            season, trend = self.decomposition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        outputs = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            fusion_input = torch.cat([out_season, out_trend], dim=-1)
            gate = self.adaptive_fusion(fusion_input)
            out = gate * out_season + (1 - gate) * out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            outputs.append(out[:, :length, :])

        return outputs

