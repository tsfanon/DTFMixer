import torch
import torch.nn as nn


class CrossScaleSeasonFusion(nn.Module):

    def __init__(self, configs):
        super().__init__()
        window = configs.down_sampling_window
        seq_len = configs.seq_len

        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(seq_len // (window ** i), seq_len // (window ** (i + 1))),
                    nn.GELU(),
                    nn.Linear(seq_len // (window ** (i + 1)), seq_len // (window ** (i + 1))),
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        outputs = [out_high.permute(0, 2, 1)]

        for idx in range(len(season_list) - 1):
            out_low = out_low + self.down_sampling_layers[idx](out_high)
            out_high = out_low
            if idx + 2 <= len(season_list) - 1:
                out_low = season_list[idx + 2]
            outputs.append(out_high.permute(0, 2, 1))

        return outputs


class CrossScaleTrendFusion(nn.Module):

    def __init__(self, configs):
        super().__init__()
        window = configs.down_sampling_window
        seq_len = configs.seq_len

        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(seq_len // (window ** (i + 1)), seq_len // (window ** i)),
                    nn.GELU(),
                    nn.Linear(seq_len // (window ** i), seq_len // (window ** i)),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        trend_rev = list(reversed(trend_list))
        out_low = trend_rev[0]
        out_high = trend_rev[1]
        outputs = [out_low.permute(0, 2, 1)]

        for idx in range(len(trend_rev) - 1):
            out_high = out_high + self.up_sampling_layers[idx](out_low)
            out_low = out_high
            if idx + 2 <= len(trend_rev) - 1:
                out_high = trend_rev[idx + 2]
            outputs.append(out_low.permute(0, 2, 1))

        outputs.reverse()
        return outputs
