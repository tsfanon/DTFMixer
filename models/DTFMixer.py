import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from models.dtf_mixer import TimeFrequencyDistillation


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence

        self.pdm_blocks = nn.ModuleList(
            [TimeFrequencyDistillation(configs) for _ in range(configs.e_layers)]
        )
        self.preprocess = series_decomp(configs.moving_avg)

        embed_in = 1 if self.channel_independence else configs.enc_in
        self.enc_embedding = DataEmbedding_wo_pos(
            embed_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.layer = configs.e_layers
        self.normalize_layers = nn.ModuleList(
            [
                Normalize(
                    configs.enc_in,
                    affine=True,
                    non_norm=(configs.use_norm == 0),
                )
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.scale_count = configs.down_sampling_layers + 1
            self.predict_layers = nn.ModuleList(
                [
                    nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(self.scale_count)
                ]
            )

            if self.channel_independence:
                self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)
                self.out_res_layers = nn.ModuleList(
                    [
                        nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.seq_len // (configs.down_sampling_window ** i),
                        )
                        for i in range(self.scale_count)
                    ]
                )
                self.regression_layers = nn.ModuleList(
                    [
                        nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(self.scale_count)
                    ]
                )

            self.scale_gates = nn.Parameter(torch.zeros(self.scale_count))
            self.scale_context_proj = nn.Sequential(
                nn.LayerNorm(self.scale_count),
                nn.Linear(self.scale_count, self.scale_count),
                nn.GELU(),
                nn.Linear(self.scale_count, self.scale_count),
            )


    def out_projection(self, dec_out, idx, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[idx](out_res)
        out_res = self.regression_layers[idx](out_res).permute(0, 2, 1)
        return dec_out + out_res

    def pre_enc(self, x_list):
        if self.channel_independence:
            return x_list, None

        seasonal, residual = [], []
        for x in x_list:
            x_1, x_2 = self.preprocess(x)
            seasonal.append(x_1)
            residual.append(x_2)
        return seasonal, residual

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        method = self.configs.down_sampling_method
        window = self.configs.down_sampling_window

        if method == "max":
            down_pool = nn.MaxPool1d(window, return_indices=False)
        elif method == "avg":
            down_pool = nn.AvgPool1d(window)
        elif method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.configs.enc_in,
                out_channels=self.configs.enc_in,
                kernel_size=3,
                padding=padding,
                stride=window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return x_enc, x_mark_enc

        x_enc = x_enc.permute(0, 2, 1)
        x_enc_current = x_enc
        x_mark_current = x_mark_enc

        x_enc_list = [x_enc_current.permute(0, 2, 1)]
        x_mark_list = [x_mark_current] if x_mark_current is not None else None

        for _ in range(self.configs.down_sampling_layers):
            x_enc_current = down_pool(x_enc_current)
            x_enc_list.append(x_enc_current.permute(0, 2, 1))

            if x_mark_current is not None:
                x_mark_current = x_mark_current[:, ::window, :]
                x_mark_list.append(x_mark_current)

        return x_enc_list, x_mark_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        batch_size = x_enc[0].size(0)

        normalized_inputs = []
        mark_list = [] if x_mark_enc is not None else None

        if x_mark_enc is not None:
            for idx, (x, x_mark) in enumerate(zip(x_enc, x_mark_enc)):
                B, T, N = x.size()
                x = self.normalize_layers[idx](x, "norm")
                if self.channel_independence:
                    reshaped = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    normalized_inputs.append(reshaped)
                    mark_list.append(x_mark.repeat(N, 1, 1))
                else:
                    normalized_inputs.append(x)
                    mark_list.append(x_mark)
        else:
            for idx, x in enumerate(x_enc):
                B, T, N = x.size()
                x = self.normalize_layers[idx](x, "norm")
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                normalized_inputs.append(x)

        processed_inputs = self.pre_enc(normalized_inputs)
        enc_out_list = []

        if mark_list is not None:
            for x, x_mark in zip(processed_inputs[0], mark_list):
                enc_out_list.append(self.enc_embedding(x, x_mark))
        else:
            for x in processed_inputs[0]:
                enc_out_list.append(self.enc_embedding(x, None))

        for block in self.pdm_blocks:
            enc_out_list = block(enc_out_list)

        dec_out_list = self.future_multi_mixing(batch_size, enc_out_list, processed_inputs)

        dec_out_stack = torch.stack(dec_out_list, dim=-1)
        scale_context = dec_out_stack.mean(dim=1).mean(dim=1)
        dynamic_logits = self.scale_context_proj(scale_context)
        prior_logits = self.scale_gates.unsqueeze(0).expand_as(dynamic_logits)
        scale_weights = torch.softmax(prior_logits + dynamic_logits, dim=-1)
        dec_out = torch.einsum("btds,bs->btd", dec_out_stack, scale_weights)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def future_multi_mixing(self, batch_size, enc_out_list, processed_inputs):
        dec_out_list = []
        residuals = None if self.channel_independence else processed_inputs[1]

        for idx, enc_out in enumerate(enc_out_list):
            dec_out = self.predict_layers[idx](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            if self.channel_independence:
                dec_out = self.projection_layer(dec_out)
                dec_out = (
                    dec_out.reshape(batch_size, self.configs.c_out, self.pred_len)
                    .permute(0, 2, 1)
                    .contiguous()
                )
            else:
                dec_out = self.out_projection(dec_out, idx, residuals[idx])
            dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise ValueError("Only forecasting tasks are currently supported.")
