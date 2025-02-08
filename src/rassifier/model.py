import numpy as np
import torch
import torch.nn as nn

from rassifier.config import ModelConfig


def masked_mean_pool(input: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    mask = (~padding_mask).unsqueeze(-1).float()
    return (input * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class Rassifier(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        output_dim: int,
        ini_queries: torch.Tensor,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(ini_queries)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        out = self.encoder(x, src_key_padding_mask=padding_mask)
        out = masked_mean_pool(out, padding_mask)
        out = self.fc(out)
        return out


def make_model(ini_queries: np.ndarray, cfg: ModelConfig):
    model_config = cfg.model_dump(exclude={"num_queries", "query_ini_random", "device"})
    queries = torch.tensor(ini_queries, dtype=torch.float)
    queries.to(cfg.device)
    return Rassifier(ini_queries=queries, **model_config)
