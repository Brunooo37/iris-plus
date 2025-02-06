import duckdb
import torch
import torch.nn as nn
from lancedb.table import Table

from ragifier.config import Config


class Ragifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
        initial_queries: torch.Tensor,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(initial_queries)
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
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x, src_key_padding_mask=padding_mask)
        out = self.fc(out)
        return out


def get_num_classes(tbl: Table):
    _ = tbl.to_lance()  # type: ignore
    num_labels = duckdb.sql("SELECT COUNT(DISTINCT label) FROM _").fetchone()
    return num_labels[0] if num_labels else 0


def make_model(tbl: Table, vector_dim: int, initial_queries: torch.Tensor, cfg: Config):
    num_classes = get_num_classes(tbl=tbl)
    model = Ragifier(
        num_classes=num_classes,
        d_model=vector_dim,
        nhead=cfg.model.nhead,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        num_layers=cfg.model.num_layers,
        initial_queries=initial_queries,
    )
    model.to(cfg.device)
    return model
