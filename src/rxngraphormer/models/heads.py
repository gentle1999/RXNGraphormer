from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import NormType
from .norms import legacy_batch_norm_type, make_head_norm, normalize_norm_type


class RegressorLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        layer_num: int = 3,
        batch_norm: bool = False,
        act_func: str = "relu",
        norm_type: NormType | str | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layer_num-1)])
        self.norm_type = normalize_norm_type(norm_type, default=legacy_batch_norm_type(batch_norm))
        if self.norm_type == "graphnorm":
            raise ValueError("graphnorm is only supported in graph encoders")
        self.batch_norm = self.norm_type == "batchnorm"
        if act_func == 'relu':
            self.act_func = F.relu
        elif act_func == 'tanh':
            self.act_func = F.tanh
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")
        self.batch_norms = nn.ModuleList([make_head_norm(self.norm_type, hidden_size) for _ in range(layer_num-1)])
        self.projection = nn.Linear(hidden_size, output_size,bias=False)

    @property
    def norm_layers(self) -> nn.ModuleList:
        return self.batch_norms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm_layer in zip(self.layers,self.batch_norms):
            x = self.act_func(norm_layer(layer(x)))
        return self.projection(x)


class ClassifierLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int = 2,
        layer_num: int = 3,
        batch_norm: bool = True,
        act_func: str = "relu",
        norm_type: NormType | str | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layer_num-1)])
        self.norm_type = normalize_norm_type(norm_type, default=legacy_batch_norm_type(batch_norm))
        if self.norm_type == "graphnorm":
            raise ValueError("graphnorm is only supported in graph encoders")
        self.batch_norm = self.norm_type == "batchnorm"
        if act_func == 'relu':
            self.act_func = F.relu
        elif act_func == 'tanh':
            self.act_func = F.tanh
        else:
            raise ValueError(f"Unsupported activation function: {act_func}")
        self.batch_norms = nn.ModuleList([make_head_norm(self.norm_type, hidden_size) for _ in range(layer_num-1)])
        self.projection = nn.Linear(hidden_size, output_size,bias=False)

    @property
    def norm_layers(self) -> nn.ModuleList:
        return self.batch_norms

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        for layer,norm_layer in zip(self.layers,self.batch_norms):
            x = self.act_func(norm_layer(layer(x)))

        return self.projection(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(x), dim=-1)


class EXTFeatEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        hidden_layer_num: int = 3,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layer_num)])
        self.projection = nn.Linear(hidden_size, output_size)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.input_batch_norm = nn.BatchNorm1d(hidden_size)
            self.hidden_batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(hidden_layer_num)])
            self.proj_batch_norm = nn.BatchNorm1d(output_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_norm:
            x = F.relu(self.input_batch_norm(self.input_layer(x)))

            for layer,batch in zip(self.hidden_layers,self.hidden_batch_norms):
                x = F.relu(batch(layer(x)))
            return self.proj_batch_norm(self.projection(x))
        else:
            x = F.relu(self.input_layer(x))
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
            return self.projection(x)
