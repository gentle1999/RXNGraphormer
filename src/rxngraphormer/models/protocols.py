"""Typing protocols for shared RXNGraphormer model capabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch


class RXNGraphEncoderProtocol(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        mol_index: object,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        atom_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


class RXNEncoderProtocol(Protocol):
    rxn_graph_encoder: RXNGraphEncoderProtocol

    def __call__(self, data: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


class DecoderProtocol(Protocol):
    layers: Sequence[torch.nn.Module]
    batch_norms: Sequence[torch.nn.Module]
    norm_layers: Sequence[torch.nn.Module]
    batch_norm: bool


class ReactionEmbeddingModel(Protocol):
    rct_encoder: RXNEncoderProtocol
    pdt_encoder: RXNEncoderProtocol
    trans_readout: str
    split_merge_method: str
    decoder: DecoderProtocol


class OptimizerGroupedModel(Protocol):
    rct_encoder: torch.nn.Module
    pdt_encoder: torch.nn.Module
    decoder: torch.nn.Module
    mid_encoder: torch.nn.Module
    mid_iteract: torch.nn.Module
    mid_decoder: torch.nn.Module


class SequenceBatchProtocol(Protocol):
    tgt_token_ids: torch.Tensor
    tgt_lens: torch.Tensor


class SequenceModelProtocol(Protocol):
    vocab: dict[str, int]
    encoder: torch.nn.Module
    attention_encoder: torch.nn.Module
    decoder: torch.nn.Module
    output_layer: torch.nn.Module

    def __call__(self, reaction_batch: object) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def infer(
        self,
        reaction_batch: object,
        batch_size: int,
        beam_size: int,
        n_best: int,
        temperature: float,
        min_length: int,
        max_length: int,
    ) -> Mapping[str, object]:
        ...


@runtime_checkable
class SupportsLogits(Protocol):
    def logits(self, data: object) -> torch.Tensor:
        ...
