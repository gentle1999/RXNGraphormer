"""Typed boundary for RDKit reaction objects.

RDKit exposes several Boost.Python classes whose stubs are incomplete across
type checkers. Keep the casts at this boundary and let preprocessing code use
structural protocols.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeAlias

from rdkit import Chem

ReactionProducts: TypeAlias = tuple[tuple[Chem.Mol, ...], ...]


class AtomProtocol(Protocol):
    def GetAtomMapNum(self) -> int:
        ...

    def GetAtomicNum(self) -> int:
        ...

    def GetIdx(self) -> int:
        ...

    def SetAtomMapNum(self, map_num: int) -> None:
        ...


class BondProtocol(Protocol):
    def GetBeginAtom(self) -> AtomProtocol:
        ...

    def GetEndAtom(self) -> AtomProtocol:
        ...


class MolProtocol(Protocol):
    def GetAtoms(self) -> Iterable[AtomProtocol]:
        ...

    def GetBonds(self) -> Iterable[BondProtocol]:
        ...


class ReactionProtocol(Protocol):
    def GetReactantTemplate(self, idx: int) -> Chem.Mol:
        ...

    def GetNumReactantTemplates(self) -> int:
        ...

    def GetAgentTemplate(self, idx: int) -> Chem.Mol:
        ...

    def GetNumAgentTemplates(self) -> int:
        ...

    def GetProductTemplate(self, idx: int) -> Chem.Mol:
        ...

    def GetNumProductTemplates(self) -> int:
        ...

    def Initialize(self) -> None:
        ...

    def RunReactants(self, reactants: tuple[Chem.Mol, ...]) -> ReactionProducts:
        ...
