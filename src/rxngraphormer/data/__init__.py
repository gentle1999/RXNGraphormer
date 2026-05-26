"""Public dataset API.

Legacy data/dataset APIs remain import-compatible from this package while their
implementation is split across focused data modules. New code should type
against `data_protocol` where possible.
"""

from ..data_protocol import ReactionDatasetProtocol, ReactionDatasetSpec, validate_graph_data
from .batch import (
    add_dense_empty_node_edge,
    add_empty_node_and_edge,
    pad_feat,
    update_batch_idx,
)
from .collate import pair_collate_fn, single_collate_fn, triple_collate_fn
from .constants import (
    ATOM_DICT,
    ATOM_FEAT_DIMS,
    ATOM_LST,
    BOND_DIR_LST,
    BOND_FEAT_DIME,
    BOND_STEREO_LST,
    BOND_TYPE_LST,
    FC_DICT,
    HYBRIDTYPE_DICT,
    HYBRIDTYPE_LST,
    MAX_NEIGHBORS,
    NUM_AROMATIC_NUM,
    NUM_ATOM_TYPE,
    NUM_BOND_DIRECTION,
    NUM_BOND_INRING,
    NUM_BOND_ISCONJ,
    NUM_BOND_STEREO,
    NUM_BOND_TYPE,
    NUM_CHIRAL_TYPE,
    NUM_DEGRESS_TYPE,
    NUM_FORMCHRG_TYPE,
    NUM_HYBRIDTYPE,
    NUM_RS_TPYE,
    NUM_VALENCE_TYPE,
    RS_TAG_DICT,
    RS_TAG_LST,
    NUM_Hs_LST,
    NUM_Hs_TYPE,
)
from .features import (
    desc_calc,
    descs,
    ext_feat_gen,
    get_fp,
    get_rdkit_desc,
    rdkit_desc_dim,
    sel_descs,
)
from .files import (
    _collated_data_count,
    _matched_raw_files,
    _raw_file_sort_key,
    _separate_collated_data,
)
from .graph import (
    calc_batch_graph_distance,
    calc_graph_distance,
    gen_onehot,
    get_agraph,
    get_bgraphs,
    mol2graphinfo,
)
from .graph_data import ReactionGraphData, as_reaction_graph_data
from .multi_reaction_dataset import MultiRXNDataset
from .pairing import PairDataset, TripleDataset
from .reaction_dataset import RXNDataset
from .reaction_graph import (
    gen_mol_in_rxn,
    generate_regression_dataset,
    get_cpd_rxn_info,
    get_rxn_pfm_info,
    get_rxn_seq_info,
)
from .sequence_dataset import RXNG2SDataset
from .splits import get_idx_split
from .tokenization import (
    gen_vocab_map,
    get_token_ids,
    get_train_val_test_token_data,
    load_vocab,
    smi_tokenizer,
)

__all__ = [
    "ATOM_DICT",
    "ATOM_FEAT_DIMS",
    "ATOM_LST",
    "BOND_DIR_LST",
    "BOND_FEAT_DIME",
    "BOND_STEREO_LST",
    "BOND_TYPE_LST",
    "FC_DICT",
    "HYBRIDTYPE_DICT",
    "HYBRIDTYPE_LST",
    "MAX_NEIGHBORS",
    "MultiRXNDataset",
    "NUM_AROMATIC_NUM",
    "NUM_ATOM_TYPE",
    "NUM_BOND_DIRECTION",
    "NUM_BOND_INRING",
    "NUM_BOND_ISCONJ",
    "NUM_BOND_STEREO",
    "NUM_BOND_TYPE",
    "NUM_CHIRAL_TYPE",
    "NUM_DEGRESS_TYPE",
    "NUM_FORMCHRG_TYPE",
    "NUM_HYBRIDTYPE",
    "NUM_Hs_LST",
    "NUM_Hs_TYPE",
    "NUM_RS_TPYE",
    "NUM_VALENCE_TYPE",
    "PairDataset",
    "ReactionDatasetProtocol",
    "ReactionDatasetSpec",
    "ReactionGraphData",
    "RXNDataset",
    "RXNG2SDataset",
    "RS_TAG_DICT",
    "RS_TAG_LST",
    "TripleDataset",
    "_collated_data_count",
    "_matched_raw_files",
    "_raw_file_sort_key",
    "_separate_collated_data",
    "add_dense_empty_node_edge",
    "add_empty_node_and_edge",
    "as_reaction_graph_data",
    "calc_batch_graph_distance",
    "calc_graph_distance",
    "desc_calc",
    "descs",
    "ext_feat_gen",
    "gen_mol_in_rxn",
    "gen_onehot",
    "gen_vocab_map",
    "generate_regression_dataset",
    "get_agraph",
    "get_bgraphs",
    "get_cpd_rxn_info",
    "get_fp",
    "get_idx_split",
    "get_rdkit_desc",
    "get_rxn_pfm_info",
    "get_rxn_seq_info",
    "get_token_ids",
    "get_train_val_test_token_data",
    "load_vocab",
    "mol2graphinfo",
    "pad_feat",
    "pair_collate_fn",
    "rdkit_desc_dim",
    "sel_descs",
    "single_collate_fn",
    "smi_tokenizer",
    "triple_collate_fn",
    "update_batch_idx",
    "validate_graph_data",
]
