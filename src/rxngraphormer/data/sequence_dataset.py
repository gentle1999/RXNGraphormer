"""Graph-to-sequence PyG dataset."""

from multiprocessing import Pool

import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from ..serialization import (
    existing_processed_file_name,
    load_processed_graph_data,
    save_processed_graph_data,
)
from .graph_data import ReactionGraphData, as_reaction_graph_data
from .reaction_graph import get_rxn_seq_info
from .tokenization import load_vocab


class RXNG2SDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        src_file,
        tgt_file,
        vocab_file,
        max_length=1024,
        transform=None,
        pre_transform=None,
        train=True,
        trunck=None,
        num_worker=8,
        multi_process=True,
        oh=True,
    ):
        self.root = root
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.vocab_file = f"{root}/{vocab_file}"
        self.vocab = load_vocab(self.vocab_file)
        self.max_length = max_length
        self.train = train
        self.trunck = trunck if trunck is not None and trunck != 0 else None
        self.num_worker = num_worker
        self.multi_process = multi_process
        self.oh = oh
        super().__init__(root, transform, pre_transform)
        loaded_data, self.slices = load_processed_graph_data(self.processed_paths[0])
        self.data = as_reaction_graph_data(loaded_data)

    @property
    def raw_file_names(self):
        return [self.src_file, self.tgt_file]

    @property
    def processed_file_names(self):
        if self.oh:
            names = [f"{self.src_file[:-4]}_{self.trunck}.safetensors"]
        else:
            names = [f"{self.src_file[:-4]}_{self.trunck}_mini.safetensors"]
        return [existing_processed_file_name(self.processed_dir, name) for name in names]

    def process(self):
        self.data_list = []
        with open(f"{self.root}/{self.src_file}") as fr:
            self.src_lines = fr.readlines()
        with open(f"{self.root}/{self.tgt_file}") as fr:
            self.tgt_lines = fr.readlines()
        assert len(self.src_lines) == len(self.tgt_lines), (
            "src and tgt file length not equal"
        )

        if self.trunck is not None:
            self.src_lines = self.src_lines[: self.trunck]
            self.tgt_lines = self.tgt_lines[: self.trunck]
        if self.multi_process:
            print(f"[INFO] {self.num_worker} workers are used to process data...")
            pool = Pool(self.num_worker)
            tasks = (
                (self.src_lines[i], self.tgt_lines[i], self.vocab, self.max_length)
                for i in range(len(self.src_lines))
            )
            results = []
            for result in tqdm(
                pool.imap(get_rxn_seq_info, tasks), total=len(self.src_lines)
            ):
                results.append(result)
            pool.close()
            pool.join()
            for rxn_inf in results:
                if rxn_inf is None:
                    continue
                (
                    x_merge,
                    edge_index_merge,
                    edge_attr_merge,
                    mol_index,
                    atom_mass_merge,
                    x_oh_merge,
                    edge_oh_attr_merge,
                    a_graphs_merge,
                    b_graphs_merge,
                    tgt_token_ids,
                    tgt_lens,
                ) = rxn_inf
                if self.oh:
                    data = ReactionGraphData(
                        x=x_merge,
                        edge_index=edge_index_merge,
                        edge_attr=edge_attr_merge,
                        x_oh=x_oh_merge,
                        edge_oh_attr=torch.cat(
                            (edge_index_merge.T, edge_oh_attr_merge), dim=1
                        ),
                        mol_index=mol_index,
                        tgt_token_ids=tgt_token_ids,
                        tgt_lens=tgt_lens,
                        a_graphs=a_graphs_merge,
                        b_graphs=b_graphs_merge,
                    )
                else:
                    data = ReactionGraphData(
                        x=x_merge,
                        edge_index=edge_index_merge,
                        edge_attr=edge_attr_merge,
                        mol_index=mol_index,
                        tgt_token_ids=tgt_token_ids,
                        tgt_lens=tgt_lens,
                    )
                self.data_list.append(data)
        else:
            for i in tqdm(range(len(self.src_lines))):
                src_line, tgt_line = self.src_lines[i], self.tgt_lines[i]
                rxn_inf = get_rxn_seq_info(
                    (src_line, tgt_line, self.vocab, self.max_length)
                )
                if rxn_inf is None:
                    continue
                (
                    x_merge,
                    edge_index_merge,
                    edge_attr_merge,
                    mol_index,
                    atom_mass_merge,
                    x_oh_merge,
                    edge_oh_attr_merge,
                    a_graphs_merge,
                    b_graphs_merge,
                    tgt_token_ids,
                    tgt_lens,
                ) = rxn_inf
                if self.oh:
                    data = ReactionGraphData(
                        x=x_merge,
                        edge_index=edge_index_merge,
                        edge_attr=edge_attr_merge,
                        x_oh=x_oh_merge,
                        edge_oh_attr=torch.cat(
                            (edge_index_merge.T, edge_oh_attr_merge), dim=1
                        ),
                        mol_index=mol_index,
                        tgt_token_ids=tgt_token_ids,
                        tgt_lens=tgt_lens,
                        a_graphs=a_graphs_merge,
                        b_graphs=b_graphs_merge,
                    )
                else:
                    data = ReactionGraphData(
                        x=x_merge,
                        edge_index=edge_index_merge,
                        edge_attr=edge_attr_merge,
                        mol_index=mol_index,
                        tgt_token_ids=tgt_token_ids,
                        tgt_lens=tgt_lens,
                    )
                self.data_list.append(data)

        if not self.data_list:
            raise ValueError(
                f"No valid sequence graph data was generated from {self.src_file} and {self.tgt_file}"
            )
        data, slices = self.collate(self.data_list)
        print(f"[INFO] {len(self.data_list)} Saving...")
        save_processed_graph_data(data, slices, self.processed_paths[0])

    def download(self):
        pass


__all__ = ["RXNG2SDataset"]
