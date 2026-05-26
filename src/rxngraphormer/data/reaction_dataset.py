"""Single-file reaction PyG dataset."""

from torch_geometric.data import InMemoryDataset

from ..serialization import (
    existing_processed_file_name,
    load_processed_graph_data,
    save_processed_graph_data,
)
from .graph_data import as_reaction_graph_data
from .reaction_processing import ReactionProcessingSettings, process_reaction_lines


class RXNDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        name="rxn_dataset.csv",
        raw_data=[],
        transform=None,
        pre_transform=None,
        train=True,
        trunck=None,
        task="regression",
        num_worker=8,
        batch_size=128,
        multi_process=False,
        ext_feat=False,
        ext_feat_type="morgan",
        ext_feat_param={"radius": 2, "nBits": 2048, "useChirality": True},
        mul_ext_readout="mean",
        tag="",
        parallel_mode="reaction",
    ):
        """
        Multi-process is useless in this procedure
        """
        self.name = name
        self.train = train
        self.raw_data = raw_data
        self.trunck = trunck if trunck is not None and trunck != 0 else None
        self.task = task
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.multi_process = multi_process
        self.ext_feat = ext_feat  # whether to extend the description, eg. Morgan fingerprint, RDKit descriptors, etc.
        self.ext_feat_type = ext_feat_type.lower()
        self.ext_feat_param = ext_feat_param
        self.mul_ext_readout = mul_ext_readout.lower()
        self.tag = tag
        self.parallel_mode = parallel_mode
        super().__init__(root, transform, pre_transform)
        loaded_data, self.slices = load_processed_graph_data(self.processed_paths[0])
        self.data = as_reaction_graph_data(loaded_data)

    @property
    def raw_file_names(self):
        return self.name

    @property
    def processed_file_names(self):
        if not self.ext_feat:
            if not self.tag:
                names = [f"{self.name[:-4]}_{self.trunck}.safetensors"]
            else:
                names = [f"{self.name[:-4]}_{self.trunck}_{self.tag}.safetensors"]
        else:
            if self.ext_feat_type == "rdkit":
                if not self.tag:
                    names = [f"{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}.safetensors"]
                else:
                    names = [f"{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.tag}.safetensors"]
            elif self.ext_feat_type in ["morgan", "atompair", "toptorsion", "rdfp"]:
                if not self.tag:
                    names = [
                        f"{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.ext_feat_param['radius']}_{self.ext_feat_param['nBits']}.safetensors"
                    ]
                else:
                    names = [
                        f"{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.ext_feat_param['radius']}_{self.ext_feat_param['nBits']}_{self.tag}.safetensors"
                    ]
            elif "++" in self.ext_feat_type:
                if not self.tag:
                    names = [
                        f"{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.mul_ext_readout}.safetensors"
                    ]
                else:
                    names = [
                        f"{self.name[:-4]}_{self.trunck}_{self.ext_feat_type}_{self.mul_ext_readout}_{self.tag}.safetensors"
                    ]
            else:
                raise ValueError(f"ext_feat_type {self.ext_feat_type} is not supported")
        return [existing_processed_file_name(self.processed_dir, name) for name in names]

    def process(self):
        with open(f"{self.root}/{self.name}") as f:
            rxn_smi_tgt_lst = [line.strip() for line in f.readlines()]
        if self.trunck is not None:
            rxn_smi_tgt_lst = rxn_smi_tgt_lst[: self.trunck]

        self.data_list = process_reaction_lines(
            rxn_smi_tgt_lst,
            ReactionProcessingSettings(
                task=self.task,
                ext_feat=self.ext_feat,
                ext_feat_type=self.ext_feat_type,
                ext_feat_param=self.ext_feat_param,
                mul_ext_readout=self.mul_ext_readout,
                multi_process=self.multi_process,
                num_workers=self.num_worker,
                batch_size=self.batch_size,
                include_atom_mass=True,
                raise_errors=True,
                log_invalid=False,
            ),
            desc=self.name,
        )

        if not self.data_list:
            raise ValueError(
                f"No valid reaction graph data was generated from {self.name}"
            )
        data, slices = self.collate(self.data_list)
        print(f"[INFO] {len(self.data_list)} Saving...")
        save_processed_graph_data(data, slices, self.processed_paths[0])

    def download(self):
        pass


__all__ = ["RXNDataset"]
