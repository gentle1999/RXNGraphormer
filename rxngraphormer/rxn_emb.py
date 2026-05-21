import os
import tempfile

import torch
from rxngraphormer.config import load_config, resolve_config_file
from tqdm import tqdm
from .checkpointing import CheckpointAdapter
from .model import masked_sequence_mean
from .model_factory import build_classification_model, build_regression_model
from .reaction import canonicalize_reaction_side, split_reaction_smiles
from .predictor import RXNGraphormerPredictor
from .utils import canonical_smiles
from .data import MultiRXNDataset,PairDataset,pair_collate_fn,single_collate_fn
from torch.nn.init import xavier_uniform_
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RXNEMB():
    def __init__(self,pretrained_model_path,random_init=False,model_type="classifier"):
        pretrained_config = load_config(resolve_config_file(pretrained_model_path))
        ckpt_file = f"{pretrained_model_path}/model/valid_checkpoint.pt"
        if model_type == "classifier":
            model = build_classification_model(pretrained_config)
        elif model_type == "regressor":
            model = build_regression_model(pretrained_config)
        else:
            raise ValueError("model_type must be either 'classifier' or 'regressor'")

        if not random_init:
            CheckpointAdapter().load_into_model(model, ckpt_file, map_location=device, mode="strict")
        else:
            print("[INFO] Randomly initialize model parameters")
            for p in model.parameters():
                if p.dim() > 1 and p.requires_grad:
                    xavier_uniform_(p)
            
        model.to(device)
        model.eval()
        self.model = model
    def gen_rxn_emb_from_dataset(self,root,
                                 rct_name_regrex="50k_rxn_type_rct_0.csv",pdt_name_regrex="50k_rxn_type_pdt_0.csv",batch_size=128):
        rct_dataset = MultiRXNDataset(root=root, name_regrex=rct_name_regrex)
        pdt_dataset = MultiRXNDataset(root=root, name_regrex=pdt_name_regrex)
        pair_dataset = PairDataset(rct_dataset,pdt_dataset)
        pair_dataloader = torch.utils.data.DataLoader(pair_dataset, batch_size=batch_size, shuffle=False,collate_fn=pair_collate_fn)
        all_rxn_emb = []
        print("[INFO] Generating reaction embedding...")
        with torch.no_grad():
            for data in tqdm(pair_dataloader):
                rct_data,pdt_data = data
                rct_data.to(device)
                pdt_data.to(device)
                rct_padded_memory_bank,rct_batch,rct_memory_lengths = self.model.rct_encoder(rct_data)
                pdt_padded_memory_bank,pdt_batch,pdt_memory_lengths = self.model.pdt_encoder(pdt_data)
                rct_rxn_transf_emb = rct_padded_memory_bank.transpose(0,1)
                pdt_rxn_transf_emb = pdt_padded_memory_bank.transpose(0,1)
                if self.model.trans_readout == 'mean':
                    rct_rxn_transf_emb_merg = masked_sequence_mean(rct_rxn_transf_emb, rct_memory_lengths)
                    pdt_rxn_transf_emb_merg = masked_sequence_mean(pdt_rxn_transf_emb, pdt_memory_lengths)
                else:
                    raise NotImplementedError(f"Unsupported trans_readout: {self.model.trans_readout}")
                diff_emb = torch.abs(rct_rxn_transf_emb_merg - pdt_rxn_transf_emb_merg)
                if self.model.split_merge_method == "all":
                    rxn_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg,diff_emb],dim=-1)
                elif self.model.split_merge_method == "only_diff":
                    rxn_emb = diff_emb
                elif self.model.split_merge_method == "rct_pdt":
                    rxn_emb = torch.cat([rct_rxn_transf_emb_merg,pdt_rxn_transf_emb_merg],dim=-1)
                for lin_layer,norm_layer in zip(self.model.decoder.layers[:-1],self.model.decoder.batch_norms[:-1]):
                    rxn_emb = lin_layer(rxn_emb)
                    rxn_emb = norm_layer(rxn_emb)
                all_rxn_emb.append(rxn_emb.detach().cpu())
        return torch.cat(all_rxn_emb,dim=0)
    def gen_half_rxn_mol_emb_from_dataset(self,root,
                                                name_regrex="50k_rxn_type_rct_0.csv",batch_size=128,mol_type="rct"):
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        dataset = MultiRXNDataset(root=root, name_regrex=name_regrex)
        pair_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=single_collate_fn)
        all_rxn_transf_emb = []
        print("[INFO] Generating reaction embedding...")
        with torch.no_grad():
            for data in tqdm(pair_dataloader):
                data.to(device)
                if mol_type == "rct":
                    padded_memory_bank,batch,memory_lengths = self.model.rct_encoder(data)
                elif mol_type == "pdt":
                    padded_memory_bank,batch,memory_lengths = self.model.pdt_encoder(data)

                rxn_transf_emb = padded_memory_bank.transpose(0,1)
                all_rxn_transf_emb.append(rxn_transf_emb.detach().cpu())
                ## TODO 把padding的部分去掉
        return all_rxn_transf_emb
        
    def gen_mol_emb_from_dataset(self,root,
                                 name_regrex="50k_rxn_type_rct_0.csv",batch_size=128,mol_type="rct"):
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        dataset = MultiRXNDataset(root=root, name_regrex=name_regrex)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=single_collate_fn)
        all_mol_emb = []
        print("[INFO] Generating molecular embedding...")
        with torch.no_grad():
            for data in tqdm(dataloader):
                data.to(device)
                x = data.x
                mol_index = data.mol_index
                edge_index = data.edge_index
                edge_attr = data.edge_attr
                if mol_type == "rct":
                    node_representation, mol_index,batch = self.model.rct_encoder.rxn_graph_encoder(x, mol_index, edge_index, edge_attr)
                elif mol_type == "pdt":
                    node_representation, mol_index,batch = self.model.pdt_encoder.rxn_graph_encoder(x, mol_index, edge_index, edge_attr)
                    
                # 获取唯一的分子标识符
                uniq_mol = mol_index.unique()

                # 使用列表解析选择对应于每个分子的行
                mol_representation = [node_representation[mol_index == mol].cpu() for mol in uniq_mol]
                all_mol_emb += mol_representation
        return all_mol_emb
    
    def gen_rxn_emb(self,rxn_smiles_lst,batch_size=128):
        assert len(rxn_smiles_lst) >= 2, "rxn_smiles_lst must contain at least 2 reactions"
        rct_smi_lst = []
        pdt_smi_lst = []
        for smi in rxn_smiles_lst:
            reactant, product = split_reaction_smiles(smi)
            rct_smi_lst.append(f"{canonicalize_reaction_side(reactant)},0")
            pdt_smi_lst.append(f"{canonicalize_reaction_side(product)},0")
        with tempfile.TemporaryDirectory(prefix="rxngraphormer_rxn_emb_") as tmp_dir:
            with open(os.path.join(tmp_dir, "rct_smiles_0.csv"), "w") as fw:
                fw.writelines("\n".join(rct_smi_lst))
            with open(os.path.join(tmp_dir, "pdt_smiles_0.csv"), "w") as fw:
                fw.writelines("\n".join(pdt_smi_lst))
            return self.gen_rxn_emb_from_dataset(
                root=tmp_dir,
                rct_name_regrex="rct_smiles_0.csv",
                pdt_name_regrex="pdt_smiles_0.csv",
                batch_size=batch_size,
            )
    
    def gen_mult_mol_emb(self,mult_mol_smiles_lst,mol_type="rct",batch_size=128):
        
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        assert len(mult_mol_smiles_lst) >= 2, "mult_mol_smiles_lst must contain at least 2 molecules"
        mol_num_lst = []
        tot_mol_smi_lst = []
        for idx,mult_mol_smi in enumerate(mult_mol_smiles_lst):
            tot_mol_smi_lst += mult_mol_smi.split(".")
            mol_num_lst += len(mult_mol_smi.split(".")) * [idx]
        tot_mol_emb = self.gen_mol_emb(tot_mol_smi_lst,mol_type=mol_type,batch_size=batch_size)
        return tot_mol_emb,mol_num_lst
    
    def gen_mol_emb(self,mol_smiles_lst,mol_type="rct",batch_size=128):
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        assert len(mol_smiles_lst) >= 2, "mol_smiles_lst must contain at least 2 molecule"
        mol_smi_lst = [f'{canonical_smiles(smi)},0' for smi in mol_smiles_lst]
        name_regrex = f"{mol_type}_smiles_0.csv"
        with tempfile.TemporaryDirectory(prefix="rxngraphormer_mol_emb_") as tmp_dir:
            with open(os.path.join(tmp_dir, name_regrex), "w") as fw:
                fw.writelines("\n".join(mol_smi_lst))
            return self.gen_mol_emb_from_dataset(
                root=tmp_dir,
                name_regrex=name_regrex,
                mol_type=mol_type,
                batch_size=batch_size,
            )
    
    def gen_half_rxn_mol_emb(self,half_rxn_smiles_lst,mol_type="rct",batch_size=128):
        assert mol_type in ["rct","pdt"], "mol_type must be either 'rct' or 'pdt'"
        assert len(half_rxn_smiles_lst) >= 2, "half_rxn_smiles_lst must contain at least 2 half reactions"
        half_rxn_smi_lst = [f'{canonical_smiles(smi)},0' for smi in half_rxn_smiles_lst]
        name_regrex = f"{mol_type}_smiles_0.csv"
        with tempfile.TemporaryDirectory(prefix="rxngraphormer_half_rxn_emb_") as tmp_dir:
            with open(os.path.join(tmp_dir, name_regrex), "w") as fw:
                fw.writelines("\n".join(half_rxn_smi_lst))
            return self.gen_half_rxn_mol_emb_from_dataset(
                root=tmp_dir,
                name_regrex=name_regrex,
                mol_type=mol_type,
                batch_size=batch_size,
            )

class RXNClassifier():
    def __init__(self,pretrained_model_path,random_init=False):
        self.predictor = RXNGraphormerPredictor(
            pretrained_model_path,
            task="classification",
            device=device,
            random_init=random_init,
        )
        self.model = self.predictor.model
    
    def rxn_pred(self,rxn_smiles_lst,batch_size=128):
        assert len(rxn_smiles_lst) >= 2, "rxn_smiles_lst must contain at least 2 reactions"
        rct_smi_lst = []
        pdt_smi_lst = []
        for smi in rxn_smiles_lst:
            reactant, product = split_reaction_smiles(smi)
            rct_smi_lst.append(f"{canonicalize_reaction_side(reactant)},0")
            pdt_smi_lst.append(f"{canonicalize_reaction_side(product)},0")
        with tempfile.TemporaryDirectory(prefix="rxngraphormer_rxn_pred_") as tmp_dir:
            with open(os.path.join(tmp_dir, "rct_smiles_0.csv"), "w") as fw:
                fw.writelines("\n".join(rct_smi_lst))
            with open(os.path.join(tmp_dir, "pdt_smiles_0.csv"), "w") as fw:
                fw.writelines("\n".join(pdt_smi_lst))
            return self.rxn_pred_from_dataset(
                root=tmp_dir,
                rct_name_regrex="rct_smiles_0.csv",
                pdt_name_regrex="pdt_smiles_0.csv",
                batch_size=batch_size,
            )
    
    def rxn_pred_from_dataset(self,root,
                                 rct_name_regrex,
                                 pdt_name_regrex,
                                 batch_size=128):
        print("[INFO] Predict whether the reaction is real...")
        result = self.predictor.predict_from_dataset(
            root,
            rct_name_regrex=rct_name_regrex,
            pdt_name_regrex=pdt_name_regrex,
            batch_size=batch_size,
        )
        return result.preds, result.confidence
