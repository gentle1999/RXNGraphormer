from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GlobalAttention

from ..data.batch import pad_feat
from ..data.graph import GraphTask, calc_batch_graph_distance
from .attention_xl import AttnEncoderXL
from .encoders import RXNGraphEncoder
from .sequence_dependencies import require_onmt as _require_onmt
from .transformer import TransformerEncoder


def _decoder_src_placeholder(memory_lengths):
    max_length = int(memory_lengths.max().item())
    return memory_lengths.new_zeros(max_length)


def _graph_task(task: object) -> GraphTask:
    if task not in {"forward_prediction", "retrosynthesis"}:
        raise ValueError(f"Unknown graph task: {task}")
    return cast(GraphTask, task)


class SeqDecoder(nn.Module):
    def __init__(self, word_vec_size, vocab,decoder_num_layers,decoder_hidden_size,
                 decoder_attn_heads,decoder_filter_size,max_relative_positions,
                 pad_token="_PAD",sos_token="_SOS",eos_token="_EOS",
                 copy_attn=False,self_attn_type="scaled-dot",position_encoding=True,
                 aan_useffn=False,full_context_alignment=False,alignment_layer=-3,
                 alignment_heads=0,dropout=0.0):
        super().__init__()
        onmt = _require_onmt()
        self.vocab = vocab
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.word_vocab_size = len(self.vocab)
        self.word_padding_idx = self.vocab[self.pad_token]
        self.decoder_embeddings = onmt["Embeddings"](
                word_vec_size=word_vec_size,
                word_vocab_size=self.word_vocab_size,
                word_padding_idx=self.word_padding_idx,
                position_encoding=position_encoding,
                dropout=dropout)
        self.tdecoder = onmt["TransformerDecoder"](
            num_layers=decoder_num_layers,
            d_model=decoder_hidden_size,
            heads=decoder_attn_heads,
            d_ff=decoder_filter_size,
            copy_attn=copy_attn,
            self_attn_type=self_attn_type,
            dropout=dropout,
            attention_dropout=dropout,
            embeddings=self.decoder_embeddings,
            max_relative_positions=max_relative_positions,
            aan_useffn=aan_useffn,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads)
        self.output_layer = nn.Linear(decoder_hidden_size, self.word_vocab_size, bias=True)

    def forward(self,padded_memory_bank, tgt_token_ids, memory_lengths):
        # tgt_token_ids is shorten version  tgt_token_ids = data.tgt_token_ids[:,:data.tgt_lens.max()]
        #memory_lengths = torch.bincount(batch).long().to(device=rxn_emb_enc.device)
        # padded_memory_bank = rxn_emb_enc.transpose(1,0)
        # print(f"padded_memory_bank.shape: {padded_memory_bank.shape}, memory_lengths.shape: {memory_lengths.shape}")
        self.tdecoder.state["src"] = _decoder_src_placeholder(memory_lengths)

        dec_in = tgt_token_ids[:, :-1] ## pop last, insert SOS for decoder input
        m = nn.ConstantPad1d((1, 0), self.vocab[self.sos_token])
        dec_in = m(dec_in)
        dec_in = dec_in.transpose(0, 1).unsqueeze(-1)
        #print("!!!!!!!",dec_in.shape,padded_memory_bank.shape,memory_lengths.shape)

        dec_outs, _ = self.tdecoder(tgt=dec_in,
                                    memory_bank=padded_memory_bank,
                                    memory_lengths=memory_lengths)
        dec_outs = self.output_layer(dec_outs)
        dec_outs = dec_outs.permute(1, 2, 0)
        predictions = torch.argmax(dec_outs, dim=1)
        return dec_outs, predictions

class G2STransformer(nn.Module):
    def __init__(self,tencoder,tdecoder,vocab):
        super().__init__()
        self.tdecoder = tdecoder  ## SeqDecoder
        self.tencoder = tencoder  ## RXNGEncoder
        self.vocab = vocab
    def forward(self,data):
        padded_memory_bank,batch,memory_lengths = self.tencoder(data)
        """
        print(padded_memory_bank.shape,
              data.tgt_token_ids.shape,
              data.tgt_token_ids[:data.tgt_lens.max()].shape, old version data.tgt_token_ids[:data.tgt_lens.max()] instead of data.tgt_token_ids
              memory_lengths.shape)
        """
        dec_outs, predictions = self.tdecoder(padded_memory_bank,data.tgt_token_ids,memory_lengths)
        return dec_outs,predictions
    def infer(self,data,batch_size,beam_size,n_best=10,temperature=1.0,min_length=0,max_length=512):
        onmt = _require_onmt()
        padded_memory_bank,batch,memory_lengths = self.tencoder(data)
        #memory_lengths = torch.bincount(batch).long().to(device=rxn_emb_enc.device)
        # padded_memory_bank = rxn_emb_enc.transpose(1,0)
        self.tdecoder.tdecoder.state["src"] = _decoder_src_placeholder(memory_lengths)
        if beam_size == 1:
            decode_strategy = onmt["GreedySearch"](pad=self.vocab["_PAD"],
                                            bos=self.vocab["_SOS"],
                                            eos=self.vocab["_EOS"],
                                            batch_size=batch_size,
                                            min_length=min_length,
                                            max_length=max_length,
                                            block_ngram_repeat=0,
                                            exclusion_tokens=set(),
                                            return_attention=False,
                                            sampling_temp=0.0,
                                            keep_topk=1)
        else:
            global_scorer = onmt["GNMTGlobalScorer"](alpha=0.0,beta=0.0,length_penalty="none",coverage_penalty="none")
            decode_strategy = onmt["BeamSearch"](
                beam_size=beam_size,
                batch_size=batch_size,
                pad=self.vocab["_PAD"],
                bos=self.vocab["_SOS"],
                eos=self.vocab["_EOS"],
                n_best=n_best,
                global_scorer=global_scorer,
                min_length=min_length,
                max_length=max_length,
                return_attention=False,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                stepwise_penalty=None,
                ratio=0.0)

        #padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch=reaction_batch)
        # adapted from onmt.translate.translator
        results: dict[str, Any | None] = {
            "predictions": None,
            "scores": None,
            "attention": None,
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None
        target_prefix = None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank=padded_memory_bank,
            src_lengths=memory_lengths,
            src_map=src_map,
            target_prefix=target_prefix)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            dec_out, dec_attn = self.tdecoder.tdecoder(tgt=decoder_input,
                                            memory_bank=memory_bank,
                                            memory_lengths=memory_lengths,
                                            step=step)

            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None

            dec_out = self.tdecoder.output_layer(dec_out)            # [t, b, h] => [t, b, v]
            dec_out = dec_out / temperature
            dec_out = dec_out.squeeze(0)                    # [t, b, v] => [b, v]
            log_probs = F.log_softmax(dec_out, dim=-1)

            # log_probs = self.model.generator(dec_out.squeeze(0))

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if any_finished:
                self.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = cast(Any, decode_strategy.scores)
        results["predictions"] = cast(Any, decode_strategy.predictions)
        results["attention"] = cast(Any, decode_strategy.attention)
        results["alignment"] = [[] for _ in range(batch_size)]

        return results

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.tdecoder.tdecoder.state["cache"] is not None:
            _recursive_map(self.tdecoder.tdecoder.state["cache"])

class RXNG2Sequencer(nn.Module):

    def __init__(self, config, vocab):
        super().__init__()
        onmt = _require_onmt()
        self.config = config
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.add_empty_node = self.config.model.add_empty_node
        self.encoder = RXNGraphEncoder(gnum_layer=self.config.model.gnum_layer,
                                    emb_dim=self.config.model.emb_dim,
                                    gnn_type=self.config.model.gnn_type,
                                    JK=self.config.model.JK,
                                    drop_ratio=self.config.model.drop_ratio,
                                    gnn_aggr=self.config.model.gnn_aggr,
                                    bond_feat_red=self.config.model.bond_feat_red,
                                    node_readout=self.config.model.node_readout,
                                    always_on_dropout=getattr(self.config.model, "forward_compat_mode", "modern") == "legacy",
                                    norm_type=getattr(self.config.model, "encoder_norm", "batchnorm"),
                                    )
        if self.config.model.att_encoder_type == 'attxl':
            self.attention_encoder = AttnEncoderXL(num_layers=self.config.model.decoder_num_layers,
                                                d_model=self.config.model.emb_dim,
                                                heads=self.config.model.num_heads,
                                                d_ff=self.config.model.encoder_filter_size,
                                                dropout=self.config.model.drop_ratio,
                                                attention_dropout=self.config.model.drop_ratio,
                                                rel_pos_buckets=self.config.model.rel_pos_buckets,
                                                enc_pos_encoding=None)
        elif self.config.model.att_encoder_type == 'attn':
            self.attention_pool = GlobalAttention(gate_nn=torch.nn.Linear(self.config.model.emb_dim, 1))
            self.attention_encoder = TransformerEncoder(num_layer=self.config.model.tnum_layer,
                                                        hidden_size=self.config.model.emb_dim,
                                                        intermediate_size=self.config.model.emb_dim,
                                                        num_heads=self.config.model.num_heads,
                                                        hidden_dropout_prob=self.config.model.attn_drop_ratio,
                                                        use_padding_mask=getattr(self.config.model, "forward_compat_mode", "modern") != "legacy")
        else:
            raise NotImplementedError(f"Attention encoder type {self.config.model.att_encoder_type} not implemented")




        self.decoder_embeddings = onmt["Embeddings"](
            word_vec_size=self.config.model.emb_dim,
            word_vocab_size=self.vocab_size,
            word_padding_idx=self.vocab["_PAD"],
            position_encoding=True,
            dropout=self.config.model.drop_ratio
        )

        self.decoder = onmt["TransformerDecoder"](
            num_layers=self.config.model.decoder_num_layers,
            d_model=self.config.model.emb_dim,
            heads=self.config.model.num_heads,
            d_ff=self.config.model.filter_size,
            copy_attn=False,
            self_attn_type="scaled-dot",
            dropout=self.config.model.drop_ratio,
            attention_dropout=self.config.model.drop_ratio,
            embeddings=self.decoder_embeddings,
            max_relative_positions=self.config.model.max_rel_pos,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=-3,
            alignment_heads=0
        )

        self.output_layer = nn.Linear(self.config.model.emb_dim, self.vocab_size, bias=True)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab["_PAD"],
            reduction="mean"
        )

    def encode_and_reshape(self, reaction_batch):
        hatom, mol_index, batch = self.encoder(
            reaction_batch.x,
            reaction_batch.mol_index,
            reaction_batch.edge_index,
            reaction_batch.edge_attr,
            atom_batch=getattr(reaction_batch, "batch", None),
        )                         # (n_atoms, h)

        if self.config.model.att_encoder_type == 'attxl':
            memory_lengths = torch.bincount(reaction_batch.batch).long()
            max_length = int(memory_lengths.max().item())
            padded_memory_bank = []
            if not self.add_empty_node:
                assert sum(memory_lengths) == hatom.size(0), \
                    f"Memory lengths calculation error, encoder output: {hatom.size(0)}, memory_lengths: {memory_lengths}"    ## V1

                memory_bank = torch.split(hatom,  memory_lengths.tolist(), dim=0)   # [n_atoms, h] => 1+b tup of (t, h)        ## V1
                for length, h in zip(memory_lengths, memory_bank):                                                              ## V1
                    m = nn.ZeroPad2d((0, 0, 0, max_length - int(length.item())))
                    padded_memory_bank.append(m(h))
            else:
                assert 1 + sum(memory_lengths) == hatom.size(0), \
                f"Memory lengths calculation error, encoder output: {hatom.size(0)}, memory_lengths: {memory_lengths}"
                memory_bank = torch.split(hatom, [1] + memory_lengths.tolist(), dim=0)   # [n_atoms, h] => 1+b tup of (t, h)
                for length, h in zip(memory_lengths, memory_bank[1:]):
                    m = nn.ZeroPad2d((0, 0, 0, max_length - int(length.item())))
                    padded_memory_bank.append(m(h))


            padded_memory_bank = torch.stack(padded_memory_bank, dim=1)     # list of b (max_t, h) => [max_t, b, h]

            memory_lengths = torch.tensor(memory_lengths,
                                        dtype=torch.long,
                                        device=padded_memory_bank.device)
            if not self.add_empty_node:
                graph_task = _graph_task(self.config.model.task)
                distances = calc_batch_graph_distance(batch=reaction_batch.batch,
                                                        edge_index=reaction_batch.edge_index,
                                                        task=graph_task)
            else:
                graph_task = _graph_task(self.config.model.task)
                distances = calc_batch_graph_distance(batch=reaction_batch.batch,
                                                        edge_index=reaction_batch.edge_index[:, 1:]-1,
                                                        task=graph_task)
            if self.attention_encoder is not None:
                padded_memory_bank = self.attention_encoder(
                    padded_memory_bank,
                    memory_lengths,
                    distances
                )
        elif self.config.model.att_encoder_type == "attn":
            memory_lengths = torch.bincount(batch).long().to(device=hatom.device)
            max_length = int(memory_lengths.max().item())
            rxn_representation = self.attention_pool(hatom,mol_index)  ## node_representation is equal to hatom
            padded_feat = pad_feat(rxn_representation,batch,self.config.model.emb_dim)
            rxn_transf_emb = self.attention_encoder(padded_feat, memory_lengths)
            padded_memory_bank = rxn_transf_emb.transpose(1,0)
        else:
            raise NotImplementedError(f"Attention encoder type {self.config.model.att_encoder_type} not implemented")

        self.decoder.state["src"] = np.zeros(max_length)    # TODO: this is hardcoded to make transformer decoder work
        #print(f"padded_memory_bank.shape: {padded_memory_bank.shape}, memory_lengths.shape: {memory_lengths.shape}")
        return padded_memory_bank, memory_lengths

    def forward(self, reaction_batch):
        #print("Enter Graph2SeqSeriesRel forward")
        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch)

        # adapted from onmt.models
        dec_in = reaction_batch.tgt_token_ids[:, :-1]                       # pop last, insert SOS for decoder input
        m = nn.ConstantPad1d((1, 0), self.vocab["_SOS"])
        dec_in = m(dec_in)
        dec_in = dec_in.transpose(0, 1).unsqueeze(-1)                       # [b, max_tgt_t] => [max_tgt_t, b, 1]
        #print("!!!!!!!",dec_in.shape,padded_memory_bank.shape,memory_lengths.shape)
        dec_outs, _ = self.decoder(
            tgt=dec_in,
            memory_bank=padded_memory_bank,
            memory_lengths=memory_lengths
        )

        dec_outs = self.output_layer(dec_outs)                                  # [t, b, h] => [t, b, v]
        dec_outs = dec_outs.permute(1, 2, 0)                                    # [t, b, v] => [b, v, t]

        loss = self.criterion(
            input=dec_outs,
            target=reaction_batch.tgt_token_ids
        )

        predictions = torch.argmax(dec_outs, dim=1)                             # [b, t]
        mask = (reaction_batch.tgt_token_ids != self.vocab["_PAD"]).long()
        accs = (predictions == reaction_batch.tgt_token_ids).float()
        accs = accs * mask
        acc = accs.sum() / mask.sum()

        return loss, acc

    def infer(self, reaction_batch,
                     batch_size: int, beam_size: int, n_best: int, temperature: float,
                     min_length: int, max_length: int):
        onmt = _require_onmt()
        if beam_size == 1:
            decode_strategy = onmt["GreedySearch"](
                pad=self.vocab["_PAD"],
                bos=self.vocab["_SOS"],
                eos=self.vocab["_EOS"],
                batch_size=batch_size,
                min_length=min_length,
                max_length=max_length,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                return_attention=False,
                sampling_temp=0.0,
                keep_topk=1
            )
        else:
            global_scorer = onmt["GNMTGlobalScorer"](alpha=0.0,
                                             beta=0.0,
                                             length_penalty="none",
                                             coverage_penalty="none")
            decode_strategy = onmt["BeamSearch"](
                beam_size=beam_size,
                batch_size=batch_size,
                pad=self.vocab["_PAD"],
                bos=self.vocab["_SOS"],
                eos=self.vocab["_EOS"],
                n_best=n_best,
                global_scorer=global_scorer,
                min_length=min_length,
                max_length=max_length,
                return_attention=False,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                stepwise_penalty=None,
                ratio=0.0
            )

        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch=reaction_batch)
        # adapted from onmt.translate.translator
        results: dict[str, Any | None] = {
            "predictions": None,
            "scores": None,
            "attention": None,
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None
        target_prefix = None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank=padded_memory_bank,
            src_lengths=memory_lengths,
            src_map=src_map,
            target_prefix=target_prefix
        )

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
            # print(f"[{step}] decoder_input.shape: {decoder_input.shape}, memory_bank.shape: {len(memory_bank)}, memory_lengths: {len(memory_lengths)}")
            # print(f"[{step}] memory_bank: {memory_bank[0]}, memory_lengths: {memory_lengths[0]}")
            dec_out, dec_attn = self.decoder(tgt=decoder_input,
                                             memory_bank=memory_bank,
                                             memory_lengths=memory_lengths,
                                             step=step)

            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None

            dec_out = self.output_layer(dec_out)            # [t, b, h] => [t, b, v]
            dec_out = dec_out / temperature
            dec_out = dec_out.squeeze(0)                    # [t, b, v] => [b, v]
            log_probs = F.log_softmax(dec_out, dim=-1)

            # log_probs = self.model.generator(dec_out.squeeze(0))

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if any_finished:
                self.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["alignment"] = [[] for _ in range(4096)]

        return results

    # adapted from onmt.decoders.transformer
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        # self.decoder.state["src"] = fn(self.decoder.state["src"], 1)
        # => self.state["src"] = self.state["src"].index_select(1, select_indices)

        if self.decoder.state["cache"] is not None:
            _recursive_map(self.decoder.state["cache"])
