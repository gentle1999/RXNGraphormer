from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, TypeVar, get_args, get_origin, get_type_hints

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


Task = Literal["classification", "regression", "sequence_generation"]
GNNType = Literal["gcn", "gin", "gat", "rgcn", "edge_gcn"]
Aggregation = Literal["add", "mean", "sum", "max"]
Readout = Literal["mean", "sum", "last", "attention"]
GraphPooling = Literal["attention", "mean", "sum"]
SplitMerge = Literal["all", "rct_pdt", "only_diff"]
Activation = Literal["relu", "tanh", "sigmoid", "identity"]
OptimizerName = Literal["Adam", "AdamW"]
SchedulerName = Literal["noamlr", "steplr", "step", "warmup", "linear", "none"]
LossName = Literal["ce", "l1", "l2", "mse", "mae"]
MidInteraction = Literal["fc", "attention", "1dconv"]
SequenceTask = Literal["retrosynthesis", "forward_prediction"]


@dataclass
class CompatMixin:
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, filename: str | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if filename is not None:
            Path(filename).write_text(text + "\n")
        return text


@dataclass
class ConfigNamespace(CompatMixin):
    """Dataclass-backed compatibility namespace for ad-hoc legacy mappings."""

    _data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self._data.items():
            setattr(self, key, _namespace_value(value))

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__ and key != "_data"

    def to_dict(self) -> dict[str, Any]:
        return {
            key: _to_plain(value)
            for key, value in self.__dict__.items()
            if key != "_data"
        }


@dataclass
class ModelConfig(CompatMixin):
    emb_dim: int = 256
    gnn_type: GNNType = "gcn"
    gnn_num_layer: int = 4
    gnum_layer: int | None = None
    gnn_aggr: Aggregation = "mean"
    node_readout: Readout = "mean"
    gnn_jk: Literal["last"] = "last"
    JK: Literal["last"] | None = None
    graph_pooling: GraphPooling = "attention"
    trans_num_layer: int = 4
    tnum_layer: int | None = None
    trans_readout: Readout = "mean"
    output_num_layer: int = 1
    drop_ratio: float = 0.0
    save_dir: str = "./save"
    num_heads: int = 2
    pretrained_model: str = ""
    pretrained_model_path: str = ""
    pretrained_model_freeze: bool = False
    pretrained_lr_scaled_coef: float = 0.95
    fine_tune: bool = False
    trainable: str = ""
    output_norm: bool = False
    split_merge_method: SplitMerge = "all"
    output_act_func: Activation = "relu"
    rct_batch_norm: bool = True
    pdt_batch_norm: bool = True
    mid_batch_norm: bool = True
    use_mid_inf: bool = False
    mid_iteract_method: MidInteraction = "attention"
    mid_layer_num: int = 1
    bond_feat_red: Literal["mean", "sum"] = "mean"
    attn_drop_ratio: float = 0.0
    encoder_filter_size: int = 2048
    rel_pos_buckets: int = 11
    rel_pos: Literal["emb_only", "enc_only", "none"] = "emb_only"
    filter_size: int = 2048
    decoder_num_layers: int = 4
    max_rel_pos: int = 0
    att_encoder_type: Literal["attxl", "attn", "rxngraphormer", "onmt"] = "attxl"
    add_empty_node: bool = False
    task: SequenceTask = "retrosynthesis"

    def __post_init__(self) -> None:
        if self.gnum_layer is not None:
            self.gnn_num_layer = self.gnum_layer
        if self.tnum_layer is not None:
            self.trans_num_layer = self.tnum_layer
        if self.JK is not None:
            self.gnn_jk = self.JK
        if self.att_encoder_type == "rxngraphormer":
            self.att_encoder_type = "attxl"
        elif self.att_encoder_type == "onmt":
            self.att_encoder_type = "attn"
        self.gnum_layer = self.gnn_num_layer
        self.tnum_layer = self.trans_num_layer
        self.JK = self.gnn_jk


@dataclass
class OptimizerConfig(CompatMixin):
    optimizer: OptimizerName = "AdamW"
    learning_rate: float = 0.4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class SchedulerConfig(CompatMixin):
    lr_decay_step_size: int = 15
    lr_decay_factor: float = 0.8
    warmup_step: int = 6000
    type: SchedulerName = "noamlr"


@dataclass
class DataConfig(CompatMixin):
    data_path: str = "./dataset"
    rct_data_file: str = ""
    pdt_data_file: str = ""
    mid_data_file: str = ""
    train_rct_data_file: str = ""
    train_pdt_data_file: str = ""
    val_rct_data_file: str = ""
    val_pdt_data_file: str = ""
    test_rct_data_file: str = ""
    test_pdt_data_file: str = ""
    train_mid_data_file: str = ""
    val_mid_data_file: str = ""
    test_mid_data_file: str = ""
    rct_name_regrex: str = ""
    pdt_name_regrex: str = ""
    mid_name_regrex: str = ""
    input_table: str = ""
    rxn_smiles_column: str = "rxn_smiles"
    rct_smiles_column: str = "rct_smiles"
    pdt_smiles_column: str = "pdt_smiles"
    mid_smiles_column: str = "mid_smiles"
    target_column: str = ""
    output_prefix: str = ""
    generate_mid: bool = False
    mapping_policy: Literal["auto", "always", "never"] = "auto"
    train_src_file: str = ""
    train_tgt_file: str = ""
    valid_src_file: str = ""
    valid_tgt_file: str = ""
    test_src_file: str = ""
    test_tgt_file: str = ""
    vocab_file: str = ""
    data_trunck: int = 0
    file_num_trunck: int = 0
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    train_ratio: float = 0.8
    valid_ratio: float = 0.2
    seed: int = 0
    multi_process: bool = False
    radius: int = 2
    task: Literal["classification", "regression", "sequence_generation"] = "classification"
    tag: str = ""

    def __post_init__(self) -> None:
        if self.input_table:
            prefix = self.output_prefix or Path(self.input_table).stem
            if not self.rct_data_file:
                self.rct_data_file = f"{prefix}_rct.csv"
            if not self.pdt_data_file:
                self.pdt_data_file = f"{prefix}_pdt.csv"
            if not self.mid_data_file:
                self.mid_data_file = f"{prefix}_mid.csv"
        if not self.rct_name_regrex and self.rct_data_file:
            self.rct_name_regrex = self.rct_data_file
        if not self.pdt_name_regrex and self.pdt_data_file:
            self.pdt_name_regrex = self.pdt_data_file
        if not self.mid_name_regrex and self.mid_data_file:
            self.mid_name_regrex = self.mid_data_file


@dataclass
class TrainingConfig(CompatMixin):
    epoch: int = 1
    loss: LossName = "l2"
    accum: int = 1
    clip_norm: float = 20.0
    log_iter_step: int = 200


@dataclass
class OthersConfig(CompatMixin):
    device: str = "cuda:0"
    tag: str = "default"
    multi_gpu: bool = False
    local_rank: int = -1
    enable_amp: bool = False
    log_step: int = 100
    save_improve: bool = True


@dataclass
class RuntimeConfig(CompatMixin):
    enable_amp: bool = False
    amp_dtype: Literal["fp16", "bf16"] = "fp16"
    deterministic: bool = False
    compile_model: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    early_stopping_patience: int = 0
    test_every_n_epochs: int = 1
    log_epoch_time: bool = True
    log_gpu_memory: bool = True


@dataclass
class InferConfig(CompatMixin):
    beam_size: int = 10
    min_length: int = 1
    max_length: int = 512
    print_iterval: int = 100


@dataclass
class EvalConfig(CompatMixin):
    trained_model_path: str = ""
    ckpt_file: str = "valid_checkpoint.pt"
    topk: int = 10
    beam_size: int = 10
    temperature: float = 1.0
    n_best: int = 10
    min_length: int = 1
    max_length: int = 512
    batch_size: int = 32
    scale: float = 1.0
    yield_constrain: bool = False
    max_batches: int | None = None
    task: Task = "regression"


@dataclass
class Config(CompatMixin):
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    others: OthersConfig = field(default_factory=OthersConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    task: Task = "regression"

    def __post_init__(self) -> None:
        if self.data.task == "classification" and self.task != "classification":
            self.data.task = self.task


T = TypeVar("T")


def load_config(path: str | Path) -> Config | EvalConfig:
    return from_dict(load_config_dict(path))


def resolve_config_file(path: str | Path, stem: str = "parameters") -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for suffix in (".toml", ".json", ".jsonc"):
        candidate = path / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {stem}.toml, {stem}.json, or {stem}.jsonc found under {path}")


def load_config_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as fh:
            return tomllib.load(fh)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    if suffix == ".jsonc":
        return json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))
    raise ValueError(f"Unsupported config format {suffix!r}; expected .toml, .json, or .jsonc")


def _strip_jsonc_comments(text: str) -> str:
    result: list[str] = []
    i = 0
    in_string = False
    escape = False
    length = len(text)

    while i < length:
        char = text[i]
        next_char = text[i + 1] if i + 1 < length else ""

        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == "/" and next_char == "/":
            result.extend("  ")
            i += 2
            while i < length and text[i] not in "\r\n":
                result.append(" ")
                i += 1
            continue

        if char == "/" and next_char == "*":
            result.extend("  ")
            i += 2
            closed = False
            while i < length:
                if text[i] == "*" and i + 1 < length and text[i + 1] == "/":
                    result.extend("  ")
                    i += 2
                    closed = True
                    break
                result.append(text[i] if text[i] in "\r\n" else " ")
                i += 1
            if not closed:
                raise ValueError("Unterminated JSONC block comment")
            continue

        result.append(char)
        i += 1

    return "".join(result)


def from_dict(data: Mapping[str, Any]) -> Config | EvalConfig:
    normalized = _normalize_mapping(data)
    if "model" not in normalized and "data" not in normalized and "training" not in normalized:
        return _build_dataclass(EvalConfig, normalized)
    return _build_dataclass(Config, normalized)


def namespace_from_dict(data: Mapping[str, Any]) -> ConfigNamespace:
    return ConfigNamespace(dict(_normalize_mapping(data)))


def _normalize_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _normalize_value(value) for key, value in data.items()}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _normalize_mapping(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return value


def _namespace_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ConfigNamespace(dict(value))
    if isinstance(value, list):
        return [_namespace_value(item) for item in value]
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, ConfigNamespace):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _build_dataclass(cls: type[T], data: Mapping[str, Any]) -> T:
    type_hints = get_type_hints(cls)
    field_map = {item.name: item for item in fields(cls)}
    kwargs = {}
    for name, item in field_map.items():
        if name not in data:
            continue
        value = data[name]
        target_type = type_hints.get(name, item.type)
        kwargs[name] = _coerce_value(target_type, value, name)
    return cls(**kwargs)


def _coerce_value(target_type: Any, value: Any, field_name: str) -> Any:
    origin = get_origin(target_type)
    args = get_args(target_type)

    if origin is Literal:
        if value not in args:
            raise ValueError(f"{field_name}={value!r} is not one of {args!r}")
        return value

    if origin in (list, tuple):
        return value

    if origin is None and is_dataclass(target_type) and isinstance(value, Mapping):
        return _build_dataclass(target_type, value)

    if origin in (types.UnionType, getattr(types, "UnionType", object)) or str(origin) == "typing.Union":
        if type(None) in args:
            non_none = [arg for arg in args if arg is not type(None)]
            if value is None:
                return None
            if non_none:
                return _coerce_value(non_none[0], value, field_name)

    if origin is not None and type(None) in args:
        non_none = [arg for arg in args if arg is not type(None)]
        if value is None:
            return None
        if non_none:
            return _coerce_value(non_none[0], value, field_name)

    if target_type is bool:
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)
    if target_type in (int, float, str):
        return target_type(value)
    return value


Box = namespace_from_dict
