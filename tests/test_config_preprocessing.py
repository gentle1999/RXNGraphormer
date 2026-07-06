import json
import runpy
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast
from unittest import mock

import pandas as pd

from rxngraphormer.config import (
    DataConfig,
    ModelConfig,
    NormType,
    RawConfigDict,
    load_config,
    load_config_dict,
    resolve_config_file,
)
from rxngraphormer.config_utils import align_config
from rxngraphormer.data.loader import DataLoaderSettings, dataloader_kwargs
from rxngraphormer.preprocessing import (
    generate_mid_smiles,
    generate_mid_smiles_batch,
    parse_reaction_smiles,
    preprocess_from_config,
    reaction_has_atom_mapping,
    read_prediction_table,
    read_reaction_table,
    split_reaction_smiles,
    write_reaction_table_files,
)


class _HeadNormConfig(Protocol):
    head_norm: str


class ConfigCompatibilityTest(unittest.TestCase):
    def test_legacy_data_file_fields_populate_name_regex_aliases(self):
        config = load_config("config_toml/bh_scratch_reproduce.toml")

        self.assertEqual(config.data.rct_name_regrex, config.data.rct_data_file)
        self.assertEqual(config.data.pdt_name_regrex, config.data.pdt_data_file)
        self.assertEqual(config.data.mid_name_regrex, config.data.mid_data_file)
        self.assertEqual(config.data.task, config.task)

    def test_input_table_populates_legacy_data_files_from_prefix(self):
        data = DataConfig(input_table="inputs/reactions.csv")

        self.assertEqual(data.rct_data_file, "reactions_rct.csv")
        self.assertEqual(data.pdt_data_file, "reactions_pdt.csv")
        self.assertEqual(data.mid_data_file, "reactions_mid.csv")
        self.assertEqual(data.rct_name_regrex, "reactions_rct.csv")
        self.assertEqual(data.pdt_name_regrex, "reactions_pdt.csv")
        self.assertEqual(data.mid_name_regrex, "reactions_mid.csv")

    def test_sequence_attention_encoder_legacy_values_load(self):
        config = load_config("config/uspto_50k_parameters.json")

        self.assertEqual(config.task, "sequence_generation")
        self.assertEqual(config.model.att_encoder_type, "attxl")

    def test_sequence_attention_encoder_aliases_normalize_to_model_values(self):
        self.assertEqual(ModelConfig(att_encoder_type="rxngraphormer").att_encoder_type, "attxl")
        self.assertEqual(ModelConfig(att_encoder_type="onmt").att_encoder_type, "attn")

    def test_model_norm_fields_preserve_legacy_batch_norm_controls(self):
        config = ModelConfig(rct_batch_norm=False)

        self.assertEqual(config.rct_norm, "none")
        self.assertEqual(config.pdt_norm, "batchnorm")
        self.assertEqual(config.mid_norm, "batchnorm")

    def test_model_encoder_norm_propagates_to_graph_encoders_when_explicit(self):
        config = ModelConfig(encoder_norm="layernorm")

        self.assertEqual(config.encoder_norm, "layernorm")
        self.assertEqual(config.rct_norm, "layernorm")
        self.assertEqual(config.pdt_norm, "layernorm")
        self.assertEqual(config.mid_norm, "layernorm")

    def test_model_norm_aliases_normalize(self):
        graph_alias = cast(NormType, "graph")
        layer_alias = cast(NormType, "layer")
        config = ModelConfig(rct_norm=graph_alias, head_norm=layer_alias)

        self.assertEqual(config.rct_norm, "graphnorm")
        self.assertEqual(config.head_norm, "layernorm")

    def test_model_head_norm_default_stays_task_specific(self):
        self.assertIsNone(ModelConfig(output_norm=True).head_norm)

        regressor_config = cast(_HeadNormConfig, align_config({"output_norm": True}, "regressor"))
        self.assertEqual(regressor_config.head_norm, "batchnorm")

        regressor_without_output_norm = cast(_HeadNormConfig, align_config({"output_norm": False}, "regressor"))
        self.assertEqual(regressor_without_output_norm.head_norm, "none")

        classifier_config = cast(_HeadNormConfig, align_config({}, "classifier"))
        self.assertEqual(classifier_config.head_norm, "batchnorm")

    def test_jsonc_config_supports_line_and_block_comments(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "parameters.jsonc"
            config_path.write_text(
                """
                {
                  // line comment
                  "task": "regression",
                  "model": {
                    "gnn_type": "gcn",
                    "save_dir": "runs/http://example.test/model" // URL-like string stays intact
                  },
                  /*
                    block comment
                  */
                  "data": {
                    "task": "regression",
                    "rct_data_file": "rct.json",
                    "pdt_data_file": "pdt.json"
                  }
                }
                """,
                encoding="utf-8",
            )

            self.assertEqual(resolve_config_file(tmp_dir), config_path)
            raw = cast(RawConfigDict, load_config_dict(config_path))
            config = load_config(config_path)

        raw_model = cast(dict[str, object], raw["model"])
        self.assertEqual(raw_model["save_dir"], "runs/http://example.test/model")
        self.assertEqual(config.task, "regression")
        self.assertEqual(config.data.rct_name_regrex, "rct.json")
        self.assertEqual(config.data.pdt_name_regrex, "pdt.json")

    def test_reaction_table_reader_supports_csv_and_parquet_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "reactions.csv"
            pd.DataFrame([{"rxn_smiles": "CCO>>CC=O", "target": "1.2"}]).to_csv(csv_path, index=False)

            table = read_reaction_table(csv_path)

        self.assertEqual(table.iloc[0]["rxn_smiles"], "CCO>>CC=O")
        self.assertEqual(table.iloc[0]["target"], "1.2")

    def test_prediction_table_reader_supports_csv_inputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "predictions.csv"
            pd.DataFrame([{"rxn_smiles": "CCO>>CC=O", "score": 0.5}]).to_csv(csv_path, index=False)

            rows = read_prediction_table(csv_path)

        self.assertEqual(rows, [{"rxn_smiles": "CCO>>CC=O", "score": "0.5"}])

    def test_dataloader_kwargs_keep_worker_prefetch_and_persistent_flags(self):
        kwargs = dataloader_kwargs(
            DataLoaderSettings(
                batch_size=4,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=5,
            ),
            shuffle=True,
            drop_last=True,
        )

        self.assertEqual(kwargs["batch_size"], 4)
        self.assertEqual(kwargs["num_workers"], 2)
        self.assertTrue(kwargs["pin_memory"])
        self.assertTrue(kwargs["persistent_workers"])
        self.assertEqual(kwargs["prefetch_factor"], 5)
        self.assertTrue(kwargs["shuffle"])
        self.assertTrue(kwargs["drop_last"])

    def test_oos_probe_records_batch_scaling_policy_when_batch_size_is_overridden(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "data"
            data_path.mkdir()
            (data_path / "train_rct.csv").write_text("a,0\nb,1\nc,0\nd,1\ne,0\n", encoding="utf-8")
            source = tmp_path / "source.json"
            output = tmp_path / "out.json"
            source.write_text(
                json.dumps(
                    {
                        "model": {"save_dir": "old"},
                        "others": {"tag": "old", "device": "cuda:0"},
                        "training": {"epoch": 5, "log_iter_step": 200},
                        "runtime": {},
                        "optimizer": {"learning_rate": 0.4, "weight_decay": 0.0},
                        "scheduler": {"type": "noamlr", "warmup_step": 5000},
                        "data": {"data_path": str(data_path), "train_rct_data_file": "train_rct.csv", "batch_size": 32},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "reproduce" / "train_lit_oos_probe.py"
            write_lit_config = runpy.run_path(str(script_path))["write_lit_config"]

            write_lit_config(
                source,
                output,
                run_root=tmp_path / "runs",
                disable_pretrained=False,
                pretrained_model_path=None,
                encoder_norm=None,
                rct_norm=None,
                pdt_norm=None,
                mid_norm=None,
                head_norm=None,
                batch_size=2,
                accum_steps=None,
                max_epochs=None,
                max_steps=5,
                lr_scaling_policy="linear",
                base_batch_size=32,
                base_learning_rate=None,
                scale_warmup_steps=True,
                scheduler_step_scale_policy="sample",
                precision="32-true",
                num_workers=2,
                persistent_workers=True,
                prefetch_factor=4,
                train_drop_last=False,
                forward_compat_mode="modern",
                log_iter_step=1,
                check_val_every_n_epoch=1,
                val_check_interval=80,
                num_sanity_val_steps=0,
                float32_matmul_precision="highest",
                benchmark=False,
                disable_progress_bar=True,
                disable_model_summary=True,
                compile_model=False,
                compile_mode="default",
            )
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(written["data"]["batch_size"], 2)
        self.assertFalse(written["data"]["train_drop_last"])
        self.assertEqual(written["training"]["max_steps"], 5)
        self.assertEqual(written["training"]["epoch"], 2)
        self.assertEqual(written["runtime"]["val_check_interval"], 80)
        self.assertEqual(written["optimizer"]["lr_scaling_policy"], "linear")
        self.assertEqual(written["optimizer"]["base_batch_size"], 32)
        self.assertEqual(written["scheduler"]["scale_warmup_steps"], True)
        self.assertEqual(written["scheduler"]["base_warmup_step"], 5000)
        self.assertEqual(written["scheduler"]["step_scale_policy"], "sample")

    def test_oos_probe_epoch_estimate_respects_train_drop_last(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "data"
            data_path.mkdir()
            (data_path / "train_rct.csv").write_text("a,0\nb,1\nc,0\nd,1\ne,0\n", encoding="utf-8")
            source = tmp_path / "source.json"
            output = tmp_path / "out.json"
            source.write_text(
                json.dumps(
                    {
                        "model": {"save_dir": "old"},
                        "others": {"tag": "old", "device": "cuda:0"},
                        "training": {"epoch": 5, "log_iter_step": 200},
                        "runtime": {},
                        "optimizer": {"learning_rate": 0.4, "weight_decay": 0.0},
                        "scheduler": {"type": "noamlr", "warmup_step": 5000},
                        "data": {"data_path": str(data_path), "train_rct_data_file": "train_rct.csv", "batch_size": 32},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            script_path = Path(__file__).resolve().parents[1] / "scripts" / "reproduce" / "train_lit_oos_probe.py"
            write_lit_config = runpy.run_path(str(script_path))["write_lit_config"]

            write_lit_config(
                source,
                output,
                run_root=tmp_path / "runs",
                disable_pretrained=False,
                pretrained_model_path=None,
                encoder_norm=None,
                rct_norm=None,
                pdt_norm=None,
                mid_norm=None,
                head_norm=None,
                batch_size=2,
                accum_steps=None,
                max_epochs=None,
                max_steps=5,
                lr_scaling_policy="none",
                base_batch_size=32,
                base_learning_rate=None,
                scale_warmup_steps=False,
                scheduler_step_scale_policy="none",
                precision="32-true",
                num_workers=0,
                persistent_workers=False,
                prefetch_factor=None,
                train_drop_last=True,
                forward_compat_mode="modern",
                log_iter_step=1,
                check_val_every_n_epoch=1,
                val_check_interval=None,
                num_sanity_val_steps=0,
                float32_matmul_precision="highest",
                benchmark=False,
                disable_progress_bar=True,
                disable_model_summary=True,
                compile_model=False,
                compile_mode="default",
            )
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(written["data"]["train_drop_last"])
        self.assertEqual(written["training"]["epoch"], 3)


class PreprocessTableTest(unittest.TestCase):
    def test_reaction_has_atom_mapping_detects_shared_map_numbers(self):
        self.assertTrue(reaction_has_atom_mapping("[CH3:1][OH:2]>>[CH3:1][OH:2]"))
        self.assertFalse(reaction_has_atom_mapping("CCO>>CC=O"))

    def test_reaction_smiles_parser_copies_agents_to_both_legacy_sides(self):
        reactants, products = split_reaction_smiles("CCO>O>CC=O")

        self.assertEqual(reactants, "CCO.O")
        self.assertEqual(products, "CC=O.O")

    def test_reaction_smiles_parser_maps_agents_when_reaction_is_mapped(self):
        parsed = parse_reaction_smiles("[CH3:1][OH:2]>O>[CH3:1][OH:2]")

        self.assertTrue(parsed.has_reactant_product_mapping)
        self.assertIn("[OH2:3]", parsed.reactants)
        self.assertIn("[OH2:3]", parsed.products)
        self.assertEqual(parsed.reactants, parsed.products)

    def test_reaction_smiles_parser_preserves_explicit_mapped_hydrogens(self):
        parsed = parse_reaction_smiles("[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4]>>[C:1]([H:3])([H:4])([H:5])[H:6]")

        self.assertIn("[H:3]", parsed.reactants)
        self.assertIn("[H:3]", parsed.products)
        self.assertNotIn("[*:1]", parsed.reactants)
        self.assertNotIn("[*:2]", parsed.reactants)

    def test_generate_mid_smiles_auto_skips_mapper_for_mapped_reaction(self):
        finder_mock = mock.Mock()
        finder_mock.get_electron_path.return_value = ("[CH3:1]>>[CH3:1]", None, None, None)
        fake_midmol = types.SimpleNamespace(
            finder=finder_mock,
            get_mid_smi_from_rxn=mock.Mock(return_value=["M"]),
            remove_atmmap=mock.Mock(side_effect=lambda smi: smi),
            gen_mech_mid_smi=mock.Mock(return_value=("SHOULD_NOT_USE", "", "")),
        )
        with (
            mock.patch.dict("sys.modules", {"rxngraphormer.midgen.midmol": fake_midmol}),
            mock.patch("rxngraphormer.preprocessing.materialization.canonical_smiles", side_effect=lambda smi: smi),
        ):
            out = generate_mid_smiles("[CH3:1]>>[CH3:1]", mapping_policy="auto")

        self.assertEqual(out, "M")
        finder_mock.get_electron_path.assert_called_once()
        fake_midmol.gen_mech_mid_smi.assert_not_called()

    def test_midmol_mapper_initialization_is_lazy(self):
        import importlib
        import sys

        import rxngraphormer.midgen as midgen_pkg

        module_name = "rxngraphormer.midgen.midmol"
        old_module = sys.modules.pop(module_name, None)
        had_attr = hasattr(midgen_pkg, "midmol")
        old_attr = getattr(midgen_pkg, "midmol", None)
        localmapper_mock = mock.Mock(return_value=mock.Mock())
        rxn_mapper_mock = mock.Mock()
        fake_localmapper = types.SimpleNamespace(localmapper=localmapper_mock)
        fake_rxnmapper = types.SimpleNamespace(RXNMapper=rxn_mapper_mock)
        try:
            with mock.patch.dict(
                "sys.modules",
                {"localmapper": fake_localmapper, "rxnmapper": fake_rxnmapper},
            ):
                midmol = importlib.import_module(module_name)

            self.assertIsNone(midmol.mapper)
            self.assertIsNone(midmol.rxn_mapper)
            localmapper_mock.assert_not_called()
            rxn_mapper_mock.assert_not_called()
        finally:
            sys.modules.pop(module_name, None)
            if old_module is not None:
                sys.modules[module_name] = old_module
            if had_attr:
                setattr(midgen_pkg, "midmol", old_attr)
            elif hasattr(midgen_pkg, "midmol"):
                delattr(midgen_pkg, "midmol")

    def test_midmol_batch_uses_localmapper_list_input(self):
        import rxngraphormer.midgen.midmol as midmol

        old_mapper = midmol.mapper
        old_rxn_mapper = midmol.rxn_mapper
        fake_mapper = mock.Mock()
        fake_mapper.get_atom_map.return_value = ["A_mapped>>B_mapped", "C_mapped>>D_mapped"]
        try:
            midmol.mapper = None
            midmol.rxn_mapper = None
            with (
                mock.patch.object(midmol, "get_localmapper", return_value=fake_mapper),
                mock.patch.object(
                    midmol,
                    "_mech_mid_smi_from_mapped_reaction",
                    side_effect=lambda rct, pdt, mapped: (f"{mapped}_mid", mapped, f"{rct}>>{pdt}"),
                ),
            ):
                out = midmol.gen_mech_mid_smis([("A", "B"), ("C", "D")])

            self.assertEqual(out[0][0], "A_mapped>>B_mapped_mid")
            self.assertEqual(out[1][0], "C_mapped>>D_mapped_mid")
            fake_mapper.get_atom_map.assert_called_once_with(["A>>B", "C>>D"])
        finally:
            midmol.mapper = old_mapper
            midmol.rxn_mapper = old_rxn_mapper

    def test_generate_mid_smiles_never_requires_atom_mapping(self):
        with self.assertRaisesRegex(ValueError, "requires atom-mapped reaction SMILES"):
            generate_mid_smiles("CCO>>CC=O", mapping_policy="never")

    def test_generate_mid_smiles_batch_uses_mapper_for_unmapped_reactions(self):
        mapper = mock.Mock(return_value=[("CC[O]", "", ""), ValueError("cannot map")])
        fake_midmol = types.SimpleNamespace(gen_mech_mid_smis=mapper)

        with mock.patch.dict("sys.modules", {"rxngraphormer.midgen.midmol": fake_midmol}):
            out = generate_mid_smiles_batch(["CCO>>CC=O", "CCN>>CC=N"], mapping_policy="auto")

        self.assertEqual(out[0], "CC[O]")
        self.assertIsInstance(out[1], ValueError)
        mapper.assert_called_once_with([("CCO", "CC=O"), ("CCN", "CC=N")])

    def test_write_reaction_table_files_uses_rxn_smiles_and_mid_smiles_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir, "reactions.csv")
            pd = __import__("pandas")
            pd.DataFrame(
                [
                    {"rxn_smiles": "CCO>>CC=O", "target": 1.2, "mid_smiles": "CC[O]"},
                    {"rxn_smiles": "CCN>>CC=N", "target": 3.4, "mid_smiles": "CC[N]"},
                ]
            ).to_csv(input_path, index=False)

            files = write_reaction_table_files(
                input_path,
                output_dir=tmp_dir,
                target_column="target",
                generate_mid=False,
            )

            self.assertEqual(files.rct_data_file, "reactions_rct.csv")
            self.assertTrue((Path(tmp_dir) / "reactions_rct.csv").exists())
            self.assertTrue((Path(tmp_dir) / "reactions_mid.csv").exists())
            self.assertEqual((Path(tmp_dir) / "reactions_rct.csv").read_text().splitlines()[0], "CCO,1.2")

    def test_write_reaction_table_files_materializes_agents_on_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir, "reactions.csv")
            pd = __import__("pandas")
            pd.DataFrame(
                [
                    {"rxn_smiles": "CCO>O>CC=O", "target": 1.2},
                ]
            ).to_csv(input_path, index=False)

            write_reaction_table_files(
                input_path,
                output_dir=tmp_dir,
                target_column="target",
                generate_mid=False,
            )

            self.assertEqual((Path(tmp_dir) / "reactions_rct.csv").read_text().strip(), "CCO.O,1.2")
            self.assertEqual((Path(tmp_dir) / "reactions_pdt.csv").read_text().strip(), "CC=O.O,1.2")

    def test_write_reaction_table_files_preserves_explicit_mapped_hydrogens(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir, "reactions.csv")
            pd = __import__("pandas")
            pd.DataFrame(
                [
                    {
                        "rxn_smiles": "[C:1](=[C:2]([H:5])[H:6])([H:3])[H:4]>>[C:1]([H:3])([H:4])([H:5])[H:6]",
                        "target": 1.2,
                    },
                ]
            ).to_csv(input_path, index=False)

            write_reaction_table_files(
                input_path,
                output_dir=tmp_dir,
                target_column="target",
                generate_mid=False,
            )

            rct_line = (Path(tmp_dir) / "reactions_rct.csv").read_text().strip()
            self.assertIn("[H:3]", rct_line)
            self.assertNotIn("[*:1]", rct_line)

    def test_preprocess_cli_materializes_input_table_before_regression_processing(self):
        calls: dict[str, object] = {}

        fake_config = SimpleNamespace(
            task="regression",
            data=SimpleNamespace(
                input_table="inputs/reactions.csv",
                data_path="dataset",
                output_prefix="",
                rxn_smiles_column="rxn_smiles",
                rct_smiles_column="rct_smiles",
                pdt_smiles_column="pdt_smiles",
                mid_smiles_column="mid_smiles",
                target_column="target",
                generate_mid=False,
                mapping_policy="auto",
                rct_data_file="",
                pdt_data_file="",
                mid_data_file="",
                rct_name_regrex="",
                pdt_name_regrex="",
                mid_name_regrex="",
                data_trunck=0,
                file_num_trunck=0,
                multi_process=False,
                preprocess_num_workers=8,
                preprocess_batch_size=128,
                preprocess_parallel_mode="reaction",
                task="regression",
            ),
            model=SimpleNamespace(use_mid_inf=False),
        )

        class FakeTableFiles:
            rct_data_file = "reactions_rct.csv"
            pdt_data_file = "reactions_pdt.csv"
            mid_data_file = ""
            rct_name_regrex = "reactions_rct.csv"
            pdt_name_regrex = "reactions_pdt.csv"
            mid_name_regrex = ""

        def fake_write(*args: object, **kwargs: object) -> FakeTableFiles:
            calls["write"] = (args, kwargs)
            return FakeTableFiles()

        dataset_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fake_rxn_dataset(*args: object, **kwargs: object) -> object:
            dataset_calls.append((args, kwargs))
            calls["datasets"] = dataset_calls
            return object()

        with (
            mock.patch("rxngraphormer.preprocessing.workflow.load_config", return_value=fake_config),
            mock.patch("rxngraphormer.preprocessing.workflow.write_reaction_table_files", side_effect=fake_write),
            mock.patch("rxngraphormer.preprocessing.workflow.RXNDataset", side_effect=fake_rxn_dataset),
        ):
            preprocess_from_config("config.json")

        self.assertIn("write", calls)
        self.assertEqual(len(dataset_calls), 2)
        self.assertFalse(dataset_calls[0][1]["multi_process"])
        self.assertEqual(dataset_calls[0][1]["num_worker"], 8)
        self.assertEqual(dataset_calls[0][1]["batch_size"], 128)
        self.assertEqual(dataset_calls[0][1]["parallel_mode"], "reaction")

    def test_preprocess_cli_passes_parallel_settings_to_classification_datasets(self):
        fake_config = SimpleNamespace(
            task="classification",
            data=SimpleNamespace(
                input_table="",
                data_path="dataset",
                rct_name_regrex="rct_*.csv",
                pdt_name_regrex="pdt_*.csv",
                data_trunck=0,
                file_num_trunck=2,
                multi_process=False,
                preprocess_num_workers=4,
                preprocess_batch_size=64,
                preprocess_parallel_mode="reaction",
                task="classification",
            ),
            model=SimpleNamespace(use_mid_inf=False),
        )
        dataset_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fake_multi_dataset(*args: object, **kwargs: object) -> object:
            dataset_calls.append((args, kwargs))
            return object()

        with (
            mock.patch("rxngraphormer.preprocessing.workflow.load_config", return_value=fake_config),
            mock.patch("rxngraphormer.preprocessing.workflow.MultiRXNDataset", side_effect=fake_multi_dataset),
        ):
            preprocess_from_config(
                "config.json",
                multi_process=True,
                preprocess_num_workers=16,
                preprocess_batch_size=256,
                preprocess_parallel_mode="file",
            )

        self.assertEqual(len(dataset_calls), 2)
        self.assertEqual(dataset_calls[0][1]["name_tag"], "rct")
        self.assertEqual(dataset_calls[1][1]["name_tag"], "pdt")
        self.assertTrue(dataset_calls[0][1]["multi_process"])
        self.assertEqual(dataset_calls[0][1]["num_worker"], 16)
        self.assertEqual(dataset_calls[0][1]["batch_size"], 256)
        self.assertEqual(dataset_calls[0][1]["parallel_mode"], "file")
        self.assertEqual(dataset_calls[1][1]["num_worker"], 16)
        self.assertEqual(dataset_calls[1][1]["parallel_mode"], "file")

    def test_preprocess_cli_generates_mid_files_for_classification_mid_info(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir)
            (data_path / "rxn_rct_0.csv").write_text("CCO,0\nCCN,1\nCCC,0\n", encoding="utf-8")
            (data_path / "rxn_pdt_0.csv").write_text("CC=O,0\nCC=N,1\nCC=C,0\n", encoding="utf-8")
            fake_config = SimpleNamespace(
                task="classification",
                data=SimpleNamespace(
                    input_table="",
                    data_path=str(data_path),
                    rct_name_regrex="rxn_rct_*.csv",
                    pdt_name_regrex="rxn_pdt_*.csv",
                    mid_name_regrex="rxn_mid_*.csv",
                    mid_data_file="",
                    generate_mid=True,
                    mapping_policy="auto",
                    data_trunck=0,
                    file_num_trunck=0,
                    multi_process=False,
                    preprocess_num_workers=4,
                    preprocess_batch_size=64,
                    preprocess_parallel_mode="reaction",
                    task="classification",
                ),
                model=SimpleNamespace(use_mid_inf=True),
            )
            dataset_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def fake_multi_dataset(*args: object, **kwargs: object) -> object:
                dataset_calls.append((args, kwargs))
                return object()

            with (
                mock.patch("rxngraphormer.preprocessing.workflow.load_config", return_value=fake_config),
                mock.patch(
                    "rxngraphormer.preprocessing.workflow.generate_mid_smiles_batch",
                    return_value=["CC[O]", "", RuntimeError("cannot map")],
                ),
                mock.patch("rxngraphormer.preprocessing.workflow.MultiRXNDataset", side_effect=fake_multi_dataset),
            ):
                preprocess_from_config("config.json")

            self.assertEqual((data_path / "rxn_midvalid_rct_0.csv").read_text(encoding="utf-8").strip(), "CCO,0")
            self.assertEqual((data_path / "rxn_midvalid_pdt_0.csv").read_text(encoding="utf-8").strip(), "CC=O,0")
            self.assertEqual((data_path / "rxn_midvalid_mid_0.csv").read_text(encoding="utf-8").strip(), "CC[O],0")
            self.assertEqual([call[1]["name_tag"] for call in dataset_calls], ["rct", "pdt", "mid"])
            self.assertEqual(
                [call[1]["name_regrex"] for call in dataset_calls],
                ["rxn_midvalid_rct_*.csv", "rxn_midvalid_pdt_*.csv", "rxn_midvalid_mid_*.csv"],
            )

    def test_preprocess_cli_generates_mid_files_in_file_parallel_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir)
            (data_path / "rxn_rct_0.csv").write_text("CCO,0\n", encoding="utf-8")
            (data_path / "rxn_pdt_0.csv").write_text("CC=O,0\n", encoding="utf-8")
            (data_path / "rxn_rct_1.csv").write_text("CCN,1\n", encoding="utf-8")
            (data_path / "rxn_pdt_1.csv").write_text("CC=N,1\n", encoding="utf-8")
            fake_config = SimpleNamespace(
                task="classification",
                data=SimpleNamespace(
                    input_table="",
                    data_path=str(data_path),
                    rct_name_regrex="rxn_rct_*.csv",
                    pdt_name_regrex="rxn_pdt_*.csv",
                    mid_name_regrex="rxn_mid_*.csv",
                    mid_data_file="",
                    generate_mid=True,
                    mapping_policy="auto",
                    data_trunck=0,
                    file_num_trunck=0,
                    multi_process=True,
                    preprocess_num_workers=8,
                    preprocess_batch_size=64,
                    preprocess_parallel_mode="file",
                    task="classification",
                ),
                model=SimpleNamespace(use_mid_inf=True),
            )
            dataset_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
            pool_processes: list[int] = []
            pool_initializers: list[object] = []

            class FakePool:
                def __init__(self, processes: int, initializer: object | None = None) -> None:
                    pool_processes.append(processes)
                    pool_initializers.append(initializer)

                def __enter__(self) -> "FakePool":
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

                def imap_unordered(self, worker: object, tasks: object) -> object:
                    return [worker(task) for task in reversed(list(tasks))]

            def fake_multi_dataset(*args: object, **kwargs: object) -> object:
                dataset_calls.append((args, kwargs))
                return object()

            with (
                mock.patch("rxngraphormer.preprocessing.workflow.load_config", return_value=fake_config),
                mock.patch(
                    "rxngraphormer.preprocessing.workflow.generate_mid_smiles_batch",
                    side_effect=[["CC[N]"], ["CC[O]"]],
                ),
                mock.patch("rxngraphormer.preprocessing.workflow.Pool", FakePool),
                mock.patch("rxngraphormer.preprocessing.workflow.MultiRXNDataset", side_effect=fake_multi_dataset),
            ):
                preprocess_from_config("config.json")

            self.assertEqual(pool_processes, [2])
            self.assertEqual(len(pool_initializers), 1)
            self.assertIsNotNone(pool_initializers[0])
            self.assertEqual((data_path / "rxn_midvalid_rct_0.csv").read_text(encoding="utf-8").strip(), "CCO,0")
            self.assertEqual((data_path / "rxn_midvalid_pdt_0.csv").read_text(encoding="utf-8").strip(), "CC=O,0")
            self.assertEqual((data_path / "rxn_midvalid_mid_0.csv").read_text(encoding="utf-8").strip(), "CC[O],0")
            self.assertEqual((data_path / "rxn_midvalid_rct_1.csv").read_text(encoding="utf-8").strip(), "CCN,1")
            self.assertEqual((data_path / "rxn_midvalid_pdt_1.csv").read_text(encoding="utf-8").strip(), "CC=N,1")
            self.assertEqual((data_path / "rxn_midvalid_mid_1.csv").read_text(encoding="utf-8").strip(), "CC[N],1")
            self.assertEqual(
                [call[1]["name_regrex"] for call in dataset_calls],
                ["rxn_midvalid_rct_*.csv", "rxn_midvalid_pdt_*.csv", "rxn_midvalid_mid_*.csv"],
            )
            self.assertTrue(all(call[1]["parallel_mode"] == "file" for call in dataset_calls))

    def test_preprocess_cli_uses_complete_existing_midvalid_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir)
            (data_path / "rxn_rct_0.csv").write_text("CCO,0\n", encoding="utf-8")
            (data_path / "rxn_pdt_0.csv").write_text("CC=O,0\n", encoding="utf-8")
            (data_path / "rxn_midvalid_rct_0.csv").write_text("CCO,0\n", encoding="utf-8")
            (data_path / "rxn_midvalid_pdt_0.csv").write_text("CC=O,0\n", encoding="utf-8")
            (data_path / "rxn_midvalid_mid_0.csv").write_text("CC[O],0\n", encoding="utf-8")
            fake_config = SimpleNamespace(
                task="classification",
                data=SimpleNamespace(
                    input_table="",
                    data_path=str(data_path),
                    rct_name_regrex="rxn_rct_*.csv",
                    pdt_name_regrex="rxn_pdt_*.csv",
                    mid_name_regrex="rxn_midvalid_mid_*.csv",
                    mid_data_file="",
                    generate_mid=True,
                    mapping_policy="auto",
                    data_trunck=0,
                    file_num_trunck=0,
                    multi_process=False,
                    preprocess_num_workers=4,
                    preprocess_batch_size=64,
                    preprocess_parallel_mode="reaction",
                    task="classification",
                ),
                model=SimpleNamespace(use_mid_inf=True),
            )
            dataset_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def fake_multi_dataset(*args: object, **kwargs: object) -> object:
                dataset_calls.append((args, kwargs))
                return object()

            with (
                mock.patch("rxngraphormer.preprocessing.workflow.load_config", return_value=fake_config),
                mock.patch("rxngraphormer.preprocessing.workflow.generate_mid_smiles_batch") as generate_batch,
                mock.patch("rxngraphormer.preprocessing.workflow.MultiRXNDataset", side_effect=fake_multi_dataset),
            ):
                preprocess_from_config("config.json")

            generate_batch.assert_not_called()
            self.assertEqual(
                [call[1]["name_regrex"] for call in dataset_calls],
                ["rxn_midvalid_rct_*.csv", "rxn_midvalid_pdt_*.csv", "rxn_midvalid_mid_*.csv"],
            )

    def test_preprocess_cli_regenerates_when_existing_midvalid_set_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir)
            (data_path / "rxn_rct_0.csv").write_text("CCO,0\n", encoding="utf-8")
            (data_path / "rxn_pdt_0.csv").write_text("CC=O,0\n", encoding="utf-8")
            (data_path / "rxn_rct_1.csv").write_text("CCN,1\n", encoding="utf-8")
            (data_path / "rxn_pdt_1.csv").write_text("CC=N,1\n", encoding="utf-8")
            (data_path / "rxn_midvalid_rct_0.csv").write_text("CCO,0\n", encoding="utf-8")
            (data_path / "rxn_midvalid_pdt_0.csv").write_text("CC=O,0\n", encoding="utf-8")
            (data_path / "rxn_midvalid_mid_0.csv").write_text("CC[O],0\n", encoding="utf-8")
            fake_config = SimpleNamespace(
                task="classification",
                data=SimpleNamespace(
                    input_table="",
                    data_path=str(data_path),
                    rct_name_regrex="rxn_rct_*.csv",
                    pdt_name_regrex="rxn_pdt_*.csv",
                    mid_name_regrex="rxn_midvalid_mid_*.csv",
                    mid_data_file="",
                    generate_mid=True,
                    mapping_policy="auto",
                    data_trunck=0,
                    file_num_trunck=0,
                    multi_process=False,
                    preprocess_num_workers=4,
                    preprocess_batch_size=64,
                    preprocess_parallel_mode="reaction",
                    task="classification",
                ),
                model=SimpleNamespace(use_mid_inf=True),
            )
            dataset_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def fake_multi_dataset(*args: object, **kwargs: object) -> object:
                dataset_calls.append((args, kwargs))
                return object()

            with (
                mock.patch("rxngraphormer.preprocessing.workflow.load_config", return_value=fake_config),
                mock.patch(
                    "rxngraphormer.preprocessing.workflow.generate_mid_smiles_batch",
                    return_value=["CC[N]"],
                ) as generate_batch,
                mock.patch("rxngraphormer.preprocessing.workflow.MultiRXNDataset", side_effect=fake_multi_dataset),
            ):
                preprocess_from_config("config.json")

            generate_batch.assert_called_once_with(["CCN>>CC=N"], mapping_policy="auto")
            self.assertEqual((data_path / "rxn_midvalid_mid_0.csv").read_text(encoding="utf-8").strip(), "CC[O],0")
            self.assertEqual((data_path / "rxn_midvalid_mid_1.csv").read_text(encoding="utf-8").strip(), "CC[N],1")
            self.assertEqual(
                [call[1]["name_regrex"] for call in dataset_calls],
                ["rxn_midvalid_rct_*.csv", "rxn_midvalid_pdt_*.csv", "rxn_midvalid_mid_*.csv"],
            )

    def test_file_parallel_worker_disables_nested_reaction_multiprocessing(self):
        from rxngraphormer.data.multi_reaction_dataset import _MultiReactionFileTask, _process_reaction_file
        from rxngraphormer.data.reaction_processing import ReactionProcessingSettings

        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_file = Path(tmp_dir) / "raw.csv"
            processed_file = Path(tmp_dir) / "processed.safetensors"
            raw_file.write_text("CCO,0\nCCN,1\n", encoding="utf-8")
            task = _MultiReactionFileTask(
                index=3,
                progress_position=7,
                raw_data_file=str(raw_file),
                processed_path=str(processed_file),
                trunck=1,
                settings=ReactionProcessingSettings(
                    task="classification",
                    multi_process=True,
                    num_workers=64,
                    batch_size=256,
                ),
            )

            with (
                mock.patch("rxngraphormer.data.multi_reaction_dataset.processed_file_exists", return_value=False),
                mock.patch(
                    "rxngraphormer.data.multi_reaction_dataset.process_reaction_lines", return_value=[object()]
                ) as process_lines,
                mock.patch(
                    "rxngraphormer.data.multi_reaction_dataset.InMemoryDataset.collate", return_value=("data", "slices")
                ),
                mock.patch("rxngraphormer.data.multi_reaction_dataset.save_processed_graph_data") as save_processed,
            ):
                result = _process_reaction_file(task)

        self.assertEqual(result, (3, str(processed_file), 1))
        process_lines.assert_called_once()
        lines_arg, settings_arg = process_lines.call_args.args[:2]
        self.assertEqual(lines_arg, ["CCO,0"])
        self.assertFalse(settings_arg.multi_process)
        self.assertEqual(settings_arg.num_workers, 1)
        self.assertTrue(process_lines.call_args.kwargs["show_progress"])
        self.assertEqual(process_lines.call_args.kwargs["progress_position"], 7)
        self.assertTrue(process_lines.call_args.kwargs["progress_leave"])
        save_processed.assert_called_once_with("data", "slices", str(processed_file))
