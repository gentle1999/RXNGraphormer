import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from rxngraphormer.config import load_train_config
from rxngraphormer.data.collate import triple_collate_fn
from rxngraphormer.data.loader import DataLoaderSettings
from rxngraphormer.data.pairing import FastBatchTripleDataset
from rxngraphormer.evaluation.classification_workflow import classification_split_files
from rxngraphormer.evaluation.regression_workflow import regression_split_files
from rxngraphormer.lightning.datamodule import RXNGraphormerDataModule
from rxngraphormer.lightning.workflow import (
    LightningFitSettings,
    build_dataloader_settings,
    fit_config,
    write_lightning_fit_config,
)


class LightningWorkflowTest(unittest.TestCase):
    def test_lightning_workflow_builds_dataloader_settings_from_config_and_overrides(self):
        config = type(
            "Config",
            (),
            {
                "data": type(
                    "Data",
                    (),
                    {
                        "batch_size": 32,
                        "num_workers": 2,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": 3,
                        "train_drop_last": False,
                    },
                )(),
            },
        )()

        settings = build_dataloader_settings(
            config,
            LightningFitSettings(
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
                train_drop_last=True,
            ),
        )

        self.assertEqual(settings.batch_size, 32)
        self.assertEqual(settings.num_workers, 8)
        self.assertTrue(settings.pin_memory)
        self.assertTrue(settings.persistent_workers)
        self.assertEqual(settings.prefetch_factor, 4)
        self.assertTrue(settings.train_drop_last)

    def test_datamodule_loads_split_manifest_indices(self):
        config = type(
            "Config",
            (),
            {
                "task": "regression",
                "data": type(
                    "Data",
                    (),
                    {
                        "batch_size": 2,
                        "num_workers": 0,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": None,
                        "train_ratio": 0.5,
                        "valid_ratio": 0.25,
                        "seed": 7,
                    },
                )(),
                "model": type("Model", (), {"use_mid_inf": False})(),
            },
        )()
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "split.json"
            manifest_path.write_text(
                json.dumps({"indices": {"train": [2, 0], "valid": [1], "test": [3]}}),
                encoding="utf-8",
            )
            datamodule = RXNGraphormerDataModule(config, split_manifest=manifest_path)

            split = datamodule._load_or_make_split(4)

        self.assertTrue(torch.equal(split["train"], torch.tensor([2, 0])))
        self.assertTrue(torch.equal(split["valid"], torch.tensor([1])))
        self.assertTrue(torch.equal(split["test"], torch.tensor([3])))

    def test_datamodule_preloads_split_graph_cache_when_enabled(self):
        config = SimpleNamespace(
            task="regression",
            data=SimpleNamespace(batch_size=2, preload_graph_cache=True),
            model=SimpleNamespace(use_mid_inf=True),
        )
        datamodule = RXNGraphormerDataModule(config)

        class FakeGraphDataset:
            def __init__(self, indices):
                self._indices = indices
                self.loaded = []

            def indices(self):
                return self._indices

            def get(self, idx):
                self.loaded.append(idx)
                return idx

            def __len__(self):
                return len(self._indices)

        rct = FakeGraphDataset([2, 0])
        pdt = FakeGraphDataset([2, 0])
        mid = FakeGraphDataset([2, 0])
        datamodule.train_dataset = datamodule._make_dataset(rct, pdt, mid)
        datamodule.valid_dataset = datamodule._make_dataset(
            FakeGraphDataset([1]), FakeGraphDataset([1]), FakeGraphDataset([1])
        )
        datamodule.test_dataset = datamodule._make_dataset(
            FakeGraphDataset([3]), FakeGraphDataset([3]), FakeGraphDataset([3])
        )

        datamodule._preload_graph_cache_if_enabled()

        self.assertEqual(rct.loaded, [2, 0])
        self.assertEqual(pdt.loaded, [2, 0])
        self.assertEqual(mid.loaded, [2, 0])

    def test_fast_batch_collate_matches_standard_collate_for_full_df_subset(self):
        config_path = Path("config_toml/fixed_axis_cv_rerun_20260612/rxngraphormer_standard_oos/ene_fold_6.toml")
        split_path = Path(
            "config_toml/full_df_axis_separate_oos_cv_max2_10fold_graphnorm_pretrain/split_manifests/ene_fold_6.json"
        )
        if not config_path.exists() or not split_path.exists():
            self.skipTest("full_df fixed rerun assets are not available")
        config = load_train_config(config_path)
        datamodule = RXNGraphormerDataModule(
            config,
            split_manifest=split_path,
            dataloader=DataLoaderSettings(batch_size=8, num_workers=0),
        )
        datamodule.setup("fit")
        indices = [0, 3, 5, 9, 12]

        standard = triple_collate_fn([datamodule.train_dataset[idx] for idx in indices])
        fast = FastBatchTripleDataset(datamodule.train_dataset)[indices]

        for standard_batch, fast_batch in zip(standard, fast):
            for key in ("x", "edge_index", "edge_attr", "atom_mass", "mol_index", "y", "batch", "ptr"):
                self.assertTrue(torch.equal(getattr(standard_batch, key), getattr(fast_batch, key)), key)

    def test_datamodule_passes_parallel_preprocess_settings_to_classification_datasets(self):
        config = SimpleNamespace(
            task="classification",
            data=SimpleNamespace(
                data_path="dataset",
                data_trunck=0,
                train_rct_data_file="",
                train_pdt_data_file="",
                val_rct_data_file="",
                val_pdt_data_file="",
                test_rct_data_file="",
                test_pdt_data_file="",
                rct_name_regrex="rct_*.csv",
                pdt_name_regrex="pdt_*.csv",
                rct_data_file="",
                pdt_data_file="",
                file_num_trunck=2,
                train_ratio=0.5,
                valid_ratio=0.25,
                seed=7,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=None,
                multi_process=True,
                preprocess_num_workers=12,
                preprocess_batch_size=256,
            ),
            model=SimpleNamespace(use_mid_inf=False),
        )

        class FakeDataset:
            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: object) -> object:
                return index

        with mock.patch(
            "rxngraphormer.lightning.datamodule.MultiRXNDataset", return_value=FakeDataset()
        ) as dataset_cls:
            datamodule = RXNGraphormerDataModule(config)
            datamodule._make_dataset = mock.Mock(return_value=object())
            datamodule.setup()

        self.assertEqual(dataset_cls.call_count, 2)
        kwargs = dataset_cls.call_args_list[0].kwargs
        self.assertTrue(kwargs["multi_process"])
        self.assertEqual(kwargs["num_worker"], 12)
        self.assertEqual(kwargs["batch_size"], 256)
        self.assertEqual(kwargs["name_tag"], "rct")

    def test_lightning_workflow_wires_fit_artifacts_and_preserves_cli_settings(self):
        config = type(
            "Config",
            (),
            {
                "task": "regression",
                "runtime": type(
                    "Runtime",
                    (),
                    {
                        "deterministic": False,
                        "compile_model": False,
                        "compile_mode": None,
                        "early_stopping_patience": 0,
                    },
                )(),
                "data": type(
                    "Data",
                    (),
                    {
                        "batch_size": 16,
                        "num_workers": 1,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": None,
                    },
                )(),
                "training": type("Training", (), {"max_steps": None, "accum": 1})(),
                "model": type("Model", (), {"save_dir": "default_save"})(),
            },
        )()
        fake_datamodule = mock.Mock(name="datamodule")
        fake_lit_module = mock.Mock(name="lit_module")
        fake_trainer = mock.Mock(name="trainer")

        settings = LightningFitSettings(
            default_root_dir="runs/lit",
            accelerator="cpu",
            devices="1",
            precision="32-true",
            max_epochs=7,
            max_steps=30,
            accum_steps=8,
            split_manifest="split.json",
            resume_from_checkpoint="resume.ckpt",
            compile_model=True,
            compile_mode="reduce-overhead",
            early_stopping_patience=5,
            val_check_interval=10,
            write_manifest=False,
        )
        with (
            mock.patch(
                "rxngraphormer.lightning.workflow.RXNGraphormerDataModule", return_value=fake_datamodule
            ) as datamodule_cls,
            mock.patch(
                "rxngraphormer.lightning.workflow.RXNGraphormerLitModule.from_config", return_value=fake_lit_module
            ) as from_config,
            mock.patch("rxngraphormer.lightning.workflow.build_trainer", return_value=fake_trainer) as trainer_builder,
            mock.patch("rxngraphormer.lightning.workflow.export_model_state_dict"),
            mock.patch("rxngraphormer.lightning.workflow.write_lightning_fit_config"),
        ):
            artifacts = fit_config(config, settings)

        datamodule_cls.assert_called_once()
        self.assertEqual(datamodule_cls.call_args.kwargs["split_manifest"], "split.json")
        from_config.assert_called_once_with(config)
        trainer_builder.assert_called_once_with(
            config,
            default_root_dir="runs/lit",
            accelerator="cpu",
            devices="1",
            precision="32-true",
            max_epochs=7,
            max_steps=30,
        )
        fake_trainer.fit.assert_called_once_with(fake_lit_module, datamodule=fake_datamodule, ckpt_path="resume.ckpt")
        self.assertIs(artifacts.trainer, fake_trainer)
        self.assertTrue(config.runtime.compile_model)
        self.assertEqual(config.runtime.compile_mode, "reduce-overhead")
        self.assertEqual(config.runtime.early_stopping_patience, 5)
        self.assertEqual(config.training.max_steps, 30)
        self.assertEqual(config.training.accum, 8)
        self.assertEqual(config.runtime.val_check_interval, 10)

    def test_lightning_workflow_derives_epoch_budget_when_max_steps_is_set(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "data"
            data_path.mkdir()
            (data_path / "train_rct.csv").write_text("a,0\nb,1\nc,0\nd,1\ne,0\n", encoding="utf-8")
            config = type(
                "Config",
                (),
                {
                    "task": "regression",
                    "runtime": type(
                        "Runtime",
                        (),
                        {
                            "deterministic": False,
                            "compile_model": False,
                            "compile_mode": None,
                            "early_stopping_patience": 0,
                        },
                    )(),
                    "data": type(
                        "Data",
                        (),
                        {
                            "data_path": str(data_path),
                            "train_rct_data_file": "train_rct.csv",
                            "rct_data_file": "",
                            "rct_name_regrex": "",
                            "data_trunck": 0,
                            "file_num_trunck": 0,
                            "train_ratio": 0.8,
                            "batch_size": 2,
                            "num_workers": 0,
                            "pin_memory": False,
                            "persistent_workers": False,
                            "prefetch_factor": None,
                        },
                    )(),
                    "training": type("Training", (), {"epoch": 99, "max_steps": None, "accum": 1})(),
                    "model": type("Model", (), {"save_dir": "default_save"})(),
                },
            )()
            fake_trainer = mock.Mock(name="trainer")

            with (
                mock.patch("rxngraphormer.lightning.workflow.RXNGraphormerDataModule", return_value=mock.Mock()),
                mock.patch(
                    "rxngraphormer.lightning.workflow.RXNGraphormerLitModule.from_config", return_value=mock.Mock()
                ),
                mock.patch(
                    "rxngraphormer.lightning.workflow.build_trainer", return_value=fake_trainer
                ) as trainer_builder,
                mock.patch("rxngraphormer.lightning.workflow.export_model_state_dict"),
                mock.patch("rxngraphormer.lightning.workflow.write_lightning_fit_config"),
            ):
                fit_config(config, LightningFitSettings(max_steps=5, write_manifest=False))

        self.assertEqual(config.training.max_steps, 5)
        self.assertEqual(config.training.epoch, 2)
        self.assertEqual(trainer_builder.call_args.args[0].training.epoch, 2)
        self.assertEqual(trainer_builder.call_args.kwargs["max_steps"], 5)

    def test_lightning_workflow_writes_manifest_and_optional_eval_report(self):
        config = type(
            "Config",
            (),
            {
                "task": "regression",
                "runtime": type(
                    "Runtime",
                    (),
                    {
                        "deterministic": False,
                        "compile_model": False,
                        "compile_mode": None,
                        "early_stopping_patience": 0,
                    },
                )(),
                "data": type(
                    "Data",
                    (),
                    {
                        "batch_size": 16,
                        "num_workers": 0,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": None,
                    },
                )(),
                "model": type("Model", (), {"save_dir": "default_save"})(),
            },
        )()
        fake_datamodule = mock.Mock(name="datamodule")
        fake_lit_module = mock.Mock(name="lit_module")
        fake_trainer = mock.Mock(name="trainer")
        fake_trainer.logger.log_dir = "runs/lit/version_0"
        fake_trainer.checkpoint_callback.best_model_path = "runs/lit/version_0/checkpoints/best.ckpt"
        fake_trainer.checkpoint_callback.last_model_path = ""
        fake_trainer.callback_metrics = {"val_mae": torch.tensor(0.5)}

        settings = LightningFitSettings(
            default_root_dir="runs/lit",
            config_path="config.json",
            eval_after_fit=True,
            eval_splits=("valid", "test"),
            eval_batch_size=64,
            eval_scale=100.0,
            eval_yield_constrain=True,
        )
        with (
            mock.patch("rxngraphormer.lightning.workflow.RXNGraphormerDataModule", return_value=fake_datamodule),
            mock.patch(
                "rxngraphormer.lightning.workflow.RXNGraphormerLitModule.from_config", return_value=fake_lit_module
            ),
            mock.patch("rxngraphormer.lightning.workflow.build_trainer", return_value=fake_trainer),
            mock.patch("rxngraphormer.lightning.workflow.export_model_state_dict"),
            mock.patch(
                "rxngraphormer.lightning.workflow.write_lightning_fit_config",
                return_value="runs/lit/version_0/parameters.json",
            ),
            mock.patch(
                "rxngraphormer.lightning.workflow.run_post_fit_regression_eval", return_value={"json": "eval.json"}
            ) as post_eval,
            mock.patch(
                "rxngraphormer.lightning.workflow.write_fit_manifest", return_value={"output_manifest": "manifest.json"}
            ) as write_manifest,
        ):
            artifacts = fit_config(config, settings)

        post_eval.assert_called_once()
        self.assertEqual(
            post_eval.call_args.kwargs["checkpoint_path"], "runs/lit/version_0/model/valid_checkpoint.safetensors"
        )
        self.assertEqual(post_eval.call_args.kwargs["splits"], ("valid", "test"))
        self.assertEqual(post_eval.call_args.kwargs["scale"], 100.0)
        write_manifest.assert_called_once()
        manifest = write_manifest.call_args.args[0]
        self.assertEqual(manifest["config_path"], "config.json")
        self.assertEqual(
            manifest["checkpoint"]["best_model_path"], "runs/lit/version_0/model/valid_checkpoint.safetensors"
        )
        self.assertEqual(manifest["checkpoint"]["trainer_best_model_path"], "runs/lit/version_0/checkpoints/best.ckpt")
        self.assertEqual(manifest["eval_reports"], {"regression": {"json": "eval.json"}})
        self.assertEqual(artifacts.manifest_paths, {"output_manifest": "manifest.json"})
        self.assertEqual(artifacts.exported_config_path, "runs/lit/version_0/parameters.json")

    def test_lightning_workflow_routes_classification_post_fit_eval(self):
        config = type(
            "Config",
            (),
            {
                "task": "classification",
                "runtime": type(
                    "Runtime",
                    (),
                    {
                        "deterministic": False,
                        "compile_model": False,
                        "compile_mode": None,
                        "early_stopping_patience": 0,
                    },
                )(),
                "data": type(
                    "Data",
                    (),
                    {
                        "batch_size": 16,
                        "num_workers": 0,
                        "pin_memory": False,
                        "persistent_workers": False,
                        "prefetch_factor": None,
                    },
                )(),
                "model": type("Model", (), {"save_dir": "default_save"})(),
            },
        )()
        fake_datamodule = mock.Mock(name="datamodule")
        fake_lit_module = mock.Mock(name="lit_module")
        fake_trainer = mock.Mock(name="trainer")
        fake_trainer.logger.log_dir = "runs/lit/version_0"
        fake_trainer.checkpoint_callback.best_model_path = "runs/lit/version_0/checkpoints/best.ckpt"
        fake_trainer.checkpoint_callback.last_model_path = ""
        fake_trainer.callback_metrics = {"val_acc": torch.tensor(0.75)}

        settings = LightningFitSettings(
            default_root_dir="runs/lit",
            config_path="config.json",
            eval_after_fit=True,
            eval_splits=("valid",),
            eval_batch_size=64,
        )
        with (
            mock.patch("rxngraphormer.lightning.workflow.RXNGraphormerDataModule", return_value=fake_datamodule),
            mock.patch(
                "rxngraphormer.lightning.workflow.RXNGraphormerLitModule.from_config", return_value=fake_lit_module
            ),
            mock.patch("rxngraphormer.lightning.workflow.build_trainer", return_value=fake_trainer),
            mock.patch("rxngraphormer.lightning.workflow.export_model_state_dict"),
            mock.patch("rxngraphormer.lightning.workflow.write_lightning_fit_config"),
            mock.patch(
                "rxngraphormer.lightning.workflow.run_post_fit_classification_eval", return_value={"json": "eval.json"}
            ) as post_eval,
            mock.patch(
                "rxngraphormer.lightning.workflow.write_fit_manifest", return_value={"output_manifest": "manifest.json"}
            ) as write_manifest,
        ):
            artifacts = fit_config(config, settings)

        post_eval.assert_called_once()
        self.assertEqual(
            post_eval.call_args.kwargs["checkpoint_path"], "runs/lit/version_0/model/valid_checkpoint.safetensors"
        )
        self.assertEqual(post_eval.call_args.kwargs["splits"], ("valid",))
        manifest = write_manifest.call_args.args[0]
        self.assertEqual(manifest["eval_reports"], {"classification": {"json": "eval.json"}})
        self.assertEqual(artifacts.eval_reports, {"classification": {"json": "eval.json"}})

    def test_lightning_workflow_writes_config_next_to_exported_checkpoint(self):
        config = SimpleNamespace(
            task="classification",
            model=SimpleNamespace(rct_norm="graphnorm", pdt_norm="graphnorm"),
            data=SimpleNamespace(batch_size=4),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_lightning_fit_config(config, output_dir=tmp_dir)
            payload = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(Path(path).name, "parameters.json")
        self.assertEqual(payload["task"], "classification")
        self.assertEqual(payload["model"]["rct_norm"], "graphnorm")

    def test_regression_eval_auto_uses_explicit_oos_split_files(self):
        config = type(
            "Config",
            (),
            {
                "data": type(
                    "Data",
                    (),
                    {
                        "rct_data_file": "",
                        "pdt_data_file": "",
                        "mid_data_file": "",
                        "train_rct_data_file": "train_rct.csv",
                        "train_pdt_data_file": "train_pdt.csv",
                        "train_mid_data_file": "train_mid.csv",
                        "val_rct_data_file": "val_rct.csv",
                        "val_pdt_data_file": "val_pdt.csv",
                        "val_mid_data_file": "val_mid.csv",
                        "test_rct_data_file": "test_rct.csv",
                        "test_pdt_data_file": "test_pdt.csv",
                        "test_mid_data_file": "test_mid.csv",
                    },
                )(),
            },
        )()

        files = regression_split_files(config, "test", specific_val=False)

        self.assertEqual(files, {"rct": "test_rct.csv", "pdt": "test_pdt.csv", "mid": "test_mid.csv"})

    def test_classification_eval_auto_uses_explicit_oos_split_files(self):
        config = type(
            "Config",
            (),
            {
                "data": type(
                    "Data",
                    (),
                    {
                        "rct_data_file": "",
                        "pdt_data_file": "",
                        "rct_name_regrex": "",
                        "pdt_name_regrex": "",
                        "train_rct_data_file": "train_rct.csv",
                        "train_pdt_data_file": "train_pdt.csv",
                        "val_rct_data_file": "val_rct.csv",
                        "val_pdt_data_file": "val_pdt.csv",
                        "test_rct_data_file": "test_rct.csv",
                        "test_pdt_data_file": "test_pdt.csv",
                    },
                )(),
            },
        )()

        files = classification_split_files(config, "valid", specific_val=False)

        self.assertEqual(files, {"rct": "val_rct.csv", "pdt": "val_pdt.csv"})
