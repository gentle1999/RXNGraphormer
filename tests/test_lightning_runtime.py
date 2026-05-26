import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np
import torch

from rxngraphormer.config import load_train_config
from rxngraphormer.evaluation import (
    classification_batch_input,
    classification_metrics,
    evaluate_classification,
    regression_metrics,
)
from rxngraphormer.lightning.callbacks import EpochRuntimeMonitor
from rxngraphormer.lightning.module import build_regression_loss
from rxngraphormer.lightning.trainer import build_callbacks, build_trainer
from rxngraphormer.runtime import DeviceManager, move_to_device, resolve_device
from rxngraphormer.training import RXNGraphormerLitModule
from rxngraphormer.training.optimization import resolve_optimization_plan
from rxngraphormer.training.schedulers import NoamLR


class LightningRuntimeTest(unittest.TestCase):
    @staticmethod
    def _classification_config():
        return type(
            "Config",
            (),
            {
                "task": "classification",
                "training": type("Training", (), {"loss": "ce"})(),
                "runtime": type("Runtime", (), {"compile_model": False})(),
                "optimizer": type("Optimizer", (), {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0})(),
                "scheduler": type("Scheduler", (), {"type": "none"})(),
                "model": type("Model", (), {"pretrained_model_path": ""})(),
            },
        )()

    class _TinyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[2.0, -2.0], [-2.0, 2.0]]))

        def logits(self, data):
            return data[0].x @ self.weight

        def forward(self, data):
            return torch.softmax(self.logits(data), dim=-1)

    class _FakeGraphBatch:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.to_calls: list[torch.device] = []

        def to(self, device, **kwargs):
            resolved = torch.device(device)
            self.to_calls.append(resolved)
            self.x = self.x.to(resolved)
            self.y = self.y.to(resolved)
            return self

    @staticmethod
    def _classification_batch():
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        return (
            LightningRuntimeTest._FakeGraphBatch(features.clone(), labels.clone()),
            LightningRuntimeTest._FakeGraphBatch(features.clone(), labels.clone()),
        )

    def test_runtime_device_manager_recursively_moves_graph_batches(self):
        manager = DeviceManager.from_value("cpu")
        batch = self._classification_batch()

        moved = manager.move({"pair": batch, "extra": [torch.ones(1)]})

        self.assertEqual(moved["pair"][0].x.device, manager.device)
        self.assertEqual(moved["pair"][1].y.device, manager.device)
        self.assertEqual(moved["extra"][0].device, manager.device)
        self.assertEqual(batch[0].to_calls, [manager.device])
        self.assertEqual(batch[1].to_calls, [manager.device])
        self.assertEqual(resolve_device("auto").type, "cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(move_to_device(torch.ones(1), "cpu").device, torch.device("cpu"))

    def test_traditional_training_and_inference_path_moves_batches_and_backpropagates(self):
        manager = DeviceManager.from_value("cpu")
        model = self._TinyClassifier()
        manager.move_module(model)
        batch = self._classification_batch()

        model_input, target = classification_batch_input(batch, manager)
        logits = model.logits(model_input)
        loss = torch.nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        self.assertEqual(batch[0].to_calls, [manager.device])
        self.assertEqual(batch[1].to_calls, [manager.device])
        self.assertIsNotNone(model.weight.grad)
        self.assertGreater(float(model.weight.grad.abs().sum()), 0.0)
        self.assertEqual(logits.device, manager.device)

        model.weight.grad = None
        result = evaluate_classification(model, [self._classification_batch()], manager, max_batches=1)
        self.assertEqual(result.metrics.count, 2)
        self.assertTrue(torch.equal(result.preds, torch.tensor([0, 1])))
        self.assertIsNone(model.weight.grad)

    def test_lightning_training_step_uses_lightning_device_and_backpropagates(self):
        module = RXNGraphormerLitModule(
            self._TinyClassifier(),
            self._classification_config(),
            task="classification",
        )
        batch = self._classification_batch()

        loss = module.training_step(batch, 0)
        loss.backward()

        self.assertEqual(batch[0].to_calls, [])
        self.assertEqual(batch[1].to_calls, [])
        self.assertIsNotNone(module.model.weight.grad)
        self.assertGreater(float(module.model.weight.grad.abs().sum()), 0.0)
        self.assertEqual(loss.device, module.device)

    def test_compile_forward_does_not_register_extra_state_dict_keys(self):
        config = type(
            "Config",
            (),
            {
                "training": type("Training", (), {"loss": "l1"})(),
                "runtime": type("Runtime", (), {"compile_model": True, "compile_mode": "default"})(),
            },
        )()
        model = torch.nn.Linear(2, 1)
        module = RXNGraphormerLitModule(model, config)
        compiled = mock.Mock(side_effect=lambda x: model(x) + 1.0)

        with mock.patch("rxngraphormer.lightning.module.torch.compile", return_value=compiled) as compile_mock:
            module.setup("fit")
            out = module(torch.ones(1, 2))

        compile_mock.assert_called_once()
        self.assertTrue(torch.allclose(out, model(torch.ones(1, 2)) + 1.0))
        self.assertEqual(sorted(module.state_dict().keys()), ["model.bias", "model.weight"])

    def test_regression_loss_supports_smooth_l1_and_huber_alias(self):
        smooth_config = type("Config", (), {"training": type("Training", (), {"loss": "smooth_l1", "huber_beta": 0.05})()})()
        huber_config = type("Config", (), {"training": type("Training", (), {"loss": "huber", "huber_beta": 0.2})()})()

        smooth_loss = build_regression_loss(smooth_config)
        huber_loss = build_regression_loss(huber_config)

        self.assertIsInstance(smooth_loss, torch.nn.SmoothL1Loss)
        self.assertIsInstance(huber_loss, torch.nn.SmoothL1Loss)
        self.assertAlmostEqual(smooth_loss.beta, 0.05)
        self.assertAlmostEqual(huber_loss.beta, 0.2)

    def test_regression_loss_rejects_nonpositive_huber_beta(self):
        config = type("Config", (), {"training": type("Training", (), {"loss": "smooth_l1", "huber_beta": 0.0})()})()

        with self.assertRaisesRegex(ValueError, "huber_beta"):
            build_regression_loss(config)

    def test_train_config_loads_smooth_l1_loss(self):
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                """
                {
                  "training": {
                    "loss": "smooth_l1",
                    "huber_beta": 0.05
                  }
                }
                """,
                encoding="utf-8",
            )

            config = load_train_config(path)

        self.assertEqual(config.training.loss, "smooth_l1")
        self.assertAlmostEqual(config.training.huber_beta, 0.05)

    def test_classification_metrics_reports_accuracy_loss_and_confidence(self):
        result = classification_metrics(
            torch.tensor([[3.0, 0.0], [0.5, 1.5]]),
            torch.tensor([0, 1]),
        )

        self.assertEqual(result.metrics.count, 2)
        self.assertAlmostEqual(result.metrics.accuracy, 1.0)
        self.assertTrue(torch.equal(result.preds, torch.tensor([0, 1])))
        self.assertGreater(result.metrics.mean_confidence, 0.5)

    def test_lit_module_classification_step_uses_logits_and_class_targets(self):
        config = type(
            "Config",
            (),
            {
                "task": "classification",
                "training": type("Training", (), {"loss": "ce"})(),
                "runtime": type("Runtime", (), {"compile_model": False})(),
                "optimizer": type("Optimizer", (), {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0})(),
                "scheduler": type("Scheduler", (), {"type": "none"})(),
                "model": type("Model", (), {"pretrained_model_path": ""})(),
            },
        )()

        class FakeClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[2.0, -2.0], [-2.0, 2.0]]))

            def logits(self, data):
                return data[0].x @ self.weight

            def forward(self, data):
                return torch.softmax(self.logits(data), dim=-1)

        class FakeBatch:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def to(self, device):
                self.x = self.x.to(device)
                self.y = self.y.to(device)
                return self

        module = RXNGraphormerLitModule(FakeClassifier(), config, task="classification")
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        batch = (FakeBatch(features, labels), FakeBatch(features, labels))

        loss = module.training_step(batch, 0)
        prediction = module.predict_step(batch, 0)

        self.assertLess(loss.item(), 0.1)
        self.assertTrue(torch.equal(prediction["prediction"].detach().cpu(), labels))

    def test_lit_module_classification_compile_uses_logits_method(self):
        config = type(
            "Config",
            (),
            {
                "task": "classification",
                "training": type("Training", (), {"loss": "ce"})(),
                "runtime": type("Runtime", (), {"compile_model": True, "compile_mode": "default"})(),
            },
        )()

        class FakeClassifier(torch.nn.Module):
            def logits(self, data):
                return data + 2.0

            def forward(self, data):
                return torch.softmax(self.logits(data), dim=-1)

        module = RXNGraphormerLitModule(FakeClassifier(), config, task="classification")
        compiled_logits = mock.Mock(side_effect=lambda x: x + 3.0)

        with mock.patch("rxngraphormer.lightning.module.torch.compile", return_value=compiled_logits) as compile_mock:
            module.setup("fit")
            out = module._forward_logits(torch.ones(1, 2))

        compile_mock.assert_called_once()
        compiled_logits.assert_called_once()
        self.assertTrue(torch.equal(out, torch.full((1, 2), 4.0)))
        self.assertEqual(module._compiled_forward, None)

    def test_classification_checkpoint_callback_monitors_val_accuracy(self):
        config = type(
            "Config",
            (),
            {
                "task": "classification",
                "runtime": type("Runtime", (), {"early_stopping_patience": 0, "log_epoch_time": False, "log_gpu_memory": False})(),
            },
        )()

        callbacks = build_callbacks(config)

        self.assertEqual(callbacks[0].monitor, "val_acc")
        self.assertEqual(callbacks[0].mode, "max")

    def test_build_trainer_prefers_fixed_optimizer_steps(self):
        config = type(
            "Config",
            (),
            {
                "task": "regression",
                "others": type("Others", (), {"tag": "fixed_steps"})(),
                "training": type(
                    "Training",
                    (),
                    {
                        "epoch": 99,
                        "max_steps": None,
                        "accum": 2,
                        "clip_norm": 1.0,
                        "log_iter_step": 5,
                    },
                )(),
                "runtime": type(
                    "Runtime",
                    (),
                    {
                        "enable_amp": False,
                        "deterministic": False,
                        "benchmark": False,
                        "val_check_interval": 20,
                        "check_val_every_n_epoch": 1,
                        "num_sanity_val_steps": 0,
                        "enable_progress_bar": False,
                        "enable_model_summary": False,
                        "early_stopping_patience": 0,
                        "log_epoch_time": False,
                        "log_gpu_memory": False,
                    },
                )(),
            },
        )()
        logger = mock.Mock(name="logger")

        with (
            mock.patch("rxngraphormer.lightning.trainer.TensorBoardLogger", return_value=logger),
            mock.patch("rxngraphormer.lightning.trainer.pl.Trainer", return_value=mock.Mock(name="trainer")) as trainer_cls,
        ):
            build_trainer(
                config,
                default_root_dir="runs/fixed_steps",
                accelerator="cpu",
                devices="1",
                precision="32-true",
                max_steps=100,
            )

        trainer_kwargs = trainer_cls.call_args.kwargs
        self.assertEqual(trainer_kwargs["max_epochs"], -1)
        self.assertEqual(trainer_kwargs["max_steps"], 100)
        self.assertEqual(trainer_kwargs["accumulate_grad_batches"], 2)
        self.assertEqual(trainer_kwargs["val_check_interval"], 20)
        self.assertIsNone(trainer_kwargs["check_val_every_n_epoch"])

    def test_epoch_runtime_monitor_logs_wall_time(self):
        callback = EpochRuntimeMonitor(log_epoch_time=True, log_gpu_memory=False)
        module = mock.Mock()

        callback.on_train_epoch_start(trainer=None, pl_module=module)
        callback.on_train_epoch_end(trainer=None, pl_module=module)

        logged = module.log_dict.call_args.args[0]
        self.assertIn("epoch_seconds", logged)
        self.assertGreaterEqual(logged["epoch_seconds"], 0.0)

    def test_finetune_optimizer_uses_scaled_pretrained_encoder_lr(self):
        config = type(
            "Config",
            (),
            {
                "training": type("Training", (), {"loss": "l1"})(),
                "runtime": type("Runtime", (), {"compile_model": False})(),
                "optimizer": type(
                    "Optimizer",
                    (),
                    {"optimizer": "AdamW", "learning_rate": 0.4, "weight_decay": 0.0},
                )(),
                "scheduler": type("Scheduler", (), {"type": "none"})(),
                "model": type(
                    "Model",
                    (),
                    {
                        "pretrained_model_path": "model_path/pretrained_classification_model",
                        "pretrained_model_freeze": False,
                        "pretrained_lr_scaled_coef": 0.1,
                        "use_mid_inf": True,
                    },
                )(),
            },
        )()

        class FakeFineTuneModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rct_encoder = torch.nn.Linear(2, 2)
                self.pdt_encoder = torch.nn.Linear(2, 2)
                self.decoder = torch.nn.Linear(2, 1)
                self.mid_encoder = torch.nn.Linear(2, 2)
                self.mid_iteract = torch.nn.Linear(2, 2)
                self.mid_decoder = torch.nn.Linear(2, 1)

        module = RXNGraphormerLitModule(FakeFineTuneModel(), config)
        optimizer = module.configure_optimizers()

        self.assertTrue(np.allclose([group["lr"] for group in optimizer.param_groups], [0.04, 0.04, 0.4, 0.4, 0.4, 0.4]))

    def test_optimization_plan_scales_learning_rate_from_effective_batch_size(self):
        config = type(
            "Config",
            (),
            {
                "optimizer": type(
                    "Optimizer",
                    (),
                    {
                        "learning_rate": 0.1,
                        "base_learning_rate": 0.1,
                        "base_batch_size": 32,
                        "lr_scaling_policy": "linear",
                    },
                )(),
                "scheduler": type("Scheduler", (), {"warmup_step": 100, "base_warmup_step": 100, "scale_warmup_steps": True})(),
                "data": type("Data", (), {"batch_size": 128})(),
                "training": type("Training", (), {"accum": 2})(),
            },
        )()

        plan = resolve_optimization_plan(config, world_size=2)

        self.assertEqual(plan.effective_batch_size, 512)
        self.assertAlmostEqual(plan.learning_rate, 1.6)
        self.assertEqual(plan.warmup_steps, 6)

    def test_optimization_plan_can_align_noam_progress_to_base_batch_samples(self):
        config = type(
            "Config",
            (),
            {
                "optimizer": type(
                    "Optimizer",
                    (),
                    {
                        "learning_rate": 0.4,
                        "base_learning_rate": 0.4,
                        "base_batch_size": 32,
                        "lr_scaling_policy": "none",
                    },
                )(),
                "scheduler": type(
                    "Scheduler",
                    (),
                    {
                        "warmup_step": 5000,
                        "base_warmup_step": 5000,
                        "scale_warmup_steps": False,
                        "step_scale_policy": "sample",
                    },
                )(),
                "data": type("Data", (), {"batch_size": 1024})(),
                "training": type("Training", (), {"accum": 1})(),
            },
        )()

        plan = resolve_optimization_plan(config)

        self.assertEqual(plan.batch_ratio, 32.0)
        self.assertEqual(plan.learning_rate, 0.4)
        self.assertEqual(plan.warmup_steps, 5000)
        self.assertEqual(plan.scheduler_step_scale_policy, "sample")
        self.assertEqual(plan.scheduler_step_scale, 32.0)

    def test_noamlr_step_scale_matches_original_schedule_at_same_sample_progress(self):
        base_param = torch.nn.Parameter(torch.tensor(1.0))
        scaled_param = torch.nn.Parameter(torch.tensor(1.0))
        base_optimizer = torch.optim.AdamW([base_param], lr=0.4)
        scaled_optimizer = torch.optim.AdamW([scaled_param], lr=0.4)
        base_scheduler = NoamLR(base_optimizer, model_size=256, warmup_steps=5000)
        scaled_scheduler = NoamLR(scaled_optimizer, model_size=256, warmup_steps=5000, step_scale=32.0)

        for _ in range(32):
            base_optimizer.step()
            base_scheduler.step()
        scaled_optimizer.step()
        scaled_scheduler.step()

        self.assertAlmostEqual(base_scheduler.get_last_lr()[0], scaled_scheduler.get_last_lr()[0])

    def test_regression_metrics_accepts_bfloat16_predictions(self):
        result = regression_metrics(
            torch.tensor([[1.0], [2.0]], dtype=torch.bfloat16),
            torch.tensor([[1.5], [1.5]], dtype=torch.bfloat16),
        )

        self.assertEqual(result.metrics.count, 2)
        self.assertEqual(result.preds.dtype, torch.float32)
