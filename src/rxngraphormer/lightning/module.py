from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler, OptimizerLRSchedulerConfig
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR

from ..config_utils import as_bool
from ..evaluation import (
    classification_batch_input,
    classification_metrics,
    regression_batch_input,
    regression_metrics,
)
from ..models import build_classification_model, build_regression_model_from_config
from ..models.protocols import OptimizerGroupedModel, SupportsLogits
from ..training.losses import build_regression_loss
from ..training.optimization import OptimizationPlan, resolve_optimization_plan
from ..training.schedulers import NoamLR, get_linear_scheduler_with_warmup


class RXNGraphormerLitModule(pl.LightningModule):
    """Lightning training shell around the legacy torch model implementation."""

    def __init__(self, model: torch.nn.Module, config: Any, *, task: str = "regression") -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.task = task
        if task == "regression":
            self.loss_func = build_regression_loss(config)
        elif task == "classification":
            if config.training.loss.lower() != "ce":
                raise NotImplementedError(f"Loss function {config.training.loss} is not implemented for classification")
            self.loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
        else:
            raise NotImplementedError("RXNGraphormerLitModule supports regression and classification tasks")
        self._compiled_forward: Callable[[object], torch.Tensor] | None = None
        self._compiled_logits: Callable[[object], torch.Tensor] | None = None
        self.validation_step_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.test_step_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.optimization_plan: OptimizationPlan | None = None
        self.save_hyperparameters(ignore=["model"])

    @classmethod
    def from_config(cls, config: Any) -> RXNGraphormerLitModule:
        if config.task == "classification":
            model = build_classification_model(config, initialize=True)
            return cls(model, config, task="classification")
        if config.task == "regression":
            model = build_regression_model_from_config(config)
            return cls(model, config, task="regression")
        raise NotImplementedError("RXNGraphormerLitModule.from_config supports regression and classification configs")

    def forward(self, batch_or_input):
        return self._forward_model(batch_or_input)

    def training_step(self, batch, batch_idx: int):
        if self.task == "classification":
            model_input, target = classification_batch_input(batch, self.device, move=False)
            logits = self._forward_logits(model_input)
            loss = self.loss_func(logits, target)
            pred = logits.argmax(dim=1)
            accuracy = (pred == target).float().mean()
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
            self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
            return loss
        model_input, target = regression_batch_input(batch, self.device, move=False)
        pred = self._forward_model(model_input)
        loss = self.loss_func(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        return loss

    def validation_step(self, batch, batch_idx: int):
        if self.task == "classification":
            model_input, target = classification_batch_input(batch, self.device, move=False)
            logits = self._forward_logits(model_input)
            loss = self.loss_func(logits, target)
            pred = logits.argmax(dim=1)
            accuracy = (pred == target).float().mean()
            self.validation_step_outputs.append((logits.detach().cpu(), target.detach().cpu()))
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
            self.log("val_acc_step", accuracy, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
            return loss
        model_input, target = regression_batch_input(batch, self.device, move=False)
        pred = self._forward_model(model_input)
        loss = self.loss_func(pred, target)
        self.validation_step_outputs.append((pred.detach().cpu(), target.detach().cpu()))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
        return loss

    def test_step(self, batch, batch_idx: int):
        if self.task == "classification":
            model_input, target = classification_batch_input(batch, self.device, move=False)
            logits = self._forward_logits(model_input)
            self.test_step_outputs.append((logits.detach().cpu(), target.detach().cpu()))
            return logits
        model_input, target = regression_batch_input(batch, self.device, move=False)
        pred = self._forward_model(model_input)
        self.test_step_outputs.append((pred.detach().cpu(), target.detach().cpu()))
        return pred

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if self.task == "classification":
            model_input, _target = classification_batch_input(batch, self.device, move=False)
            logits = self._forward_logits(model_input)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            return {
                "logits": logits,
                "probabilities": probabilities,
                "prediction": prediction,
                "confidence": confidence,
            }
        model_input, _target = regression_batch_input(batch, self.device, move=False)
        return self._forward_model(model_input)

    def setup(self, stage: str | None = None) -> None:
        runtime = getattr(self.config, "runtime", None)
        if not as_bool(getattr(runtime, "compile_model", False)):
            return
        if self.task == "classification" and self._compiled_logits is not None:
            return
        if self.task != "classification" and self._compiled_forward is not None:
            return
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch version")
        compile_mode = getattr(runtime, "compile_mode", "default")
        mode = None if compile_mode == "default" else compile_mode
        compile_fn = cast(Callable[..., object], torch.compile)
        if self.task == "classification" and isinstance(self.model, SupportsLogits):
            self._compiled_logits = cast(Callable[[object], torch.Tensor], compile_fn(self.model.logits, mode=mode))
        else:
            self._compiled_forward = cast(Callable[[object], torch.Tensor], compile_fn(self.model, mode=mode))

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val", self.validation_step_outputs)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test", self.test_step_outputs)
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_name = self.config.optimizer.optimizer.lower()
        optimizer_cls = AdamW if optimizer_name == "adamw" else Adam
        self.optimization_plan = resolve_optimization_plan(self.config, world_size=self._trainer_world_size())
        parameters = self._optimizer_parameters()
        optimizer = optimizer_cls(
            parameters,
            lr=self.optimization_plan.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
        scheduler_type = self.config.scheduler.type.lower()
        if scheduler_type == "steplr":
            scheduler = StepLR(
                optimizer,
                step_size=self.config.scheduler.lr_decay_step_size,
                gamma=self.config.scheduler.lr_decay_factor,
            )
            return OptimizerLRSchedulerConfig(optimizer=optimizer, lr_scheduler=scheduler)
        if scheduler_type == "warmup":
            scheduler = get_linear_scheduler_with_warmup(
                optimizer,
                num_warmup_steps=self.optimization_plan.warmup_steps,
                num_training_steps=self._estimated_stepping_batches(),
            )
            return OptimizerLRSchedulerConfig(
                optimizer=optimizer,
                lr_scheduler={"scheduler": scheduler, "interval": "step"},
            )
        if scheduler_type == "noamlr":
            scheduler = NoamLR(
                optimizer,
                model_size=self.config.model.emb_dim,
                warmup_steps=self.optimization_plan.warmup_steps,
                step_scale=self.optimization_plan.scheduler_step_scale,
            )
            return OptimizerLRSchedulerConfig(
                optimizer=optimizer,
                lr_scheduler={"scheduler": scheduler, "interval": "step"},
            )
        return optimizer

    def _optimizer_parameters(self) -> Iterable[torch.nn.Parameter] | list[dict[str, object]]:
        plan = self.optimization_plan or resolve_optimization_plan(self.config, world_size=self._trainer_world_size())
        pretrained_path = getattr(self.config.model, "pretrained_model_path", "")
        pretrained_freeze = as_bool(getattr(self.config.model, "pretrained_model_freeze", False))
        if not pretrained_path or pretrained_freeze:
            return filter(lambda param: param.requires_grad, self.model.parameters())
        model = cast(OptimizerGroupedModel, self.model)
        learning_rate = plan.learning_rate
        pretrained_lr = learning_rate * float(getattr(self.config.model, "pretrained_lr_scaled_coef", 1.0))
        groups = [
            {
                "params": filter(lambda param: param.requires_grad, model.rct_encoder.parameters()),
                "lr": pretrained_lr,
            },
            {
                "params": filter(lambda param: param.requires_grad, model.pdt_encoder.parameters()),
                "lr": pretrained_lr,
            },
            {
                "params": filter(lambda param: param.requires_grad, model.decoder.parameters()),
                "lr": learning_rate,
            },
        ]
        if as_bool(getattr(self.config.model, "use_mid_inf", False)):
            groups.extend(
                [
                    {
                        "params": filter(lambda param: param.requires_grad, model.mid_encoder.parameters()),
                        "lr": learning_rate,
                    },
                    {
                        "params": filter(lambda param: param.requires_grad, model.mid_iteract.parameters()),
                        "lr": learning_rate,
                    },
                    {
                        "params": filter(lambda param: param.requires_grad, model.mid_decoder.parameters()),
                        "lr": learning_rate,
                    },
                ]
            )
        return groups

    def on_train_start(self) -> None:
        if self.optimization_plan is None:
            return
        self.log("optimization/effective_batch_size", float(self.optimization_plan.effective_batch_size), prog_bar=False)
        self.log("optimization/learning_rate", self.optimization_plan.learning_rate, prog_bar=False)
        self.log("optimization/warmup_steps", float(self.optimization_plan.warmup_steps), prog_bar=False)
        self.log("optimization/scheduler_step_scale", self.optimization_plan.scheduler_step_scale, prog_bar=False)
        logger = self.logger
        experiment = getattr(logger, "experiment", None) if logger is not None else None
        add_scalar = getattr(experiment, "add_scalar", None)
        if callable(add_scalar):
            for key, value in self.optimization_plan.to_dict().items():
                if isinstance(value, (int, float, bool)):
                    add_scalar(f"optimization/{key}", float(value), self.global_step)

    def _log_epoch_metrics(self, prefix: str, chunks: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        if not chunks:
            return
        outputs = torch.cat([item[0] for item in chunks], dim=0)
        targets = torch.cat([item[1] for item in chunks], dim=0)
        if self.task == "classification":
            classification_result = classification_metrics(outputs, targets)
            self.log(f"{prefix}_loss", classification_result.metrics.loss, prog_bar=False, sync_dist=True)
            self.log(f"{prefix}_acc", classification_result.metrics.accuracy, prog_bar=True, sync_dist=True)
            self.log(f"{prefix}_confidence", classification_result.metrics.mean_confidence, prog_bar=False, sync_dist=True)
            return
        regression_result = regression_metrics(outputs, targets)
        self.log(f"{prefix}_mae", regression_result.metrics.mae, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}_rmse", regression_result.metrics.rmse, prog_bar=False, sync_dist=True)
        self.log(f"{prefix}_r2", regression_result.metrics.r2, prog_bar=False, sync_dist=True)

    def _forward_model(self, model_input):
        if self._compiled_forward is not None:
            return self._compiled_forward(model_input)
        return cast(torch.Tensor, self.model(model_input))

    def _forward_logits(self, model_input):
        if self._compiled_logits is not None:
            return self._compiled_logits(model_input)
        if isinstance(self.model, SupportsLogits):
            return self.model.logits(model_input)
        model = self._compiled_forward if self._compiled_forward is not None else self.model
        probabilities = cast(torch.Tensor, model(model_input))
        return probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()

    def _trainer_world_size(self) -> int:
        try:
            trainer = self.trainer
        except RuntimeError:
            return 1
        return max(1, int(getattr(trainer, "world_size", 1) or 1))

    def _estimated_stepping_batches(self) -> int:
        try:
            trainer = self.trainer
        except RuntimeError:
            return max(1, int(getattr(self.config, "training").epoch))
        steps = getattr(trainer, "estimated_stepping_batches", None)
        if isinstance(steps, int) and steps > 0:
            return steps
        if isinstance(steps, float) and steps > 0:
            return int(steps)
        return max(1, int(getattr(self.config, "training").epoch))
