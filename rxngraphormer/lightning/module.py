from __future__ import annotations

from typing import Any

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    pl = None

from ..evaluator import regression_batch_input, regression_metrics
from ..model_factory import build_regression_model_from_config
from ..scheduler import NoamLR, get_linear_scheduler_with_warmup
from ..utils import as_bool


class RXNGraphormerLitModule(pl.LightningModule if pl is not None else torch.nn.Module):
    """Lightning training shell around the legacy torch model implementation."""

    def __init__(self, model: torch.nn.Module, config: Any, *, task: str = "regression") -> None:
        if pl is None:
            raise ImportError("lightning is required to use RXNGraphormerLitModule")
        super().__init__()
        self.model = model
        self.config = config
        self.task = task
        if task != "regression":
            raise NotImplementedError("The first Lightning shell supports regression models")
        self.loss_func = torch.nn.L1Loss() if config.training.loss.lower() in {"l1", "mae"} else torch.nn.MSELoss()
        self._compiled_forward: list[Any] | None = None
        self.validation_step_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.test_step_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.save_hyperparameters(ignore=["model"])

    @classmethod
    def from_config(cls, config: Any) -> "RXNGraphormerLitModule":
        model = build_regression_model_from_config(config)
        return cls(model, config, task="regression")

    def forward(self, batch_or_input):
        return self._forward_model(batch_or_input)

    def training_step(self, batch, batch_idx: int):
        model_input, target = regression_batch_input(batch, self.device)
        pred = self._forward_model(model_input)
        loss = self.loss_func(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=target.shape[0])
        return loss

    def validation_step(self, batch, batch_idx: int):
        model_input, target = regression_batch_input(batch, self.device)
        pred = self._forward_model(model_input)
        loss = self.loss_func(pred, target)
        self.validation_step_outputs.append((pred.detach().cpu(), target.detach().cpu()))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=target.shape[0])
        return loss

    def test_step(self, batch, batch_idx: int):
        model_input, target = regression_batch_input(batch, self.device)
        pred = self._forward_model(model_input)
        self.test_step_outputs.append((pred.detach().cpu(), target.detach().cpu()))
        return pred

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        model_input, _target = regression_batch_input(batch, self.device)
        return self._forward_model(model_input)

    def setup(self, stage: str | None = None) -> None:
        runtime = getattr(self.config, "runtime", None)
        if not as_bool(getattr(runtime, "compile_model", False)) or self._compiled_forward is not None:
            return
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch version")
        compile_mode = getattr(runtime, "compile_mode", "default")
        mode = None if compile_mode == "default" else compile_mode
        self._compiled_forward = [torch.compile(self.model, mode=mode)]

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_regression_metrics("val", self.validation_step_outputs)
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self._log_epoch_regression_metrics("test", self.test_step_outputs)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer_name = self.config.optimizer.optimizer.lower()
        optimizer_cls = AdamW if optimizer_name == "adamw" else Adam
        parameters = self._optimizer_parameters()
        optimizer = optimizer_cls(
            parameters,
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
        scheduler_type = self.config.scheduler.type.lower()
        if scheduler_type == "steplr":
            scheduler = StepLR(
                optimizer,
                step_size=self.config.scheduler.lr_decay_step_size,
                gamma=self.config.scheduler.lr_decay_factor,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        if scheduler_type == "warmup":
            scheduler = get_linear_scheduler_with_warmup(
                optimizer,
                num_warmup_steps=self.config.scheduler.warmup_step,
                num_training_steps=self.config.training.epoch,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        if scheduler_type == "noamlr":
            scheduler = NoamLR(optimizer, model_size=self.config.model.emb_dim, warmup_steps=self.config.scheduler.warmup_step)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer

    def _optimizer_parameters(self):
        pretrained_path = getattr(self.config.model, "pretrained_model_path", "")
        pretrained_freeze = as_bool(getattr(self.config.model, "pretrained_model_freeze", False))
        if not pretrained_path or pretrained_freeze:
            return filter(lambda param: param.requires_grad, self.model.parameters())
        scaled_lr = self.config.optimizer.learning_rate * float(getattr(self.config.model, "pretrained_lr_scaled_coef", 1.0))
        groups = [
            {
                "params": filter(lambda param: param.requires_grad, self.model.rct_encoder.parameters()),
                "lr": scaled_lr,
            },
            {
                "params": filter(lambda param: param.requires_grad, self.model.pdt_encoder.parameters()),
                "lr": scaled_lr,
            },
            {
                "params": filter(lambda param: param.requires_grad, self.model.decoder.parameters()),
                "lr": self.config.optimizer.learning_rate,
            },
        ]
        if as_bool(getattr(self.config.model, "use_mid_inf", False)):
            groups.extend(
                [
                    {
                        "params": filter(lambda param: param.requires_grad, self.model.mid_encoder.parameters()),
                        "lr": self.config.optimizer.learning_rate,
                    },
                    {
                        "params": filter(lambda param: param.requires_grad, self.model.mid_iteract.parameters()),
                        "lr": self.config.optimizer.learning_rate,
                    },
                    {
                        "params": filter(lambda param: param.requires_grad, self.model.mid_decoder.parameters()),
                        "lr": self.config.optimizer.learning_rate,
                    },
                ]
            )
        return groups

    def _log_epoch_regression_metrics(self, prefix: str, chunks: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        if not chunks:
            return
        preds = torch.cat([item[0] for item in chunks], dim=0)
        targets = torch.cat([item[1] for item in chunks], dim=0)
        result = regression_metrics(preds, targets)
        self.log(f"{prefix}_mae", result.metrics.mae, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}_rmse", result.metrics.rmse, prog_bar=False, sync_dist=True)
        self.log(f"{prefix}_r2", result.metrics.r2, prog_bar=False, sync_dist=True)

    def _forward_model(self, model_input):
        if self._compiled_forward is not None:
            return self._compiled_forward[0](model_input)
        return self.model(model_input)
