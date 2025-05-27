from typing import Any, Optional, Union, Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities import rank_zero_only
from overrides import overrides
import torch
from torch import Tensor
import torch.nn.functional as F
from absl import logging
import time
import os
from torch.optim import Optimizer
from copy import deepcopy


class Queue:
    def __init__(self, max_len=50):
        self.items = [1]
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


class GradientClip(Callback):
    def __init__(self, max_grad_norm='Q', Q=Queue(3000)) -> None:
        super().__init__()
        # self.max_norm = max_norm
        self.gradnorm_queue = Q
        if max_grad_norm == 'Q':
            self.max_grad_norm = max_grad_norm
        else:
            self.max_grad_norm = float(max_grad_norm)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        # zero graidents if they are not finite
        # if not all([torch.isfinite(t.grad).all() for t in pl_module.parameters()]):
        #     logging.warning("Gradients are not finite number")
        #     pl_module.zero_grad()
        #     return None
        if self.max_grad_norm == 'Q':
            max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
            max_grad_norm = max_grad_norm.item()
        else:
            max_grad_norm = self.max_grad_norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), max_norm=max_grad_norm, norm_type=2.0
        )

        if self.max_grad_norm == 'Q':
            if float(grad_norm) > max_grad_norm:
                self.gradnorm_queue.add(float(max_grad_norm))
            else:
                self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            logging.info(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}",
            )
        pl_module.log_dict(
            {
                "grad_norm": grad_norm.item(),
                'max_grad_norm': max_grad_norm,
            },
            on_step=True,
            prog_bar=False,
            logger=True,
            batch_size=pl_module.cfg.train.batch_size,
        )


class DebugCallback(Callback):
    # gradient clupping for
    def __init__(self) -> None:
        super().__init__()
        # self.max_norm = max_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if not all([torch.isfinite(t.grad).all() for t in pl_module.parameters()]):
            for t in pl_module.parameters():
                if not torch.isfinite(t.grad).all():
                    print(t.name, t.grad)
            raise ValueError("gradient is not finite number")

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        self._start_time = time.time()

    def on_before_backward(
        self, trainer: Trainer, pl_module: LightningModule, loss: Tensor
    ) -> None:
        super().on_before_backward(trainer, pl_module, loss)
        _cur_time = time.time()
        logging.info(
            f"from trainbatch start to before backward took {_cur_time - self._start_time} secs"
        )

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_after_backward(trainer, pl_module)
        _cur_time = time.time()
        logging.info(
            f"from trainbatch start to after backward took {_cur_time - self._start_time} secs"
        )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        super().on_before_optimizer_step(trainer, pl_module, optimizer)
        _cur_time = time.time()
        logging.info(
            f"from trainbatch start to before optimizer step took {_cur_time - self._start_time} secs"
        )

    def on_before_zero_grad(
        self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer
    ) -> None:
        super().on_before_zero_grad(trainer, pl_module, optimizer)
        _cur_time = time.time()
        logging.info(
            f"from trainbatch start to before zero grad took {_cur_time - self._start_time} secs"
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        _cur_time = time.time()
        logging.info(f"train batch took {_cur_time - self._start_time} secs")


class NormalizerCallback(Callback):
    # for data inputs we need to normalize the data, before the data outputs we
    def __init__(self, normalizer_dict) -> None:
        super().__init__()
        self.normalizer_dict = normalizer_dict
        self.pos_normalizer = torch.tensor(self.normalizer_dict.pos, dtype=torch.float32)
        self.device = None

    def quantize(self, pos, h):
        # quantize the latent space
        h = F.one_hot(torch.argmax(h, dim=-1), num_classes=h.shape[-1])
        return pos, h

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        if self.device is None:
            self.device = batch.protein_pos.device
            self.pos_normalizer = self.pos_normalizer.to(self.device)
        batch.protein_pos = batch.protein_pos / self.pos_normalizer
        batch.ligand_pos = batch.ligand_pos / self.pos_normalizer
        # batch.x = batch.x / self.normalizer_dict.one_hot
        # #batch.charges = batch.charges / self.normalizer_dict.charges - 1
        # # print(batch.charges)
        # batch.charges = (2*batch.charges - 1)/self.normalizer_dict.charges - 1 #normalizer as k_c
        # print(batch.charges)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super().on_validation_batch_start(trainer, pl_module, batch, batch_idx)
        if self.device is None:
            self.device = batch.protein_pos.device
            self.pos_normalizer = self.pos_normalizer.to(self.device)
        batch.protein_pos = batch.protein_pos / self.pos_normalizer
        batch.ligand_pos = batch.ligand_pos / self.pos_normalizer

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super().on_test_batch_start(trainer, pl_module, batch, batch_idx)
        if self.device is None:
            self.device = batch.protein_pos.device
            self.pos_normalizer = self.pos_normalizer.to(self.device)
        batch.protein_pos = batch.protein_pos / self.pos_normalizer
        batch.ligand_pos = batch.ligand_pos / self.pos_normalizer
      

class RecoverCallback(Callback):
    def __init__(self, latest_ckpt, recover_trigger_loss=1e3, resume=False) -> None:
        super().__init__()
        self.latest_ckpt = latest_ckpt
        self.recover_trigger_loss = recover_trigger_loss
        self.resume = resume

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if os.path.exists(self.latest_ckpt) and self.resume:
            print(f"recover from checkpoint: {self.latest_ckpt}")
            checkpoint = torch.load(self.latest_ckpt)
            pl_module.load_state_dict(checkpoint["state_dict"])
            # pl_module.load_from_checkpoint(self.latest_ckpt)
        elif not os.path.exists(self.latest_ckpt) and self.resume:
            print(
                f"checkpoint {self.latest_ckpt} not found, training from scratch"
            )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if "loss" not in outputs:
            return None

        if outputs["loss"] > self.recover_trigger_loss:
            logging.warning(
                f"loss too large: {outputs}\n recovering from checkpoint: {self.latest_ckpt}"
            )
            if os.path.exists(self.latest_ckpt):
                checkpoint = torch.load(self.latest_ckpt)
                pl_module.load_state_dict(checkpoint["state_dict"])
            else:
                for layer in pl_module.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
                logging.warning(
                    f"checkpoint {self.latest_ckpt} not found, training from scratch"
                )

        else:
            pass


class EMACallback(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.
    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16
    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

    def __init__(
        self,
        decay: float = 0.9999,
        ema_device: Optional[Union[torch.device, str]] = None,
        pin_memory=True,
    ):
        super().__init__()
        self.decay = decay
        self.ema_device: str = (
            f"{ema_device}" if ema_device else None
        )  # perform ema on different device from the model
        self.ema_pin_memory = (
            pin_memory if torch.cuda.is_available() else False
        )  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.
        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @overrides
    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: pl.LightningModule
    ) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {
                    k: tensor.to(device=self.ema_device)
                    for k, tensor in self.ema_state_dict.items()
                }

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {
                    k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()
                }

        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs
    ) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(
                    self.decay * ema_value + (1.0 - self.decay) * value,
                    non_blocking=True,
                )

    @overrides
    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))

        trainer.strategy.broadcast(self.ema_state_dict, 0)

        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), (
            f"There are some keys missing in the ema static dictionary broadcasted. "
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        )
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    @overrides
    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_validation_start(trainer, pl_module)

    @overrides
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_validation_end(trainer, pl_module)

    @overrides
    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint["ema_state_dict"] = self.ema_state_dict
        checkpoint["_ema_state_dict_ready"] = self._ema_state_dict_ready
        # return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    @overrides
    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        if checkpoint is None:
            self._ema_state_dict_ready = False
        else:
            self._ema_state_dict_ready = checkpoint["_ema_state_dict_ready"]
            self.ema_state_dict = checkpoint["ema_state_dict"] 