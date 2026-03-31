import math
import shutil
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Optional
from abc import ABC
from tqdm import tqdm

import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin, DataLoaderConfiguration
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from dataclasses import dataclass, field
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule
from utils.utils import check_bloat16_supported
from omegaconf import OmegaConf

from model.overwatch import initialize_overwatch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from model.training.metrics import VLAMetrics

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# default deepspeed configs
@dataclass
class DefaultDeepSpeedConfig:
    bf16: dict = field(default_factory=lambda: {
        "enabled": True,
        # "loss_scale": 0,
        # "initial_scale_power": 16,
        # "loss_scale_window": 1000,
        # "hysteresis": 2,
        # "min_loss_scale": 1
    })
    fp16: dict = field(default_factory=lambda: {
        "enabled": False,
        # "loss_scale": 0,
        # "initial_scale_power": 16,
        # "loss_scale_window": 1000,
        # "hysteresis": 2,
        # "min_loss_scale": 1
    })
    zero_optimization: dict = field(default_factory=lambda: {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": True
    })
    train_batch_size: str = "auto"
    train_micro_batch_size_per_gpu: str = "auto"
    wall_clock_breakdown: bool = False

class AcceleratorStrategy_InternVL(ABC):
    def __init__(
        self,
        vla,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = False,
        enable_mixed_precision_training: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        model_cfg: str = "qwen",
        save_checkpoint: bool = True,
        use_token_loss: bool = True,
        use_latent_loss: bool = True,
        token_loss_weight: float = 1.0,
        val_freq: int = 1000,
    ) -> None:
        self.vla = vla  # QwenVLA
        self.model_cfg = model_cfg
        self.save = save_checkpoint
        self.use_token_loss = use_token_loss
        self.use_latent_loss = use_latent_loss
        self.token_loss_weight = token_loss_weight
        self.val_freq = val_freq
        
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.mixed_precision_dtype = mixed_precision_dtype
        self.worker_init_fn = worker_init_fn

        self.optimizer, self.lr_scheduler = None, None

        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
    ) -> None:
        if self.accelerator.is_main_process:
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if train_loss is None:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
            else:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

            torch.save(self.vla.cpu().state_dict(), checkpoint_path)
            # shutil.copy(checkpoint_path, checkpoint_dir / f"{self.model_cfg}_latest-checkpoint.pt")

    def run_setup(self, n_train_examples: int) -> None:
        self.deepspeed_config = OmegaConf.structured(DefaultDeepSpeedConfig)
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=OmegaConf.to_container(self.deepspeed_config, resolve=True),
            gradient_accumulation_steps=self.grad_accumulation_steps,
        )
        self.accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            gradient_accumulation_steps=self.grad_accumulation_steps,
            mixed_precision="bf16" if self.enable_mixed_precision_training else "no",
        )

        # === Optimizer & Scheduler ===
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        decay, no_decay = [], []
        for name, param in self.vla.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(groups, lr=self.learning_rate)

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
        elif self.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        elif self.lr_scheduler_type == "step-decay":
            self.lr_scheduler = StepLR(self.optimizer, step_size=20000, gamma=0.5)
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {self.lr_scheduler_type}")

    def clip_grad_norm(self) -> None:
        self.accelerator.clip_grad_norm_(self.vla.parameters(), self.max_grad_norm)

    def run_training(
        self,
        dataset: IterableDataset,
        collator,
        metrics,
        save_step: int = 2000,
        val_dataset: Optional[IterableDataset] = None,
    ) -> None:
        epoch = 0
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.per_device_batch_size,
                sampler=None,
                collate_fn=collator,
                num_workers=0,
                worker_init_fn=self.worker_init_fn,
            )
            val_dataloader = self.accelerator.prepare(val_dataloader)
        self.vla, self.optimizer, self.lr_scheduler, dataloader = self.accelerator.prepare(
            self.vla, self.optimizer, self.lr_scheduler, dataloader
        )

        status = metrics.get_status()
        steps_per_epoch = len(dataset) // self.global_batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = 1

        total_steps = (
            self.epochs * (steps_per_epoch // self.grad_accumulation_steps)
            if self.max_steps is None
            else self.max_steps
        )
        with tqdm(
            total=total_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vla.train()
            self.optimizer.zero_grad()
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ret_loss = self.vla(batch, training=True)
                if self.use_latent_loss and self.token_loss_weight > 0.0:
                    loss = ret_loss["action_loss"]+ret_loss["latent_action_loss"]+ret_loss["action_token_loss"] * self.token_loss_weight
                    metrics.commit(loss=loss, action_loss=ret_loss["action_loss"], latent_loss=ret_loss["latent_action_loss"], action_token_loss=ret_loss["action_token_loss"])
                elif self.use_latent_loss:
                    loss = ret_loss["action_loss"]+ret_loss["latent_action_loss"]
                    metrics.commit(loss=loss, action_loss=ret_loss["action_loss"],latent_loss=ret_loss["latent_action_loss"])
                elif self.token_loss_weight == 0.0:
                    loss = ret_loss["action_loss"]
                    metrics.commit(loss=loss, action_loss=ret_loss["action_loss"])
                else:
                    loss = ret_loss["action_loss"]+ret_loss["action_token_loss"] * self.token_loss_weight
                    metrics.commit(loss=loss, action_loss=ret_loss["action_loss"],action_token_loss=ret_loss["action_token_loss"])
                # print(ret_loss["action_loss"], ret_loss["action_token_loss"], loss)
                normalized_loss = loss / self.grad_accumulation_steps
                self.accelerator.backward(normalized_loss)

                if val_dataset is not None and (metrics.global_step + 1) % self.val_freq == 0:
                    self.vla.eval()
                    val_loss_total = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                ret_loss = self.vla(val_batch, training=False)
                            val_loss = ret_loss["action_loss"]
                            val_loss_total += val_loss.item()
                            val_batches += 1
                    avg_val_loss = val_loss_total / val_batches
                    metrics.commit(val_loss=avg_val_loss)
                    self.vla.train()
                    # print(avg_val_loss)

                if (batch_count + 1) % self.grad_accumulation_steps == 0:
                    metrics.commit(update_step_time=True)
                    self.clip_grad_norm()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0], epoch=epoch)
                    
                    gs = metrics.global_step
                    self.accelerator.wait_for_everyone()
                    if self.save and (gs % save_step == 0) and (gs < self.max_steps):
                        if self.accelerator.is_main_process:
                            unwrapped = self.accelerator.unwrap_model(self.vla)
                            # state_dict = self.accelerator.get_state_dict(unwrapped)
                            state_dict = {
                                k: v.detach().cpu().half()
                                for k, v in unwrapped.state_dict().items()
                            }
                            checkpoint_dir = metrics.run_dir / "checkpoints"
                            checkpoint_dir.mkdir(parents=True, exist_ok=True)
                            ckpt_path = checkpoint_dir / f"step-{gs:06d}-epoch-{epoch:02d}.pt"
                            self.accelerator.save(state_dict, ckpt_path)
                        self.accelerator.wait_for_everyone()
                    
                    status = metrics.push()
                    progress.update()
                    progress.set_description(status)

                    if self.max_steps is not None and metrics.global_step >= self.max_steps:
                        print("Training complete, reached max steps.", flush=True)
                        print(f"Before save: loss={loss.item():.4f}")
                        if self.save:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                        dist.barrier()
                        return

                previous_epoch = epoch
                epoch = (metrics.global_step + 1) // (len(dataset) // self.global_batch_size)
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or epoch > previous_epoch or metrics.global_step == 1 or (metrics.global_step + 1) % save_step == 0:
                    dist.barrier()
                    torch.cuda.empty_cache()
                    if terminate:
                        return
                

class AcceleratorStrategy_Qwen(ABC):
    def __init__(
        self,
        vla,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = False,
        enable_mixed_precision_training: bool = True,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        model_cfg: str = "qwen",
        save_checkpoint: bool = True,
    ) -> None:
        self.vla = vla  # QwenVLA
        self.model_cfg = model_cfg
        self.save = save_checkpoint
        
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.mixed_precision_dtype = mixed_precision_dtype
        self.worker_init_fn = worker_init_fn

        self.optimizer, self.lr_scheduler = None, None

        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
    ) -> None:
        if self.accelerator.is_main_process:
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if train_loss is None:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
            else:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

            torch.save(self.vla.cpu().state_dict(), checkpoint_path)

    def run_setup(self, n_train_examples: int) -> None:
        self.deepspeed_config = OmegaConf.structured(DefaultDeepSpeedConfig)
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=OmegaConf.to_container(self.deepspeed_config, resolve=True),
            gradient_accumulation_steps=self.grad_accumulation_steps,
        )
        self.accelerator = Accelerator(
            deepspeed_plugin=deepspeed_plugin,
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            gradient_accumulation_steps=self.grad_accumulation_steps,
            mixed_precision="bf16" if self.enable_mixed_precision_training else "no",
        )

        # === Optimizer & Scheduler ===
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        decay, no_decay = [], []
        for name, param in self.vla.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(groups, lr=self.learning_rate)

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
        elif self.lr_scheduler_type == "constant":
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        elif self.lr_scheduler_type == "step-decay":
            self.lr_scheduler = StepLR(self.optimizer, step_size=8000, gamma=0.5)
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {self.lr_scheduler_type}")

    def clip_grad_norm(self) -> None:
        self.accelerator.clip_grad_norm_(self.vla.parameters(), self.max_grad_norm)

    def run_training(
        self,
        dataset: IterableDataset,
        collator,
        metrics,
        save_step: int = 2000,
    ) -> None:
        epoch = 0
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )
        self.vla, self.optimizer, self.lr_scheduler, dataloader = self.accelerator.prepare(
            self.vla, self.optimizer, self.lr_scheduler, dataloader
        )

        status = metrics.get_status()
        steps_per_epoch = len(dataset) // self.global_batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = 1

        total_steps = (
            self.epochs * (steps_per_epoch // self.grad_accumulation_steps)
            if self.max_steps is None
            else self.max_steps
        )
        # print(total_steps, self.grad_accumulation_steps)
        with tqdm(
            total=total_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vla.train()
            self.optimizer.zero_grad()
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if self.model_cfg == "change_vla":
                    loss = self.vla(batch, training=True)
                    metrics.commit(loss=loss, action_loss=loss)
                else:
                    latent_loss, action_loss=self.vla(batch, training=True)
                    loss = latent_loss + action_loss
                    metrics.commit(loss=loss, latent_loss=latent_loss, action_loss=action_loss)

                normalized_loss = loss / self.grad_accumulation_steps
                self.accelerator.backward(normalized_loss)

                if (batch_count + 1) % self.grad_accumulation_steps == 0:
                    metrics.commit(update_step_time=True)
                    self.clip_grad_norm()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0], epoch=epoch)
                    status = metrics.push()
                    progress.update()
                    progress.set_description(status)

                    if self.max_steps is not None and metrics.global_step >= self.max_steps:
                        print("Training complete, reached max steps.", flush=True)
                        print(f"Before save: loss={loss.item():.4f}, latent={latent_loss.item():.4f}, action={action_loss.item():.4f}")
                        if self.save:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                        dist.barrier()
                        return

                previous_epoch = epoch
                epoch = (metrics.global_step + 1) // (len(dataset) // self.global_batch_size)
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or epoch > previous_epoch or metrics.global_step == 1 or (metrics.global_step + 1) % save_step == 0:
                    dist.barrier()
                    torch.cuda.empty_cache()
                    if terminate:
                        return
                print(f"Training step {batch_count}, epoch {epoch}, global step {metrics.global_step}", flush=True)
