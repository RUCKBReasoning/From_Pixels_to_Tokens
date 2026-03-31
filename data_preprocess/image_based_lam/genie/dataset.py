import math
from os import listdir, makedirs, path
from random import choices, randint
from typing import Any, Callable, Dict

import cv2 as cv
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data import get_worker_info
import torchvision.transforms as transforms
from dataclasses import dataclass

# from prismatic.util import set_global_seed
# from prismatic.util.data_utils import CollatorForLatentAction, CollatorForMultiViewVideo
# from prismatic.vla.datasets import RLDSDataset, EpisodicRLDSDataset, RLDSBatchTransformVideo
                                    


# def exists(var) -> bool:
#     return var is not None


# def default(var, val) -> Any:
#     return var if exists(var) else val


# def default_worker_init_fn(worker_id: int) -> None:
#     torch.manual_seed(torch.initial_seed() + worker_id)
#     worker_info = get_worker_info()

#     if exists(worker_info):
#         dataset = worker_info.dataset
#         glob_start = dataset._start
#         glob_end = dataset._end

#         per_worker = int((glob_end - glob_start) / worker_info.num_workers)
#         worker_id = worker_info.id

#         dataset._start = glob_start + worker_id * per_worker
#         dataset._end = min(dataset._start + per_worker, glob_end)


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
            self,
            *args,
            batch_size: int = 8,
            num_workers: int = 64,
            train_shuffle: bool = True,
            val_shuffle: bool = False,
            val_batch_size: int = None,
            worker_init_fn: Callable = None,
            collate_fn: Callable = None,
            train_sampler: Callable = None,
            test_sampler: Callable = None,
            val_sampler: Callable = None
    ) -> None:
        super(LightningDataset, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = 0    # For RLDS parallelism
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        # shuffle unspecified for iteratable datasets
        # self.train_shuffle = train_shuffle
        # self.val_shuffle = val_shuffle

        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            # shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.val_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            # shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.test_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.val_batch_size,
            # shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )



# from PIL import Image
# import random

# @dataclass
# class random_crop_resize():
#     def __init__(
#         self,
#         target_size=224
#     ):
#         self.target_size = target_size
#         self.to_tensor = transforms.ToTensor()
    
#     def __call__(self, image):
#         width, height = image.size

#         if width < height:
#             crop_size = width
#         else:
#             crop_size = height

#         left = random.randint(0, width - crop_size)
#         top = random.randint(0, height - crop_size)

#         image_cropped = image.crop((left, top, left + crop_size, top + crop_size))
#         image_resized = image_cropped.resize((self.target_size, self.target_size), Image.BILINEAR)
#         image_resized = self.to_tensor(image_resized)
        
#         return image_resized

from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from typing import Dict, List, Optional, Sequence
from PIL import Image

class RLDSBatchTransformLAM:
    """Minimal RLDS transform for LAM training.

    Converts the Libero start and goal images into a two-frame video tensor.
    """

    def __init__(self, resolution: int = 224) -> None:
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, rlds_batch: Dict) -> Dict:
        start = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        goal = Image.fromarray(rlds_batch["goal_image"]["image_primary"])

        video = torch.stack([self.image_transform(start), self.image_transform(goal)], dim=0)
        # instruction = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        instr = rlds_batch["task"]["language_instruction"]
        if isinstance(instr, (np.ndarray, list, tuple)):
            instr = random.choice(instr)
        if isinstance(instr, bytes):
            instr = instr.decode("utf-8")
        assert isinstance(instr, str), f"Unexpected type: {type(instr)}"
        lang = instr.lower()
        
        return {
            "videos": video,
            "task_instruction": instr,
            "dataset_name": rlds_batch["dataset_name"],
            "actions": actions,
        }


class CollatorForLAM:
    """Collate Libero RLDS samples for LAM training."""

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        videos = torch.stack([instance["videos"] for instance in instances])
        dataset_names = [instance["dataset_name"] for instance in instances]
        instructions = [instance["task_instruction"] for instance in instances]
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        return {
            "videos": videos,
            "dataset_names": dataset_names,
            "task_instruction": instructions,
            "actions": actions,
        }
import random
import numpy as np
import os
from latentvla.data_provider.datasets import RLDSDataset

def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return worker_init_function if get_worker_init_fn else None
def worker_init_function(worker_id: int) -> None:
    """
    Borrowed directly from PyTorch-Lightning; inspired by this issue comment in the PyTorch repo:
        > Ref: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    Intuition: You can think of the seed sequence spawn function as a "janky" torch.Generator() or jax.PRNGKey that
    you can run iterative splitting on to get new (predictable) randomness.

    :param worker_id: Identifier for the given worker [0, num_workers) for the Dataloader in question.
    """
    # Get current `rank` (if running distributed) and `process_seed`
    global_rank, process_seed = int(os.environ["LOCAL_RANK"]), torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k) produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)

class LightningLAMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root_dir: str,
        data_mix: List[str],
        per_device_batch_size: int = 16,
        resize_resolution: int = 256,
        goal_image_step: int = 1,
        shuffle_buffer_size: int = 100_000,
        num_workers: int = 0,
        image_aug: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.per_device_batch_size = per_device_batch_size
        self.resize_resolution = resize_resolution
        self.goal_image_step = goal_image_step
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_workers = num_workers
        self.image_aug = image_aug
        self.seed = seed

        self.worker_init_fn = set_global_seed(
            seed, get_worker_init_fn=True
        )

        self.save_hyperparameters()

    def setup(self, stage: str = None):
        transform = RLDSBatchTransformLAM(self.resize_resolution)

        if stage in (None, "fit"):
            self.train_dataset = RLDSDataset(
                Path(self.data_root_dir),
                self.data_mix,
                batch_transform=transform,
                resize_resolution=(self.resize_resolution, self.resize_resolution),
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=True,
                image_aug=self.image_aug,
                window_size=self.goal_image_step,
                goal_image_step=self.goal_image_step,
            )

            # Reuse the training dataset when a separate validation split is unavailable.
            self.val_dataset = RLDSDataset(
                Path(self.data_root_dir),
                self.data_mix,
                batch_transform=transform,
                resize_resolution=(self.resize_resolution, self.resize_resolution),
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=False,
                image_aug=False,
                window_size=self.goal_image_step,
                goal_image_step=self.goal_image_step,
            )

        if stage == "test":
            self.test_dataset = RLDSDataset(
                Path(self.data_root_dir),
                self.data_mix,
                batch_transform=transform,
                resize_resolution=(self.resize_resolution, self.resize_resolution),
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=True,
                image_aug=False,
                window_size=self.goal_image_step,
                goal_image_step=self.goal_image_step,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=False,  # RLDS handles shuffling internally.
            sampler=None,
            collate_fn=CollatorForLAM(),
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=CollatorForLAM(),
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            sampler=None,
            collate_fn=CollatorForLAM(),
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
        )


class LightningOpenX(LightningDataset):
    """
    This dataset samples video recorded using a random agent
    playing the gym environments defined in the Procgen Benchmark,
    see Cobbe et al. ICML (2020).
    """

    def __init__(
            self,
            data_root: str,
            data_mix: str,
            batch_size:int = 16,
            resolution: int = 256,
            num_frames: int = 16,
            episodic: bool = False,
            shuffle_buffer_size: int = 100_000,
            image_aug:bool = False,
            **kwargs
    ) -> None:
        super(LightningOpenX, self).__init__(**kwargs)

        self.data_root_dir = data_root
        self.data_mix = data_mix

        self.batch_size = batch_size
        self.resolution = (resolution, resolution)
        self.num_frames = num_frames

        self.episodic = episodic
        self.shuffle_buffer_size = shuffle_buffer_size
        self.image_aug = image_aug

        self.num_workers = 0    # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        self.worker_init_fn = set_global_seed(42, get_worker_init_fn=True)

        self.batch_transform = RLDSBatchTransformVideo(
            image_transform=transforms.ToTensor() 
        )
        self.collate_fn = CollatorForLatentAction()

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        cls = RLDSDataset if not self.episodic else EpisodicRLDSDataset
        if stage == "fit":
            self.train_dataset = cls(
                self.data_root_dir,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.resolution,
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=True,
                image_aug=self.image_aug,
                training_phase='lam',
            )
            self.val_dataset = cls(
                self.data_root_dir,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.resolution,
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=False,
                image_aug=False,
                training_phase='lam',
            )
        elif stage == "test":
            self.test_dataset = cls(
                self.data_root_dir,
                self.data_mix,
                self.batch_transform,
                resize_resolution=self.resolution,
                shuffle_buffer_size=self.shuffle_buffer_size,
                train=True,
                image_aug=False,
                training_phase='lam',
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

