"""
datasets.py

PyTorch Dataset Definitions for Prismatic models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Iterator, List

from torch.utils.data import IterableDataset

from latentvla.data_provider.data_utils import RLDSBatchTransform
from latentvla.data_provider.rlds import make_interleaved_dataset
from latentvla.data_provider.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from latentvla.data_provider.rlds.utils.data_utils import NormalizationType
from latentvla.models.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: List[str],
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
        window_size: int = 8,
        goal_image_step: int = 8,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        mixture_spec = []
        for mix_name in self.data_mix:
            if mix_name in OXE_NAMED_MIXTURES:
                mixture_spec.extend(OXE_NAMED_MIXTURES[mix_name])
            else:
                mixture_spec.append((mix_name, 1.0))

        
        if "aloha" in self.data_mix[0]:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        elif "jaka" in self.data_mix[0]:
            load_camera_views = ("primary", "wrist")
        else:
            load_camera_views = ("primary", "wrist")
        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=window_size-1,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            goal_image_step=goal_image_step, 
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        print("[DEBUG __getitem__]", idx, flush=True)
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
