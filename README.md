<div align="center">
  <h1>From Pixels to Tokens: A Systematic Study of Latent Action Supervision for Vision-Language-Action Models</h1>
  <p>
    <a href="#training-latent-action-models"><img alt="latent action models" src="https://img.shields.io/badge/Latent_Action-Image_+_Action-f28c28"></a>
    <a href="#training-latentvla"><img alt="vla training" src="https://img.shields.io/badge/VLA-Qwen3--VL--2B-0f766e"></a>
    <a href="#method"><img alt="method" src="https://img.shields.io/badge/Method-4_Strategies-7c3aed"></a>
  </p>
</div>

<p align="center">
  <img src="asserts/Figure_1.png" alt="LatentVLA overview" width="100%">
</p>

## Overview

This work studies how latent action supervision can be integrated into Vision-Language-Action (VLA) models under a unified training framework. We focuses on how different latent supervision strategies change downstream VLA policy learning.

Our implementation is built on a shared `Qwen3-VL-2B` backbone and compares four representative strategies:

| Model       | Latent supervision          | Role in VLA training                                      |
| ----------- | --------------------------- | --------------------------------------------------------- |
| `Baseline`  | None                        | Direct action prediction without latent supervision       |
| `LA-Align`  | Image-based latent actions  | Align internal VLM representations with latent embeddings |
| `LA-Direct` | Image-based latent actions  | Directly decode latent actions as discrete tokens         |
| `LA-Cond`   | Image-based latent actions  | Jointly decode latent actions and action representations  |
| `LA-Tok`    | Action-based latent actions | Map actions into discrete latent tokens                   |

This project follows two complementary perspectives from the paper:

- Image-based latent actions for trajectory regularization
- Action-based latent actions for target-space unification

## Method

<p align="center">
  <img src="asserts/Figure_2.png" alt="LatentVLA architecture" width="100%">
</p>

All methods share the same VLA backbone and action head, and differ only in how latent supervision is injected. The main VLA implementations live in [`latentvla/models/vla`](./latentvla/models/vla).

## Installation

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

The repository assumes RLDS-style datasets for both latent action preprocessing and VLA training.

## Training Latent Action Models

### A. Image-based latent action model

The image-based latent action model is in [`data_preprocess/image_based_lam`](./data_preprocess/image_based_lam). It follows a UniVLA-style image-based latent action pipeline and is the part used to produce latent supervision for `LA-Align`, `LA-Direct`, and `LA-Cond`.

Before post-training on your dataset, download the two public [UniVLA checkpoints](https://github.com/OpenDriveLab/UniVLA):

- Stage-1 checkpoint
- Stage-2 checkpoint

These checkpoints are used as initialization because it performs dataset-specific post-training rather than training the image-based latent model entirely from scratch.

#### Training

Edit [`data_preprocess/image_based_lam/config/lam-stage-2.yaml`](./data_preprocess/image_based_lam/config/lam-stage-2.yaml):

- `model.lam_path`
- `model.stage_one_ckpt`
- `data.data_root_dir`
- `data.data_mix`
- `trainer.devices`
- logging and checkpoint paths

Then run:

```bash
cd data_preprocess/image_based_lam
torchrun --standalone --nnodes 1 --nproc-per-node 1 main.py fit \
  --config config/lam-stage-2.yaml
```

### B. Annotate RLDS data with image-based latent labels

After training the image-based model, use [`data_preprocess/image_based_lam/latent.py`](./data_preprocess/image_based_lam/latent.py) to annotate trajectories with latent labels. The current script gives a `LIBERO`-style TFRecord example and writes:

- `steps/latent_idx`
- `steps/latent_z`

Before running it, edit the checkpoint path inside the script:

```python
lam_path = "your_lam_checkpoint.pth"
```

Then run:

```bash
cd data_preprocess/image_based_lam
python latent.py /path/to/file.tfrecord
```

The script writes a new TFRecord file into a sibling `output/` directory next to the input file.

### C. Action-based latent action model

The action-based latent action model is in [`data_preprocess/action_based_lam`](./data_preprocess/action_based_lam). This one is trained from scratch in this repository and is used by `LA-Tok`.

The tokenizer learns a VQ-style discrete latent space over action chunks and saves checkpoints of the form `tokenizer_step_*.pt`.

You can launch training with [`data_preprocess/action_based_lam/action.sh`](./data_preprocess/action_based_lam/action.sh) after editing:

- `--data-root-dir`
- `--data-mix`
- `--results-dir`
- `--num-steps`

Run:

```bash
cd data_preprocess/action_based_lam
bash action.sh
```

## Training LatentVLA

The main training entry is [`exp/train_vla.py`](./exp/train_vla.py).

Before VLA training, first download the `Qwen3-VL-2B` [checkpoint](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and set:

```bash
--vlm_path /path/to/Qwen3-VL-2B
```

Supported `--vla_id` values:

- `baseline`
- `la_align`
- `la_direct`
- `la_cond`
- `la_tok`

Before launching training, make sure you set:

- `--vlm_path` to the downloaded `Qwen3-VL-2B` checkpoint
- `--data_root_dir` to the RLDS dataset root
- `--data_mix` to the target dataset split or mixture
- `--action_tokenizer_ckpt` when training `la_tok`

Example baseline command:

```bash
torchrun --nnodes=1 --nproc_per_node=1 exp/train_vla.py \
  --seed 42 \
  --run_root_dir runs \
  --save_checkpoint True \
  --vla_id baseline \
  --vlm_path /path/to/Qwen3-VL-2B \
  --vlm_model_id Qwen3 \
  --default_image_size 224 \
  --data_root_dir /path/to/rlds_data \
  --data_mix '["libero_goal"]' \
  --shuffle_buffer_size 128 \
  --image_aug True \
  --window_size 8 \
  --use_wrist_image True \
  --use_proprio True \
  --type training \
  --epochs 10 \
  --max_steps 20000 \
  --global_batch_size 128 \
  --per_device_batch_size 32 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type constant \
  --warmup_ratio 0.03 \
  --save_step 20000 \
  --wandb_project latentvla \
  --use_wandb True
```

For the other variants, change `--vla_id`:

```bash
--vla_id la_align
--vla_id la_direct
--vla_id la_cond
--vla_id la_tok
```

For `la_tok`, also add:

```bash
--action_tokenizer_ckpt /path/to/tokenizer_step_xxxxx.pt
```

## Notes

- Robot-specific constants are selected in [`latentvla/models/constants.py`](./latentvla/models/constants.py) by parsing command-line arguments. If your dataset name does not clearly indicate the robot platform, adjust that file manually.
- The codebase expects RLDS-format training data.
- Some preprocessing scripts still contain placeholder paths and should be edited before first use.
- `swanlab` logging is opt-in. It will only initialize when you explicitly set `ENABLE_SWANLAB=1`.

## Acknowledgements

We thank [OpenVLA](https://github.com/openvla/openvla), [UniVLA](https://github.com/OpenDriveLab/UniVLA), [StarVLA](https://github.com/starVLA/starVLA), and [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) for their open-sourced work!
