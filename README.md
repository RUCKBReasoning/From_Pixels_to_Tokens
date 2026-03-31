## VLA Training Project

This repository contains code for training a Vision-Language-Action (VLA) model with 4 latent supervision strategies.

---

### Training Script

You can launch VLA training with `torchrun` using `exp/train_vla.py`.  
Example command:

```bash
torchrun --nnodes=1 --nproc_per_node=1 exp/train_vla.py \
  --seed 42 \
  --run_root_dir runs \
  --save_checkpoint True \
  --vla_id baseline \
  --vlm_path path_to_vlm \
  --vlm_model_id Qwen3 \
  --default_image_size 224 \
  --data_root_dir path_to_dir \
  --data_mix data_mix \
  --shuffle_buffer_size 128 \
  --image_aug True \
  --window_size 8 \
  --use_wrist_image True \
  --use_proprio True \
  --use_pro_version True \
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
  --wandb_project vla \
  --token_loss_weight 0.0 \
  --from_pretrained True \
  --use_wandb True
```

---

### Repository Structure

- **`exp/train_vla.py`**  
  Main training script for the VLA model. Handles argument parsing, dataset setup, model creation, and training loop.

- **`model/models`**  
  Core model architectures:
  - **`Baseline.py`**: Baseline VLA architecture.
  - **`LA_Align.py`**: Strategy 1.
  - **`LA_Cond.py`**: Strategy 3.
  - **`LA_Direct.py`**: Strategy 2.
  - **`LA_Tok.py`**: Strategy 4.
  - **`action_heads.py` / `latent_heads.py`**: Action and latent heads used by the main architectures.
  - **`constants.py`**: Shared constants and configuration values.
  - **`vla/utils.py`**: Helper functions for building and using VLA models.

- **`model/data_provider`**  
  Data loading and preprocessing:
  - **`datasets.py`**: Dataset definitions and builders.
  - **`data_utils.py` / `utils.py`**: General data helper functions.
  - **`materialize.py`**: Utilities for preparing / materializing datasets on disk.
  - **`rlds/`**: RLDS-style dataset handling, observation and trajectory transforms, mixture configs, and utilities (e.g., `oxe` configs and transforms, goal relabeling, task augmentation).

- **`model/training`**  
  Training utilities:
  - **`accelerator.py`**: Multi-GPU / distributed training helpers.
  - **`metrics.py`**: Metric computation and logging utilities.
  - **`training_utils.py`**: General training loop helpers (scheduling, checkpointing, etc.).

- **`model/overwatch`**  
  Monitoring and orchestration helpers for training (e.g., logging, run management).

- **`utils`**  
  General-purpose utilities:
  - **`msgpack_numpy.py`**: Numpy serialization helpers.
  - **`utils.py`**: Miscellaneous helper functions shared across the project.

- **`requirements.txt` / `pyproject.toml`**  
  Python dependencies and project configuration.

---

### Getting Started

- **Install dependencies**:  
  ```bash
  pip install -r requirements.txt
  ```
- **Set environment variables and run training** using the example `torchrun` command above, adjusting paths (`vlm_path`, `data_root_dir`, etc.) as needed.