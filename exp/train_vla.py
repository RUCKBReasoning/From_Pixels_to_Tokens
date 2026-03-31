import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
import importlib
import torch.distributed as dist
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoConfig
from dataclasses import dataclass, field
import dataclasses
from datetime import datetime
import draccus
from accelerate import PartialState
import glob
from collections import OrderedDict

from latentvla.overwatch import initialize_overwatch
from utils.utils import set_global_seed
from latentvla.models.action_tokenizer import ActionEncoder, VectorQuantizerEMA
from latentvla.training import AcceleratorStrategy_InternVL
from latentvla.training.metrics import VLAMetrics
from latentvla.data_provider.rlds.utils.data_utils import save_dataset_statistics
from latentvla.data_provider.datasets import RLDSDataset
from latentvla.data_provider.data_utils import PaddedCollatorForQwen3
from latentvla.models.vla import Baseline, LA_Cond_VLA, LA_Align_VLA, LA_Direct_VLA, LA_Tok_VLA

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

@dataclass
class FinetuneConfig:
    # save
    seed: int = 42
    run_root_dir: str = "runs"
    save_checkpoint: bool = True

    # vla
    vla_id: str = "vla_id"

    # vlm
    vlm_path: str = "path_to_vlm"
    vlm_model_id: str = "Qwen3"
    default_image_size: int = 224
    codebook_size: int = 16
    use_latent: bool = True
    pretrained_checkpoint: str = "path_to_pretrained_model"
    from_pretrained: bool = False
    
    # data
    data_root_dir: str = "path_to_rlds_data"
    data_mix: list[str] = field(default_factory=lambda: ["libero_goal"])
    shuffle_buffer_size: int = 128
    image_aug: bool = True
    window_size: int = 8
    use_wrist_image: bool = True
    use_proprio: bool = True
    use_pro_version: bool = False
    num_images_in_input: int = 2
    action_tokenizer_ckpt: str = "path_to_action_tokenizer_ckpt"

    # training
    type: str = "training"
    epochs: int = 10
    max_steps: int = 100000
    global_batch_size: int = 64
    per_device_batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.03
    save_step: int = 5000
    token_loss_weight: float = 1.0
    use_token: bool = False
    use_val: bool = False

    # wandb
    wandb_project: str = "vla"
    use_wandb: bool = False

def load_vlm(cfg):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.vlm_path, trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(cfg.vlm_path)
    tokenizer = processor.tokenizer
    config = model.config
    return model, tokenizer, processor, config

def load_vla(cfg, vlm, processor):
    if cfg.vla_id == "la_direct":
        vla = LA_Direct_VLA(vlm=vlm, num_images=cfg.num_images_in_input, use_proprio=cfg.use_proprio,action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version)
    elif cfg.vla_id == "la_cond":
        vla = LA_Cond_VLA(vlm=vlm, num_images=cfg.num_images_in_input, use_proprio=cfg.use_proprio,action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version)
    elif cfg.vla_id == "baseline":
        vla = Baseline(vlm=vlm, num_images=cfg.num_images_in_input, use_proprio=cfg.use_proprio,action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version)
    elif cfg.vla_id == "la_tok":
        vla = LA_Tok_VLA(vlm=vlm, num_images=cfg.num_images_in_input, use_proprio=cfg.use_proprio,action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version)
    else:
        vla = LA_Align_VLA(vlm=vlm, num_images=cfg.num_images_in_input, use_proprio=cfg.use_proprio, action_token_id=processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0], use_pro_version = cfg.use_pro_version)
    if cfg.from_pretrained:
        ckpt_dir = os.path.join(cfg.pretrained_checkpoint, "checkpoints")
        ckpt_list = sorted(glob.glob(os.path.join(ckpt_dir, "step-*-epoch-*-loss=*.pt")))
        ckpt_path = ckpt_list[-1]
        state_dict = torch.load(ckpt_path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        missing, unexpected = vla.load_state_dict(new_state_dict, strict=True)
        
        print("unexpeacted:", unexpected)
        print("missing:", missing)
    return vla

def get_batch_transform(cfg, processor):
    from latentvla.data_provider.data_utils import (
        RLDSBatchTransformQwen3,
        RLDSBatchTransformQwen3Joint,
        RLDSBatchTransformQwen3Uni,
    )
    transform_map = {
        "baseline": RLDSBatchTransformQwen3,
        "la_align": RLDSBatchTransformQwen3,
        "la_direct": RLDSBatchTransformQwen3Joint,
        "la_cond": RLDSBatchTransformQwen3Uni,
    }

    transform_cls = transform_map.get(cfg.vla_id, RLDSBatchTransformQwen3)
    return transform_cls(
        processor=processor,
        use_wrist_image=cfg.use_wrist_image,
        use_proprio=cfg.use_proprio,
    )


def load_action_tokenizer(ckpt_path, device="cuda"):
    encoder = ActionEncoder().to(device)
    vq = VectorQuantizerEMA().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    vq.load_state_dict(ckpt["vq"])

    encoder.eval()
    vq.eval()

    return encoder, vq

@draccus.wrap()
def train(cfg: FinetuneConfig) -> None:
    overwatch.info("VLA Training ...")
    constants = importlib.import_module("latentvla.models.constants")

    print(f"[INFO] Using VLA config for {cfg.vla_id}:")
    print(f"  ACTION_TOKEN_BEGIN_IDX = {constants.ACTION_TOKEN_BEGIN_IDX}")

    # Configure Unique Run Name & Save Directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{cfg.vla_id}+{cfg.data_mix}+{cfg.vlm_model_id}+type-{cfg.type}+{timestamp}"

    # Start =>> Build Directories and Set Randomness
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (Path(cfg.run_root_dir) / run_id), exist_ok=True)
    os.makedirs(Path(cfg.run_root_dir) / run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        OmegaConf.save(cfg, run_dir / "config.yaml")
        
    # GPU setup
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    device = distributed_state.device
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Load vlm
    vlm, tokenizer, processor, config = load_vlm(cfg)
    
    # add latent tokens
    if cfg.vla_id == "la_direct" or cfg.vla_id == "la_cond":
        special_tokens_dict = {'additional_special_tokens': [f'<ACT_{i}>' for i in range(cfg.codebook_size)]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Latent tokens range: {tokenizer.convert_tokens_to_ids(special_tokens_dict['additional_special_tokens'][0])} ~ {tokenizer.convert_tokens_to_ids(special_tokens_dict['additional_special_tokens'][-1])}, num = {num_added_toks}")
    
    if cfg.vla_id == "la_tok":        
        encoder, vq = load_action_tokenizer(
            ckpt_path=cfg.action_tokenizer_ckpt,
            device=device,
        )
        special_tokens_dict = {'additional_special_tokens': [f'<LA_{i}>' for i in range(256)]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Action tokens range: {tokenizer.convert_tokens_to_ids(special_tokens_dict['additional_special_tokens'][0])} ~ {tokenizer.convert_tokens_to_ids(special_tokens_dict['additional_special_tokens'][-1])}, num = {num_added_toks}")
        print("New vocab size:", len(tokenizer))

    # Get VLA Dataset & Collator
    overwatch.info(f"Creating VLA Dataset with Mixture `{cfg.data_mix}`")
    
    if cfg.vla_id == "la_tok":
        from latentvla.data_provider.data_utils import RLDSBatchTransformQwen3Token
        batch_transform = RLDSBatchTransformQwen3Token(
            processor=processor,
            use_wrist_image=cfg.use_wrist_image,
            use_proprio=cfg.use_proprio,
            action_encoder=encoder,
            action_vq=vq,
        )
    else:
        batch_transform = get_batch_transform(cfg, processor)
    
    vla_dataset = RLDSDataset(
        Path(cfg.data_root_dir),
        cfg.data_mix,
        batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=128,
        train=train,
        image_aug=True,
        window_size=cfg.window_size,
        goal_image_step=cfg.window_size,
    )
    print(f"Dataset length = {len(vla_dataset)}", flush=True)
    if cfg.use_val:
        val_dataset = RLDSDataset(
            Path(cfg.data_root_dir),
            cfg.data_mix,
            batch_transform,
            resize_resolution=(224, 224),
            shuffle_buffer_size=128,
            train=False,
            image_aug=True,
            window_size=cfg.window_size,
            goal_image_step=cfg.window_size,
        )
        print(f"Val Dataset length = {len(val_dataset)}", flush=True)
    
    # Create Collator
    collator = PaddedCollatorForQwen3(
        tokenizer.model_max_length, 
        tokenizer.pad_token_id, 
        padding_side="right"
    )

    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)     
    vla = load_vla(cfg, vlm, processor)

    # [Validate] Model should be in Full Precision!
    total = sum(p.numel() for p in vlm.parameters())
    trainable = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    # latent_head_params = sum(p.numel() for p in vla.latent_action_head.parameters())
    action_head_params = sum(p.numel() for p in vla.action_head.parameters())

    print(f"# Parameters of VLA: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable, {action_head_params/1e6:.2f}M action head", flush=True)
    
    overwatch.info(
        f"# Parameters of VLA (in millions): {total / 10**6:.3f} Total, {trainable / 10**6:.3f} Trainable"
    )

    # Create Train Strategy
    overwatch.info("Initializing Train Strategy `AcceleratorStrategy`")
    train_strategy = AcceleratorStrategy_InternVL(
        vla=vla,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        worker_init_fn=worker_init_fn,
        model_cfg=cfg.vla_id,
        save_checkpoint=cfg.save_checkpoint,
        use_latent_loss=(cfg.vla_id == "la_align" and cfg.use_latent),
        token_loss_weight=cfg.token_loss_weight,
    )
    train_strategy.run_setup(n_train_examples=len(vla_dataset))
    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info("Creating Metrics with Active Trackers => `jsonl, wandb`")
    metrics = VLAMetrics(
        ("jsonl", "wandb"),
        run_id,
        run_dir,
        dataclasses.asdict(cfg),
        wandb_project=cfg.wandb_project,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
        wandb=cfg.use_wandb,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_training(
        vla_dataset,
        collator,
        metrics,
        save_step=cfg.save_step,
        val_dataset=val_dataset if cfg.use_val else None,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()