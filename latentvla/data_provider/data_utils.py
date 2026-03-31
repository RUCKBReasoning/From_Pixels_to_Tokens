"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple, Any, Dict, Type, Callable, Optional, Any

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import tensorflow as tf
from PIL import Image
from torchvision import transforms
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer, AutoProcessor

from latentvla.models.action_tokenizer import ActionTokenizer
num_image_token = 256
IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = "<img>", "</img>", "<IMG_CONTEXT>"
from latentvla.models.constants import (
    IGNORE_INDEX,
    PROPRIO_DIM,
    NUM_ACTIONS_CHUNK
)
from latentvla.data_provider.utils import load_image

def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()}

@dataclass
class PaddedCollatorForQwen3:
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32
    
    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        image_grid_thw = torch.stack([
            inst["image_grid_thw"].squeeze(0)
            for inst in instances
        ], dim=0)
        latent_action_idx = [torch.as_tensor(instance["latent_action_idx"]) for instance in instances]
        latent_action_idx = torch.stack(latent_action_idx, dim=0)
        
        latent_action_z = [torch.as_tensor(instance["latent_action_z"]) for instance in instances]
        latent_action_z = torch.stack(latent_action_z, dim=0)
        
        dataset_names = [inst["dataset_name"] for inst in instances] if "dataset_name" in instances[0] else None

        assert self.padding_side == "right", f"Invalid Tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        
        attention_mask = input_ids.ne(self.pad_token_id)

        # Stack all `pixel_values` --> depending on type is torch.Tensor or Dict[str, torch.Tensor]
        if isinstance(pixel_values[0], torch.Tensor):
            if "pixel_values_wrist" in instances[0]:
                pixel_values_wrist = [instance["pixel_values_wrist"] for instance in instances]
                pixel_values = torch.cat((torch.stack(pixel_values), torch.stack(pixel_values_wrist)), dim=1)
            else:
                pixel_values = torch.stack(pixel_values)
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        # Stack all actions
        actions = [torch.from_numpy(np.copy(instance["actions"])) for instance in instances]
        actions = torch.stack(actions)

        # Stack proprio
        if "proprio" in instances[0]:
            proprio = [instance["proprio"] for instance in instances]
            proprio = torch.Tensor(np.squeeze(np.stack(proprio)))
        else:
            proprio = None

        output = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            proprio=proprio,
            attention_mask=attention_mask,
            actions=actions,
            labels=labels,
            latent_action_idx=latent_action_idx,
            latent_action_z=latent_action_z,
            image_grid_thw =image_grid_thw
        )
        if dataset_names is not None:
            output["dataset_names"] = dataset_names

        return output 

@dataclass
class RLDSBatchTransformQwen3:
    processor: AutoProcessor
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name, actions = rlds_batch["dataset_name"], rlds_batch["action"]
        latent_action_idx, latent_action_z = rlds_batch["latent_idx"], rlds_batch["latent_z"]
        imgs = []
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        imgs.append(img.resize((224, 224)))
        if self.use_wrist_image:
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    imgs.append(img_wrist.resize((224, 224)))
            
        instr = rlds_batch["task"]["language_instruction"]
        if isinstance(instr, (np.ndarray, list, tuple)):
            instr = random.choice(instr)
        if isinstance(instr, bytes):
            instr = instr.decode("utf-8")
        assert isinstance(instr, str), f"Unexpected type: {type(instr)}"
        lang = instr.lower()

        lang = DATASET_INSTR_MAP.get(dataset_name.decode("utf-8"), lang)
        
        action_token = "🔍"
        action_tokens = action_token* (NUM_ACTIONS_CHUNK*5+1)
        prompt_suffix = f"Please predict the next {NUM_ACTIONS_CHUNK*5} robot actions: {action_tokens}"
        
        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": lang + prompt_suffix})
        msg = [{"role": "user", "content": content}]
        batch_inputs = self.processor.apply_chat_template(
            msg,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        return_dict = dict(
            input_ids=batch_inputs["input_ids"][0],
            attention_mask=batch_inputs["attention_mask"][0],
            pixel_values=batch_inputs["pixel_values"],
            image_grid_thw=batch_inputs["image_grid_thw"],
            actions=actions,
            labels=batch_inputs["input_ids"][0].clone(),
            dataset_name=dataset_name,
            latent_action_idx=latent_action_idx,
            latent_action_z=latent_action_z
        )
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            if proprio.shape[-1] != PROPRIO_DIM:
                proprio = tf.concat([proprio[:, :6], proprio[:, 7:]], axis=1)
            return_dict["proprio"] = proprio

        return return_dict

@dataclass
class RLDSBatchTransformQwen3Token:
    processor: AutoProcessor
    use_wrist_image: bool = False
    use_proprio: bool = False
    action_encoder: nn.Module = None
    action_vq: nn.Module = None

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name, actions = rlds_batch["dataset_name"], rlds_batch["action"]
        # actions: (25,14)
        device = next(self.action_encoder.parameters()).device
        act = torch.tensor(actions, dtype=torch.float32).unsqueeze(0).to(device)
        # => (1,25,14)
        with torch.no_grad():
            H = self.action_encoder(act)          # (1,25,128)
            H_q, token_idx, _ = self.action_vq(H) # token_idx: (1,25)
        tokens = token_idx.squeeze(0).tolist()
        la_tokens = [f"<LA_{t}>" for t in tokens]
        action_tokens_str = "".join(la_tokens)
        
        latent_action_idx, latent_action_z = rlds_batch["latent_idx"], rlds_batch["latent_z"]
        imgs = []
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        imgs.append(img.resize((224, 224)))
        if self.use_wrist_image:
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    imgs.append(img_wrist.resize((224, 224)))
            
        instr = rlds_batch["task"]["language_instruction"]
        if isinstance(instr, (np.ndarray, list, tuple)):
            instr = random.choice(instr)
        if isinstance(instr, bytes):
            instr = instr.decode("utf-8")
        assert isinstance(instr, str), f"Unexpected type: {type(instr)}"
        lang = instr.lower()

        lang = DATASET_INSTR_MAP.get(dataset_name.decode("utf-8"), lang)
     
        prompt_suffix = (
            f"Please predict the next {NUM_ACTIONS_CHUNK} robot actions: "
            f"{action_tokens_str}"
        )
        
        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": lang + prompt_suffix})
        msg = [{"role": "user", "content": content}]
        batch_inputs = self.processor.apply_chat_template(
            msg,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        input_ids = batch_inputs["input_ids"][0]
        labels = torch.full_like(input_ids, -100)
        mask = (input_ids >= 151669) & (input_ids <= 151924)
        labels[mask] = input_ids[mask]

        action_token_id=self.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]
        input_ids[mask] = action_token_id

        # print(input_ids, labels)

        # print(batch_inputs["input_ids"][0])
        return_dict = dict(
            input_ids=input_ids,
            attention_mask=batch_inputs["attention_mask"][0],
            pixel_values=batch_inputs["pixel_values"],
            image_grid_thw=batch_inputs["image_grid_thw"],
            actions=actions,
            labels=batch_inputs["input_ids"][0].clone(),
            dataset_name=dataset_name,
            latent_action_idx=latent_action_idx,
            latent_action_z=latent_action_z
        )
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            if proprio.shape[-1] != PROPRIO_DIM:
                proprio = tf.concat([proprio[:, :6], proprio[:, 7:]], axis=1)
            return_dict["proprio"] = proprio

        return return_dict
@dataclass
class RLDSBatchTransformQwen3Joint:
    processor: AutoProcessor
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name, actions = rlds_batch["dataset_name"], rlds_batch["action"]
        latent_action_idx = rlds_batch["latent_idx"]   # shape = [chunk, 4]
        latent_action_z  = rlds_batch["latent_z"]

        # -----------------------------------------
        # 1. load images
        # -----------------------------------------
        imgs = []
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        imgs.append(img.resize((224, 224)))

        if self.use_wrist_image:
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    imgs.append(img_wrist.resize((224, 224)))
            
        # -----------------------------------------
        # 2. instruction
        # -----------------------------------------
        instr = rlds_batch["task"]["language_instruction"]
        if isinstance(instr, (np.ndarray, list, tuple)):
            instr = random.choice(instr)
        if isinstance(instr, bytes):
            instr = instr.decode("utf-8")
        assert isinstance(instr, str), f"Unexpected type: {type(instr)}"
        lang = instr.lower()
        total_latent_tokens = NUM_ACTIONS_CHUNK * 4

        dummy_placeholder = "🔍"
        prompt_suffix = (
            f"Please predict the next {NUM_ACTIONS_CHUNK} robot actions: "
            + dummy_placeholder * (total_latent_tokens + 1)
        )
        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": lang + prompt_suffix})
        msg = [{"role": "user", "content": content}]

        batch_inputs = self.processor.apply_chat_template(
            msg,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        input_ids = batch_inputs["input_ids"][0]
        labels    = input_ids.clone()

        dummy_id = self.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]

        latent_ids_str_list = [
            f"<ACT_{int(tok)}>" for row in latent_action_idx for tok in row
        ]
        latent_token_ids = [
            self.processor.tokenizer(tok_str, add_special_tokens=False)["input_ids"][0]
            for tok_str in latent_ids_str_list
        ]

        assert len(latent_token_ids) == total_latent_tokens
        act_pos = (labels == dummy_id).nonzero(as_tuple=False).squeeze(-1)

        assert len(act_pos) == total_latent_tokens, \
            f"dummy count={len(act_pos)}, expected={total_latent_tokens}"
        
        labels[:] = -100

        for i, pos in enumerate(act_pos):
            labels[pos] = latent_token_ids[i]
        return_dict = dict(
            input_ids=input_ids,
            attention_mask=batch_inputs["attention_mask"][0],
            pixel_values=batch_inputs["pixel_values"],
            image_grid_thw=batch_inputs["image_grid_thw"],
            actions=actions,
            dataset_name=dataset_name,
            labels=labels,
            latent_action_idx=latent_action_idx,
            latent_action_z=latent_action_z
        )

        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            if proprio.shape[-1] != PROPRIO_DIM:
                proprio = tf.concat([proprio[:, :6], proprio[:, 7:]], axis=1)
            return_dict["proprio"] = proprio

        return return_dict

@dataclass
class RLDSBatchTransformQwen3Uni:
    processor: AutoProcessor
    use_wrist_image: bool = False
    use_proprio: bool = False

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name, actions = rlds_batch["dataset_name"], rlds_batch["action"]
        latent_action_idx = rlds_batch["latent_idx"]   # shape = [chunk, 4]
        latent_action_z  = rlds_batch["latent_z"]

        imgs = []
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        imgs.append(img.resize((224, 224)))

        if self.use_wrist_image:
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    imgs.append(img_wrist.resize((224, 224)))

        instr = rlds_batch["task"]["language_instruction"]
        if isinstance(instr, (np.ndarray, list, tuple)):
            instr = random.choice(instr)
        if isinstance(instr, bytes):
            instr = instr.decode("utf-8")
        assert isinstance(instr, str), f"Unexpected type: {type(instr)}"
        lang = instr.lower()

        total_latent_tokens = NUM_ACTIONS_CHUNK * 4
        total_dummy_tokens = total_latent_tokens + NUM_ACTIONS_CHUNK

        dummy_placeholder = "🔍"
        prompt_suffix = (
            f"Please predict the next {NUM_ACTIONS_CHUNK} robot actions: "
            + dummy_placeholder * (total_dummy_tokens + 1)
        )

        content = [{"type": "image", "image": img} for img in imgs]
        content.append({"type": "text", "text": lang + prompt_suffix})
        msg = [{"role": "user", "content": content}]

        batch_inputs = self.processor.apply_chat_template(
            msg,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        input_ids = batch_inputs["input_ids"][0]
        labels    = input_ids.clone()

        dummy_id = self.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]

        latent_ids_str_list = [
            f"<ACT_{int(tok)}>" for row in latent_action_idx for tok in row
        ]
        latent_token_ids = [
            self.processor.tokenizer(tok_str, add_special_tokens=False)["input_ids"][0]
            for tok_str in latent_ids_str_list
        ]

        assert len(latent_token_ids) == total_latent_tokens
        
        all_dummy_pos = (labels == dummy_id).nonzero(as_tuple=False).squeeze(-1)
        assert len(all_dummy_pos) == total_dummy_tokens, \
            f"dummy count={len(all_dummy_pos)}, expected={total_dummy_tokens}"

        latent_dummy_pos = all_dummy_pos[:total_latent_tokens]
        ignore_dummy_pos = all_dummy_pos[total_latent_tokens:]

        labels[:] = -100

        for i, pos in enumerate(latent_dummy_pos):
            labels[pos] = latent_token_ids[i]

        return_dict = dict(
            input_ids=input_ids,
            attention_mask=batch_inputs["attention_mask"][0],
            pixel_values=batch_inputs["pixel_values"],
            image_grid_thw=batch_inputs["image_grid_thw"],
            actions=actions,
            dataset_name=dataset_name,
            labels=labels,
            latent_action_idx=latent_action_idx,
            latent_action_z=latent_action_z
        )

        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            if proprio.shape[-1] != PROPRIO_DIM:
                proprio = tf.concat([proprio[:, :6], proprio[:, 7:]], axis=1)
            return_dict["proprio"] = proprio

        return return_dict
   
@dataclass
class RLDSBatchTransform:
    def __init__(self, processor, latent, latent_model):
        self.processor = processor
        self.latent = latent
        self.latent_model = latent_model

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = rlds_batch["dataset_name"]


        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        img_k = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        if self.latent:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            img_tensor = transform(img).unsqueeze(0).cuda()
            img_k_tensor = transform(img_k).unsqueeze(0).cuda()
            video = torch.stack([img_tensor, img_k_tensor], dim=2)  # (B, C, T, H, W)
            with torch.no_grad():
                latent_actions = self.latent_model.inference(
                    video,
                    return_only_codebook_ids=True
                )
                 
        img = img.resize((56, 56))
        H, W = img.size[1], img.size[0]
        patch_size = 14
        image_grid_thw = [(1, H // patch_size, W // patch_size)]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": f"What action should the robot take to {lang}?"},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        action = torch.tensor(rlds_batch["action"][0], dtype=torch.bfloat16)

        return dict(
            input_ids=inputs["input_ids"][0],
            attention_mask=inputs["attention_mask"][0],
            pixel_values=inputs.get("pixel_values", None) if "pixel_values" in inputs else None,
            labels=action,
            dataset_name=dataset_name,
            image_grid_thw=image_grid_thw,
            latent_actions=latent_actions if self.latent else None,
        )
    
@dataclass
class RLDSBatchTransformLatentAction:
    def __init__(self, processor, latent_model, window_size=12):
        self.processor = processor
        self.latent_model = latent_model
        self.window_size = window_size

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = rlds_batch["dataset_name"]

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][-1])
        img_k = Image.fromarray(rlds_batch["goal_image"]["image_primary"])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        with torch.no_grad():
            initial_pixel_values = transform(img).unsqueeze(0).cuda()
            target_pixel_values = transform(img_k).unsqueeze(0).cuda()
            video = torch.stack([initial_pixel_values, target_pixel_values], dim=2)  # (B, C, T, H, W)
            latent_actions = self.latent_model.inference(
                video,
                return_only_codebook_ids=True
            )
        action_vocab = [f'<ACT_{i.item()}>' for i in latent_actions.squeeze(0)]
        action_tokens = ''
        for i, action in enumerate(action_vocab):
            action_tokens += action
            
        img = [Image.fromarray(rlds_batch["observation"]["image_primary"][-1])]
        img = [i.resize((224, 224)) for i in img]
        W, H = img[0].size
        patch_size = 14
        image_grid_thw = [(len(img), H // patch_size, W // patch_size)]

        image_contents = [{"type": "image", "image": i} for i in img]
        messages = [
            {
                "role": "user",
                "content":  image_contents+[
                    {"type": "text", "text": f"What action should the robot take to {lang}?"},
                ],
            },
            # {
            #     "role": "assistant",
            #     "content": [
            #         {"type": "text", "text": action_tokens},
            #     ],
            # },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        action = torch.tensor(rlds_batch["action"][-self.window_size:], dtype=torch.bfloat16)
        labels = inputs["input_ids"][0].clone()
        labels[: -(len(action_vocab) + 1)] = IGNORE_INDEX
        # print(inputs["input_ids"].shape, inputs["attention_mask"].shape, inputs.get("pixel_values", None).shape)
        return dict(
            input_ids=inputs["input_ids"][0],
            attention_mask=inputs["attention_mask"][0],
            pixel_values=inputs.get("pixel_values", None) if "pixel_values" in inputs else None,
            actions=action,
            labels=labels,
            dataset_name=dataset_name,
            image_grid_thw=image_grid_thw,
            latent_actions=latent_actions,
        )
