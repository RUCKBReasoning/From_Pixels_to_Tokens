import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from latentvla.models.action_heads import L1RegressionActionHead, ProprioProjector
from latentvla.models.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM
)
from latentvla.models.vla.utils import _gather_action_token_embeddings

class Baseline(nn.Module):
    def __init__(
        self, vlm, num_images, use_proprio, action_token_id, use_pro_version=False):
        super().__init__()
        self.vlm = vlm
        self.action_token_id = action_token_id
        self.num_images = num_images
        
        for param in self.vlm.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        self.use_proprio = use_proprio
        self.vlm = get_peft_model(self.vlm, lora_config)

        self.proprio_projector = ProprioProjector(
            llm_dim = 2048,
            proprio_dim = PROPRIO_DIM
        )
        self.action_head = L1RegressionActionHead(
            input_dim = 2048,
            hidden_dim = 2048,
            action_dim = ACTION_DIM,
            num_blocks = 11,
            num_action_chunk = NUM_ACTIONS_CHUNK,
            use_pro_version = use_pro_version
        )

    def forward(self, batch, training=True):
        B, N, _ = batch["image_grid_thw"].shape
        image_grid_thw = batch["image_grid_thw"].reshape(B*N, 3)
        vlm_outputs = self.vlm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Get action masks needed for logging
            num_patches = 256 * self.num_images
            multi_layer_hidden_states = []
            
            for layer_hidden in vlm_outputs.hidden_states[-12:]:   # [B, L, H]
                B, L, H = layer_hidden.shape
                image_hidden = layer_hidden[:, :num_patches]       # [B, P, H]
                text_hidden = layer_hidden

                action_hidden = _gather_action_token_embeddings(
                    last_hidden=text_hidden,
                    input_ids=batch["input_ids"][:, :],
                    action_token_id=self.action_token_id,
                    num_chunk=NUM_ACTIONS_CHUNK*5,
                )  # [B, NUM_ACTIONS_CHUNK, H]
                image_latent = image_hidden.unsqueeze(1)                  # [B, 1, P, H]
                action_latent = action_hidden.unsqueeze(1)                # [B, 1, A, H]

                all_hidden = torch.cat((image_latent, action_latent), dim=2)
                multi_layer_hidden_states.append(all_hidden)
            
            multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim = 1)
            predicted_actions = self.action_head.predict_action(
                multi_layer_hidden_states,
                proprio=batch["proprio"] if self.use_proprio else None,
                proprio_projector=self.proprio_projector if self.use_proprio else None,
                phase="Training" if training else "Inference",
            )
            ground_truth_actions = batch["actions"].to(torch.bfloat16)
            action_loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)
        return dict(
            action_loss=action_loss
        )