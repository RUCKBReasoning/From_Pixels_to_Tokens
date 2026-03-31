import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentAlignProjector(nn.Module):
    """
    Align a subset of LLM tokens [B, 56, 1024] with latent actions [B, 32, 128].
    """
    def __init__(self, llm_dim=1024, latent_dim=128, align_loss_type="cosine", use_norm=False):
        super().__init__()
        self.llm_dim = llm_dim
        self.latent_dim = latent_dim
        self.align_loss_type = align_loss_type

        # MLP projector
        self.fc1 = nn.Linear(llm_dim, 2 * latent_dim)
        self.fc2 = nn.Linear(2 * latent_dim, latent_dim)
        self.act_fn = nn.GELU()
        self.latent_fc = nn.Linear(latent_dim, latent_dim)
        
        self.norm = nn.LayerNorm(llm_dim) if use_norm else None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def align_dimension(self, llm_emb):
        # [B, 56, 1024] → [B, 56, 128]
        if self.norm is not None:
            llm_emb = self.norm(llm_emb)
        projected = self.fc1(llm_emb)
        projected = self.act_fn(projected)
        projected = self.fc2(projected)
        return projected

    def compute_align_loss_cosine(self, llm_proj, latent_emb):
        """
        llm_proj: [B, 56, 128]
        latent_emb: [B, 32, 128]
        """
        latent_mean = latent_emb.mean(dim=2)  # [B, 25, 128]

        # normalize
        llm_proj = F.normalize(llm_proj, dim=-1)
        latent_mean = F.normalize(latent_mean, dim=-1)

        # element-wise cosine
        cos = (llm_proj * latent_mean).sum(dim=-1)  # [B, 25]
        loss = 1 - cos.mean()  # scalar
        return loss

    def compute_align_loss_infonce(self, llm_proj, latent_emb, temperature=0.1):
        """
        llm_proj:   [B, 25, 128]
        latent_emb: [B, 25, 4, 128]
        """
        B, T, _ = llm_proj.shape

        # 1. pool latent chunk (4 actions)
        latent_mean = latent_emb.mean(dim=2)   # [B, 25, 128]

        # 2. normalize
        llm_proj = F.normalize(llm_proj, dim=-1)
        latent_mean = F.normalize(latent_mean, dim=-1)

        # 3. compute pairwise similarity
        # [B, 25, 25]
        sim = torch.einsum("bid,bjd->bij", llm_proj, latent_mean)  

        # 4. apply temperature
        sim = sim / temperature

        # 5. labels: token_i should match latent_i
        labels = torch.arange(T, device=sim.device).unsqueeze(0).repeat(B, 1)

        loss = F.cross_entropy(sim, labels)
        return loss

    def compute_align_loss_contrastive_margin(self, llm_proj, latent_emb, margin=0.3):
        """
        llm_proj: [B, 25, 128]
        latent_emb: [B, 25, 4, 128]
        """
        B, T, _ = llm_proj.shape

        # pool chunk (4 vectors)
        latent_mean = latent_emb.mean(dim=2)             # [B, 25, 128]

        # normalize
        llm_proj = F.normalize(llm_proj, dim=-1)
        latent_mean = F.normalize(latent_mean, dim=-1)

        # compute pairwise sim: [B, 25, 25]
        sim = torch.einsum("bid,bjd->bij", llm_proj, latent_mean)

        # positive is the diagonal: [B, 25]
        pos = sim.diagonal(dim1=1, dim2=2)

        # turn pos into [B, 25, 1] so it can broadcast
        pos = pos.unsqueeze(-1)

        # margin ranking loss for each negative
        # loss = relu(margin - pos + neg)
        loss = F.relu(margin - pos + sim)

        # mask diagonal (pos-pos)
        mask = ~torch.eye(T, dtype=torch.bool, device=sim.device)
        loss = loss[:, mask].mean()

        return loss

    def forward(self, llm_emb, latent_emb):
        llm_proj = self.align_dimension(llm_emb)
        # latent_emb = self.latent_fc(latent_emb)
        if self.align_loss_type == "cosine":
            return self.compute_align_loss_cosine(llm_proj, latent_emb)
        elif self.align_loss_type == "infonce":
            return self.compute_align_loss_contrastive_margin(llm_proj, latent_emb)
        else:
            raise NotImplementedError(f"Loss type {self.align_loss_type} not implemented.")


class CrossAttentionAlignProjector(nn.Module):
    """
    Cross-attention alignment (LLM → Latent).
    LLM token embeddings [B, 56, 1024] learn to align with fixed latent actions [B, 32, 128].
    """
    def __init__(self, llm_dim=1024, latent_dim=128, num_heads=4, align_loss_type="cosine", use_norm=False):
        super().__init__()
        self.llm_dim = llm_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.align_loss_type = align_loss_type

        # Project LLM to latent space
        self.llm_proj = nn.Linear(llm_dim, latent_dim)

        # Key/Value projections for latent
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)

        # Multi-head cross-attention: Q=LLM, K/V=Latent
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)

        # Optional normalization
        self.norm = nn.LayerNorm(llm_dim) if use_norm else None

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, llm_emb, latent_emb):
        """
        llm_emb: [B, 56, 1024]  (trainable)
        latent_emb: [B, 32, 128] (ground truth, frozen)
        """
        if self.norm is not None:
            llm_emb = self.norm(llm_emb)

        # Project LLM into latent feature space
        llm_proj = self.llm_proj(llm_emb)  # [B, 56, 128]

        # Use the latent features as frozen key/value tensors.
        with torch.no_grad():
            K = self.k_proj(latent_emb)
            V = self.v_proj(latent_emb)

        # LLM tokens attend to latent representation
        attn_output, attn_weights = self.attn(llm_proj, K, V)  # [B, 56, 128], [B, 56, 32]

        # Alignment loss
        if self.align_loss_type == "cosine":
            return self.compute_align_loss_cosine(attn_output, latent_emb)
        elif self.align_loss_type == "mse":
            return F.mse_loss(attn_output.mean(dim=1), latent_emb.mean(dim=1))
        else:
            raise NotImplementedError(f"Loss type {self.align_loss_type} not implemented.")

    def compute_align_loss_cosine(self, pred, target):
        # mean-pool both sides to compare global alignment
        pred = pred.mean(dim=1)    # [B, 128]
        target = target.mean(dim=1)  # [B, 128]
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        loss = 1 - (pred * target).sum(dim=-1).mean()
        return loss
