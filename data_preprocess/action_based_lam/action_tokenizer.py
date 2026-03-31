import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
from data_provider.datasets import RLDSDataset
from tqdm import tqdm
import swanlab
from latentvla.models.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

def dct_1d(x):
    # x: (B, T, D)
    return torch.fft.dct(x, norm='ortho', type=2, dim=1)

def fft_1d(x):
    fft = torch.fft.fft(x, dim=1)
    return torch.cat([fft.real, fft.imag], dim=-1)

class ActionEncoder(nn.Module):
    def __init__(self, action_dim=ACTION_DIM, hidden_dim=128, freq_type="fft"):
        super().__init__()
        self.freq_type = freq_type

        input_dim = action_dim * 2 if freq_type == "dct" else action_dim * 3

        self.in_proj = nn.Linear(input_dim, hidden_dim)

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        """
        x: (B, 25, 14)
        return H: (B, 25, hidden_dim)
        """
        if self.freq_type == "dct":
            freq = dct_1d(x)
        else:
            freq = fft_1d(x)

        x_aug = torch.cat([x, freq], dim=-1)

        h = self.in_proj(x_aug)  # (B,25,H)

        h_ = h.permute(0, 2, 1)
        h_ = F.relu(self.conv1(h_))
        h_ = F.relu(self.conv2(h_))
        h_ = F.relu(self.conv3(h_))
        h = h_.permute(0, 2, 1)

        H = self.transformer(h)
        return H

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings=256, embedding_dim=128, decay=0.99, eps=1e-5):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.randn(num_embeddings, embedding_dim))

    def forward(self, z):
        """
        z: (B, T, D)
        output:
          z_q: quantized output
          indices: (B,T)
          loss: commitment loss
        """
        B, T, D = z.shape
        z_flat = z.reshape(B * T, D)

        # L2 distances
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )  # (B*T, K)

        indices = torch.argmin(dist, dim=1)  # (B*T)
        z_q = self.embedding[indices].view(B, T, D)

        if self.training:
            one_hot = F.one_hot(indices, self.num_embeddings).float()
            cluster_size = one_hot.sum(0)
            self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            ema_w = one_hot.T @ z_flat
            self.ema_w.mul_(self.decay).add_(ema_w, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / \
                           (n + self.num_embeddings * self.eps) * n
            self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))

        # commitment loss
        loss_vq = F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()

        return z_q, indices.view(B, T), loss_vq

class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim=128, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, z_q):
        return self.net(z_q)


# ============================================
# Masking
# ============================================
def apply_mask(H, mask_ratio=0.15):
    B, T, D = H.shape
    mask = (torch.rand(B, T) < mask_ratio).to(H.device)
    H_masked = H.clone()
    H_masked[mask] = 0.0
    return H_masked, mask

def build_collate():
    def _collate(batch):
        actions = []
        for sample in batch:
            act = torch.tensor(sample["action"], dtype=torch.float32)
            if act.dim() == 1:
                act = act.unsqueeze(0)
            if act.size(0) < NUM_ACTIONS_CHUNK:
                act = torch.cat([act, act[-1:].repeat(NUM_ACTIONS_CHUNK - act.size(0), 1)], dim=0)
            else:
                act = act[:NUM_ACTIONS_CHUNK]
            actions.append(act)

        return {
            "actions": torch.stack(actions, dim=0)
        }

    return _collate

@dataclass
class ActionConfig:
    data_root_dir: str
    data_mix: List[str] = field(default_factory=lambda: ["libero_goal"])
    batch_size: int = 8
    num_steps: int = 50_000
    lr: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    save_every: int = 5000
    results_dir: str = "results_action"
    use_swanlab: bool = False

def train_tokenizer(
    encoder, vq, decoder,
    dataloader, optimizer,
    num_steps=10000,
    save_every=5000,
    device="cuda",
    mask_ratio=0.15,
    use_swanlab=False,
):
    encoder.train()
    vq.train()
    decoder.train()
    
    progress = tqdm(range(num_steps))
    dataloader_iter = iter(dataloader)
    
    for step in progress:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            
        actions = batch["actions"].to(device)  # (B,25,14)

        # ----- Encode -----
        H = encoder(actions)

        # ----- Mask -----
        H_masked, mask = apply_mask(H, mask_ratio)

        # ----- VQ -----
        H_q, indices, loss_vq = vq(H_masked)

        # ----- Decode -----
        recon = decoder(H_q)   # (B,25,14)

        # ----- Loss 1: reconstruction -----
        loss_recon = F.mse_loss(recon, actions)

        # ----- Loss 2: masked latent prediction -----
        # Predict latent again from recon
        pred_H = encoder(recon)
        loss_mask = F.mse_loss(pred_H[mask], H[mask])

        # ----- Total loss -----
        loss = loss_recon + loss_mask + 0.25 * loss_vq

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"[step {step}] loss={loss.item():.4f}, "
                  f"recon={loss_recon.item():.4f}, mask={loss_mask.item():.4f}, "
                  f"vq={loss_vq.item():.4f}")
        if use_swanlab:
            swanlab.log({
                "loss": loss.item(),
                "loss_recon": loss_recon.item(),
                "loss_mask": loss_mask.item(),
                "loss_vq": loss_vq.item()
            })
        # Save
        if (step + 1) % save_every == 0:
            out = Path(cfg.results_dir)
            out.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "encoder": encoder.state_dict(),
                "vq": vq.state_dict(),
                "decoder": decoder.state_dict(),
            }
            torch.save(ckpt, out / f"tokenizer_step_{step+1}.pt")

def main(cfg: ActionConfig):

    device = "cuda"

    # --- build dataset ----
    dataset = RLDSDataset(
        Path(cfg.data_root_dir),
        cfg.data_mix,
        batch_transform=None,
        resize_resolution=(224, 224),
        shuffle_buffer_size=128,
        train=True,
        image_aug=False,
        window_size=NUM_ACTIONS_CHUNK,
        goal_image_step=32,
        return_raw_data=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=build_collate(),
        num_workers=4,
        pin_memory=True,
    )

    # --- build tokenizer modules ----
    encoder = ActionEncoder().to(device)
    vq = VectorQuantizerEMA().to(device)
    decoder = ActionDecoder().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(vq.parameters()) +
        list(decoder.parameters()), lr=cfg.lr
    )
    if cfg.use_swanlab:
        swanlab.init(
            project="action_tokenizer",
            config=cfg.__dict__,  # log config
        )

    train_tokenizer(
        encoder, vq, decoder,
        dataloader, optimizer,
        num_steps=cfg.num_steps,
        save_every=cfg.save_every,
        device=device,
        use_swanlab=cfg.use_swanlab,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root-dir", type=str, required=True)
    parser.add_argument("--data-mix", type=str, nargs="+", default=["libero_goal"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--results-dir", type=str, default="results_laq_flow")
    parser.add_argument("--use_swanlab", type=bool, default=False)

    args = parser.parse_args()
    from pathlib import Path

    if len(args.data_mix) == 1 and args.data_mix[0] == "aloha":
        robottwin_root = Path("path_to/RoboTwin")
        assert robottwin_root.exists(), f"{robottwin_root} not found"

        args.data_mix = sorted([
            f"{p.name}"
            for p in robottwin_root.iterdir()
            if p.is_dir()
        ])

        print("[INFO] Auto-loaded RoboTwin datasets:")
        for name in args.data_mix:
            print("  ", name)

    cfg = ActionConfig(
        data_root_dir=args.data_root_dir,
        data_mix=args.data_mix,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        save_every=args.save_every,
        results_dir=args.results_dir,
        use_swanlab=args.use_swanlab,
    )
    main(cfg)
