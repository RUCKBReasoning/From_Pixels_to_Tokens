import torch
import torch.nn as nn
from model.models.constants import NUM_ACTIONS_CHUNK

def _gather_action_token_embeddings(
        last_hidden: torch.Tensor,   # [B, L, H]
        input_ids: torch.Tensor,     # [B, L]
        action_token_id=None,
        num_chunk=NUM_ACTIONS_CHUNK,
    ) -> torch.Tensor:
        if action_token_id is None:
            raise ValueError("action_token_id should be provided.")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < num_chunk).any():
            insufficient = (counts < num_chunk).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"tokens not enough {num_chunk}: {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))

        topk_pos = masked_pos.topk(k=num_chunk, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values                     # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)   # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, h):  
        scores = torch.matmul(h, self.query)
        weights = torch.softmax(scores, dim=1)  # [B, 4]
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)

        return pooled
