import torch

# IGNORE_INDEX = -100
# ACTION_DIM = 7
# ACTION_TOKEN_BEGIN_IDX = 151386

from model.models.constants import (
    ACTION_DIM,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    LATENT_ACTION_DIM
)

def get_current_action_mask(token_ids):
    # import pdb; pdb.set_trace()
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids):
    # import pdb; pdb.set_trace()
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = cumsum > ACTION_DIM

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask
