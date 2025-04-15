import torch
import matplotlib.pyplot as plt

def entropy_weighted_attention(attentions):
    """
    Compute a weighted average of attention heads based on their entropy.
    Higher entropy heads will receive lower weights.
    """
    num_heads = attentions.size(1)
    # Calculate the entropy value for each header
    entropy = -torch.sum(attentions * torch.log(attentions + 1e-6), dim=-1)  # Compute entropy for each head
    weights = torch.exp(-entropy)  # The negative exponent of the entropy value is used as a weighting
    weights = weights / weights.sum(dim=1, keepdim=True)  # normalized weight

    # Weighted average of all attention heads
    weighted_attention = (attentions * weights.unsqueeze(-1)).sum(dim=1)
    return weighted_attention

def attention_rollout(attentions, discard_ratio=0.5):
    """
    Compute the attention rollout based on the attention weights from the ViT model,
    using entropy-weighted averaging for attention heads.
    
    Parameters:
    - attentions: List of attention weights from each layer.
    - discard_ratio: Percentage of attention heads to discard (less important ones).
    
    Returns:
    - rollout_attention: Final attention map after rollout.
    """
    num_tokens = attentions[0].size(-1)
    rollout_attention = torch.eye(num_tokens).to(attentions[0].device)

    for idx, attention in enumerate(attentions):
        # Fusing Attention Heads Using Weighted Strategies
        weighted_attention = entropy_weighted_attention(attention)

        # Print statistics for each layer of the Attention Matrix
        print(f"Layer {idx} - Mean: {weighted_attention.mean().item()}, Min: {weighted_attention.min().item()}, Max: {weighted_attention.max().item()}")

        # Add a small value to prevent division by 0
        epsilon = 1e-6
        weighted_attention = weighted_attention + epsilon
        
        # Normalizing attention
        weighted_attention = weighted_attention / weighted_attention.sum(dim=-1, keepdim=True)

        # Cumulative attention per layer
        rollout_attention = torch.matmul(weighted_attention, rollout_attention)

    return rollout_attention

def visualize_attention_map(attention_map):
    """
    Visualize the attention map using matplotlib.
    """
    attention_map = attention_map.cpu().numpy() 
    plt.imshow(attention_map, cmap='viridis')
    plt.colorbar()
    plt.show()