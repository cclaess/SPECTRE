import torch
import torch.nn as nn


class FeatureVisionTransformer(nn.Module):
    def __init__(
        self,
        patch_dim: int = 768,
        embed_dim: int = 768,
        num_patches: int = 36,
        depth: int = 4,
        heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        causal: bool = True
    ):
        """
        A Vision Transformer that accepts already flattened embedding tokens from a previous layer as input 
        (call them patches for consistency).

        Args:
            patch_dim (int): Dimension of each flattened patch 
            embed_dim (int): Dimension of the patch embeddings.
            num_patches (int): Number of patches in the input.
            depth (int): Number of transformer encoder layers.
            heads (int): Number of attention heads.
            mlp_dim (int): Dimension of the feed-forward (MLP) layer in each transformer block.
            dropout (float): Dropout rate.
            causal (bool): If True, the transformer applies a causal mask.
        """
        super().__init__()
        # Linear projection of flattened patches to embedding dimension.
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_dim, self.embed_dim)
        
        # Learnable classification token.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Learnable positional embeddings for each patch + the cls token.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder layer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Final layer normalization.
        self.norm = nn.LayerNorm(embed_dim)

        self.causal = causal

    def forward(self, patches: torch.Tensor, return_cls_token = True) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer.

        Args:
            patches (torch.Tensor): Input tensor of shape (batch, num_patches, patch_dim)
                containing flattened patches.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, num_patches + 1, embed_dim)
                after processing through the transformer.
        """
        b = patches.shape[0]  # Batch size.
        
        # Project flattened patches to embedding dimension.
        x = self.proj(patches)  # Shape: (b, n, embed_dim)
        
        # Prepend the class token to each sequence.
        cls_tokens = self.cls_token.expand(b, -1, -1)  # Shape: (b, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (b, n+1, embed_dim)
        
        # Add positional embeddings.
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)
        
        # Transformer expects input as (sequence_length, batch, embed_dim)
        x = x.transpose(0, 1)
        # If causal mode is enabled, create a causal mask.
        if self.causal:
            seq_len = x.size(0)
            # Create an upper-triangular matrix, where True indicates masked positions.
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            x = self.transformer(x, mask=causal_mask, is_causal=True)
        else:
            x = self.transformer(x)
        x = x.transpose(0, 1)
        
        # Apply final layer normalization.
        x = self.norm(x)

        if return_cls_token:
            return x[:, 0]
        return x