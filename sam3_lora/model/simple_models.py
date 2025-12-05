"""
Simple models for testing and demonstration.

These models can be used for testing LoRA without requiring SAM3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparatedMultiheadAttention(nn.Module):
    """
    Multihead Attention with separate Linear layers for Q, K, V.
    
    This allows LoRA to target individual projections (q_proj, k_proj, v_proj),
    unlike nn.MultiheadAttention which fuses them.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout_p = dropout
        self.batch_first = True  # Required for compatibility with TransformerEncoderLayer

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, is_causal=False, **kwargs):
        # Inputs are (Batch, Seq, Dim) assuming batch_first=True usage
        B, L, D = query.shape
        
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # F.scaled_dot_product_attention expects (Batch, Heads, Seq, HeadDim)
        # key_padding_mask: (Batch, Seq)
        
        # Handle masks if provided
        # Note: nn.TransformerEncoderLayer handles mask preparation, but SDPA handles it efficiently too
        
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )
        
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(output), None


class SeparatedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer using SeparatedMultiheadAttention."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", layer_norm_eps=1e-5, batch_first=True, 
                 norm_first=False, bias=True):
        # Force batch_first=True for our implementation simplicity
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, 
                         layer_norm_eps, True, norm_first, bias)
        
        # Replace self_attn with our separated version
        self.self_attn = SeparatedMultiheadAttention(d_model, nhead, dropout=dropout, bias=bias)


class SimpleTransformer(nn.Module):
    """
    Simple transformer for testing LoRA injection.

    This is a lightweight model that demonstrates LoRA functionality
    without requiring the full SAM3 model.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
    ):
        super().__init__()

        # Encoder
        # Use our separated layer to enable full LoRA testing
        encoder_layer = SeparatedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Decoder
        # For decoder, we keep standard for now or could replace too
        # (Simpler to keep standard for demo unless requested)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Output head
        self.head = nn.Linear(d_model, 10)

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return self.head(output)


class SimpleSegmentationModel(nn.Module):
    """
    Simple segmentation model for demonstration.

    This is a minimal model that can be used to demonstrate LoRA training
    without the full SAM3 architecture.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
    ):
        super().__init__()

        # Simple encoder with Separated Attention for LoRA support
        self.encoder = SeparatedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        # Simple head
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x.mean(dim=1))