import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenCompressor(nn.Module):
    def __init__(self, input_dim, block_size, stride):
        super(TokenCompressor, self).__init__()
        self.block_size = block_size
        self.stride = stride
        self.linear = nn.Linear(input_dim * block_size, input_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        compressed_tokens = []
        for i in range(0, seq_len - self.block_size + 1, self.stride):
            block = x[:, i:i + self.block_size, :].contiguous().view(batch_size, -1)
            compressed_token = self.linear(block)
            compressed_tokens.append(compressed_token)
        return torch.stack(compressed_tokens, dim=1)
    

class TokenSelector(nn.Module):
    def __init__(self, input_dim, selection_block_size, top_k):
        super(TokenSelector, self).__init__()
        self.selection_block_size = selection_block_size
        self.top_k = top_k
        self.scoring = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        scores = self.scoring(x).squeeze(-1)  # (batch_size, seq_len)
        selected_indices = []
        for i in range(0, seq_len, self.selection_block_size):
            block_scores = scores[:, i:i + self.selection_block_size]
            _, top_indices = torch.topk(block_scores, self.top_k, dim=-1)
            selected_indices.append(top_indices + i)
        selected_indices = torch.cat(selected_indices, dim=-1)
        selected_tokens = torch.gather(x, 1, selected_indices.unsqueeze(-1).expand(-1, -1, input_dim))
        return selected_tokens


class SlidingWindowAttention(nn.Module):
    def __init__(self, input_dim, window_size):
        super(SlidingWindowAttention, self).__init__()
        self.window_size = window_size
        self.scale = input_dim ** -0.5

    def forward(self, q, k, v):
        # q, k, v: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = q.size()
        output = torch.zeros_like(q)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            q_i = q[:, i:i + 1, :]  # (batch_size, 1, input_dim)
            k_window = k[:, start:end, :]  # (batch_size, window, input_dim)
            v_window = v[:, start:end, :]  # (batch_size, window, input_dim)
            attn_weights = torch.bmm(q_i, k_window.transpose(1, 2)) * self.scale  # (batch_size, 1, window)
            attn_weights = F.softmax(attn_weights, dim=-1)
            output[:, i:i + 1, :] = torch.bmm(attn_weights, v_window)  # (batch_size, 1, input_dim)
        return output


class NativeSparseAttention(nn.Module):
    def __init__(self, input_dim, num_heads, compress_params, select_params, window_size):
        super(NativeSparseAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, input_dim * 3)
        self.o_proj = nn.Linear(input_dim, input_dim)

        self.token_compressor = TokenCompressor(input_dim, **compress_params)
        self.token_selector = TokenSelector(input_dim, **select_params)
        self.sliding_window_attn = SlidingWindowAttention(self.head_dim, window_size)

        self.gate_proj = nn.Linear(input_dim, 3)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_proj(x).view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute attention outputs
        compressed_k = self.token_compressor(k)
        compressed_v = self.token_compressor(v)
        selected_k = self.token_selector(k)
        selected_v = self.token_selector(v)

        compressed_attn_output = self.sliding_window_attn(q, compressed_k, compressed_v)
        selected_attn_output = self.sliding_window_attn(q, selected_k, selected_v)
        sliding_attn_output = self.sliding_window_attn(q, k, v)

        # Learnable gating mechanism
        gates = torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        combined_output = (
            gates[..., 0] * compressed_attn_output +
            gates[..., 1] * selected_attn_output +
            gates[..., 2] * sliding_attn_output
        )

        output = self.o_proj(combined_output.view(batch_size, seq_len, -1))
        return output
