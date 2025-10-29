"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List

from mup_implementations import standard_param_impl

class Capture(nn.Module):
    def __init__(_):
        super().__init__()
    def forward(_, x: torch.Tensor):
        return x

def normalization(config):
    if config.normalization == "RMSNorm":
        return RMSNorm(config)
    elif config.normalization == "LayerNorm":
        return LayerNorm(config)
    else:
        raise ValueError(f"Unknown normalization type: {config.normalization}. Supported: 'RMSNorm', 'LayerNorm'")

class L2Norm(nn.Module):
    def __init__(self, config, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = (x**2).mean(dim=-1, keepdim=True)
        normed = x / torch.sqrt(mean + self.eps)
        return self.mup_multiplier * normed

class RMSNorm(nn.Module):
    def __init__(self, config, eps=1e-3):
        super().__init__()
        ndim = config.n_embd
        self.mup_multiplier = config.mup_multiplier if hasattr(config, 'mup_multiplier') else 1
        self.weight = nn.Parameter(
            torch.ones(ndim) / self.mup_multiplier
        )
        self.eps = eps

    def forward(self, input):
        mean = (input**2).mean(dim=-1, keepdim=True)
        normed = input / torch.sqrt(mean + self.eps)
        return self.mup_multiplier * self.weight * normed


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config, bias=None, shape=None):
        super().__init__()
        print(f"Initialized a Layer Norm", flush=True)
        ndim = config.n_embd
        if bias is None:
            bias = config.bias if hasattr(config, 'bias') else False

        self.mup_multiplier = config.mup_multiplier if hasattr(config, 'mup_multiplier') else 1

        # self.weight = nn.Parameter(
        #     torch.ones(ndim) / self.mup_multiplier
        # )
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input, 
            self.weight.shape, 
            # self.mup_multiplier * self.weight, 
            self.weight,
            self.bias, 1e-5)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Lifted from xLLM: Credit Max Ma
    bsz, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, :, None, :].expand(bsz, slen, n_kv_heads, n_rep, head_dim)
    x = x.reshape(bsz, slen, n_kv_heads * n_rep, head_dim)
    return x

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    
    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.
    
    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        
    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.
    
    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.
    
    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        
    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    
    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.
    
    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0, f"Expected config.n_head {config.n_head} to be divisible by config.n_kv_head {config.n_kv_head}"
        self.impl = config.impl
        self.n_kv_head = config.n_kv_head

        self.n_kv_reps = config.n_head // self.n_kv_head
        self.head_dim = config.n_embd // config.n_head

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # n_embd / n_kv_reps = n_kv_head * head_size
        self.c_k = nn.Linear(config.n_embd, config.n_embd // self.n_kv_reps, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd // self.n_kv_reps, bias=config.bias)
        # self.c_kv.kv = True

        self.kv_capture = Capture()

        # RoPE configuration
        self.use_rope = getattr(config, 'use_rope', False)
        if self.use_rope:
            # Precompute RoPE frequencies
            self.register_buffer(
                "freqs_cis", 
                precompute_freqs_cis(self.head_dim, config.block_size, theta=getattr(config, 'rope_theta', 10000.0)),
                persistent=False
            )

        self.q_prelayer_norm = nn.Identity()
        print(f"Config: q prelayer_norm: {config.q_prelayer_normalization}, k prelayer_norm: {config.k_prelayer_normalization}")
        if config.q_prelayer_normalization == 'LayerNorm':
            self.q_prelayer_norm = LayerNorm(config)
        elif config.q_prelayer_normalization == 'L2Norm':
            self.q_prelayer_norm = L2Norm(config)
        elif config.k_prelayer_normalization == 'L2NormScale':
            self.q_prelayer_norm = RMSNorm(config)
        elif config.q_prelayer_normalization == 'LayerNormWithBias':
            self.q_prelayer_norm = LayerNorm(config, bias=True)

        self.k_prelayer_norm = nn.Identity()
        if config.k_prelayer_normalization == 'LayerNorm':
            self.k_prelayer_norm = LayerNorm(config)
        elif config.k_prelayer_normalization == 'L2Norm':
            self.k_prelayer_norm = L2Norm(config)
        elif config.k_prelayer_normalization == 'L2NormScale':
            self.k_prelayer_norm = RMSNorm(config)
        elif config.k_prelayer_normalization == 'LayerNormWithBias':
            self.k_prelayer_norm = LayerNorm(config, bias=True)
            

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self.fc_mult = self.impl['hidden']['output_multiplier'](config.mup_multiplier)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.fc_mult * self.c_q(x)
        k = self.fc_mult * self.c_k(x)
        v = self.fc_mult * self.c_v(x)
        # k, v = ( self.fc_mult * self.c_kv(x) ).split(self.n_embd // self.n_kv_reps, dim=2)

        q = self.q_prelayer_norm(q)        
        k = self.k_prelayer_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        k = k.view(B, T, self.n_kv_head, C // self.n_head) # (B, T, n_kvh, hs)
        v = v.view(B, T, self.n_kv_head, C // self.n_head) # (B, T, n_kvh, hs)

        k = repeat_kv(k, self.n_kv_reps).transpose(1, 2) # (B, nh, T, hs)
        v = repeat_kv(v, self.n_kv_reps).transpose(1, 2) # (B, nh, T, hs)

        # Apply RoPE if enabled
        if self.use_rope:
            freqs_cis = self.freqs_cis[:T]
            q, k = apply_rotary_emb(q, k, freqs_cis)

        if 'kv_layer' in self.impl.keys():
            r = self.config.n_head // self.config.n_kv_head
            k = self.impl['kv_layer']['output_multiplier'](self.config.mup_multiplier, r) * k
            v = self.impl['kv_layer']['output_multiplier'](self.config.mup_multiplier, r) * v

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True,
                scale=self.impl['attention_scale'](k.size(-1))
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * self.impl['attention_scale'](k.size(-1)) # (B, nh, T, T)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

            att = att

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

            y = self.kv_capture(y)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = y 

        # output projection
        y = self.resid_dropout( self.fc_mult * self.c_proj(y) )
        return y

class _Expert(nn.Module):
    """Single FFN expert: GELU(FC)->FC"""
    def __init__(self, config):
        super().__init__()
        h = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, h,  bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h,            config.n_embd, bias=config.bias)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.fc_mult = config.impl['hidden']['output_multiplier'](config.mup_multiplier)

    def forward(self, x):
        x = self.fc_mult * self.c_fc(x)
        x = self.gelu(x)
        x = self.fc_mult * self.c_proj(x)
        x = self.dropout(x)
        return x

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.topk = max(1, min(config.router_topk, self.num_experts))
        in_dim = config.n_embd
        hidden = config.moe_ffn_hidden_size
        E = self.num_experts
        self.config = config

        self.w_gating = nn.Linear(in_dim, E, bias=False)
        self.router_bias = nn.Parameter(torch.zeros(E))
        self.null_expert_bias = config.moe_null_expert_bias

        self.fc1_weight = nn.Parameter(torch.randn(E, in_dim, hidden))
        self.fc1_bias = nn.Parameter(torch.zeros(E, hidden)) if config.bias else None
        self.fc2_weight = nn.Parameter(torch.randn(E, hidden, in_dim))
        self.fc2_bias = nn.Parameter(torch.zeros(E, in_dim)) if config.bias else None

        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def _forward_sparse(self, x, topk_idx, gates):
        B, T, C = x.shape
        k = topk_idx.size(2)

        # Memory-efficient, group-by-expert + batched matmul approach.
        device = x.device
        dtype = x.dtype

        # Flatten token-expert selections and corresponding inputs
        flat_idx = topk_idx.reshape(-1)                  # (N,) where N = B*T*k
        N = flat_idx.numel()
        x_exp = x.unsqueeze(2).expand(-1, -1, k, -1).reshape(N, C)  # (N, C)

        # Find unique experts and inverse mapping (maps each position -> 0..U-1)
        unique_experts, inverse = torch.unique(flat_idx, return_inverse=True)
        U = unique_experts.numel()
        if U == 0:
            return torch.zeros(B, T, k, C, device=device, dtype=dtype)

        # Sort positions by expert to make contiguous groups
        perm = torch.argsort(inverse)
        inverse_sorted = inverse[perm]
        x_sorted = x_exp[perm]  # (N, C)

        # Counts per expert and grouping boundaries
        counts = torch.bincount(inverse_sorted, minlength=U)
        max_count = int(counts.max().item())
        starts = torch.cat((torch.tensor([0], device=device, dtype=counts.dtype), counts.cumsum(0)[:-1]))

        # Pack inputs into (U, max_count, C) with zero-padding
        x_grouped = torch.zeros(U, max_count, C, device=device, dtype=dtype)
        for i in range(U):
            cnt = int(counts[i].item())
            if cnt == 0:
                continue
            s = int(starts[i].item())
            x_grouped[i, :cnt] = x_sorted[s:s+cnt]

        # Gather per-expert parameters once (U is small)
        weight1_u = self.fc1_weight[unique_experts]  # (U, C, H)
        if self.fc1_bias is not None:
            bias1_u = self.fc1_bias[unique_experts]      # (U, H)

        # Batched matmul: (U, max_count, C) x (U, C, H) -> (U, max_count, H)
        h_grouped = torch.bmm(x_grouped, weight1_u)
        h_grouped *= self.config.impl['moe_ffn_1']['output_multiplier'](self.config.mup_multiplier, self.config.moe_ffn_mup_multiplier, self.config.num_experts, self.topk)
        if self.fc1_bias is not None:
            h_grouped.add_(bias1_u.unsqueeze(1))  # (U, max_count, H)

        # Unpack grouped outputs back to sorted-flat layout
        h_sorted_flat = torch.empty(N, h_grouped.size(-1), device=device, dtype=dtype)
        for i in range(U):
            cnt = int(counts[i].item())
            if cnt == 0:
                continue
            s = int(starts[i].item())
            h_sorted_flat[s:s+cnt] = h_grouped[i, :cnt]

        # Unpermute to original flat order and reshape to (B, T, k, H)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N, device=device)
        h_flat = h_sorted_flat[inv_perm]
        H = h_flat.size(-1)
        h = h_flat.view(B, T, k, H)

        # Activation + dropout
        h = self.act(h)
        h = self.dropout(h)

        # Second projection: follow same grouped-batched strategy
        # Flatten h for second projection
        h_exp = h.reshape(N, H)
        h_sorted = h_exp[perm]

        weight2_u = self.fc2_weight[unique_experts]  # (U, H, C)
        if self.fc2_bias is not None:
            bias2_u = self.fc2_bias[unique_experts]      # (U, C)

        # Pack h_sorted into (U, max_count, H)
        h_grouped2 = torch.zeros(U, max_count, H, device=device, dtype=dtype)
        for i in range(U):
            cnt = int(counts[i].item())
            if cnt == 0:
                continue
            s = int(starts[i].item())
            h_grouped2[i, :cnt] = h_sorted[s:s+cnt]

        # Batched matmul: (U, max_count, H) x (U, H, C) -> (U, max_count, C)
        out_grouped = torch.bmm(h_grouped2, weight2_u)
        out_grouped *= self.config.impl['moe_ffn_2']['output_multiplier'](self.config.mup_multiplier, self.config.moe_ffn_mup_multiplier, self.config.num_experts, self.topk)
        if self.fc2_bias is not None:
            out_grouped.add_(bias2_u.unsqueeze(1))  # (U, max_count, C)

        # Unpack back to flat sorted -> unpermute -> reshape
        out_sorted_flat = torch.empty(N, C, device=device, dtype=dtype)
        for i in range(U):
            cnt = int(counts[i].item())
            if cnt == 0:
                continue
            s = int(starts[i].item())
            out_sorted_flat[s:s+cnt] = out_grouped[i, :cnt]

        out_flat = out_sorted_flat[inv_perm]
        out_view = out_flat.view(B, T, k, C)

        # Multiply per-position outputs by corresponding gate values and scatter-add into final output y
        y = torch.zeros(B, T, C, device=device, dtype=out_view.dtype)
        gate_vals = gates.gather(2, topk_idx)               # (B, T, k)
        y = (out_view * gate_vals.unsqueeze(-1)).sum(dim=2)  # (B, T, C)

        return y

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        device = x.device
        E = self.num_experts

        logits = self.w_gating(x) + self.router_bias.view(1, 1, E)
        print(f"Gating logits: mean {logits.mean().item():.4f}, std {logits.std().item():.4f}, min {logits.min().item():.4f}, max {logits.max().item():.4f}")
        gates = F.softmax(logits, dim=-1)  # (B, T, E)

        # Compute aux loss
        importance = gates.sum(dim=(0, 1))  # (E,)
        importance_mean = importance / (B * T)
        if self.config.mup:
            aux = (E**2 * (importance_mean ** 2).sum()).to(device)
        else:
            aux = (E * (importance_mean ** 2).sum()).to(device)
        # aux = ((importance_mean ** 2).sum()).to(device)

        if self.config.moe_random_router:
            logits = torch.rand_like(logits)
            gates = logits

        topk_vals, topk_idx = gates.topk(self.topk, dim=-1)  # (B,T,topk)
        if self.config.mup:
            topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + self.null_expert_bias)
        sparse_gates = torch.zeros_like(gates)
        sparse_gates.scatter_(2, topk_idx, topk_vals)
        gates = sparse_gates

        y = self._forward_sparse(x, topk_idx, gates)

        return y, aux


class Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln_1 = normalization(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = normalization(config)

        # choose ffn implementation (MLP or MoE)
        # first layer (layer_idx=0) is always dense MLP, rest can be MoE
        self.use_moe = getattr(config, 'use_moe', False) and (layer_idx != 0)
        if self.use_moe:
            self.ffn = MoE(config)
        else:
            self.ffn = MLP(config)

        self.depth_mult = config.impl['depth_scale'](config.n_layer) if hasattr(config, 'impl') and 'depth_scale' in config.impl else 1.0

    def forward(self, x):
        x = x + self.depth_mult * self.attn(self.ln_1(x))
        if self.use_moe:
            ffn_out = self.ffn(self.ln_2(x))
            if isinstance(ffn_out, tuple):
                # MoE returns (output, aux_loss)
                y, aux = ffn_out
            else:
                y, aux = ffn_out, 0.0
            return x + self.depth_mult * y, aux
        else:
            # Fast path for non-MoE: no tuple unpacking overhead
            y = self.ffn(self.ln_2(x))
            return x + self.depth_mult * y
@dataclass                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    mup: bool = False
    mup_multiplier: float = 1.0
    init_std: float = 0.02
    impl: dict = field(default_factory=standard_param_impl) # implementation details, see standard_param_impl and tpv_left_impl
    normalization: str = "LayerNorm"
    q_prelayer_normalization: str = 'None'
    k_prelayer_normalization: str = 'None'
    complete_p_layers: bool = False # if True, the depth multiplier is 1/n_layer, otherwise 1.0
    # RoPE configuration
    use_rope: bool = False # Enable/disable Rotary Position Embedding
    rope_theta: float = 10000.0 # RoPE theta parameter for frequency scaling
    # MoE configuration
    use_moe: bool = False
    router_topk: int = 8
    num_experts: int = 64
    moe_seq_aux_loss_coeff: float = 1e-4
    moe_ffn_hidden_size: int = 128
    moe_ffn_mup_multiplier: float = 1.0
    moe_null_expert_bias: float = 0.0
    moe_random_router: bool = False
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.impl = config.impl if hasattr(config, 'impl') else standard_param_impl

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(reversed([Block(config, layer_idx=config.n_layer - 1 - i) for i in range(config.n_layer)])),
            ln_f = normalization(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.emb_mult = self.impl['embedding']['output_multiplier'](config.mup_multiplier)
        self.lm_mult = self.impl['unembedding']['output_multiplier'](config.mup_multiplier)

        # KYLE: Untie final layer weights
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # self._init_weights(None)
    
        # report number of parameters
        print("Total parameters: %.2fM" % (self.get_num_params(non_embedding=False)/1e6,))
        print("Total non-embedding parameters: %.2fM" % (self.get_num_params(non_embedding=True)/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            n_params -= self.lm_head.weight.numel() 
        return n_params

    def _get_weight_groups(self, kv=False):
        # TODO: NO BIAS
        embedding_type = [self.transformer.wte.weight, 
                          self.transformer.wpe.weight]
        hidden_type = []
        kv_type = []
        unembedding_type = [self.lm_head.weight]
        router_type = []
        router_bias_type = []
        moe_ffn_type_1 = []
        moe_ffn_type_2 = []
        layer_norms = []

        for block in reversed(self.transformer.h):
            hidden_type.append(block.attn.c_q.weight)
            kv_type.append(block.attn.c_k.weight)
            kv_type.append(block.attn.c_v.weight)
            hidden_type.append(block.attn.c_proj.weight)
            # handle different FFN types (MLP or MoE)
            if hasattr(block, 'ffn'):
                ffn = block.ffn
                if isinstance(ffn, MLP):
                    hidden_type.append(ffn.c_fc.weight)
                    hidden_type.append(ffn.c_proj.weight)
                else:
                    # include gating network params
                    if hasattr(ffn, 'w_gating'):
                        router_type.append(ffn.w_gating.weight)
                        if hasattr(ffn.w_gating, 'bias') and ffn.w_gating.bias is not None:
                            router_bias_type.append(ffn.w_gating.bias)
                    # router bias
                    if hasattr(ffn, 'router_bias'):
                        router_bias_type.append(ffn.router_bias)
                    # expert weights (stacked tensors)
                    if hasattr(ffn, 'fc1_weight'):
                        moe_ffn_type_1.append(ffn.fc1_weight)
                    if hasattr(ffn, 'fc1_bias') and ffn.fc1_bias is not None:
                        print(ffn.fc1_bias)
                        raise ValueError("FC biases are not supported in MoE implementation")
                    if hasattr(ffn, 'fc2_weight'):
                        moe_ffn_type_2.append(ffn.fc2_weight)
                    if hasattr(ffn, 'fc2_bias') and ffn.fc2_bias is not None:
                        print(ffn.fc2_bias)
                        raise ValueError("FC biases are not supported in MoE implementation")
            else:
                # backwards compatibility with attribute name 'mlp'
                if hasattr(block, 'mlp'):
                    hidden_type.append(block.mlp.c_fc.weight)
                    hidden_type.append(block.mlp.c_proj.weight)

        for n, p in self.named_parameters():
            if ("bias" in n) and (self.config.mup) and \
                (p is not None) and \
                ('router' not in n) and \
                ('gating' not in n):
                raise ValueError(f"Biases are not supported in {self.impl['name']} implementation, found {n}")

        if kv:
            return embedding_type, hidden_type, kv_type, unembedding_type, router_type, router_bias_type, moe_ffn_type_1, moe_ffn_type_2
        else:
            return embedding_type, hidden_type + kv_type, unembedding_type, router_type, router_bias_type, moe_ffn_type_1, moe_ffn_type_2
        
    def _init_weights(self, module):
        if not self.config.mup:
            return
            
        n0 = self.config.n_embd / self.config.mup_multiplier
        et, ht, kv, ut, rt, rbt, mft1, mft2 = self._get_weight_groups(kv=True)
        for p in et:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['embedding']['init_std'](self.config.mup_multiplier))

        for p in kv:
            target_mult = 1.0

            if 'kv_layer' in self.impl.keys():
                r = self.config.n_head // self.config.n_kv_head
                torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['kv_layer']['init_std'](self.config.mup_multiplier, r))
            else:
                torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['hidden']['init_std'](self.config.mup_multiplier))

        for p in ht:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['hidden']['init_std'](self.config.mup_multiplier))

        for p in ut:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['unembedding']['init_std'](self.config.mup_multiplier))

        for p in rt:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['router']['init_std'](self.config.n_embd, self.config.num_experts))

        for p in rbt:
            torch.nn.init.zeros_(p)

        for p in mft1:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['moe_ffn_1']['init_std'](self.config.n_embd, self.config.moe_ffn_hidden_size, self.config.num_experts, self.config.router_topk))
        
        for p in mft2:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['moe_ffn_2']['init_std'](self.config.n_embd, self.config.moe_ffn_hidden_size, self.config.num_experts, self.config.router_topk))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop( self.emb_mult * (tok_emb + pos_emb) )


        # iterate through blocks and accumulate MoE aux losses if present
        if getattr(self.config, 'use_moe', False):
            # MoE path: handle aux losses
            moe_aux = torch.tensor(0.0, device=device)
            for block in self.transformer.h:
                out = block(x)
                if isinstance(out, tuple):
                    x, aux = out
                    if isinstance(aux, torch.Tensor):
                        moe_aux = moe_aux + aux
                    else:
                        moe_aux = moe_aux + torch.tensor(float(aux), device=device)
                else:
                    x = out
        else:
            # Fast path for non-MoE: no aux loss overhead
            moe_aux = None
            for block in self.transformer.h:
                x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits_all = self.lm_mult * self.lm_head(x)
            lm_loss = F.cross_entropy(logits_all.view(-1, logits_all.size(-1)), targets.view(-1), ignore_index=-1)
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_mult * self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            # combine language loss with MoE aux loss if MoE is enabled
            if getattr(self.config, 'use_moe', False) and moe_aux is not None:
                loss = lm_loss + self.config.moe_seq_aux_loss_coeff * moe_aux
            else:
                loss = lm_loss
        else:
            logits = self.lm_mult * self.lm_head(x[:, [-1], :])
            loss = None

        if getattr(self.config, 'use_moe', False):
            return logits, loss, moe_aux if moe_aux is not None else torch.tensor(0.0, device=device), lm_loss
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:  
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device_type, adaptive_optimizer=False, enable_fsdp=False):
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = False # fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        if enable_fsdp:
            # With FSDP, we need to use a different approach since parameters are flattened
            # We'll create parameter groups based on parameter names rather than parameter objects
            def should_exclude_from_decay(param_name):
                # Exclude embedding layers from weight decay
                return 'wte' in param_name or 'wpe' in param_name or 'ln_' in param_name or 'bias' in param_name
            
            decay_params = []
            no_decay_params_list = []
            
            for name, param in self.named_parameters():
                if should_exclude_from_decay(name):
                    no_decay_params_list.append(param)
                else:
                    decay_params.append(param)
            
            # Apply hidden layer learning rate scaling to both groups
            hidden_lr_scale = self.impl['hidden']['lr_scale'](self.config.mup_multiplier)
            
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate * hidden_lr_scale},
                {'params': no_decay_params_list, 'weight_decay': 0.0, 'lr': learning_rate * hidden_lr_scale}
            ]
            
            if adaptive_optimizer:
                # use the OWDA optimizer, which is an AdamW variant with dynamic weight decay
                from oawda import OWDAAdam
                print("Using OAWDA optimizer with FSDP")
                optimizer = OWDAAdam(optim_groups, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, **extra_args)
            else:
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, **extra_args)
                print(f"using fused AdamW with FSDP: {use_fused}, hidden_lr_scale: {hidden_lr_scale:.6f}")
            
            return optimizer
        
        # Original MuP parameter grouping approach for non-FSDP
        optim_groups = []
        embedding_type, hidden_type, kv_type, unembedding_type, router_type, router_bias_type, moe_ffn_type1, moe_ffn_type2 = self._get_weight_groups(kv=True)

        et_group = {}
        et_group['params'] = embedding_type
        et_group['lr_scale'] = self.impl['embedding']['lr_scale'](self.config.mup_multiplier)
        et_group['wd_scale'] = self.impl['embedding']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(et_group)

        ht_group = {}
        ht_group['params'] = hidden_type
        ht_group['lr_scale'] = self.impl['hidden']['lr_scale'](self.config.mup_multiplier)
        ht_group['wd_scale'] = self.impl['hidden']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(ht_group)

        kv_group = {}
        kv_group['params'] = kv_type
        if 'kv_layer' in self.impl.keys():
            r = self.config.n_head // self.config.n_kv_head
            kv_group['lr_scale'] = self.impl['kv_layer']['lr_scale'](self.config.mup_multiplier, r)
            kv_group['wd_scale'] = self.impl['kv_layer']['wd_scale'](self.config.mup_multiplier, r)
        else:
            kv_group['lr_scale'] = self.impl['hidden']['lr_scale'](self.config.mup_multiplier)
            kv_group['wd_scale'] = self.impl['hidden']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(kv_group)

        ut_group = {}
        ut_group['params'] = unembedding_type
        ut_group['lr_scale'] = self.impl['unembedding']['lr_scale'](self.config.mup_multiplier)
        ut_group['wd_scale'] = self.impl['unembedding']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(ut_group)

        rt_group = {}
        if len(router_type) > 0:
            rt_group['params'] = router_type
            if 'router' in self.impl:
                router_group_dict = self.impl['router']
                rt_group['lr_scale'] = router_group_dict['lr_scale'](self.config.n_embd, self.config.num_experts)
                rt_group['wd_scale'] = router_group_dict['wd_scale'](self.config.n_embd, self.config.num_experts)
            else:
                router_group_dict = self.impl['hidden']
                rt_group['lr_scale'] = router_group_dict['lr_scale'](self.config.mup_multiplier)
                rt_group['wd_scale'] = router_group_dict['wd_scale'](self.config.mup_multiplier)
            optim_groups.append(rt_group)

        rbt_group = {}
        if len(router_bias_type) > 0:
            rbt_group['params'] = router_bias_type
            if 'router' in self.impl:
                router_bias_group_dict = self.impl['router_bias']
                rbt_group['lr_scale'] = router_bias_group_dict['lr_scale'](self.config.mup_multiplier, self.config.num_experts)
                rbt_group['wd_scale'] = router_bias_group_dict['wd_scale'](self.config.mup_multiplier, self.config.num_experts) 
            else:
                router_bias_group_dict = self.impl['hidden']    
                rbt_group['lr_scale'] = router_bias_group_dict['lr_scale'](self.config.mup_multiplier)
                rbt_group['wd_scale'] = router_bias_group_dict['wd_scale'](self.config.mup_multiplier)
            optim_groups.append(rbt_group)

        mft_group_1 = {}
        if len(moe_ffn_type1) > 0:
            mft_group_1['params'] = moe_ffn_type1
            if 'moe_ffn_1' in self.impl:
                print("using moe_ffn_1 param impl")
                moe_ffn_group_dict = self.impl['moe_ffn_1']
                mft_group_1['lr_scale'] = moe_ffn_group_dict['lr_scale'](
                    self.config.mup_multiplier, 
                    self.config.moe_ffn_mup_multiplier,
                    self.config.num_experts,
                    self.config.router_topk
                )
                mft_group_1['wd_scale'] = moe_ffn_group_dict['wd_scale'](
                    self.config.mup_multiplier, 
                    self.config.moe_ffn_mup_multiplier,
                    self.config.num_experts,
                    self.config.router_topk
                )
            else:
                moe_ffn_group_dict = self.impl['hidden']
                mft_group_1['lr_scale'] = moe_ffn_group_dict['lr_scale'](self.config.mup_multiplier)
                mft_group_1['wd_scale'] = moe_ffn_group_dict['wd_scale'](self.config.mup_multiplier)
            optim_groups.append(mft_group_1)

        mft_group_2 = {}
        if len(moe_ffn_type2) > 0:
            mft_group_2['params'] = moe_ffn_type2
            if 'moe_ffn_2' in self.impl:
                print(f"Using separate moe_ffn_2 param group with impl {self.impl['name']}")
                moe_ffn_group_dict = self.impl['moe_ffn_2']
                mft_group_2['lr_scale'] = moe_ffn_group_dict['lr_scale'](
                    self.config.mup_multiplier, 
                    self.config.moe_ffn_mup_multiplier,
                    self.config.num_experts,
                    self.config.router_topk
                )
                mft_group_2['wd_scale'] = moe_ffn_group_dict['wd_scale'](
                    self.config.mup_multiplier, 
                    self.config.moe_ffn_mup_multiplier,
                    self.config.num_experts,
                    self.config.router_topk
                )
            else:
                moe_ffn_group_dict = self.impl['hidden']
                mft_group_2['lr_scale'] = moe_ffn_group_dict['lr_scale'](self.config.mup_multiplier)
                mft_group_2['wd_scale'] = moe_ffn_group_dict['wd_scale'](self.config.mup_multiplier)
            optim_groups.append(mft_group_2)

        layer_norms = [p for n, p in self.named_parameters() if 'ln_' in n]
        layer_norms_group = {
            'params': layer_norms,
            'weight_decay': 0.0,  # no weight decay for layer norms
            'lr_scale': self.impl['normalization']['lr_scale'](self.config.mup_multiplier),      # no scaling for layer norms
            'wd_scale': 1.0       # no scaling for layer norms
        }
        optim_groups.append(layer_norms_group)

        if adaptive_optimizer:
            # use the OWDA optimizer, which is an AdamW variant with dynamic weight decay
            from oawda import OWDAAdam
            print("Using OAWDA optimizer")
            optimizer = OWDAAdam(optim_groups, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, **extra_args)
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, **extra_args)
            print(f"using fused AdamW: {use_fused}")
            # optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)

        # Apply MuP scaling (only for non-FSDP case, FSDP case returns early)
        for group in optimizer.param_groups:
            group['weight_decay'] = group['wd_scale'] * group['weight_decay'] * weight_decay
            group['lr'] = group['lr_scale'] * group['lr']

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        This version provides a different estimate when using a Mixture-of-Experts (MoE):
        - For non-MoE models we keep the original heuristic.
        - For MoE models we replace the standard FFN cost with the cost of executing
          `router_topk` experts per token (each expert does two linear projections).
        """
        # basic sizes
        N = self.get_num_params()
        cfg = self.config
        L = cfg.n_layer
        H = cfg.n_head
        Q = cfg.n_embd // cfg.n_head
        T = cfg.block_size

        # original coarse PaLM-style heuristic
        flops_per_token_orig = 6 * N + 12 * L * H * Q * T

        use_moe = getattr(cfg, 'use_moe', False)
        if not use_moe:
            flops_per_fwdbwd = flops_per_token_orig * T
            flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        else:
            # Estimate attention + other non-FFN work by subtracting a heuristic
            # standard FFN cost from the original term, then add MoE cost.
            # Standard MLP hidden size is typically 4 * n_embd.
            n_embd = cfg.n_embd
            std_ffn_hidden = 4 * n_embd
            # heuristic standard FFN flops per token per layer (two matmuls: n_embd->hidden and hidden->n_embd)
            std_ffn_flops_per_token_per_layer = 2 * n_embd * std_ffn_hidden
            std_ffn_total = L * std_ffn_flops_per_token_per_layer

            # approximate the non-FFN component contained in the original heuristic
            attention_and_other = max(flops_per_token_orig - std_ffn_total, 0.0)

            # MoE cost: each selected expert does two matmuls of sizes (n_embd x moe_hidden) and (moe_hidden x n_embd)
            topk = max(1, min(getattr(cfg, 'router_topk', 1), getattr(cfg, 'num_experts', 1)))
            moe_hidden = getattr(cfg, 'moe_ffn_hidden_size', std_ffn_hidden)
            moe_ffn_flops_per_token_per_layer = topk * (2 * n_embd * moe_hidden)
            moe_ffn_total = L * moe_ffn_flops_per_token_per_layer

            # final per-token cost: non-FFN work + MoE-executed FFN work + parameter-related term (6*N)
            flops_per_token = 6 * N + attention_and_other + moe_ffn_total
            flops_per_fwdbwd = flops_per_token * T
            flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # express our flops throughput as ratio of H200 bf16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        # H200 bf16 peak FLOPS
        flops_promised = 1979e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from 