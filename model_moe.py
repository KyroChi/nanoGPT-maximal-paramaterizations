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

from mup_implementations import standard_param_impl

# -------------------------
# Utilities / small helpers
# -------------------------

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
        self.mup_multiplier = getattr(config, 'mup_multiplier', 1.0)

    def forward(self, x):
        mean = (x**2).mean(dim=-1, keepdim=True)
        normed = x / torch.sqrt(mean + self.eps)
        return self.mup_multiplier * normed

class RMSNorm(nn.Module):
    def __init__(self, config, eps=1e-3):
        super().__init__()
        ndim = config.n_embd
        self.mup_multiplier = config.mup_multiplier if hasattr(config, 'mup_multiplier') else 1
        self.weight = nn.Parameter(torch.ones(ndim) / self.mup_multiplier)
        self.eps = eps

    def forward(self, input):
        mean = (input**2).mean(dim=-1, keepdim=True)
        normed = input / torch.sqrt(mean + self.eps)
        return self.mup_multiplier * self.weight * normed

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""
    def __init__(self, config, bias=None, shape=None):
        super().__init__()
        ndim = config.n_embd
        if bias is None:
            bias = config.bias if hasattr(config, 'bias') else False
        self.mup_multiplier = config.mup_multiplier if hasattr(config, 'mup_multiplier') else 1
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Lifted from xLLM: Credit Max Ma
    bsz, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, :, None, :].expand(bsz, slen, n_kv_heads, n_rep, head_dim)
    x = x.reshape(bsz, slen, n_kv_heads * n_rep, head_dim)
    return x

# -------------------------
# Attention (unchanged)
# -------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0, f"Expected config.n_head {config.n_head} to be divisible by config.n_kv_head {config.n_kv_head}"
        self.impl = config.impl
        self.n_kv_head = config.n_kv_head
        self.n_kv_reps = config.n_head // self.n_kv_head

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd // self.n_kv_reps, bias=config.bias)
        self.c_kv.kv = True

        self.kv_capture = Capture()

        self.q_prelayer_norm = None
        if config.q_prelayer_normalization == 'LayerNorm':
            self.q_prelayer_norm = LayerNorm(config)
        elif config.q_prelayer_normalization == 'L2Norm':
            self.q_prelayer_norm = L2Norm(config)
        elif config.k_prelayer_normalization == 'L2NormScale':
            self.q_prelayer_norm = RMSNorm(config)
        elif config.q_prelayer_normalization == 'LayerNormWithBias':
            self.q_prelayer_norm = LayerNorm(config, bias=True)

        self.k_prelayer_norm = None
        if config.k_prelayer_normalization == 'LayerNorm':
            self.k_prelayer_norm = LayerNorm(config)
        elif config.k_prelayer_normalization == 'L2Norm':
            self.k_prelayer_norm = L2Norm(config)
        elif config.k_prelayer_normalization == 'L2NormScale':
            self.k_prelayer_norm = RMSNorm(config)
        elif config.k_prelayer_normalization == 'LayerNormWithBias':
            self.k_prelayer_norm = LayerNorm(config, bias=True)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self.fc_mult = self.impl['hidden']['output_multiplier'](config.mup_multiplier)

    def forward(self, x):
        B, T, C = x.size()
        q = self.fc_mult * self.c_q(x)
        k, v = ( self.fc_mult * self.c_kv(x) ).split(self.n_embd // self.n_kv_reps, dim=2)

        if self.q_prelayer_norm is not None:
            q = self.q_prelayer_norm(q)
        if self.k_prelayer_norm is not None:
            k = self.k_prelayer_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, C // self.n_head)
        v = v.view(B, T, self.n_kv_head, C // self.n_head)
        k = repeat_kv(k, self.n_kv_reps).transpose(1, 2)
        v = repeat_kv(v, self.n_kv_reps).transpose(1, 2)

        if 'kv_layer' in self.impl.keys():
            r = self.config.n_head // self.config.n_kv_head
            k = self.impl['kv_layer']['output_multiplier'](self.config.mup_multiplier, r) * k
            v = self.impl['kv_layer']['output_multiplier'](self.config.mup_multiplier, r) * v

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True,
                scale=self.impl['attention_scale'](k.size(-1))
            )
        else:
            att = (q @ k.transpose(-2, -1)) * self.impl['attention_scale'](k.size(-1))
            att = att.masked_fill(self.bias[:,:,:x.size(1),:x.size(1)] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            y = self.kv_capture(y)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout( self.fc_mult * self.c_proj(y) )
        return y

# -------------------------
# Vanilla dense MLP (kept for non-MoE)
# -------------------------

class _Expert(nn.Module):
    """Single FFN expert: GELU(FC)->FC"""
    def __init__(self, config, hidden=None):
        super().__init__()
        h = hidden if hidden is not None else 4 * config.n_embd
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

# -------------------------
# MoE: changes (1,3,4,5,6,7)
# -------------------------

class MoERouter(nn.Module):
    """
    Router with:
    - score function: sigmoid
    - pre-softmax log mapping (optional)
    - fp64 internal dtype (optional)
    - top-k selection with scaling
    - expert bias with EMA load tracking and bias update rate
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.n_embd
        self.num_experts = config.num_experts
        self.topk = config.moe_router_topk
        self.pre_softmax = config.moe_router_pre_softmax
        self.score_fn = config.moe_router_score_function
        self.scaling = config.moe_router_topk_scaling_factor
        self.enable_bias = config.moe_router_enable_expert_bias
        self.bias_update_rate = config.moe_router_bias_update_rate
        self.dtype64 = (config.moe_router_dtype.lower() == 'fp64')

        self.proj = nn.Linear(self.d_model, self.num_experts, bias=True)
        # persistent stats/bias
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        self.register_buffer("ema_load", torch.zeros(self.num_experts))
        self.ema_alpha = getattr(config, "moe_router_ema_alpha", 0.1)
        # debug flag
        self.debug = getattr(config, 'moe_debug', False)

    def _score(self, x):  # x: [N,E] pre-activation
        if self.score_fn == 'sigmoid':
            s = torch.sigmoid(x)
        else:
            s = x
        return s

    @torch.no_grad()
    def _update_bias_from_usage(self, usage_frac):
        # usage_frac: [E], sums to 1 across experts on average
        self.ema_load.mul_(1.0 - self.ema_alpha).add_(self.ema_alpha * usage_frac)
        target = 1.0 / float(self.num_experts)
        self.expert_bias.add_(self.bias_update_rate * (target - self.ema_load))

    def forward(self, x, B=None, T=None):
        """
        x: [B,T,C] flattened inside MoE layer -> we accept [N,C] here
        returns:
          topk_idx: Long[N, K]
          topk_w:   Float[N, K]  (renormalized)
          probs_full: Float[N, E]  (for aux loss / usage)
        """
        N, C = x.shape
        z = self.proj(x.to(torch.float64) if self.dtype64 else x)
        if self.dtype64:
            z = z.to(torch.float64)
        # score function
        s = self._score(z)
        if self.enable_bias:
            s = s + self.expert_bias.to(s.dtype)
        # pre-softmax mapping
        logits = torch.log(s.clamp_min(1e-12)) if self.pre_softmax else s
        logits = logits * self.scaling
        probs_full = F.softmax(logits, dim=-1)  # [N,E]

        # usage stats for bias update (expected fraction per expert)
        with torch.no_grad():
            usage = probs_full.mean(dim=0)  # [E]
            self._update_bias_from_usage(usage)
            if self.debug:
                print(f"[MoERouter] N={N} C={C} probs_full={probs_full.shape} usage_mean={usage.mean().item():.6f} usage_max={usage.max().item():.6f}", flush=True)
                print(f"[MoERouter] expert_bias min/max: {self.expert_bias.min().item():.6f}/{self.expert_bias.max().item():.6f}", flush=True)

        # top-k selection (then renormalize within selected)
        topk_w, topk_idx = probs_full.topk(self.topk, dim=-1)  # [N,K]
        denom = topk_w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        topk_w = topk_w / denom

        return topk_idx, topk_w, probs_full

class _TokenDispatcher:
    """
    Packs tokens per expert and fuses scatter/gather back to original order.
    This is a single-GPU stand-in for alltoall; it keeps the interface similar.
    """
    def __init__(self, num_experts, capacity_factor=1.25):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.debug = False

    def _capacity(self, N, K):
        # heuristic capacity per expert (can be None to disable)
        avg = math.ceil((N * K) / self.num_experts)
        cap = int(math.ceil(avg * self.capacity_factor))
        return max(cap, 1)

    def dispatch(self, x_flat, topk_idx, topk_w, capacity_masking=True):
        """
        x_flat: [N,C]; topk_idx/topk_w: [N,K]
        returns:
          expert_inputs: list of tensors (Ei, C) per expert (concatenated)
          assign_info: list of (global_indices, weights) per expert
          dropped_fraction: scalar tensor with fraction of (token,expert) pairs dropped
        """
        N, C = x_flat.shape
        K = topk_idx.size(1)
        device = x_flat.device

        # build per-expert assignment lists
        per_e_indices = [[] for _ in range(self.num_experts)]
        per_e_weights = [[] for _ in range(self.num_experts)]
        total_pairs = N * K

        # flatten all pairs for sorting by weight if we need to drop
        pairs_e = []
        for n in range(N):
            for k in range(K):
                e = int(topk_idx[n, k])
                w = float(topk_w[n, k])
                pairs_e.append((e, n, w))
        # Optionally cap capacity by expert, drop lowest-weight assignments
        dropped = 0
        if capacity_masking:
            cap = self._capacity(N, K)
            # group per expert, then keep highest weights
            buckets = [[] for _ in range(self.num_experts)]
            for e, n, w in pairs_e:
                buckets[e].append((w, n))
            for e in range(self.num_experts):
                if not buckets[e]:
                    continue
                # sort by weight descending
                buckets[e].sort(key=lambda t: t[0], reverse=True)
                kept = buckets[e][:cap]
                dropped += max(len(buckets[e]) - cap, 0)
                per_e_indices[e] = [n for (_, n) in kept]
                per_e_weights[e] = [w for (w, _) in kept]
        else:
            for e, n, w in pairs_e:
                per_e_indices[e].append(n)
                per_e_weights[e].append(w)

        # materialize expert inputs
        expert_inputs = []
        assign_info = []
        for e in range(self.num_experts):
            if len(per_e_indices[e]) == 0:
                expert_inputs.append(torch.empty(0, x_flat.size(-1), device=device, dtype=x_flat.dtype))
                assign_info.append((torch.empty(0, dtype=torch.long, device=device),
                                    torch.empty(0, dtype=x_flat.dtype, device=device)))
            else:
                idx = torch.tensor(per_e_indices[e], dtype=torch.long, device=device)
                w = torch.tensor(per_e_weights[e], dtype=x_flat.dtype, device=device)
                expert_inputs.append(x_flat.index_select(0, idx))  # (E_i, C)
                assign_info.append((idx, w))

        dropped_fraction = torch.tensor(dropped / max(total_pairs, 1), device=device, dtype=x_flat.dtype)
        if getattr(self, 'debug', False):
            counts = [len(per_e_indices[e]) for e in range(self.num_experts)]
            print(f"[_TokenDispatcher.dispatch] N={N} K={K} cap={self._capacity(N,K)} dropped={dropped} dropped_frac={dropped_fraction.item():.6f} counts={counts}", flush=True)
        return expert_inputs, assign_info, dropped_fraction

    def combine(self, expert_outputs, assign_info, N, C):
        """
        expert_outputs: list of (Ei, C) per expert
        assign_info: list of (global_idx, weights)
        returns y_flat: [N,C], combined by weighted sum over assigned experts
        """
        device = expert_outputs[0].device if expert_outputs else 'cpu'
        dtype = expert_outputs[0].dtype if expert_outputs else torch.float32
        y = torch.zeros(N, C, device=device, dtype=dtype)
        for e, (out_e, (idx, w)) in enumerate(zip(expert_outputs, assign_info)):
            if out_e.numel() == 0:
                continue
            # fused scatter-add of weighted outputs
            y.index_add_(0, idx, (out_e * w.unsqueeze(-1)).to(dtype))
        if getattr(self, 'debug', False):
            out_counts = [out.numel() for out in expert_outputs]
            print(f"[_TokenDispatcher.combine] N={N} C={C} out_counts={out_counts} result_shape={tuple(y.shape)}", flush=True)
        return y

class MoEFeedForward(nn.Module):
    """
    MoE FFN block:
      y = SharedFFN(x) + sum_{selected experts} w_i * Expert_e(x)
    Includes:
      - Router (3)
      - Token dispatch / permute fusion style pack/combine (4)
      - Capacity/masking (5)
      - Per-sequence aux loss (6)
      - Proper gradient flow through selected paths (7)
    """
    def __init__(self, config):
        super().__init__()
        assert config.num_experts > 1, "MoE requires num_experts > 1"
        self.config = config
        self.num_experts = config.num_experts
        self.k = config.moe_router_topk
        self.dropout = nn.Dropout(config.dropout)

        # Shared expert branch
        se_hidden = max(0, int(config.moe_shared_expert_intermediate_size))
        self.shared = _Expert(config, hidden=se_hidden if se_hidden > 0 else 4 * config.n_embd)

        # Experts
        e_hidden = int(config.moe_ffn_hidden_size)
        self.experts = nn.ModuleList([_Expert(config, hidden=e_hidden) for _ in range(self.num_experts)])

        # Router
        self.router = MoERouter(config)

        # Dispatcher (single-GPU simulation of alltoall)
        self.dispatcher = _TokenDispatcher(self.num_experts, capacity_factor=config.moe_capacity_factor)
        # debug flag propagation
        self.debug = getattr(config, 'moe_debug', False)
        self.router.debug = self.debug
        self.dispatcher.debug = self.debug

        # book-keeping of last computed aux loss for the block
        self.last_aux_loss = None

        # scaling (mup)
        self.fc_mult = config.impl['hidden']['output_multiplier'](config.mup_multiplier)

    def forward(self, x):
        """
        x: [B,T,C]
        returns: [B,T,C]
        side-effect: sets self.last_aux_loss (scalar tensor)
        """
        B, T, C = x.shape
        x_flat = x.reshape(B*T, C)

        # Router (3)
        topk_idx, topk_w, probs_full = self.router(x_flat, B=B, T=T)  # [N,K], [N,K], [N,E]
        if self.debug:
            print(f"[MoEFeedForward] B={B} T={T} C={C} x_flat={x_flat.shape} topk_idx={topk_idx.shape} topk_w={topk_w.shape} probs_full={probs_full.shape}", flush=True)

        # Token dispatch (4) with capacity/masking (5)
        expert_inputs, assign_info, dropped_frac = self.dispatcher.dispatch(
            x_flat, topk_idx, topk_w, capacity_masking=True
        )
        if self.debug:
            per_sizes = [ei.size(0) for ei in expert_inputs]
            print(f"[MoEFeedForward] after dispatch per_expert_input_sizes={per_sizes} dropped_frac={dropped_frac.item():.6f}", flush=True)

        # Compute expert outputs
        expert_outputs = []
        # Vectorized expert computation: pad per-expert inputs to (E, Lmax, C)
        lengths = [ei.size(0) for ei in expert_inputs]
        total_tokens = sum(lengths)
        if total_tokens == 0:
            # nothing to do
            expert_outputs = expert_inputs
        else:
            E = self.num_experts
            max_len = max(lengths)
            device = x_flat.device
            dtype = x_flat.dtype
            # padded inputs shape: (E, L, C)
            inputs_padded = torch.zeros(E, max_len, C, device=device, dtype=dtype)
            for e in range(E):
                le = lengths[e]
                if le > 0:
                    inputs_padded[e, :le] = expert_inputs[e]

            # Stack expert weights for batched computation
            # fc1: (h, C), fc2: (C, h)
            W1 = torch.stack([ex.fc1.weight for ex in self.experts]).to(dtype)
            b1 = torch.stack([ex.fc1.bias if ex.fc1.bias is not None else torch.zeros(ex.fc1.out_features, device=device, dtype=dtype) for ex in self.experts]).to(dtype)
            # hidden: (E, L, h)
            hidden = torch.einsum('elc,ehc->elh', inputs_padded, W1) + b1[:, None, :]
            hidden = F.gelu(hidden)

            W2 = torch.stack([ex.fc2.weight for ex in self.experts]).to(dtype)  # (E, C, h)
            b2 = torch.stack([ex.fc2.bias if ex.fc2.bias is not None else torch.zeros(ex.fc2.out_features, device=device, dtype=dtype) for ex in self.experts]).to(dtype)
            out_padded = torch.einsum('elh,ech->elc', hidden, W2) + b2[:, None, :]

            # Unpack per-expert outputs back to list of tensors
            for e in range(E):
                le = lengths[e]
                if le == 0:
                    expert_outputs.append(torch.empty(0, C, device=device, dtype=dtype))
                else:
                    ye = out_padded[e, :le]
                    expert_outputs.append(ye)
                if self.debug:
                    print(f"[MoEFeedForward] expert {e} input_shape={(lengths[e], C)} output_shape={(expert_outputs[-1].shape)}", flush=True)

        # Combine back (fused scatter-add) (4)
        y_moe_flat = self.dispatcher.combine(expert_outputs, assign_info, N=B*T, C=C)
        if self.debug:
            print(f"[MoEFeedForward] y_moe_flat={y_moe_flat.shape}", flush=True)

        # Shared expert branch (can overlap with dispatch in a multi-stream setup)
        y_shared = self.shared(x)

        # Aux load-balancing loss (6): per-sequence uniformity (seq_aux_loss)
        # Compute p_e(seq) using the router probabilities (before capacity drops),
        # normalized within each sequence over experts.
        with torch.no_grad():
            probs_seq = probs_full.reshape(B, T, self.num_experts).sum(dim=1)  # [B,E], sum over tokens
            probs_seq = probs_seq / probs_seq.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # p_e(seq)
        # Encourage uniformity: N * sum_e p_e^2 (minimum at uniform)
        aux_loss = (self.num_experts * (probs_seq**2).sum(dim=-1)).mean()

        # Encourage not dropping assignments (optional small term)
        aux_loss = aux_loss + dropped_frac  # tiny extra regularization

        self.last_aux_loss = aux_loss
        if self.debug:
            print(f"[MoEFeedForward] aux_loss={aux_loss.item():.6f}", flush=True)

        # Sum shared + MoE and apply dropout
        y = y_moe_flat.reshape(B, T, C) + y_shared
        y = self.dropout(self.fc_mult * y)
        return y

# -------------------------
# Transformer Block
# -------------------------

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = normalization(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = normalization(config)

        # Choose FFN: MoE if enabled, else dense MLP
        if getattr(config, "use_moe", True) and getattr(config, "num_experts", 0) and config.num_experts > 1:
            self.ffn = MoEFeedForward(config)
        else:
            self.ffn = MLP(config)

        self.depth_mult = config.impl['depth_scale'](config.n_layer) if hasattr(config, 'impl') and 'depth_scale' in config.impl else 1.0

    def forward(self, x):
        x = x + self.depth_mult * self.attn(self.ln_1(x))
        return x + self.depth_mult * self.ffn(self.ln_2(x))

# -------------------------
# Config
# -------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    mup: bool = False
    mup_multiplier: float = 1.0
    init_std: float = 0.02
    impl: dict = field(default_factory=standard_param_impl)
    normalization: str = "LayerNorm"
    q_prelayer_normalization: str = 'None'
    k_prelayer_normalization: str = 'None'
    complete_p_layers: bool = False
    use_moe: bool = True

    # -------- MoE-related knobs (1,3,4,5,6,7) ----------
    num_experts: int = 0                                 # if <=1, disables MoE
    moe_ffn_hidden_size: int = 128                       # per-expert hidden
    moe_shared_expert_intermediate_size: int = 128       # shared branch hidden
    moe_router_score_function: str = 'sigmoid'           # (3)
    moe_router_pre_softmax: bool = True                  # (3)
    moe_router_topk: int = 1                             # (3)
    moe_router_bias_update_rate: float = 0.0             # (3) set >0 to enable bias nudging
    moe_router_enable_expert_bias: bool = False          # (3)
    moe_router_topk_saling_dummy: float = 1.0            # kept for compatibility; not used
    moe_router_topk_scaling_factor: float = 1.0          # (3)
    moe_router_dtype: str = 'fp32'                       # 'fp32' or 'fp64' (3)
    moe_router_ema_alpha: float = 0.1                    # (3) smoothing for usage EMA
    moe_token_dispatcher_type: str = 'alltoall'          # (4) simulated single-GPU
    moe_capacity_factor: float = 1.25                    # (5)
    moe_router_load_balancing_type: str = 'seq_aux_loss' # (6)
    moe_aux_loss_coeff: float = 0.0                      # (6) set >0 to enable
    moe_debug: bool = False                              # enable verbose MoE debug prints

# -------------------------
# GPT model
# -------------------------

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
            h = nn.ModuleList(reversed([Block(config) for _ in range(config.n_layer)])),
            ln_f = normalization(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.emb_mult = self.impl['embedding']['output_multiplier'](config.mup_multiplier)
        self.lm_mult = self.impl['unembedding']['output_multiplier'](config.mup_multiplier)

        self.apply(self._init_weights)

        print("Total parameters: %.2fM" % (self.get_num_params(non_embedding=False)/1e6,))
        print("Total non-embedding parameters: %.2fM" % (self.get_num_params(non_embedding=True)/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            n_params -= self.lm_head.weight.numel()
        return n_params

    def _get_weight_groups(self, kv=False):
        # Collect params by type; include MoE params in 'hidden_type'
        embedding_type = [self.transformer.wte.weight, self.transformer.wpe.weight]
        hidden_type = []
        kv_type = []
        unembedding_type = [self.lm_head.weight]

        for block in reversed(self.transformer.h):
            hidden_type.append(block.attn.c_q.weight)
            kv_type.append(block.attn.c_kv.weight)
            hidden_type.append(block.attn.c_proj.weight)

            # FFN / MoE params
            if isinstance(block.ffn, MLP):
                hidden_type.append(block.ffn.c_fc.weight)
                hidden_type.append(block.ffn.c_proj.weight)
            else:
                # MoE: include router + experts + shared
                for n, p in block.ffn.named_parameters():
                    if n.endswith(".weight"):
                        hidden_type.append(p)

        for n, p in self.named_parameters():
            if "bias" in n and self.config.mup:
                raise ValueError(f"Biases are not supported in {self.impl['name']} implementation, found {n}")

        if kv:
            return embedding_type, hidden_type, kv_type, unembedding_type
        else:
            return embedding_type, hidden_type + kv_type, unembedding_type

    def _init_weights(self, module):
        if not self.config.mup:
            return
        et, ht, kv, ut = self._get_weight_groups(kv=True)
        for p in et:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['embedding']['init_std'](self.config.mup_multiplier))
        for p in kv:
            if 'kv_layer' in self.impl.keys():
                r = self.config.n_head // self.config.n_kv_head
                torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['kv_layer']['init_std'](self.config.mup_multiplier, r))
            else:
                torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['hidden']['init_std'](self.config.mup_multiplier))
        for p in ht:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['hidden']['init_std'](self.config.mup_multiplier))
        for p in ut:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['unembedding']['init_std'](self.config.mup_multiplier))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop( self.emb_mult * (tok_emb + pos_emb) )

        # accumulate MoE aux loss across blocks
        moe_aux_total = None

        for block in self.transformer.h:
            x = block(x)
            # collect any MoE aux
            if hasattr(block, "ffn") and isinstance(block.ffn, MoEFeedForward):
                aux = block.ffn.last_aux_loss
                if aux is not None:
                    moe_aux_total = (aux if moe_aux_total is None else moe_aux_total + aux)

        x = self.transformer.ln_f(x)

        logits = self.lm_mult * self.lm_head(x)

        loss = None
        if targets is not None:
            # standard token-level cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # add aux load-balancing loss
            if moe_aux_total is not None and self.config.moe_aux_loss_coeff > 0.0:
                loss = loss + self.config.moe_aux_loss_coeff * moe_aux_total

            # inference-time optimization (keep full logits by default; if you want last-token-only, uncomment)
            # logits = self.lm_mult * self.lm_head(x[:, [-1], :])

        if self.config.use_moe:
            return logits, loss, moe_aux_total
        else:
            return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device_type, adaptive_optimizer=False):
        use_fused = False
        extra_args = dict(fused=True) if use_fused else dict()

        optim_groups = []
        embedding_type, hidden_type, kv_type, unembedding_type = self._get_weight_groups(kv=True)

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

        layer_norms = [p for n, p in self.named_parameters() if 'ln_' in n]
        layer_norms_group = {
            'params': layer_norms,
            'weight_decay': 0.0,
            'lr_scale': self.impl['normalization']['lr_scale'](self.config.mup_multiplier),
            'wd_scale': 1.0
        }
        optim_groups.append(layer_norms_group)

        if adaptive_optimizer:
            from oawda import OWDAAdam
            print("Using OAWDA optimizer")
            optimizer = OWDAAdam(optim_groups, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, **extra_args)
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, **extra_args)
            print(f"using fused AdamW: {use_fused}")

        for group in optimizer.param_groups:
            group['weight_decay'] = group['wd_scale'] * group['weight_decay'] * weight_decay
            group['lr'] = group['lr_scale'] * group['lr']

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 985e12 # H200 bfloat16 peak
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
