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
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# -------------------------
# MoE helper functions
# -------------------------

def log_mean(x, dim):
    """Compute log of mean in a numerically stable way."""
    return torch.logsumexp(x, dim=dim) - torch.log(
        torch.tensor(x.shape[dim], dtype=torch.float32, device=x.device)
    )


def entropy_reg(logits: torch.Tensor, mean_over_batch: bool = True):
    entropy_l = lambda l: -(l * l.exp()).sum(-1)
    logprobs = F.log_softmax(logits, dim=-1)
    if mean_over_batch:
        logprobs = log_mean(logprobs, 0)

    return -entropy_l(logprobs).mean()


# two losses below are adapted from
# https://github.com/google/flaxformer/blob/b725bd2a51d70e866d819c92de166fbf24425e6a/flaxformer/architectures/moe/routing.py
def load_balancing_loss(logits: torch.Tensor, expert_indices: torch.Tensor) -> float:
    _, num_experts = logits.shape

    expert_mask = F.one_hot(expert_indices, num_experts)
    expert_mask, _ = torch.max(expert_mask, dim=-2)

    tokens_per_expert = torch.mean(expert_mask.float(), dim=0, dtype=torch.float32)

    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = log_mean(logprobs, dim=0)
    router_prob_per_expert = torch.exp(logprobs)
    return (
        torch.mean( 
            tokens_per_expert * router_prob_per_expert,
            dtype=torch.float32,
        )
        * num_experts
    )


def router_z_loss(router_logits: torch.Tensor) -> float:
    """Compute router z-loss.

     The router z-loss was introduced in Designing Effective Sparse Expert Models
     (https://arxiv.org/abs/2202.08906). It encourages router logits to remain
     small in an effort to improve stability.

    Args:
      router_logits: <float>[batch_size * sequence_length, num_experts]
        router logits

    Returns:
      Scalar router z-loss.
    """
    num_tokens, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss, dtype=torch.float32) / num_tokens


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
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

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class MoE(nn.Module):
    def __init__(self, config, num_experts=None, topk=None):
        super().__init__()
        # Allow per-layer override of num_experts and topk
        if num_experts is not None:
            self.num_experts = num_experts
        else:
            self.num_experts = config.moe_num_experts
        if topk is not None:
            self.topk = max(1, min(topk, self.num_experts))
        else:
            self.topk = max(1, min(config.moe_num_experts_per_tok, self.num_experts))
        in_dim = config.n_embd
        hidden = config.moe_ffn_hidden_size if config.moe_ffn_hidden_size > 0 else 4 * config.n_embd
        E = self.num_experts
        self.config = config
        self.single_layer_experts = config.moe_single_layer_experts

        self.w_gating = nn.Linear(in_dim, E, bias=False)
        self.router_bias = nn.Parameter(torch.zeros(E))

        if self.single_layer_experts:
            # Single layer experts: shared fc1, then route to per-expert fc2
            self.fc1 = nn.Linear(in_dim, hidden, bias=config.bias)
            self.fc2_weight = nn.Parameter(torch.randn(E, hidden, in_dim) * 0.02)
            self.fc2_bias = nn.Parameter(torch.zeros(E, in_dim)) if config.bias else None
        else:
            # Standard MoE: route first, then per-expert fc1 and fc2
            self.fc1_weight = nn.Parameter(torch.randn(E, in_dim, hidden) * 0.02)
            self.fc1_bias = nn.Parameter(torch.zeros(E, hidden)) if config.bias else None
            self.fc2_weight = nn.Parameter(torch.randn(E, hidden, in_dim) * 0.02)
            self.fc2_bias = nn.Parameter(torch.zeros(E, in_dim)) if config.bias else None

        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def _forward_sparse(self, x, topk_idx, topk_vals):
        """Memory-efficient, group-by-expert + batched matmul approach."""
        B, T, C = x.shape
        k = topk_idx.size(2)
        device = x.device
        dtype = x.dtype

        flat_idx = topk_idx.reshape(-1)  # (N,) where N = B*T*k
        N = flat_idx.numel()
        x_exp = x.unsqueeze(2).expand(-1, -1, k, -1).reshape(N, C)  # (N, C)

        unique_experts, inverse = torch.unique(flat_idx, return_inverse=True)
        U = unique_experts.numel()
        if U == 0:
            return torch.zeros(B, T, k, C, device=device, dtype=dtype)

        perm = torch.argsort(inverse)
        inverse_sorted = inverse[perm]
        x_sorted = x_exp[perm]  # (N, C)

        counts = torch.bincount(inverse_sorted, minlength=U)
        max_count = int(counts.max().item())
        starts = torch.cat((torch.tensor([0], device=device, dtype=counts.dtype), counts.cumsum(0)[:-1]))
        
        # Convert to Python lists once to avoid repeated GPU-CPU syncs in loops
        counts_list = counts.cpu().tolist()
        starts_list = starts.cpu().tolist()

        x_grouped = torch.zeros(U, max_count, C, device=device, dtype=dtype)
        for i in range(U):
            cnt = counts_list[i]
            if cnt == 0:
                continue
            s = starts_list[i]
            x_grouped[i, :cnt] = x_sorted[s:s+cnt]

        weight1_u = self.fc1_weight[unique_experts]  # (U, C, H)
        if self.fc1_bias is not None:
            bias1_u = self.fc1_bias[unique_experts]  # (U, H)

        h_grouped = torch.bmm(x_grouped, weight1_u)
        if self.fc1_bias is not None:
            h_grouped = h_grouped + bias1_u.unsqueeze(1)

        H = h_grouped.size(-1)
        h_sorted_flat = torch.empty(N, H, device=device, dtype=dtype)
        for i in range(U):
            cnt = counts_list[i]
            if cnt == 0:
                continue
            s = starts_list[i]
            h_sorted_flat[s:s+cnt] = h_grouped[i, :cnt]

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N, device=device)
        h_flat = h_sorted_flat[inv_perm]
        h = h_flat.view(B, T, k, H)

        h = self.act(h)
        h = self.dropout(h)
        h_exp = h.reshape(N, H)
        h_sorted = h_exp[perm]

        weight2_u = self.fc2_weight[unique_experts]  # (U, H, C)
        if self.fc2_bias is not None:
            bias2_u = self.fc2_bias[unique_experts]  # (U, C)

        h_grouped2 = torch.zeros(U, max_count, H, device=device, dtype=dtype)
        for i in range(U):
            cnt = counts_list[i]
            if cnt == 0:
                continue
            s = starts_list[i]
            h_grouped2[i, :cnt] = h_sorted[s:s+cnt]

        out_grouped = torch.bmm(h_grouped2, weight2_u)
        if self.fc2_bias is not None:
            out_grouped = out_grouped + bias2_u.unsqueeze(1)

        out_sorted_flat = torch.empty(N, C, device=device, dtype=dtype)
        for i in range(U):
            cnt = counts_list[i]
            if cnt == 0:
                continue
            s = starts_list[i]
            out_sorted_flat[s:s+cnt] = out_grouped[i, :cnt]

        out_flat = out_sorted_flat[inv_perm]
        out_view = out_flat.view(B, T, k, C)

        # Use topk_vals directly instead of gathering from sparse_gates
        y = (out_view * topk_vals.unsqueeze(-1)).sum(dim=2)  # (B, T, C)

        return y

    def _forward_sparse_single_layer(self, h, topk_idx, topk_vals):
        """Forward pass for single layer experts: h is already fc1+activation output."""
        B, T, H = h.shape
        k = topk_idx.size(2)
        device = h.device
        dtype = h.dtype

        flat_idx = topk_idx.reshape(-1)  # (N,) where N = B*T*k
        N = flat_idx.numel()
        h_exp = h.unsqueeze(2).expand(-1, -1, k, -1).reshape(N, H)  # (N, H)

        unique_experts, inverse = torch.unique(flat_idx, return_inverse=True)
        U = unique_experts.numel()
        if U == 0:
            return torch.zeros(B, T, k, self.config.n_embd, device=device, dtype=dtype)

        perm = torch.argsort(inverse)
        inverse_sorted = inverse[perm]
        h_sorted = h_exp[perm]  # (N, H)

        counts = torch.bincount(inverse_sorted, minlength=U)
        max_count = int(counts.max().item())
        starts = torch.cat((torch.tensor([0], device=device, dtype=counts.dtype), counts.cumsum(0)[:-1]))
        
        # Convert to Python lists once to avoid repeated GPU-CPU syncs in loops
        counts_list = counts.cpu().tolist()
        starts_list = starts.cpu().tolist()

        h_grouped = torch.zeros(U, max_count, H, device=device, dtype=dtype)
        for i in range(U):
            cnt = counts_list[i]
            if cnt == 0:
                continue
            s = starts_list[i]
            h_grouped[i, :cnt] = h_sorted[s:s+cnt]

        weight2_u = self.fc2_weight[unique_experts]  # (U, H, C)
        if self.fc2_bias is not None:
            bias2_u = self.fc2_bias[unique_experts]  # (U, C)

        out_grouped = torch.bmm(h_grouped, weight2_u)
        if self.fc2_bias is not None:
            out_grouped = out_grouped + bias2_u.unsqueeze(1)

        C = out_grouped.size(-1)
        out_sorted_flat = torch.empty(N, C, device=device, dtype=dtype)
        for i in range(U):
            cnt = counts_list[i]
            if cnt == 0:
                continue
            s = starts_list[i]
            out_sorted_flat[s:s+cnt] = out_grouped[i, :cnt]

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N, device=device)
        out_flat = out_sorted_flat[inv_perm]
        out_view = out_flat.view(B, T, k, C)

        y = (out_view * topk_vals.unsqueeze(-1)).sum(dim=2)  # (B, T, C)
        return y

    def forward(self, x):
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor of shape (B, T, C)
            
        Returns:
            y: Output tensor of shape (B, T, C)
            aux_info: Dict with router_logits and aux_loss for load balancing
        """
        B, T, C = x.shape
        device = x.device
        E = self.num_experts

        if self.single_layer_experts:
            # Single layer experts: apply shared fc1 + activation first
            h = self.fc1(x)  # (B, T, H)
            h = self.act(h)
            h = self.dropout(h)
            
            # Then route based on the hidden representation (or could use x, but h makes more sense)
            # Route based on original input x for consistency with standard MoE
            logits = self.w_gating(x) + self.router_bias.view(1, 1, E)
            gates = F.softmax(logits, dim=-1)  # (B, T, E)

            importance = gates.sum(dim=(0, 1))  # (E,)
            importance_mean = importance / (B * T)
            aux_loss = (E * (importance_mean ** 2).sum()).to(device)

            topk_vals, topk_idx = gates.topk(self.topk, dim=-1)  # (B, T, topk)
            # Renormalize top-k values
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            y = self._forward_sparse_single_layer(h, topk_idx, topk_vals)
        else:
            # Standard MoE: route first, then per-expert processing
            logits = self.w_gating(x) + self.router_bias.view(1, 1, E)
            gates = F.softmax(logits, dim=-1)  # (B, T, E)

            importance = gates.sum(dim=(0, 1))  # (E,)
            importance_mean = importance / (B * T)
            aux_loss = (E * (importance_mean ** 2).sum()).to(device)

            topk_vals, topk_idx = gates.topk(self.topk, dim=-1)  # (B, T, topk)
            # Renormalize top-k values
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            y = self._forward_sparse(x, topk_idx, topk_vals)

        return y, {
            "router_logits": logits.view(-1, E),
            "selected_experts": topk_idx.view(-1, self.topk),
            "aux_loss": aux_loss,
        }


class Block(nn.Module):

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.use_moe = config.use_moe
        if config.use_moe:
            # Handle per-layer configuration for num_experts and topk
            num_experts = None
            topk = None
            if layer_idx is not None:
                # Check if moe_num_experts is a list
                if isinstance(config.moe_num_experts, (list, tuple)):
                    if layer_idx < len(config.moe_num_experts):
                        num_experts = config.moe_num_experts[layer_idx]
                    else:
                        # Fallback to last value if list is shorter than n_layer
                        num_experts = config.moe_num_experts[-1]
                # Check if moe_num_experts_per_tok is a list
                if isinstance(config.moe_num_experts_per_tok, (list, tuple)):
                    if layer_idx < len(config.moe_num_experts_per_tok):
                        topk = config.moe_num_experts_per_tok[layer_idx]
                    else:
                        # Fallback to last value if list is shorter than n_layer
                        topk = config.moe_num_experts_per_tok[-1]
            self.mlp = MoE(config, num_experts=num_experts, topk=topk)
        else:
            self.mlp = MLP(config)
        # Store aux info from last forward pass (for MoE aux loss)
        self.last_moe_aux_info = None

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.use_moe:
            mlp_out, aux_info = self.mlp(self.ln_2(x))
            x = x + mlp_out
            # Return aux_info as second return value for GPT to accumulate
            return x, aux_info
        else:
            x = x + self.mlp(self.ln_2(x))
            return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # MoE settings
    use_moe: bool = False
    moe_num_experts: int = 8  # Can be int or list[int] for per-layer configuration
    moe_num_experts_per_tok: int = 2  # Can be int or list[int] for per-layer configuration (top-k experts per token)
    moe_ffn_hidden_size: int = 0  # 0 means use 4 * n_embd (default MLP hidden size)
    moe_single_layer_experts: bool = False  # If True, use shared fc1+activation before routing, then per-expert fc2

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

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
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_moe_aux_loss(self, aux_loss_weight=0.01):
        """
        Get the auxiliary MoE load balancing loss.
        Should be called after forward pass. Returns 0 if use_moe is False.
        
        The aux loss encourages balanced expert utilization across tokens.
        
        Args:
            aux_loss_weight: Weight to apply to the auxiliary loss
            
        Returns:
            Weighted auxiliary loss (scalar tensor or 0.0)
        """
        if not self.config.use_moe:
            return 0.0
        
        # Use the accumulated aux loss from the forward pass
        if hasattr(self, '_last_aux_loss'):
            return aux_loss_weight * self._last_aux_loss
        return 0.0

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Accumulate MoE aux loss during forward pass
        total_aux_loss = 0.0
        num_moe_blocks = 0
        for block in self.transformer.h:
            if self.config.use_moe and block.use_moe:
                x, aux_info = block(x)
                if isinstance(aux_info, dict) and "aux_loss" in aux_info:
                    total_aux_loss += aux_info["aux_loss"]
                    num_moe_blocks += 1
            else:
                x = block(x)
        x = self.transformer.ln_f(x)
        
        # Store accumulated aux loss for get_moe_aux_loss()
        self._last_aux_loss = total_aux_loss / num_moe_blocks if num_moe_blocks > 0 else 0.0

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def estimate_flops_per_token(self, tokens_per_iter):
        """
        Estimate the number of flops per token processed by the model.
        This is used to estimate the throughput in TFLOPS.
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        return flops_per_token / tokens_per_iter

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
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
