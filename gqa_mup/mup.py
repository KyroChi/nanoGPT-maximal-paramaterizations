"""
    muP implementation dicts.

    Each dict describes how to scale init, learning rate, weight decay,
    and output multipliers for the different parameter groups of a
    Transformer, as a function of model width m (or head dim d, depth L,
    GQA ratio r, etc.).

    Importable objects:
        standard_param_impl
        standard_param_impl_completep_depth_scaling
        standard_param_impl_tpvi_depth_scaling
        impl_dict          – maps string names to implementation dicts
"""

# ---------------------------------------------------------------------------
# Standard-parameterization implementations
# ---------------------------------------------------------------------------

# KV and router weights are 1.0 by default.
standard_param_impl = {
    'name':                     'SP',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d**(1/2),
    'depth_scale':              lambda L: 1.0,
}

standard_param_impl_completep_depth_scaling = {
    'name':                     'SP with Complete-P depth scaling',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d**(1/2),
    'depth_scale':              lambda L: 1 / L,
}

standard_param_impl_tpvi_depth_scaling = {
    'name':                     'SP with TP6 depth scaling',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d**(1/2),
    'depth_scale':              lambda L: 1 / L,
}

# ---------------------------------------------------------------------------
# muP implementations
# ---------------------------------------------------------------------------

# TODO: This is not the correct standard param implementation.
_standard_param_impl_legacy = {
    'name':                     'SP',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 0.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d**(1/2),
    'depth_scale':              lambda L: 1.0,
}

tpv_left_impl = {
    'name':                     'TPV-L (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 0.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 1 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: 1 / (m**(1/2) * (1 + r**(1/2))),
        'lr_scale':             lambda m, r: 1 / (m * r**(1/2)),
        'wd_scale':             lambda m, r: m * r**(1/2),
        'output_multiplier':    lambda m, r: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1 / m
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}

tpv_left_impl_no_kv = {
    'name':                     'TPV-L (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 0.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 1 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: m,
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}

tpv_left_impl_failing_hidden = {
    'name':                     'TPV-L Failing Hidden (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 0.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 0,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: m,
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}

tpv_left_impl_unit_wd = {
    'name':                     'TPV-L (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 0.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 1 / m,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: 1 / (m**(1/2) * (1 + r**(1/2))),
        'lr_scale':             lambda m, r: 1 / (m * r**(1/2)),
        'wd_scale':             lambda m, r: 1.0,
        'output_multiplier':    lambda m, r: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1 / m
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}

tpv_left_impl_new_kv = {
    'name':                     'TPV-L, new KV (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 1 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: 1 / m**(1/2),
        'lr_scale':             lambda m, r: (r + r**(1/2)) / m,
        'wd_scale':             lambda m, r: m / (r + r**(1/2)),
        'output_multiplier':    lambda m, r: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: m,
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}

tpv_left_impl_new_kv_2 = {
    'name':                     'TPV-L, new KV (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 0.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 1 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: 1 / m**(1/2),
        'lr_scale':             lambda m, r: ( 1 + r**(1/2) ) / ( m ),
        'wd_scale':             lambda m, r: ( m ) / ( 1 + r**(1/2) ),
        'output_multiplier':    lambda m, r: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1 / m
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}

tpv_left_impl_new_kv_static = {
    'name':                     'TPV-L, new KV (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 1 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: 1 / m**(1/2),
        'lr_scale':             lambda m, r: (r + r**(1/2)) / (2 * m),
        'wd_scale':             lambda m, r: 2 * m / (r + r**(1/2)),
        'output_multiplier':    lambda m, r: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1 / m
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}

# Table from Cerebras which fixed the learning rate across
# all of the layers.
tpv_right_impl = {
    'name':                     'TPV-R (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: m**(1/2),
        'lr_scale':             lambda m: 1,
        'wd_scale':             lambda m: 1,
        'output_multiplier':    lambda m: 1 / m
    },
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1 / m
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1.0,
}

# Untied weights.
# Table from IFM which fixes learning rate and ensure that
# the outputs land in bf16 range.
xllm_impl = {
    'name':                     'xLLM (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: 0,
        'output_multiplier':    lambda m: m,
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0 / m,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1.0 / L,
}

mengxi_impl = {
    'name':                     'xLLM (muP) Mengxi Candidate KV Scaling',
    'embedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: m,
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: r / m**(1/2),
        'lr_scale':             lambda m, r: 1 / m,
        'wd_scale':             lambda m, r: m,
        'output_multiplier':    lambda m, r: 1.0 / r,
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0,
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0 / m,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1.0,
}

kyle_impl = {
    'name':                     'xLLM (muP) Kyle Candidate KV Scaling',
    'embedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: 0,
        'output_multiplier':    lambda m: m,
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: 1 / ( (2**(1/2) + r**(1/2)) * (2 * m**(1/2)) ),
        'lr_scale':             lambda m, r: 1 / m,
        'wd_scale':             lambda m, r: m,
        'output_multiplier':    lambda m, r: 2 / (2**(1/2) + r**(1/2)),
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0,
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0 / m,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1.0 / L,
}

muS_impl = {
    'name':                     'muS',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0,
    },
    'hidden': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0 / m**(1/2),
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0 / m**(1/2)
    },
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0 / m,
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1.0,
}

moe_base = {
    'name':                     'TPV-L, new KV (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 0.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 1 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: 1 / m**(1/2),
        'lr_scale':             lambda m, r: (1 + r**(1/2)) / (2 * m),
        'wd_scale':             lambda m, r: 2 * m / (1 + r**(1/2)),
        'output_multiplier':    lambda m, r: 1.0
    },
    'unembedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1 / m
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}

moe_fsdp = {
    'name':                     'TPV-L, new KV (muP)',
    'embedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: 0.0,
        'output_multiplier':    lambda m: m,
    },
    'hidden': {
        'init_std':             lambda m: 1 / m**(1/2),
        'lr_scale':             lambda m: 1 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: (1 + r**(1/2)) / (2 * m**(1/2)),
        'lr_scale':             lambda m, r: 1 / m,
        'wd_scale':             lambda m, r: m,
        'output_multiplier':    lambda m, r: 2 / (1 + r**(1/2)),
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0,
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0 / m,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1 / L
}

# ---------------------------------------------------------------------------
# Lookup dict  –  maps string names to implementation dicts
# ---------------------------------------------------------------------------

impl_dict = {
    'standard_param_impl': standard_param_impl,
    'tpv_left_impl': tpv_left_impl,
    'tpv_left_impl_failing_hidden': tpv_left_impl_failing_hidden,
    'tpv_left_impl_unit_wd': tpv_left_impl_unit_wd,
    'tpv_left_impl_new_kv': tpv_left_impl_new_kv,
    'tpv_left_impl_no_kv': tpv_left_impl_no_kv,
    'tpv_left_impl_new_kv_static': tpv_left_impl_new_kv_static,
    'tpv_left_impl_new_kv_2': tpv_left_impl_new_kv_2,
    'tpv_right_impl': tpv_right_impl,
    'xllm_impl': xllm_impl,
    'mengxi_impl': mengxi_impl,
    'kyle_impl': kyle_impl,
    'muS_impl': muS_impl,
    'moe_base': moe_base,
    'moe_fsdp': moe_fsdp,
}
