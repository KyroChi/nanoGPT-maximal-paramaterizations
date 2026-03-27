"""
muP implementation dicts for GQA-muP.

Each dict describes how to scale init, learning rate, weight decay,
and output multipliers for the different parameter groups of a
Transformer, as a function of model width m (or head dim d, depth L,
GQA ratio r, etc.).

Implementations:
    sp                  – Standard parameterization (baseline, no muP)
    mup                 – muP with depth scaling (no GQA-specific KV correction)
    mup_no_kv           – muP treating KV same as hidden (ablation baseline)
    gqa_mup             – GQA-muP (ours, the paper's contribution)
    gqa_mup_alt         – GQA-muP alternative KV correction (new_kv_2)

    impl_dict           – maps string names to implementation dicts

All implementations set embedding wd_scale=0.0 (embeddings should not be
weight-decayed).
"""

# ---------------------------------------------------------------------------
# Standard parameterization (SP) — baseline, no muP scaling
# ---------------------------------------------------------------------------

standard_param_impl = {
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

# Alias
sp = standard_param_impl

# ---------------------------------------------------------------------------
# muP — with depth scaling, KV layers have a GQA-aware kv_layer group
#        This is the "tpv_left_impl" from the original codebase.
# ---------------------------------------------------------------------------

mup = {
    'name':                     'muP',
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

# ---------------------------------------------------------------------------
# muP without KV correction — treats KV weights same as other hidden weights.
#        Ablation baseline: "what if we don't correct for GQA at all?"
#        This is the "tpv_left_impl_no_kv" from the original codebase.
# ---------------------------------------------------------------------------

mup_no_kv = {
    'name':                     'muP (no KV correction)',
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

# ---------------------------------------------------------------------------
# GQA-muP — our paper's contribution.
#        KV correction: lr_scale = (r + r^½) / (2m), wd_scale = 2m / (r + r^½)
#        This is the "tpv_left_impl_new_kv_static" from the original codebase.
# ---------------------------------------------------------------------------

gqa_mup = {
    'name':                     'GQA-muP',
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

# ---------------------------------------------------------------------------
# GQA-muP (alt) — alternative KV correction factor.
#        KV correction: lr_scale = (1 + r^½) / m, wd_scale = m / (1 + r^½)
#        This is the "tpv_left_impl_new_kv_2" from the original codebase.
# ---------------------------------------------------------------------------

gqa_mup_alt = {
    'name':                     'GQA-muP (alt)',
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
        'lr_scale':             lambda m, r: (1 + r**(1/2)) / m,
        'wd_scale':             lambda m, r: m / (1 + r**(1/2)),
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

# ---------------------------------------------------------------------------
# Lookup dict — maps string names to implementation dicts
# ---------------------------------------------------------------------------

impl_dict = {
    # Clean names
    'sp':               sp,
    'mup':              mup,
    'mup_no_kv':        mup_no_kv,
    'gqa_mup':          gqa_mup,
    'gqa_mup_alt':      gqa_mup_alt,
    # Legacy aliases (so old configs still work)
    'standard_param_impl':          sp,
    'tpv_left_impl':                mup,
    'tpv_left_impl_no_kv':          mup_no_kv,
    'tpv_left_impl_new_kv_static':  gqa_mup,
    'tpv_left_impl_new_kv_2':       gqa_mup_alt,
}
