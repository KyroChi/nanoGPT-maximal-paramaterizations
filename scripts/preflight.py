#!/usr/bin/env python3
"""Quick preflight check: torch, CUDA, imports, forward+backward pass."""
import sys

print("1. Checking torch + CUDA...")
import torch
if not torch.cuda.is_available():
    print("FAIL: CUDA not available")
    sys.exit(1)
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   CUDA: {torch.version.cuda}")
print(f"   PyTorch: {torch.__version__}")

print("2. Checking imports...")
from gqa_mup.model import GPT, GPTConfig
from gqa_mup.mup import impl_dict
print(f"   impls: {list(impl_dict.keys())}")

print("3. Forward + backward pass...")
m = GPT(GPTConfig(
    n_embd=256, n_head=4, n_kv_head=2, n_layer=3,
    vocab_size=50304, block_size=1024,
    mup=True, mup_multiplier=1.0, impl=impl_dict['gqa_mup'],
)).cuda().bfloat16()
x = torch.randint(0, 50304, (2, 1024), device='cuda')
logits, loss = m(x, x)
print(f"   forward OK, loss={loss.item():.4f}")
loss.backward()
print(f"   backward OK")

print("\nALL CHECKS PASSED")
