#!/usr/bin/env python3
"""
Benchmark script for tuning batch size on a single GPU.

Sweeps microbatch size for each model configuration, measures throughput
(tokens/sec) and iter time, and recommends the batch size that minimizes
total experiment wall-clock time.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --n_embd 1536 --n_layer 12
    python scripts/benchmark.py --batch_sizes 4,8,16,32,64,128
    python scripts/benchmark.py --target_tokens 500_000_000  # 500M tokens
"""

import argparse
import gc
import json
import sys
import time

import torch
import torch.nn as nn

# Add repo root to path so we can import gqa_mup
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gqa_mup.model import GPTConfig, GPT
from gqa_mup.mup import impl_dict


def benchmark_config(n_embd, n_layer, n_head, n_kv_head, batch_size,
                     block_size=1024, dtype=torch.bfloat16, compile_model=True,
                     warmup_iters=3, bench_iters=10):
    """Benchmark a single (model, batch_size) config. Returns tokens/sec or None if OOM."""
    device = 'cuda'
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        config = GPTConfig(
            n_embd=n_embd, n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head,
            block_size=block_size, vocab_size=50304, bias=False, dropout=0.0,
            mup=True, mup_multiplier=n_embd / 256, init_std=0.02,
            impl=impl_dict['tpv_left_impl_new_kv_2'],
        )
        model = GPT(config).to(device, dtype=dtype)

        if compile_model:
            model = torch.compile(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        # Synthetic data
        x = torch.randint(0, 50304, (batch_size, block_size), device=device)
        y = torch.randint(0, 50304, (batch_size, block_size), device=device)

        tokens_per_step = batch_size * block_size

        # Warmup (also triggers compile)
        for _ in range(warmup_iters):
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        sec_per_iter = elapsed / bench_iters
        tokens_per_sec = tokens_per_step / sec_per_iter
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

        # Clean up
        del model, optimizer, scaler, x, y, loss
        torch.cuda.empty_cache()
        gc.collect()

        return {
            'sec_per_iter': sec_per_iter,
            'tokens_per_sec': tokens_per_sec,
            'peak_mem_gb': peak_mem_gb,
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark batch sizes for GQA-muP models')
    parser.add_argument('--batch_sizes', type=str, default='4,8,16,32,48,64,96,128',
                        help='Comma-separated microbatch sizes to try')
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=None,
                        help='If set, only benchmark this width')
    parser.add_argument('--n_layer', type=int, default=None,
                        help='If set, only benchmark this depth')
    parser.add_argument('--compile', action='store_true', default=True)
    parser.add_argument('--no-compile', dest='compile', action='store_false')
    parser.add_argument('--warmup_iters', type=int, default=3)
    parser.add_argument('--bench_iters', type=int, default=10)
    parser.add_argument('--target_tokens', type=float, default=500e6,
                        help='Target tokens per experiment for time estimates (default: 500M)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results as JSON to this path')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    # Model configs to benchmark
    # These are the configs from the mu-transfer experiment
    if args.n_embd and args.n_layer:
        n_embd = args.n_embd
        n_layer = args.n_layer
        n_head = n_embd // 64
        models = [
            {'name': f'{n_embd}w', 'n_embd': n_embd, 'n_layer': n_layer,
             'n_head': n_head, 'n_kv_head': max(1, n_head // 3)},
        ]
    else:
        models = [
            # Proxy
            {'name': '768w',  'n_embd': 768,  'n_layer': 12, 'n_head': 12, 'n_kv_head': 4},
            # Target
            {'name': '1536w', 'n_embd': 1536, 'n_layer': 12, 'n_head': 24, 'n_kv_head': 4},
            # Also test different kv heads at target width
            {'name': '1536w_kv1',  'n_embd': 1536, 'n_layer': 12, 'n_head': 24, 'n_kv_head': 1},
            {'name': '1536w_kv12', 'n_embd': 1536, 'n_layer': 12, 'n_head': 24, 'n_kv_head': 12},
        ]

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"Compile: {args.compile}")
    print(f"Block size: {args.block_size}")
    print(f"Target tokens: {args.target_tokens:.0e}")
    print()

    all_results = {}

    for model in models:
        name = model['name']
        n_embd = model['n_embd']
        n_layer = model['n_layer']
        n_head = model['n_head']
        n_kv_head = model['n_kv_head']

        # Compute param count
        per_layer = 2*n_embd**2 + 2*n_embd*(n_kv_head*(n_embd//n_head)) + 2*n_embd*4*n_embd
        non_layer = 2*50304*n_embd + args.block_size*n_embd
        params = non_layer + n_layer * per_layer

        print(f"{'='*75}")
        print(f"Model: {name} | {n_embd}w, {n_layer}L, {n_head}h, {n_kv_head}kv | {params/1e6:.0f}M params")
        print(f"{'='*75}")
        print(f"{'batch':>6} {'sec/iter':>10} {'tok/sec':>12} {'MFU%':>7} {'mem GB':>8} {'status':>8}")
        print(f"{'-'*75}")

        # A100 BF16 peak
        a100_peak_tflops = 312e12

        model_results = []
        max_batch = None

        for bs in batch_sizes:
            result = benchmark_config(
                n_embd=n_embd, n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head,
                batch_size=bs, block_size=args.block_size, compile_model=args.compile,
                warmup_iters=args.warmup_iters, bench_iters=args.bench_iters,
            )

            if result is None:
                print(f"{bs:>6} {'':>10} {'':>12} {'':>7} {'':>8} {'OOM':>8}")
                break
            else:
                spi = result['sec_per_iter']
                tps = result['tokens_per_sec']
                mem = result['peak_mem_gb']

                # Estimate MFU
                flops_per_token = 6 * params
                achieved_flops = flops_per_token * tps
                mfu = achieved_flops / a100_peak_tflops * 100

                result['batch_size'] = bs
                result['mfu_pct'] = mfu
                model_results.append(result)
                max_batch = bs

                print(f"{bs:>6} {spi:>10.4f} {tps:>12,.0f} {mfu:>6.1f}% {mem:>7.1f}G {'OK':>8}")

        if not model_results:
            print("  All batch sizes OOM!")
            print()
            continue

        # Find optimal batch size
        # "Optimal" = highest tokens/sec, since we want to minimize wall-clock time
        # for a fixed total number of tokens
        best = max(model_results, key=lambda r: r['tokens_per_sec'])
        best_bs = best['batch_size']
        best_tps = best['tokens_per_sec']

        print()
        print(f"  >> Best throughput: batch_size={best_bs}, {best_tps:,.0f} tok/sec, "
              f"MFU={best['mfu_pct']:.1f}%, {best['sec_per_iter']:.4f} sec/iter")

        # Time estimates for this model
        print()
        print(f"  Time estimates at batch_size={best_bs}:")
        for label, total_tok in [("500M tok", 500e6), ("1B tok", 1e9), ("2B tok", 2e9)]:
            iters = total_tok / (best_bs * args.block_size)
            hours = iters * best['sec_per_iter'] / 3600
            print(f"    {label}: {iters:>8,.0f} iters, {hours:>6.1f} hours")

        # Show how grad_accum affects things
        # tokens/sec is ~constant regardless of grad_accum (it's just more microsteps)
        # So total time = target_tokens / tokens_per_sec
        total_hours = args.target_tokens / best_tps / 3600
        print(f"    {args.target_tokens:.0e} tokens: {total_hours:.1f} hours "
              f"(any grad_accum, since tok/sec is the same)")

        all_results[name] = model_results
        print()

    # Cross-model recommendation
    print("=" * 75)
    print("RECOMMENDATION: Unified batch size across all models")
    print("=" * 75)
    print()

    # Find the largest batch that works for ALL models
    # For each model, find max batch that didn't OOM
    max_per_model = {}
    for name, results in all_results.items():
        if results:
            max_per_model[name] = max(r['batch_size'] for r in results)

    if not max_per_model:
        print("No successful benchmarks!")
        return

    unified_max = min(max_per_model.values())
    print(f"Max batch that fits ALL models: {unified_max}")
    print()

    # For each candidate batch size, compute total experiment time
    # assuming we run all models with that batch size
    print(f"{'batch':>6} | ", end="")
    for name in all_results:
        print(f"{name:>14} ", end="")
    print(f"| {'total tok/s':>12} | {'time for':>10}")
    print(f"{'':>6} | ", end="")
    for name in all_results:
        print(f"{'tok/sec':>14} ", end="")
    print(f"| {'(all models)':>12} | {f'{args.target_tokens:.0e} tok':>10}")
    print("-" * (10 + 16*len(all_results) + 30))

    best_unified = None
    best_unified_time = float('inf')

    for bs in batch_sizes:
        if bs > unified_max:
            break
        # Check this bs worked for all models
        all_have = True
        total_tps = 0
        row = f"{bs:>6} | "
        for name, results in all_results.items():
            match = [r for r in results if r['batch_size'] == bs]
            if not match:
                all_have = False
                break
            tps = match[0]['tokens_per_sec']
            total_tps += tps
            row += f"{tps:>14,.0f} "

        if not all_have:
            continue

        # Total time = sum of (target_tokens / tps) for each model
        # But really we care about the SLOWEST model since experiments are sequential
        # Per-model time at this batch
        slowest_tps = min(
            next(r['tokens_per_sec'] for r in results if r['batch_size'] == bs)
            for results in all_results.values()
        )
        # For unified batch: experiment time is dominated by the slowest model
        # But total = sum across all models
        total_time_h = sum(
            args.target_tokens / next(r['tokens_per_sec'] for r in results if r['batch_size'] == bs) / 3600
            for results in all_results.values()
        )

        row += f"| {total_tps:>12,.0f} | {total_time_h:>9.1f}h"
        print(row)

        if total_time_h < best_unified_time:
            best_unified_time = total_time_h
            best_unified = bs

    print()
    if best_unified:
        print(f"  >> RECOMMENDED: batch_size={best_unified}")
        print(f"     Total time for all {len(all_results)} model configs "
              f"@ {args.target_tokens:.0e} tokens each: {best_unified_time:.1f} hours")
        print()
        print(f"  To use this in experiment configs:")
        print(f"    'batch_size': {best_unified},")
        print(f"    'gradient_accumulation_steps': <desired_effective_batch> // {best_unified},")

    # Save JSON if requested
    if args.output:
        out = {
            'gpu': gpu_name,
            'gpu_mem_gb': gpu_mem,
            'compile': args.compile,
            'block_size': args.block_size,
            'recommended_batch_size': best_unified,
            'models': {},
        }
        for name, results in all_results.items():
            out['models'][name] = results
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  Results saved to {args.output}")


if __name__ == '__main__':
    main()
