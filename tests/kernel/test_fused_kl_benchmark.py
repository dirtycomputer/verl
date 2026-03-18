"""
Benchmark and correctness test for fused full-vocab KL divergence kernel.

Tests:
1. Correctness: Triton kernel (or fallback) vs naive PyTorch reference
2. Performance: wall-clock time comparison
3. Memory: peak memory usage comparison (GPU only)

Usage:
    python tests/kernel/test_fused_kl_benchmark.py
"""

import time
import sys
import os

import torch
import torch.nn.functional as F

# Direct import to avoid verl.__init__ dependency on ray
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import importlib.util

spec = importlib.util.spec_from_file_location("fused_kl", "verl/utils/kernel/fused_kl.py")
fused_kl_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fused_kl_mod)

fused_kl_from_logits = fused_kl_mod.fused_kl_from_logits
_kl_from_logits_pytorch = fused_kl_mod._kl_from_logits_pytorch
HAVE_TRITON = fused_kl_mod.HAVE_TRITON
HAS_CUDA = torch.cuda.is_available()

print("=" * 70)
print("Environment")
print("=" * 70)
print(f"  PyTorch:      {torch.__version__}")
print(f"  CUDA:         {HAS_CUDA}")
if HAS_CUDA:
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
try:
    import triton
    print(f"  Triton:       {triton.__version__}")
except ImportError:
    print("  Triton:       Not installed")
print(f"  HAVE_TRITON:  {HAVE_TRITON}")
print(f"  Kernel mode:  {'Triton fused' if (HAVE_TRITON and HAS_CUDA) else 'PyTorch fallback'}")
print()


# ============================================================
# Helper: naive reference implementation (no optimization)
# ============================================================
def kl_naive_reference(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Naive full-vocab KL: materialize all intermediates."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)  # (B, S, V)
    log_q = F.log_softmax(logits_q.float(), dim=-1)  # (B, S, V)
    p = log_p.exp()  # (B, S, V)
    kld = (p * (log_p - log_q)).sum(dim=-1)  # (B, S)
    return torch.clamp(kld, min=0.0)


# ============================================================
# Part 1: Correctness Tests
# ============================================================
print("=" * 70)
print("Part 1: Correctness Tests")
print("=" * 70)

test_configs = [
    ("small",   2,   4,    10),
    ("medium",  4,   32,   1000),
    ("llama",   2,   128,  32000),
    ("large-v", 2,   64,   128256),
]

device = "cuda" if HAS_CUDA else "cpu"

all_correct = True
for name, B, S, V in test_configs:
    # Skip very large configs on CPU (too slow)
    if device == "cpu" and V > 32000:
        print(f"  [{name:>8s}] B={B}, S={S}, V={V:>6d}  -- SKIPPED (CPU, too large)")
        continue

    torch.manual_seed(42)
    logits_p = torch.randn(B, S, V, device=device, dtype=torch.float32)
    logits_q = torch.randn(B, S, V, device=device, dtype=torch.float32)

    kl_ref = kl_naive_reference(logits_p, logits_q)
    kl_fused = fused_kl_from_logits(logits_p, logits_q)

    max_diff = (kl_ref - kl_fused).abs().max().item()
    mean_diff = (kl_ref - kl_fused).abs().mean().item()
    is_close = torch.allclose(kl_ref, kl_fused, atol=1e-3, rtol=1e-3)

    status = "PASS" if is_close else "FAIL"
    if not is_close:
        all_correct = False
    print(
        f"  [{name:>8s}] B={B}, S={S}, V={V:>6d}  "
        f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  {status}"
    )

    # Also test identical distributions -> 0
    kl_same = fused_kl_from_logits(logits_p, logits_p)
    same_ok = torch.allclose(kl_same, torch.zeros_like(kl_same), atol=1e-5)
    if not same_ok:
        all_correct = False
        print(f"           KL(p||p) != 0! max={kl_same.max().item():.2e}  FAIL")

print(f"\n  Overall correctness: {'ALL PASSED' if all_correct else 'SOME FAILED'}")
print()


# ============================================================
# Part 2: Performance Benchmark
# ============================================================
print("=" * 70)
print("Part 2: Performance Benchmark")
print("=" * 70)

if HAS_CUDA:
    bench_configs = [
        ("small",   4,   64,   32000),
        ("medium",  8,   128,  32000),
        ("large",   16,  256,  32000),
        ("xl",      32,  512,  32000),
        ("128k-v",  4,   64,   128256),
    ]
else:
    bench_configs = [
        ("small",   2,   16,   1000),
        ("medium",  4,   32,   5000),
        ("large",   4,   64,   10000),
        ("llama",   2,   32,   32000),
    ]

warmup_iters = 3
bench_iters = 10


def bench_fn(fn, logits_p, logits_q, n_warmup, n_iter):
    """Benchmark a function, return avg time in ms."""
    if HAS_CUDA:
        torch.cuda.synchronize()
    for _ in range(n_warmup):
        _ = fn(logits_p, logits_q)
    if HAS_CUDA:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = fn(logits_p, logits_q)
    if HAS_CUDA:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iter * 1000
    return elapsed


print(f"\n  {'Config':>10s}  {'B':>3s}  {'S':>4s}  {'V':>7s}  "
      f"{'Naive(ms)':>10s}  {'Fused(ms)':>10s}  {'Speedup':>8s}")
print("  " + "-" * 62)

for name, B, S, V in bench_configs:
    torch.manual_seed(42)
    logits_p = torch.randn(B, S, V, device=device, dtype=torch.float32)
    logits_q = torch.randn(B, S, V, device=device, dtype=torch.float32)

    t_naive = bench_fn(kl_naive_reference, logits_p, logits_q, warmup_iters, bench_iters)
    t_fused = bench_fn(fused_kl_from_logits, logits_p, logits_q, warmup_iters, bench_iters)
    speedup = t_naive / t_fused if t_fused > 0 else float("inf")

    print(
        f"  {name:>10s}  {B:>3d}  {S:>4d}  {V:>7d}  "
        f"{t_naive:>10.2f}  {t_fused:>10.2f}  {speedup:>7.2f}x"
    )

print()


# ============================================================
# Part 3: Memory Benchmark (GPU only)
# ============================================================
print("=" * 70)
print("Part 3: Memory Benchmark")
print("=" * 70)

if not HAS_CUDA:
    print("\n  No GPU available. Showing theoretical memory analysis instead.\n")
    print(f"  {'Config':>10s}  {'B':>3s}  {'S':>4s}  {'V':>7s}  "
          f"{'Naive Peak':>12s}  {'Fused Peak':>12s}  {'Savings':>10s}")
    print("  " + "-" * 66)

    for name, B, S, V in [
        ("small",   4,   64,   32000),
        ("medium",  8,   128,  32000),
        ("large",   16,  256,  32000),
        ("xl",      32,  512,  32000),
        ("128k-v",  4,   64,   128256),
    ]:
        # Naive: 3 intermediates of (B, S, V) float32 = 4 bytes
        # log_p, log_q, p (or p*(log_p-log_q))
        naive_bytes = 3 * B * S * V * 4
        # Fused (Triton): only output (B, S) float32
        fused_bytes = B * S * 4
        savings_pct = (1.0 - fused_bytes / naive_bytes) * 100

        def fmt_bytes(b):
            if b >= 1e9:
                return f"{b/1e9:.2f} GB"
            elif b >= 1e6:
                return f"{b/1e6:.1f} MB"
            else:
                return f"{b/1e3:.1f} KB"

        print(
            f"  {name:>10s}  {B:>3d}  {S:>4d}  {V:>7d}  "
            f"{fmt_bytes(naive_bytes):>12s}  {fmt_bytes(fused_bytes):>12s}  {savings_pct:>9.1f}%"
        )

else:
    mem_configs = [
        ("small",   4,   64,   32000),
        ("medium",  8,   128,  32000),
        ("large",   16,  256,  32000),
    ]

    print(f"\n  {'Config':>10s}  {'B':>3s}  {'S':>4s}  {'V':>7s}  "
          f"{'Naive Peak':>12s}  {'Fused Peak':>12s}  {'Savings':>10s}")
    print("  " + "-" * 66)

    for name, B, S, V in mem_configs:
        torch.manual_seed(42)

        # Measure naive
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logits_p = torch.randn(B, S, V, device="cuda", dtype=torch.float32)
        logits_q = torch.randn(B, S, V, device="cuda", dtype=torch.float32)
        baseline_mem = torch.cuda.memory_allocated()

        _ = kl_naive_reference(logits_p, logits_q)
        torch.cuda.synchronize()
        naive_peak = torch.cuda.max_memory_allocated() - baseline_mem

        # Measure fused
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logits_p = torch.randn(B, S, V, device="cuda", dtype=torch.float32)
        logits_q = torch.randn(B, S, V, device="cuda", dtype=torch.float32)
        baseline_mem = torch.cuda.memory_allocated()

        _ = fused_kl_from_logits(logits_p, logits_q)
        torch.cuda.synchronize()
        fused_peak = torch.cuda.max_memory_allocated() - baseline_mem

        def fmt_bytes(b):
            if b >= 1e9:
                return f"{b/1e9:.2f} GB"
            elif b >= 1e6:
                return f"{b/1e6:.1f} MB"
            else:
                return f"{b/1e3:.1f} KB"

        savings = (1.0 - fused_peak / naive_peak) * 100 if naive_peak > 0 else 0
        print(
            f"  {name:>10s}  {B:>3d}  {S:>4d}  {V:>7d}  "
            f"{fmt_bytes(naive_peak):>12s}  {fmt_bytes(fused_peak):>12s}  {savings:>9.1f}%"
        )

print()
print("=" * 70)
print("Summary")
print("=" * 70)
if not HAS_CUDA:
    print("  Running on CPU - both paths use PyTorch fallback.")
    print("  Triton kernel benefits (fused log_softmax + KL) only apply on GPU.")
    print("  Theoretical memory savings shown above for reference.")
else:
    print("  Triton kernel fuses log_softmax + KL into one pass,")
    print("  avoiding (B, S, V) intermediate tensor materialization.")
print()
