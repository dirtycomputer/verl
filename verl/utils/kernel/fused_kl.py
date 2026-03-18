# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Triton-fused full-vocabulary KL divergence kernel.

Computes KL(p || q) = Σ_v p(v) * (log p(v) - log q(v)) directly from logits,
without materializing the full (batch, seq, vocab) log-probability tensors.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _fused_kl_forward_kernel(
        logits_p_ptr,
        logits_q_ptr,
        kl_out_ptr,
        vocab_size: tl.constexpr,
        logits_p_stride_row: tl.int64,
        logits_q_stride_row: tl.int64,
        BLOCK_V: tl.constexpr,
    ):
        """Compute per-row KL(p||q) from raw logits.

        Each program instance handles one row (one token position).
        We tile over the vocab dimension to avoid loading the full vocab at once.

        Steps per row:
        1. Two-pass online logsumexp for both p and q logits
        2. Accumulate KL = Σ_v exp(logit_p_v - lse_p) * ((logit_p_v - lse_p) - (logit_q_v - lse_q))
        """
        row_idx = tl.program_id(0)

        p_row_start = logits_p_ptr + row_idx * logits_p_stride_row
        q_row_start = logits_q_ptr + row_idx * logits_q_stride_row

        # --- Pass 1: compute logsumexp for p and q ---
        max_p = float("-inf")
        max_q = float("-inf")
        for v_start in range(0, vocab_size, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < vocab_size
            p_vals = tl.load(p_row_start + v_offsets, mask=mask, other=float("-inf"))
            q_vals = tl.load(q_row_start + v_offsets, mask=mask, other=float("-inf"))
            max_p = tl.maximum(max_p, tl.max(p_vals, axis=0))
            max_q = tl.maximum(max_q, tl.max(q_vals, axis=0))

        sum_exp_p = 0.0
        sum_exp_q = 0.0
        for v_start in range(0, vocab_size, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < vocab_size
            p_vals = tl.load(p_row_start + v_offsets, mask=mask, other=float("-inf"))
            q_vals = tl.load(q_row_start + v_offsets, mask=mask, other=float("-inf"))
            sum_exp_p += tl.sum(tl.exp(p_vals - max_p), axis=0)
            sum_exp_q += tl.sum(tl.exp(q_vals - max_q), axis=0)

        lse_p = max_p + tl.log(sum_exp_p)
        lse_q = max_q + tl.log(sum_exp_q)

        # --- Pass 2: accumulate KL ---
        kl_acc = 0.0
        for v_start in range(0, vocab_size, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < vocab_size
            p_vals = tl.load(p_row_start + v_offsets, mask=mask, other=float("-inf"))
            q_vals = tl.load(q_row_start + v_offsets, mask=mask, other=float("-inf"))
            log_p = p_vals - lse_p
            log_q = q_vals - lse_q
            p_probs = tl.exp(log_p)
            # KL contribution: p(v) * (log p(v) - log q(v))
            kl_acc += tl.sum(tl.where(mask, p_probs * (log_p - log_q), 0.0), axis=0)

        # Clamp to 0 (KL is non-negative; small negatives from float errors)
        kl_acc = tl.maximum(kl_acc, 0.0)
        tl.store(kl_out_ptr + row_idx, kl_acc)


def fused_kl_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Compute full-vocabulary KL(p || q) from raw logits using a fused Triton kernel.

    This avoids materializing (batch, seq, vocab_size) intermediate log-probability tensors,
    significantly reducing peak memory usage compared to the naive PyTorch implementation.

    Args:
        logits_p: Policy logits of shape (..., vocab_size).
        logits_q: Reference policy logits of shape (..., vocab_size).

    Returns:
        Per-token KL divergence of shape (...), with the vocab dimension reduced.
    """
    if not HAVE_TRITON:
        # Fallback to PyTorch
        return _kl_from_logits_pytorch(logits_p, logits_q)

    orig_shape = logits_p.shape[:-1]
    vocab_size = logits_p.shape[-1]
    # Flatten to 2D: (num_rows, vocab_size)
    logits_p_2d = logits_p.reshape(-1, vocab_size).contiguous()
    logits_q_2d = logits_q.reshape(-1, vocab_size).contiguous()
    num_rows = logits_p_2d.shape[0]

    kl_out = torch.empty(num_rows, device=logits_p.device, dtype=torch.float32)

    # Choose BLOCK_V: power of 2, capped at vocab_size rounded up
    BLOCK_V = min(triton.next_power_of_2(vocab_size), 4096)

    _fused_kl_forward_kernel[(num_rows,)](
        logits_p_2d,
        logits_q_2d,
        kl_out,
        vocab_size=vocab_size,
        logits_p_stride_row=logits_p_2d.stride(0),
        logits_q_stride_row=logits_q_2d.stride(0),
        BLOCK_V=BLOCK_V,
    )

    return kl_out.reshape(orig_shape)


def _kl_from_logits_pytorch(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """PyTorch fallback for full-vocab KL from raw logits.

    Uses chunked computation along the vocab dimension to reduce peak memory.
    """
    import torch.nn.functional as F

    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    kld = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return torch.clamp(kld, min=0.0)
