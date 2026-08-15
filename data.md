# Lab3 GDN Prefill Optimization Log

## Baseline (v7, commit 05020da)
Date: 2026-08-15
Status: ALL PASS (8/8)

| Case | B | T | Hq | Hv | State | Latency (ms) |
|------|---|---|----|----|-------|-------------|
| short_tail_state | 1 | 1025 | 2 | 8 | yes | 0.458 |
| chain_equal | 1 | 8192 | 4 | 4 | no | 3.083 |
| parallel_equal | 1 | 2048 | 16 | 16 | no | 1.635 |
| parallel_gva | 1 | 2048 | 4 | 16 | no | 1.690 |
| long_low_gva | 1 | 32768 | 2 | 8 | no | 13.081 |
| batch_split_gva | 4 | 8192 | 2 | 8 | no | 10.214 |
| wide_gva_state | 1 | 8192 | 16 | 64 | yes | 17.632 |
| deep_gva_state | 1 | 16384 | 8 | 32 | yes | 20.564 |

Notes: First fully working version. Fixed layout conflict (T.copy instead of T.Parallel for fragments), GVA Q-head expansion, tail chunk g_shared padding.
Threads: 128, Shared mem: ~219KB

---

## Iteration History

| Round | Commit | Optimization | Status | Key Latency Change |
|-------|--------|-------------|--------|-------------------|
| 0 | 05020da | Baseline | PASS 8/8 | - |
| 1 | 6230b2a | threads 128->256 | PASS 8/8 | ~20% faster all cases |
| 2 | 1eefb33 | T.copy V loading | PASS 8/8 | within noise |

### v8 Detail (threads=256)
short_tail_state: 0.364ms (-20.5%), chain_equal: 2.397ms (-22.3%), parallel_equal: 1.309ms (-19.9%)
parallel_gva: 1.369ms (-19.0%), long_low_gva: 10.301ms (-21.3%), batch_split_gva: 8.170ms (-20.0%)
wide_gva_state: 14.217ms (-19.4%), deep_gva_state: 16.427ms (-20.1%)

### v9 Detail (T.copy V loading)
short_tail_state: 0.362, chain_equal: 2.387, parallel_equal: 1.320, parallel_gva: 1.374
long_low_gva: 10.329, batch_split_gva: 7.988, wide_gva_state: 14.185, deep_gva_state: 16.421
No significant change, keeping for code cleanliness.

## v10 (commit 2842b8f, merge kv_scratch_shared)
Date: 2026-08-15
Status: ALL PASS (8/8)
Optimization: Merged k_decay_shared and v_beta_shared into single kv_scratch_shared buffer (saves 16KB shared mem)

| Case | Latency (ms) | vs v9 |
|------|-------------|-------|
| short_tail_state | 0.365 | +0.8% |
| chain_equal | 2.416 | +1.2% |
| parallel_equal | 1.316 | -0.3% |
| parallel_gva | 1.362 | -0.9% |
| long_low_gva | 10.302 | -0.3% |
| batch_split_gva | 8.057 | +0.9% |
| wide_gva_state | 14.115 | -0.5% |
| deep_gva_state | 16.616 | +1.2% |

Notes: Within noise. Merged buffer saves 16KB shared mem but doesn't change occupancy (still 1 block/SM).

## v11 (commit 1006002, precompute exp_g + inv_exp_g) - FAILED
Date: 2026-08-15
Status: ALL FAIL (0/8) - Compilation error
Error: "A and B must have the same dtype" during JIT compilation
Cause: Likely dtype inference issue with 1.0/eg division or shared memory access in fragment operations
Action: Reset to v10, trying conservative version (v11b)

## v11b (commit 130e58b, precompute exp_g only)
Date: 2026-08-15
Status: TESTING
Optimization: Precompute exp_g once, use in K_decay, output_from_state, state decay.
Keep decay_mask and K_decay_last using T.exp2 (they involve g differences).

## v11b Results (precompute exp_g only)
Status: ALL PASS (8/8)

| Case | Latency (ms) | vs v10 | vs baseline |
|------|-------------|-------|-------------|
| short_tail_state | 0.362 | -0.8% | -21.0% |
| chain_equal | 2.472 | +2.3% | -19.8% |
| parallel_equal | 1.288 | -2.1% | -21.2% |
| parallel_gva | 1.371 | +0.7% | -18.9% |
| long_low_gva | 10.130 | -1.7% | -22.6% |
| batch_split_gva | 7.809 | -3.1% | -23.5% |
| wide_gva_state | 13.816 | -2.1% | -21.6% |
| deep_gva_state | 16.166 | -2.7% | -21.4% |

Notes: Larger cases benefit more (more chunks = more exp2 savings). chain_equal slightly worse.
Overall ~1% improvement vs v10, ~21% vs baseline.

## v16 (commit 79af99e, fuse output writes) - SUCCESS
Date: 2026-08-15
Status: ALL PASS (8/8)
Optimization: Fused two separate output writes (output_from_state + output_in_chunk) into a single pass
- Eliminated T.copy(temp_frag, scratch_fp32) after Q@state GEMM
- Eliminated T.copy(out_chunk_frag, scratch_fp32) after scores@V_new GEMM
- Eliminated one global memory write (output_from_state was written then read back for +=)
- Combined: output = scale * (exp(g) * Q@state + scores@V_new) in one T.Parallel loop

| Case | v13 (ms) | v16 (ms) | vs v13 | vs baseline |
|------|----------|----------|--------|-------------|
| short_tail_state | 0.359 | 0.353 | -1.7% | -22.9% |
| chain_equal | 2.449 | 2.361 | -3.6% | -23.4% |
| parallel_equal | 1.282 | 1.278 | -0.3% | -21.8% |
| parallel_gva | 1.356 | 1.319 | -2.7% | -21.9% |
| long_low_gva | 10.015 | 9.978 | -0.4% | -23.7% |
| batch_split_gva | 7.726 | 7.670 | -0.7% | -24.9% |
| wide_gva_state | 13.688 | 13.513 | -1.3% | -23.4% |
| deep_gva_state | 15.969 | 15.793 | -1.1% | -23.2% |

Notes: chain_equal and parallel_gva benefited most (>2%). Geometric mean ~1.4% vs v13, ~23% vs baseline.
## v17 (commit df86047, BF16 v_new + scores + fuse state decay) - SUCCESS, MAJOR WIN
Date: 2026-08-15
Status: ALL PASS (8/8)
Optimization: Convert v_new_shared and scores_shared from FP32 to BF16
- scores@V_new GEMM: FP32 x FP32 -> BF16 x BF16 (tensor core ~2x faster)
- K_decay_last^T @ V_new GEMM: FP32^T x FP32 -> BF16^T x BF16 (tensor core ~2x faster)
- K_decay_last now uses kv_scratch_shared (BF16) instead of scratch_fp32 (FP32)
- Fused state decay with state update (saves one pass over 128x128 state_shared)
- Eliminated T.copy(u_frag, v_new_shared), combined subtraction+cast
- Combined scores causal mask + dtype cast into single loop
- Added T.copy(temp_frag, scratch_fp32) before fused output (layout conflict fix)
- Saved 24KB shared memory (16KB v_new + 8KB scores)

| Case | v16 (ms) | v17 (ms) | vs v16 | vs baseline |
|------|----------|----------|--------|-------------|
| short_tail_state | 0.353 | 0.302 | -14.4% | -34.1% |
| chain_equal | 2.361 | 1.916 | -18.9% | -37.8% |
| parallel_equal | 1.278 | 1.066 | -16.6% | -34.8% |
| parallel_gva | 1.319 | 1.111 | -15.8% | -34.2% |
| long_low_gva | 9.978 | 8.516 | -14.7% | -34.9% |
| batch_split_gva | 7.670 | 6.589 | -14.1% | -35.5% |
| wide_gva_state | 13.513 | 11.727 | -13.2% | -33.5% |
| deep_gva_state | 15.793 | 13.603 | -13.9% | -33.9% |

Notes: Geometric mean ~15.3% faster than v16, ~34.9% faster than baseline.
Only 2 FP32 GEMMs remain (W@state, Q@state). All other GEMMs are now BF16.

---
[Report Round 13]
时间：2026-08-15 15:36
本轮优化方向：BF16 conversion of v_new_shared + scores_shared, fused state decay+update
测试结果：成功 | [延迟: 0.302-13.603 ms] | [相比基线提升: ~35%]
当前最优版本 Commit ID: df86047
---
继续执行下一轮...
## v18 (commit f077670, BF16 state GEMMs) - SUCCESS, MAJOR WIN
Date: 2026-08-15
Status: ALL PASS (8/8)
Optimization: Add BF16 state copy (state_bf16) for W@state and Q@state GEMMs
- W@state: FP32 x FP32 -> BF16 x BF16 (W cast to kv_scratch_shared BF16)
- Q@state: FP32 x FP32 -> BF16 x BF16 (q_shared used directly, no Q cast!)
- Eliminated Q FP32 cast loop (saved 8192 element-wise ops per chunk)
- Added state_bf16 refresh per chunk (16384 elements)
- Net: ALL 7 GEMMs now BF16, zero FP32 GEMMs remaining
- Shared mem: +32KB state_bf16, total ~209KB (under 228KB limit)

| Case | v17 (ms) | v18 (ms) | vs v17 | vs baseline |
|------|----------|----------|--------|-------------|
| short_tail_state | 0.302 | 0.267 | -11.6% | -41.7% |
| chain_equal | 1.916 | 1.689 | -11.8% | -45.2% |
| parallel_equal | 1.066 | 0.929 | -12.8% | -43.2% |
| parallel_gva | 1.111 | 0.978 | -12.0% | -42.1% |
| long_low_gva | 8.516 | 7.417 | -12.9% | -43.3% |
| batch_split_gva | 6.589 | 5.886 | -10.7% | -42.4% |
| wide_gva_state | 11.727 | 10.527 | -10.2% | -40.3% |
| deep_gva_state | 13.603 | 11.787 | -13.4% | -42.7% |

Notes: Geometric mean ~12% faster than v17, ~42.7% faster than baseline.
ALL GEMMs are now BF16. Remaining optimization targets: element-wise ops, T.copy, memory access.
## v19 (commit 99e7f9a, fuse state_bf16 refresh with state update) - SUCCESS
Date: 2026-08-15
Status: ALL PASS (8/8)
Optimization: Moved state_bf16 refresh from start-of-chunk to end-of-chunk, fused with state update
- Saves one pass over 128x128 state_shared per chunk
- Initial state_bf16 cast done once before the loop

| Case | v18 (ms) | v19 (ms) | vs v18 | vs baseline |
|------|----------|----------|--------|-------------|
| short_tail_state | 0.267 | 0.259 | -3.0% | -43.5% |
| chain_equal | 1.689 | 1.632 | -3.4% | -47.1% |
| parallel_equal | 0.929 | 0.928 | -0.1% | -43.3% |
| parallel_gva | 0.978 | 0.965 | -1.3% | -42.9% |
| long_low_gva | 7.417 | 7.156 | -3.5% | -45.3% |
| batch_split_gva | 5.886 | 5.640 | -4.2% | -44.8% |
| wide_gva_state | 10.527 | 10.139 | -3.7% | -42.5% |
| deep_gva_state | 11.787 | 11.692 | -0.8% | -43.1% |

Notes: Long-sequence cases benefit most (more chunks = more saved passes).
## v20 (commit 8a3a4b2, precompute beta * exp_g) - SUCCESS
Date: 2026-08-15
Status: ALL PASS (8/8)
Optimization: Precompute beta * exp_g into beta_exp_g_shared, use in K_decay
- Saves one multiply per element in K_decay computation
- Adds 256 bytes shared memory (negligible)

| Case | v19 (ms) | v20 (ms) | vs v19 | vs baseline |
|------|----------|----------|--------|-------------|
| short_tail_state | 0.259 | 0.256 | -1.2% | -43.9% |
| chain_equal | 1.632 | 1.588 | -2.7% | -48.5% |
| parallel_equal | 0.928 | 0.904 | -2.6% | -44.7% |
| parallel_gva | 0.965 | 0.976 | +1.1% | -42.2% |
| long_low_gva | 7.156 | 7.122 | -0.5% | -45.6% |
| batch_split_gva | 5.640 | 5.620 | -0.4% | -45.0% |
| wide_gva_state | 10.139 | 10.104 | -0.3% | -42.7% |
| deep_gva_state | 11.692 | 11.619 | -0.6% | -43.5% |

Notes: Geometric mean ~0.9% improvement. Some cases >2% faster, one case slightly slower (noise).

## v23 (commit 0cbd94c, GEMM output to shared) - FAIL (compile error)
Date: 2026-08-15
Status: FAIL 0/8 - "local_buf must be a fragment"
Optimization: T.gemm output directly to scratch_fp32 shared memory
- TileLang requires T.gemm output to be a fragment, not shared memory
- Reverted

## v23b (commit 63b1727, v_beta_shared + direct W cast) - SUCCESS, MAJOR WIN
Date: 2026-08-15
Status: ALL PASS (8/8)
Optimization: Decouple V_beta from kv_scratch_shared using dedicated v_beta_shared
- T.copy(w_frag, kv_scratch_shared) with cross-dtype FP32->BF16 conversion
- Eliminates T.copy(w_frag, scratch_fp32) + cast loop = ~64KB less traffic per chunk
- Cost: +16KB shared (v_beta_shared), total ~225KB (under 228KB limit)

| Case | v21b (ms) | v23b (ms) | vs v21b | vs baseline |
|------|----------|-----------|--------|-------------|
| short_tail_state | 0.256 | 0.248 | -3.1% | -45.9% |
| chain_equal | 1.579 | 1.478 | -6.4% | -52.0% |
| parallel_equal | 0.904 | 0.841 | -7.0% | -48.6% |
| parallel_gva | 0.965 | 0.923 | -4.4% | -45.4% |
| long_low_gva | 7.092 | 6.915 | -2.5% | -47.2% |
| batch_split_gva | 5.597 | 5.446 | -2.7% | -46.7% |
| wide_gva_state | 10.110 | 9.575 | -5.3% | -45.7% |
| deep_gva_state | 11.671 | 10.990 | -5.8% | -46.6% |

Notes: Geometric mean ~4.7% faster than v21b, ~46.6% faster than baseline.
Key discovery: T.copy supports cross-dtype (FP32->BF16) conversion!
