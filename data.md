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
