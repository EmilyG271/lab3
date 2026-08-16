import torch
import tilelang
import tilelang.language as T


CHUNK_SIZE = 64
HEAD_DIM_K = 128
HEAD_DIM_V = 128
LOG2E = 1.4426950408889634
SCALE = HEAD_DIM_K ** -0.5


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 1,
    },
)
def _gdn_prefill_kernel(H, Hg, split_v, dtype, accum_dtype):
    head_dim_v_split = HEAD_DIM_V // split_v
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")

    qk_shape = (batch_size, num_tokens, H, HEAD_DIM_K)
    k_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    v_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)
    output_shape = (batch_size, num_tokens, H, HEAD_DIM_V)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=dtype),
        k: T.Tensor(k_shape, dtype=dtype),
        v: T.Tensor(v_shape, dtype=dtype),
        g_cumsum: T.Tensor(gate_shape, dtype=accum_dtype),
        beta: T.Tensor(gate_shape, dtype=accum_dtype),
        A: T.Tensor(a_shape, dtype=dtype),
        initial_state: T.Tensor(state_shape, dtype=accum_dtype),
        output: T.Tensor(output_shape, dtype=dtype),
        final_state: T.Tensor(state_shape, dtype=accum_dtype),
        num_chunks: T.int32,
    ):
        with T.Kernel(batch_size * H * split_v, threads=256) as (block_id,):
            bb = block_id // (H * split_v)
            rest = block_id % (H * split_v)
            hh = rest // split_v
            vv = rest % split_v
            v_offset = vv * head_dim_v_split
            hhg = hh // (H // Hg) if H != Hg else hh

           # State persists across chunks (FP32 for accuracy)
            state_frag = T.alloc_fragment((HEAD_DIM_K, head_dim_v_split), dtype=accum_dtype)
            state_bf16 = T.alloc_shared((HEAD_DIM_K, head_dim_v_split), dtype=dtype)

            # Input data in BF16
            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=dtype)
            g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)

            # Precomputed exp(g) to avoid redundant exp2 calls
            exp_g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            inv_exp_g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)

            # K_decay/K_decay_last + K_state/V_beta scratch
            if split_v > 1:
                k_decay_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
                kv_scratch_shared = T.alloc_shared((CHUNK_SIZE, head_dim_v_split), dtype=dtype)
            else:
                kv_scratch_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
                k_decay_shared = kv_scratch_shared
            # V_beta buffer
            v_beta_shared = T.alloc_shared((CHUNK_SIZE, head_dim_v_split), dtype=dtype)
            # Double-buffered V: separate prefetch buffer to give V prefetch the whole chunk to complete
            v_next_shared = T.alloc_shared((CHUNK_SIZE, head_dim_v_split), dtype=dtype)

            # Alias v_new_shared with v_beta_shared (v_beta_shared is free after V*beta loop)
            v_new_shared = v_beta_shared
            scores_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=dtype)

            # Fragments (registers, FP32 accumulators)
            u_frag = T.alloc_fragment((CHUNK_SIZE, head_dim_v_split), dtype=accum_dtype)
            out_chunk_frag = T.alloc_fragment((CHUNK_SIZE, head_dim_v_split), dtype=accum_dtype)
            scores_frag = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)
            q_state_frag = T.alloc_fragment((CHUNK_SIZE, head_dim_v_split), dtype=accum_dtype)

            # Initialize state and BF16 copy
            for d1, d2 in T.Parallel(HEAD_DIM_K, head_dim_v_split):
                state_frag[d1, d2] = initial_state[bb, hh, d1, v_offset + d2]
            T.copy(state_frag, state_bf16)

            for chunk_idx in T.serial(num_chunks):
                left = chunk_idx * CHUNK_SIZE
                right = left + CHUNK_SIZE

                # Compute next chunk bounds early for staggered prefetches
                next_left = (chunk_idx + 1) * CHUNK_SIZE
                next_right = next_left + CHUNK_SIZE
                has_next = next_right <= num_tokens

                # Load chunk data with tail handling
                if right <= num_tokens:
                    if chunk_idx == 0:
                        T.async_copy(q[bb, left:right, hh, 0:HEAD_DIM_K], q_shared)
                        T.async_copy(k[bb, left:right, hhg, 0:HEAD_DIM_K], k_shared)
                        T.async_copy(A[bb, left:right, hh, 0:CHUNK_SIZE], a_shared)
                        T.async_copy(v[bb, left:right, hh, v_offset:v_offset + head_dim_v_split], v_beta_shared)
                        for i in T.Parallel(CHUNK_SIZE):
                            g_shared[i] = g_cumsum[bb, left + i, hh]
                            beta_shared[i] = beta[bb, left + i, hh]
                            exp_g_shared[i] = T.exp2(g_shared[i] * LOG2E)
                            inv_exp_g_shared[i] = T.exp2(-g_shared[i] * LOG2E)
                    else:
                        # g/beta prefetched from previous chunk, compute exp_g/inv_exp_g only
                        for i in T.Parallel(CHUNK_SIZE):
                            exp_g_shared[i] = T.exp2(g_shared[i] * LOG2E)
                            inv_exp_g_shared[i] = T.exp2(-g_shared[i] * LOG2E)
                else:
                    for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                        if left + i < num_tokens:
                            q_shared[i, d] = q[bb, left + i, hh, d]
                            k_shared[i, d] = k[bb, left + i, hhg, d]
                        else:
                            q_shared[i, d] = 0
                            k_shared[i, d] = 0
                    for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        if left + i < num_tokens:
                            a_shared[i, j] = A[bb, left + i, hh, j]
                        else:
                            a_shared[i, j] = 0
                    for i in T.Parallel(CHUNK_SIZE):
                        if left + i < num_tokens:
                            g_shared[i] = g_cumsum[bb, left + i, hh]
                            beta_shared[i] = beta[bb, left + i, hh]
                        else:
                            # Pad g with last valid cumsum so g_last is correct
                            g_shared[i] = g_cumsum[bb, num_tokens - 1, hh]
                            beta_shared[i] = 0
                        exp_g_shared[i] = T.exp2(g_shared[i] * LOG2E)
                        inv_exp_g_shared[i] = T.exp2(-g_shared[i] * LOG2E)

                # exp(g) precomputed in g/beta load loop above

                # Wait for async global memory loads (non-tail case only)
                if right <= num_tokens:
                    T.ptx_wait_group(0)
                    if split_v == 1:
                        # Double-buffered V: copy prefetched V from v_next_shared to working buffer
                        if chunk_idx > 0:
                            T.copy(v_next_shared, v_beta_shared)
                        # Issue V prefetch early (into v_next_shared, has whole chunk to complete)
                        if has_next:
                            T.async_copy(v[bb, next_left:next_right, hh, v_offset:v_offset + head_dim_v_split], v_next_shared)

                # Issue Q@K^T, K@state, and Q@state concurrently (GEMM 4 has no dependency on GEMMs 1+2)
                T.wgmma_gemm(q_shared, k_shared, scores_frag, transpose_B=True, clear_accum=True)
                T.wgmma_gemm(k_shared, state_bf16, u_frag, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
                T.wgmma_gemm(q_shared, state_bf16, q_state_frag, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
                T.warpgroup_arrive()
                T.warpgroup_commit_batch()
                # Scale state by exp_g_last while WGMMA executes (register-only, no conflict)
                exp_g_last = exp_g_shared[CHUNK_SIZE - 1]
                for d1, d2 in T.Parallel(HEAD_DIM_K, head_dim_v_split):
                    state_frag[d1, d2] = state_frag[d1, d2] * exp_g_last
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand(scores_frag)
                T.warpgroup_fence_operand(u_frag)
                T.warpgroup_fence_operand(q_state_frag)
                # Fused: scale K_state by beta*exp_g and compute V*beta - K_state directly
                if right <= num_tokens:
                    for i, d in T.Parallel(CHUNK_SIZE, head_dim_v_split):
                        kv_scratch_shared[i, d] = T.cast(
                            T.cast(v_beta_shared[i, d], accum_dtype) * beta_shared[i]
                            - u_frag[i, d] * beta_shared[i] * exp_g_shared[i],
                            dtype,
                        )
                else:
                    for i, d in T.Parallel(CHUNK_SIZE, head_dim_v_split):
                        if left + i < num_tokens:
                            kv_scratch_shared[i, d] = T.cast(
                                T.cast(v[bb, left + i, hh, v_offset + d], accum_dtype) * beta_shared[i]
                                - u_frag[i, d] * beta_shared[i] * exp_g_shared[i],
                                dtype,
                            )
                        else:
                            kv_scratch_shared[i, d] = 0

                # Issue GEMM 3 async (A@kv -> u_frag), q_state already computed
                T.wgmma_gemm(a_shared, kv_scratch_shared, u_frag, clear_accum=True)
                T.warpgroup_arrive()
                T.warpgroup_commit_batch()

                # Scale scores while GEMM 3 executes (scores_frag ready, no conflict with kv_scratch)
                for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if i >= j:
                        scores_shared[i, j] = T.cast(
                            scores_frag[i, j] * exp_g_shared[i] * inv_exp_g_shared[j],
                            dtype,
                        )
                    else:
                        scores_shared[i, j] = 0

                # Wait for GEMM 3
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand(u_frag)

                # Copy V_new to shared (GEMM 3 done, GEMM 4 still running)
                T.copy(u_frag, v_new_shared)

                # Prefetch A, Q, g, beta (all buffers free, overlaps with GEMM 4)
                if has_next:
                    T.async_copy(A[bb, next_left:next_right, hh, 0:CHUNK_SIZE], a_shared)
                    T.async_copy(q[bb, next_left:next_right, hh, 0:HEAD_DIM_K], q_shared)
                    T.async_copy(g_cumsum[bb, next_left:next_right, hh], g_shared)
                    T.async_copy(beta[bb, next_left:next_right, hh], beta_shared)

                # K_decay (exp_g_last already computed, state already scaled)
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    k_decay_shared[i, d] = T.cast(
                        T.cast(k_shared[i, d], accum_dtype)
                        * exp_g_last * inv_exp_g_shared[i],
                        dtype,
                    )

                # Issue GEMMs 5+6 concurrently
                T.wgmma_gemm(scores_shared, v_new_shared, out_chunk_frag, clear_accum=True)
                T.wgmma_gemm(k_decay_shared, v_new_shared, state_frag, transpose_A=True, clear_accum=False)
                T.warpgroup_arrive()
                T.warpgroup_commit_batch()

                # Prefetch K (overlaps with all async GEMMs)
                if has_next:
                    T.async_copy(k[bb, next_left:next_right, hhg, 0:HEAD_DIM_K], k_shared)

                # Wait for all GEMMs (groups 2 and 3)
                T.warpgroup_wait(0)
                T.warpgroup_fence_operand(q_state_frag)
                T.warpgroup_fence_operand(out_chunk_frag)
                T.warpgroup_fence_operand(state_frag)

                # Output write (all accumulators ready)
                if right <= num_tokens:
                    for i, d in T.Parallel(CHUNK_SIZE, head_dim_v_split):
                        output[bb, left + i, hh, v_offset + d] = T.cast(
                            SCALE * (exp_g_shared[i] * q_state_frag[i, d] + out_chunk_frag[i, d]),
                            dtype,
                        )
                else:
                    for i, d in T.Parallel(CHUNK_SIZE, head_dim_v_split):
                        if left + i < num_tokens:
                            output[bb, left + i, hh, v_offset + d] = T.cast(
                                SCALE * (exp_g_shared[i] * q_state_frag[i, d] + out_chunk_frag[i, d]),
                                dtype,
                            )
                # For split_v=2: V prefetch at end of chunk (old approach, no T.copy overhead)
                if split_v == 2:
                    if has_next:
                        T.async_copy(v[bb, next_left:next_right, hh, v_offset:v_offset + head_dim_v_split], v_beta_shared)
                # Refresh state_bf16 from state_frag
                T.copy(state_frag, state_bf16)

            # Write final state
            for d1, d2 in T.Parallel(HEAD_DIM_K, head_dim_v_split):
                final_state[bb, hh, d1, v_offset + d2] = state_frag[d1, d2]

    return kernel


def gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_tokens, num_heads_qk, _ = q.shape
    num_heads_v = v.shape[2]
    num_chunks = tilelang.cdiv(num_tokens, CHUNK_SIZE)

    # Expand Q heads for GVA (one Q/K head maps to multiple V heads)
    if num_heads_qk != num_heads_v:
        q = q.repeat_interleave(num_heads_v // num_heads_qk, dim=2)

    if initial_state is None:
        initial_state = torch.zeros(
            (batch_size, num_heads_v, HEAD_DIM_K, HEAD_DIM_V),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        initial_state = initial_state.to(torch.float32)

    output = torch.empty(
        (batch_size, num_tokens, num_heads_v, HEAD_DIM_V),
        dtype=torch.bfloat16,
        device=q.device,
    )
    final_state = torch.empty(
        (batch_size, num_heads_v, HEAD_DIM_K, HEAD_DIM_V),
        dtype=torch.float32,
        device=q.device,
    )

    split_v = 2 if batch_size * num_heads_v <= 4 else 1
    kernel = _gdn_prefill_kernel(
        num_heads_v,
        num_heads_qk,
        split_v,
        dtype="bfloat16",
        accum_dtype="float32",
    )
    kernel(
        q, k, v, g_cumsum, beta, A,
        initial_state, output, final_state,
        num_chunks,
    )
    return output, final_state
