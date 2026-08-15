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
    },
)
def _gdn_prefill_kernel(H, Hg, dtype, accum_dtype):
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
        with T.Kernel(batch_size * H, threads=128) as (block_id,):
            bb = block_id // H
            hh = block_id % H
            hhg = hh // (H // Hg) if H != Hg else hh

            # State persists across chunks (FP32 for accuracy)
            state_shared = T.alloc_shared((HEAD_DIM_K, HEAD_DIM_V), dtype=accum_dtype)

            # Input data in BF16
            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=dtype)
            g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)

            # BF16 scratch for GEMMs with A (both BF16)
            k_decay_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=dtype)
            v_beta_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_V), dtype=dtype)

            # FP32 scratch for GEMMs with state (both FP32)
            scratch_fp32 = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=accum_dtype)
            v_new_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_V), dtype=accum_dtype)
            scores_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)

            # Fragments (registers, FP32 accumulators)
            # Each fragment used as GEMM output must have consistent layout
            w_frag = T.alloc_fragment((CHUNK_SIZE, HEAD_DIM_K), dtype=accum_dtype)
            u_frag = T.alloc_fragment((CHUNK_SIZE, HEAD_DIM_V), dtype=accum_dtype)
            temp_frag = T.alloc_fragment((CHUNK_SIZE, HEAD_DIM_V), dtype=accum_dtype)
            out_chunk_frag = T.alloc_fragment((CHUNK_SIZE, HEAD_DIM_V), dtype=accum_dtype)
            scores_frag = T.alloc_fragment((CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype)
            state_update_frag = T.alloc_fragment((HEAD_DIM_K, HEAD_DIM_V), dtype=accum_dtype)

            # Initialize state
            T.copy(initial_state[bb, hh, :, :], state_shared)

            for chunk_idx in T.serial(num_chunks):
                left = chunk_idx * CHUNK_SIZE
                right = left + CHUNK_SIZE

                # Load chunk data with tail handling
                if right <= num_tokens:
                    T.copy(q[bb, left:right, hh, 0:HEAD_DIM_K], q_shared)
                    T.copy(k[bb, left:right, hhg, 0:HEAD_DIM_K], k_shared)
                    T.copy(A[bb, left:right, hh, 0:CHUNK_SIZE], a_shared)
                    for i in T.Parallel(CHUNK_SIZE):
                        g_shared[i] = g_cumsum[bb, left + i, hh]
                        beta_shared[i] = beta[bb, left + i, hh]
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
                            g_shared[i] = 0
                            beta_shared[i] = 0

                # K_decay = K * beta * exp(g)  -> BF16 (for GEMM with A which is BF16)
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    k_decay_shared[i, d] = T.cast(
                        T.cast(k_shared[i, d], accum_dtype)
                        * beta_shared[i]
                        * T.exp2(g_shared[i] * LOG2E),
                        dtype,
                    )

                # V_beta = V * beta  -> BF16 (for GEMM with A)
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    v_beta_shared[i, d] = T.cast(
                        T.cast(v[bb, left + i, hh, d], accum_dtype)
                        * beta_shared[i] if left + i < num_tokens else 0,
                        dtype,
                    )

                # W = A @ K_decay  (BF16 x BF16 -> FP32 frag)
                T.gemm(a_shared, k_decay_shared, w_frag, clear_accum=True)
                # Copy W to FP32 scratch for W@S GEMM
                T.copy(w_frag, scratch_fp32)

                # U = A @ V_beta  (BF16 x BF16 -> FP32 frag)
                T.gemm(a_shared, v_beta_shared, u_frag, clear_accum=True)

                # V_new = U - W @ S  (FP32 x FP32 -> FP32 frag)
                T.gemm(scratch_fp32, state_shared, temp_frag, clear_accum=True)
                # Use separate loops to avoid layout conflict between u_frag and temp_frag
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    v_new_shared[i, d] = u_frag[i, d]
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    v_new_shared[i, d] = v_new_shared[i, d] - temp_frag[i, d]

                # output_from_state = scale * exp(g) * (Q @ S)
                # Need Q in FP32 for GEMM with state (FP32)
                # Reuse scratch_fp32 (W no longer needed)
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    scratch_fp32[i, d] = T.cast(q_shared[i, d], accum_dtype)
                T.gemm(scratch_fp32, state_shared, temp_frag, clear_accum=True)
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    temp_frag[i, d] = SCALE * T.exp2(g_shared[i] * LOG2E) * temp_frag[i, d]

                # scores = Q @ K^T  (BF16 x BF16 -> FP32 frag)
                T.gemm(q_shared, k_shared, scores_frag, transpose_B=True, clear_accum=True)
                # Apply decay mask
                for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if i >= j:
                        scores_frag[i, j] = scores_frag[i, j] * T.exp2(
                        (g_shared[i] - g_shared[j]) * LOG2E)
                    else:
                        scores_frag[i, j] = 0
                T.copy(scores_frag, scores_shared)

                # output_in_chunk = scale * (scores * decay) @ V_new  (FP32 x FP32)
                T.gemm(scores_shared, v_new_shared, out_chunk_frag, clear_accum=True)

                # Write output in two steps to avoid layout conflict between temp_frag and out_chunk_frag
                # Step 1: write out_state (temp_frag) to output
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    if left + i < num_tokens:
                        output[bb, left + i, hh, d] = T.cast(temp_frag[i, d], dtype)
                # Step 2: add scaled out_chunk (out_chunk_frag) to output
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    if left + i < num_tokens:
                        output[bb, left + i, hh, d] = output[bb, left + i, hh, d] + T.cast(SCALE * out_chunk_frag[i, d], dtype)

                # State update: state = exp(g_last) * state + K_decay_last^T @ V_new
                g_last = g_shared[CHUNK_SIZE - 1]
                exp_g_last = T.exp2(g_last * LOG2E)

                # Scale state
                for d1, d2 in T.Parallel(HEAD_DIM_K, HEAD_DIM_V):
                    state_shared[d1, d2] = state_shared[d1, d2] * exp_g_last

                # K_decay_last = K * exp(g_last - g)  -> FP32 (for GEMM with V_new FP32)
                # Reuse scratch_fp32 (Q no longer needed)
                for i, d in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    scratch_fp32[i, d] = (
                        T.cast(k_shared[i, d], accum_dtype)
                        * T.exp2((g_last - g_shared[i]) * LOG2E)
                    )

                # state += K_decay_last^T @ V_new  (FP32^T x FP32 -> FP32)
                T.gemm(
                    scratch_fp32, v_new_shared, state_update_frag,
                    transpose_A=True, clear_accum=True,
                )
                for d1, d2 in T.Parallel(HEAD_DIM_K, HEAD_DIM_V):
                    state_shared[d1, d2] = state_shared[d1, d2] + state_update_frag[d1, d2]

            # Write final state
            T.copy(state_shared, final_state[bb, hh, :, :])

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

    kernel = _gdn_prefill_kernel(
        num_heads_v,
        num_heads_qk,
        dtype="bfloat16",
        accum_dtype="float32",
    )
    kernel(
        q, k, v, g_cumsum, beta, A,
        initial_state, output, final_state,
        num_chunks,
    )
    return output, final_state
