
import math
import torch
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr, M_ptr, P_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_mb, stride_mq,
    stride_pb, stride_pq, stride_pk,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # NOTE(chris): We want K to be transposed actually
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )

    M_block_ptr = tl.make_block_ptr(
        M_ptr + batch_index * stride_mb,
        shape=(N_QUERIES, ),
        strides=(stride_mq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, ),
    )

    P_block_ptr = tl.make_block_ptr(
        P_ptr + batch_index * stride_pb,
        shape=(N_QUERIES, N_KEYS),
        strides=(stride_pq, stride_pk),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, K_TILE_SIZE),
        order=(1, 0),
    )


    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    m_block = tl.full((Q_TILE_SIZE, ), -float("inf"), dtype=tl.float32)
    p_block = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, ), padding_option="zero")
    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):

        # We aren't sharding by D so maybe don't need that boundary check
        k = tl.load(K_block_ptr, boundary_check=(0, ), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, ), padding_option="zero")

        # The scale should be applied to the dot product, not to k directly
        attention_scores = tl.dot(q, k) * scale
        # tl.device_print("attention scores: ", attention_scores)
        # Max over the keys
        new_m_block = tl.maximum(m_block, tl.max(attention_scores, axis=1))

        p_block = tl.math.exp(attention_scores - new_m_block[:, None])
        tl.device_print("p block: ", p_block)
        alpha = tl.math.exp(m_block - new_m_block)
        l = alpha * l + tl.sum(p_block, axis=1)
        # tl.device_print("output: ", p_block)
        # l += tl.sum(p_block, axis=1)
        # tl.device_print("output after sum: ", l)

        output = alpha[:, None] * output
        p_block = tl.cast(p_block, dtype=v.dtype)
        output = tl.dot(p_block, v, output)

        # Update pointers
        m_block = new_m_block

        K_block_ptr = K_block_ptr.advance((D, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, D))

    output = output / l[:, None]
    l = m_block + tl.math.log(l)

    output = output.to(O_block_ptr.type.element_ty)
    l = l.to(L_block_ptr.type.element_ty)

    tl.store(O_block_ptr, output, boundary_check=(0, ))
    tl.store(L_block_ptr, l, boundary_check=(0, ))
    tl.store(M_block_ptr, m_block, boundary_check=(0, ))
    tl.store(P_block_ptr, p_block, boundary_check=(0, 1))
        

class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        assert Q.is_contiguous()
        assert K.is_contiguous()
        assert V.is_contiguous()

        bsz, nq, d = Q.size()
        _, nk, d = K.size()

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        O = torch.empty((bsz, nq, d), device=Q.device)
        L = torch.empty((bsz, nq), device=Q.device)
        M = torch.empty((bsz, nq), device=Q.device)
        P = torch.empty((bsz, nq, nq), device=Q.device)

        launch_grid = (
            math.ceil(nq / ctx.Q_TILE_SIZE),
            bsz
        )

        scale = 1 / d**0.5
        
        flash_fwd_kernel[launch_grid](
            Q, K, V,
            O, L, M, P,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            M.stride(0), M.stride(1),
            P.stride(0), P.stride(1), P.stride(2),
            nq, nk,
            scale,
            d,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
        )

        ctx.save_for_backward(L)

        qk = torch.einsum("... qd, ... kd->... qk", Q, K) * scale
        ref_ls = torch.logsumexp(qk, dim=-1)
        print(f"ref_ls: {ref_ls}")

        print(f"ref max: {torch.amax(qk, dim=-1, keepdim=False)}")

        print(L)
        print(M)
        print(P)
        return O


    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError