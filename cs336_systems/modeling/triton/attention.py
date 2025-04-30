
import math
import torch
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
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


    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    m_block = tl.full((Q_TILE_SIZE, ), -float("inf"), dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, ), padding_option="zero")
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):

        # We aren't sharding by D so maybe don't need that boundary check
        k = tl.load(K_block_ptr, boundary_check=(0, ), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, ), padding_option="zero")

        attention_scores = tl.dot(q, k) * scale

        if is_causal:
            q_pos = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_pos = tl.arange(0, K_TILE_SIZE) + j * K_TILE_SIZE
            mask = q_pos[:, None] >= k_pos[None, :]
            attention_scores += tl.where(mask, 0, -1e6)
        # tl.device_print("attention scores: ", k)
        # Max over the keys
        new_m_block = tl.maximum(m_block, tl.max(attention_scores, axis=1))

        # Broadcast on the key dimension for the max
        p_block = tl.math.exp(attention_scores - new_m_block[:, None])
        p_block = tl.cast(p_block, dtype=v.dtype)
        alpha = tl.math.exp(m_block - new_m_block)
        l = alpha * l + tl.sum(p_block, axis=1)
        # tl.device_print("output: ", p_block)
        # l += tl.sum(p_block, axis=1)
        # tl.device_print("output after sum: ", l)

        output = alpha[:, None] * output
        output = tl.dot(p_block, v, output)

        # Update pointers
        m_block = new_m_block

        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    output = output / l[:, None]
    l = m_block + tl.math.log(l)

    output = output.to(O_block_ptr.type.element_ty)
    l = l.to(L_block_ptr.type.element_ty)

    tl.store(O_block_ptr, output, boundary_check=(0, ))
    tl.store(L_block_ptr, l, boundary_check=(0, ))
        

class TritonAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        assert Q.is_contiguous()
        assert K.is_contiguous()
        assert V.is_contiguous()

        bsz, nq, d = Q.size()
        _, nk, d = K.size()

        scale = 1 / d**0.5

        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal
        ctx.scale = scale

        O = torch.empty((bsz, nq, d), device=Q.device)
        L = torch.empty((bsz, nq), device=Q.device)

        launch_grid = (
            math.ceil(nq / ctx.Q_TILE_SIZE),
            bsz
        )

        
        flash_fwd_kernel[launch_grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            nq, nk,
            scale,
            d,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            is_causal,
        )

        ctx.save_for_backward(L, Q, K, V, O)

        # qk = torch.einsum("... qd, ... kd->... qk", Q, K) * scale
        # ref_ls = torch.logsumexp(qk, dim=-1)
        # print(f"ref_ls: {ref_ls}")

        # print(L)
        return O


    @staticmethod
    @torch.compile
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        bsz, nq, d = Q.size()
        _, nk, _ = K.size()

        D = torch.sum(O * grad_out, dim=-1, keepdim=True)

        S = torch.einsum("... q d, ... k d -> ... q k", Q, K) * ctx.scale

        if ctx.is_causal:
            S = torch.where(
                torch.arange(nq, device=S.device)[None, :, None] >= torch.arange(nk, device=S.device)[None, None, :],
                S,
                -1e6
            )

        P = torch.exp(S - L.unsqueeze(-1))
        dV = torch.einsum("... q k, ... q d -> ... k d", P, grad_out)
        dP = torch.einsum("... q d, ... k d -> ... q k", grad_out, V)
        dS = torch.einsum("... q k, ... q k -> ... q k", P, (dP - D))
        dQ = torch.einsum("... q k, ... k d -> ... q d", dS, K) * ctx.scale
        dK = torch.einsum("... q k, ... q d -> ... k d", dS, Q) * ctx.scale

        return dQ, dK, dV, None
