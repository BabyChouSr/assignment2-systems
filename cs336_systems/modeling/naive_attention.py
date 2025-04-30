import torch
import math
from einops import einsum

class NaiveAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        bsz = Q.size(0)
        num_queries, d = Q.size(1), Q.size(2)
        num_keys = K.size(1)
        scale = 1 / math.sqrt(d)

        ctx.is_causal = is_causal
        ctx.scale = scale

        QUERY_TILE_SIZE = 16
        KEY_TILE_SIZE = 16

        # query_indices = torch.arange(0, math.ceil(num_queries / QUERY_TILE_SIZE))
        # key_indices = torch.arange(0, math.ceil(num_keys / KEY_TILE_SIZE))

        O = torch.zeros_like(Q).to(Q.device)
        L = torch.zeros((bsz, num_queries)).to(K.device)

        for q_idx in range(0, math.ceil(num_queries / QUERY_TILE_SIZE)):
            q_start = q_idx * QUERY_TILE_SIZE

            q_block = Q[:, q_start:q_start + QUERY_TILE_SIZE, :]
            o_block = O[:, q_start:q_start + QUERY_TILE_SIZE, :]
            l_block = L[:, q_start:q_start + QUERY_TILE_SIZE]
            m_block = torch.ones((bsz, QUERY_TILE_SIZE)).to(Q.device) * -float('inf')
            for k_idx in range(0, math.ceil(num_keys / KEY_TILE_SIZE)):
                k_start = k_idx * KEY_TILE_SIZE

                k_block = K[:, k_start: k_start + KEY_TILE_SIZE, :]
                v_block = V[:, k_start: k_start + KEY_TILE_SIZE, :]

                attention_scores = torch.einsum("... q d, ... kd->... qk", q_block, k_block) * scale

                # ref_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d)
                # is_close =  torch.allclose(attention_scores, ref_scores, rtol=1e-3, atol=1e-3)
                # print("NaiveAttention vs. reference allclose:", is_close)
                # if not is_close:
                #     print("Max abs diff:", (attention_scores - ref_scores).abs().max().item())
                #     print("Mean abs diff:", (attention_scores - ref_scores).abs().mean().item())
                # assert is_close, "NaiveAttention output does not match reference scaled_dot_product_attention"

                prev_m_block = m_block.detach().clone()
                m_block = torch.maximum(m_block, torch.amax(attention_scores, dim=-1, keepdim=False))

                # Need to unsqueeze here to make sure that the maximum is broadcasted across the key dimension correctly
                p_block = torch.exp(attention_scores - m_block.unsqueeze(-1))
                l_block = torch.exp(prev_m_block - m_block) * l_block + torch.sum(p_block, dim=-1, keepdim=False)
                o_diag_exp = torch.exp(prev_m_block - m_block).unsqueeze(-1) * o_block
                o_block = o_diag_exp + torch.einsum("... q k, ... k d -> ... q d", p_block, v_block)

            # o_block = torch.einsum("... q, ... q d -> q d", torch.inverse(torch.diag_embed(l_block)), o_block)
            # o_block = o_block / l_block.unsqueeze(-1)
            # l_block = m_block + torch.log(l_block)

            O[:, q_start:q_start + QUERY_TILE_SIZE, :] = o_block / l_block.unsqueeze(-1)
            L[:, q_start:q_start + QUERY_TILE_SIZE] = m_block + torch.log(l_block)

        ctx.save_for_backward(L, Q, K, V, O)
        
        # print(f"Naive p block: {p_block}")

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