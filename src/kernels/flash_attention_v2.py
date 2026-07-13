import math
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_v2_kernel(
        Q, K, V,
        O, l, m,
        stride_qh, stride_qn,
        stride_kh, stride_kn,
        stride_vh, stride_vn,
        stride_oh, stride_on,
        stride_lh,
        stride_mh,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_D: tl.constexpr,
        N_q: tl.constexpr, N_v: tl.constexpr,
        D_head: tl.constexpr, B_c: tl.constexpr, B_r: tl.constexpr,
        T_r: tl.constexpr, T_c: tl.constexpr,
        sm_scale,
):
    head_idx = tl.program_id(0)
    tr_idx = tl.program_id(1)

    Q_ptr = Q + head_idx * stride_qh
    K_ptr = K + head_idx * stride_kh
    V_ptr = V + head_idx * stride_vh
    O_ptr = O + head_idx * stride_oh
    l_ptr = l + head_idx * stride_lh
    m_ptr = m + head_idx * stride_mh

    offs_q = tr_idx * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)

    qo_mask = (offs_q[:, None] < N_q) & (offs_d[None, :] < D_head)
    q_ptrs_2d = Q_ptr + offs_q[:, None] * stride_qn + offs_d[None, :]
    q = tl.load(q_ptrs_2d, mask=qo_mask, other=0.0)

    m_i = tl.full([BLOCK_R], value=float('-inf'), dtype=tl.float16)
    l_i = tl.zeros([BLOCK_R], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_R, BLOCK_D], dtype=tl.float32)

    for tc in range(T_c):
        offs_kv = tc * BLOCK_C + tl.arange(0, BLOCK_C)
        kv_mask = (offs_kv[:, None] < N_v) & (offs_d[None, :] < D_head)

        k_ptrs_2d = K_ptr + offs_kv[:, None] * stride_kn + offs_d[None, :]
        k = tl.load(k_ptrs_2d, mask=kv_mask, other=0.0)

        v_ptrs_2d = V_ptr + offs_kv[:, None] * stride_vn + offs_d[None, :]
        v = tl.load(v_ptrs_2d, mask=kv_mask, other=0.0)

        s = tl.dot(q, k.T) * sm_scale

        s_mask = (offs_q[:, None] < N_q) & (offs_kv[None, :] < N_v)
        s = tl.where(s_mask, s, float('-inf'))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_i = o_i * alpha[:, None]
        o_i += tl.dot(p.to(v.dtype), v)
        m_i = m_new.to(m_i.dtype)

    o_i = o_i / l_i[:, None]

    o_ptrs_2d = O_ptr + offs_q[:, None] * stride_on + offs_d[None, :]
    tl.store(o_ptrs_2d, o_i, mask=qo_mask)


def multiheaded_attention_triton(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        W_qkv, W_out,
        b_qkv, b_out,
        num_heads=1,
        device="cuda",
        block_size=64) -> torch.Tensor:

    N_q, D = query.shape
    N_v, _ = key.shape
    D_head = D // num_heads

    Q = torch.matmul(query, W_qkv[0:D, :].T) + b_qkv[0:D][None, :]
    K = torch.matmul(key, W_qkv[D:2*D, :].T) + b_qkv[D:2*D][None, :]
    V = torch.matmul(value, W_qkv[2*D:3*D, :].T) + b_qkv[2*D:3*D][None, :]

    Q = Q.reshape(N_q, num_heads, D_head).permute(1, 0, 2).contiguous()
    K = K.reshape(N_v, num_heads, D_head).permute(1, 0, 2).contiguous()
    V = V.reshape(N_v, num_heads, D_head).permute(1, 0, 2).contiguous()

    BLOCK_D = triton.next_power_of_2(D_head)
    B_c = min(triton.next_power_of_2(N_v), block_size)
    B_r = min(triton.next_power_of_2(N_q), block_size)
    BLOCK_C = B_c
    BLOCK_R = B_r

    T_c = triton.cdiv(N_v, B_c)
    T_r = triton.cdiv(N_q, B_r)

    O = torch.zeros_like(Q)
    l = torch.zeros((num_heads, N_q), device=device, dtype=torch.float32)
    m = torch.full((num_heads, N_q), fill_value=float('-inf'), device=device, dtype=torch.float32)

    grid = (num_heads, T_r)
    num_warps = 4 if D_head <= block_size else 8
    sm_scale = 1.0 / math.sqrt(D_head)

    _flash_attn_v2_kernel[grid](
        Q, K, V,
        O, l, m,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        l.stride(0),
        m.stride(0),
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        BLOCK_D=BLOCK_D,
        N_q=N_q, N_v=N_v,
        D_head=D_head, B_c=B_c, B_r=B_r,
        T_r=T_r, T_c=T_c,
        sm_scale=sm_scale,
        num_warps=num_warps,
    )

    O = O.permute(1, 0, 2).contiguous().reshape(N_q, D)
    out = torch.matmul(O, W_out.T) + b_out[None, :]
    return out
