import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 64,  'BLOCK_C': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R': 128, 'BLOCK_C': 64},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_R': 64,  'BLOCK_C': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_R': 128, 'BLOCK_C': 128}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_R': 64,  'BLOCK_C': 32},  num_warps=4, num_stages=5),
        triton.Config({'BLOCK_R': 32,  'BLOCK_C': 64},  num_warps=4, num_stages=5),
    ],
    key=['N_q', 'N_v', 'D_head'],
)
@triton.jit
def _fx_flash_attn_v2_int8kv(
    Q_ptr, K_int8_ptr, V_int8_ptr, O_ptr,
    k_scale_val, v_scale_val, sm_scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N_q, N_v,
    D_head: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx  = tl.program_id(1)
    tr_idx    = tl.program_id(2)

    Q = Q_ptr     + batch_idx * stride_qb + head_idx * stride_qh
    K = K_int8_ptr + batch_idx * stride_kb + head_idx * stride_kh
    V = V_int8_ptr + batch_idx * stride_vb + head_idx * stride_vh
    O = O_ptr     + batch_idx * stride_ob + head_idx * stride_oh

    offs_q = tr_idx * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)

    qo_mask = (offs_q[:, None] < N_q) & (offs_d[None, :] < D_head)
    q_ptrs = Q + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=qo_mask, other=0.0)

    m_i = tl.full([BLOCK_R], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_R], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_R, BLOCK_D], dtype=tl.float32)

    T_c = tl.cdiv(N_v, BLOCK_C)

    for tc in range(T_c):
        offs_kv = tc * BLOCK_C + tl.arange(0, BLOCK_C)

        # Pre-transpose K load to bypass .T compiler limitation
        kv_mask_T = (offs_d[:, None] < D_head) & (offs_kv[None, :] < N_v)
        k_ptrs = K + offs_d[:, None] * stride_kd + offs_kv[None, :] * stride_kn
        k_int8 = tl.load(k_ptrs, mask=kv_mask_T, other=0)
        k = (k_int8.to(tl.float16) * k_scale_val).to(tl.float16)

        kv_mask = (offs_kv[:, None] < N_v) & (offs_d[None, :] < D_head)
        v_ptrs = V + offs_kv[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_int8 = tl.load(v_ptrs, mask=kv_mask, other=0)
        v = (v_int8.to(tl.float16) * v_scale_val).to(tl.float16)

        s = tl.dot(q, k, out_dtype=tl.float32) * sm_scale

        s_mask = (offs_q[:, None] < N_q) & (offs_kv[None, :] < N_v)
        s = tl.where(s_mask, s, float("-inf"))

        m_ij  = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_i = o_i * alpha[:, None]
        o_i += tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        m_i = m_new

    o_i = o_i / l_i[:, None]

    o_ptrs = O + offs_q[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_i.to(tl.float16), mask=qo_mask)


def fx_flash_attn_v2_wrapper(q, k_tuple, v_tuple):
    k_int8, k_scale = k_tuple
    v_int8, v_scale = v_tuple

    B, H, N_q, D_head = q.shape
    N_v = k_int8.shape[2]

    BLOCK_D = triton.next_power_of_2(D_head)
    O = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D_head)

    grid = lambda META: (B, H, triton.cdiv(N_q, META['BLOCK_R']))

    _fx_flash_attn_v2_int8kv[grid](
        q, k_int8, v_int8, O,
        k_scale.item(), v_scale.item(), sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        v_int8.stride(0), v_int8.stride(1), v_int8.stride(2), v_int8.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N_q, N_v,
        D_head=D_head,
        BLOCK_D=BLOCK_D,
    )
    return O
