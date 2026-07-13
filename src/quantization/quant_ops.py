import math
import torch


@torch.compile
def quantize_to_int8(t: torch.Tensor):
    scale = t.abs().max() / 127.0
    q_t = torch.round(t / scale)
    q_t = torch.clamp(q_t, -128, 127).to(torch.int8)
    return q_t, scale


def dequantize_to_fp16(q_t: torch.Tensor, scale: torch.Tensor):
    return (q_t.to(torch.float16) * scale)


@torch.compile
def fused_qk_dequantize(q, k_int8, k_scale):
    k_fp16 = (k_int8.to(torch.float16) * k_scale).to(torch.float16)
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.matmul(q, k_fp16.transpose(-2, -1)) * sm_scale


@torch.compile
def fused_av_dequantize(attn, v_int8, v_scale):
    v_fp16 = (v_int8.to(torch.float16) * v_scale).to(torch.float16)
    return torch.matmul(attn, v_fp16)


def true_unfused_int8_baseline(q, k_tuple, v_tuple):
    k_int8, k_scale = k_tuple
    v_int8, v_scale = v_tuple

    scores = fused_qk_dequantize(q, k_int8, k_scale)
    attn = torch.nn.functional.softmax(scores, dim=-1)
    out = fused_av_dequantize(attn, v_int8, v_scale)
    return out
