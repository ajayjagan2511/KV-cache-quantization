import operator
import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from src.quantization.quant_ops import quantize_to_int8, dequantize_to_fp16, true_unfused_int8_baseline
from src.kernels.int8_flash_attn import fx_flash_attn_v2_wrapper


def transform_to_triton_batched(m: nn.Module, use_fused_FA: bool = False) -> nn.Module:
    symbolic_traced = torch.fx.symbolic_trace(m)
    attention_outputs = []

    for node in symbolic_traced.graph.nodes:
        if node.target == torch.matmul:
            arg0 = node.args[0]
            if isinstance(arg0, torch.fx.Node) and ("softmax" in str(arg0.target)):
                final_matmul = node
                softmax_node = arg0
                v_node = final_matmul.args[1]

                div_node = softmax_node.args[0]
                qk_matmul = div_node.args[0]

                q_node = qk_matmul.args[0]
                k_transpose_node = qk_matmul.args[1]
                k_node = k_transpose_node.args[0]

                attention_outputs.append({
                    "final_out": final_matmul,
                    "softmax": softmax_node,
                    "div_node": div_node,
                    "qk_matmul": qk_matmul,
                    "q": q_node,
                    "k": k_node,
                    "v": v_node,
                    "k_transpose": k_transpose_node,
                })

    attention_node = fx_flash_attn_v2_wrapper if use_fused_FA else true_unfused_int8_baseline

    for att_block in attention_outputs:
        with symbolic_traced.graph.inserting_before(att_block["qk_matmul"]):
            quantized_k = symbolic_traced.graph.call_function(quantize_to_int8, args=(att_block["k"],))
        with symbolic_traced.graph.inserting_before(att_block["qk_matmul"]):
            quantized_v = symbolic_traced.graph.call_function(quantize_to_int8, args=(att_block["v"],))

        with symbolic_traced.graph.inserting_after(quantized_v):
            triton_node = symbolic_traced.graph.call_function(
                attention_node,
                args=(att_block["q"], quantized_k, quantized_v),
            )

        att_block["final_out"].replace_all_uses_with(triton_node)

        symbolic_traced.graph.erase_node(att_block["final_out"])
        symbolic_traced.graph.erase_node(att_block["softmax"])
        symbolic_traced.graph.erase_node(att_block["div_node"])
        symbolic_traced.graph.erase_node(att_block["qk_matmul"])
        symbolic_traced.graph.erase_node(att_block["k_transpose"])

    symbolic_traced.graph.lint()
    symbolic_traced.recompile()

    return symbolic_traced
