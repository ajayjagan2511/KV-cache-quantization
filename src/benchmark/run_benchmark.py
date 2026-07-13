import gc
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.attention import NaiveMultiHeadAttentionBatched
from src.compiler.fx_pass import transform_to_triton_batched
from src.kernels.flash_attention_v2 import multiheaded_attention_triton


SEQ_LENGTHS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def benchmark_attention(setting: str = "original"):
    assert setting in ("original", "flash_int8_kv"), (
        f"Unknown setting '{setting}'. Choose 'original' or 'flash_int8_kv'."
    )

    torch.manual_seed(42)
    d = 64
    heads = 4
    d_model = d * heads

    results = []

    for N in SEQ_LENGTHS:
        print(f"[{setting}] Benchmarking Sequence Length: {N}...")

        try:
            X = torch.rand(1, N, d_model, dtype=torch.float16, device="cuda")
        except RuntimeError:
            print(f"  -> Fatal OOM allocating input tensor at {N}. Stopping.")
            break

        pytorch_mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, dtype=torch.float16, batch_first=True
        ).cuda()

        naive_mha = NaiveMultiHeadAttentionBatched(
            W=pytorch_mha.in_proj_weight.T, bias=pytorch_mha.in_proj_bias,
            Wo=pytorch_mha.out_proj.weight.T, bias_o=pytorch_mha.out_proj.bias,
            d_model=d_model, heads=heads,
        ).half().cuda()

        compiled_mha    = transform_to_triton_batched(naive_mha, use_fused_FA=False)
        compiled_FA_mha = transform_to_triton_batched(naive_mha, use_fused_FA=True)

        if setting == "original":
            models = {
                "PyTorch Native":          lambda: pytorch_mha(X, X, X, need_weights=False)[0],
                "Naive Unrolled":          lambda: naive_mha(X),
                "Triton Unfused (INT8 KV)": lambda: compiled_mha(X),
            }
        else:
            models = {
                "PyTorch Native":              lambda: pytorch_mha(X, X, X, need_weights=False)[0],
                "Flash Attn V2 (Triton)":      lambda: multiheaded_attention_triton(
                    X[0], X[0], X[0],
                    W_qkv=pytorch_mha.in_proj_weight,
                    W_out=pytorch_mha.out_proj.weight,
                    b_qkv=pytorch_mha.in_proj_bias,
                    b_out=pytorch_mha.out_proj.bias,
                    num_heads=heads, device="cuda", block_size=64,
                ),
                "Flash Attn V2 + INT8 KV (FX)": lambda: compiled_FA_mha(X),
            }

        seq_label = f"{N//1024}k" if N >= 1024 else str(N)

        for model_name, forward_fn in models.items():
            gc.collect()
            torch.cuda.empty_cache()

            try:
                with torch.inference_mode():
                    for _ in range(3):
                        _ = forward_fn()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt   = torch.cuda.Event(enable_timing=True)
                    start_evt.record()
                    for _ in range(10):
                        _ = forward_fn()
                    end_evt.record()
                    torch.cuda.synchronize()

                latency_ms  = start_evt.elapsed_time(end_evt) / 10.0
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  -> [OOM] {model_name} at {seq_label}")
                    latency_ms, peak_mem_mb = None, None
                    torch.cuda.empty_cache()
                else:
                    raise

            results.append({
                "Seq_Val":         N,
                "Seq_Label":       seq_label,
                "Model":           model_name,
                "Latency (ms)":    latency_ms,
                "Peak Memory (MB)": peak_mem_mb,
                "Setting":         setting,
            })

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, setting: str):
    if setting == "original":
        colors = {
            "PyTorch Native":          "#94a3b8",
            "Naive Unrolled":          "#ef4444",
            "Triton Unfused (INT8 KV)": "#0f172a",
        }
        chart_title = "Attention Architecture: FX Compilation vs O(N²) Baseline"
    else:
        colors = {
            "PyTorch Native":               "#94a3b8",
            "Flash Attn V2 (Triton)":       "#10b981",
            "Flash Attn V2 + INT8 KV (FX)": "#6d28d9",
        }
        chart_title = "Attention Architecture: Compiled INT8 Flash vs Native Flash"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Forward Pass Latency (ms)", "Peak HBM Allocation (MB)"),
        horizontal_spacing=0.1,
    )

    for model_name, color in colors.items():
        model_data = df[df["Model"] == model_name]

        fig.add_trace(go.Scatter(
            x=model_data["Seq_Label"],
            y=model_data["Latency (ms)"],
            mode="lines+markers",
            name=model_name,
            line=dict(color=color, width=2.5, shape="spline"),
            marker=dict(size=7, opacity=0.9),
            legendgroup=model_name,
            connectgaps=False,
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=model_data["Seq_Label"],
            y=model_data["Peak Memory (MB)"],
            name=model_name,
            marker_color=color,
            marker_line_width=0,
            legendgroup=model_name,
            showlegend=False,
        ), row=1, col=2)

    axis_styling = dict(
        showgrid=True, gridwidth=1, gridcolor="#f1f5f9",
        showline=True, linewidth=1, linecolor="#e2e8f0",
        tickfont=dict(color="#64748b", size=12),
        title_font=dict(size=14, color="#475569"),
    )

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=22, color="#0f172a"), x=0.01, y=0.95),
        height=600, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(t=120, b=40, l=40, r=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0,
            font=dict(size=13, color="#475569"), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        barmode="group",
        bargap=0.15, bargroupgap=0.05,
    )

    fig.update_xaxes(title_text="Sequence Length (Log2 Progression)", **axis_styling, row=1, col=1)
    fig.update_yaxes(title_text="Latency (ms) — Log Scale", type="log", **axis_styling, row=1, col=1)
    fig.update_xaxes(title_text="Sequence Length (Log2 Progression)", **axis_styling, row=1, col=2)
    fig.update_yaxes(title_text="Memory (MB)", **axis_styling, row=1, col=2)

    fig.show()
