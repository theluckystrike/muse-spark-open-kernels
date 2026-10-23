# theluckystrike · GitHub · OSS must win

Kernels for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev) (Meta × Paradigm × Andromeda, Oct 22 – Nov 19, 2026): faster decode on NVIDIA H200 for an open 256-expert, 4-bit MoE model, served by SGLang.

What this repo is for. Every kernel we submit is licensed Apache-2.0 by the challenge rules, so the best work can go upstream into SGLang. This repo will hold our kernels, the measurements behind them, and the notes on what did not work, published after the kernel boards close on Nov 4.

Track our row on the boards: team `theluckystrike`.

## Problems

| problem | what it is | status |
|---|---|---|
| muse_moe_w4a16 | routed experts, NVFP4 weights, BF16 activations | custom CUDA W4A16 GEMM, scored |
| muse_decode_layer | the whole decoder layer at verify shapes | fused glue + custom MoE, scored |
| muse_bf16_proj | dense BF16 projections and the router | skinny GEMM path, scored |
| muse_attention_decode | gated attention with an FP8 KV cache | fused sliding-window kernel, in progress |

## Method

- Measure first. Every design starts from the judge's own profile and a bandwidth table (bytes moved / time against 4.8 TB/s).
- One check, many variants. The judge queue is slow, so each run times several variants inside `prepare()` and prints a table.
- A local judge on a rented GPU runs the same image, tests and timing formula before any submission.
- Short sentences, one fact each, in every report (ASD-STE100 style), so agents and humans read the same notes.

Code and full write-ups land here after Nov 4. Until then this page is the placeholder.
