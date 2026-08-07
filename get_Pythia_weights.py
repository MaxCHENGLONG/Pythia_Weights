# Pythia Attention and MLP Weight Extraction
# -------------------------------------------
# Download EleutherAI Pythia checkpoints and re-store every weight matrix in the
# [in_features, out_features] convention .
#
# Requires transformers >= 5.x (tested on 5.9.0 / torch 2.12).
# On transformers 4.x this fails on the `dtype=` kwarg, and - worse - config.rope_parameters
# does not exist there, so the rotary fields in metadata.json would silently fall back to
# partial_rotary_factor=1.0 (rotary_ndims recorded as head_dim instead of head_dim // 4).
#
# Usage:
#   python get_Pyhia_weights.py                                   # pythia-70m @ step143000
#   python get_Pyhia_weights.py --revision step1000
#   python get_Pyhia_weights.py --revision step0 step1 step128     # several at once
#   python get_Pyhia_weights.py --all-steps --purge-cache          # all 154 revisions
#   python get_Pyhia_weights.py --size 160m --deduped
#   python get_Pyhia_weights.py --size 1b --out-dir /data          # -> /data/pythia-1b/<revision>/
#
# pythia-70m architecture (GPTNeoXForCausalLM):
#   - 6 transformer layers (NO parameter sharing, unlike ALBERT)
#   - hidden_size = 512
#   - intermediate_size = 2048 (4x hidden_size)
#   - num_heads = 8, head_dim = 64
#   - vocab_size = 50304, untied input/output embeddings
#   - rotary position embeddings on the first 25% of each head (16 of 64 dims)
#   - LayerNorm (with bias) everywhere; all projections carry a bias
#   - use_parallel_residual = True:
#       h_out = h + attn(input_layernorm(h)) + mlp(post_attention_layernorm(h))
#     i.e. the MLP reads the *pre-attention* residual, not the post-attention one.
#
# GPTNeoX fused QKV layout (the easy thing to get wrong):
#   attention.query_key_value.weight has shape [3 * hidden, hidden] but it is
#   NOT [Q; K; V] stacked in three blocks. transformers does:
#       qkv = self.query_key_value(h).view(*input_shape, -1, 3 * head_size)
#       query, key, value = qkv.chunk(3, dim=-1)
#   so the output rows are interleaved *per head*:
#       [h0_Q(64), h0_K(64), h0_V(64), h1_Q(64), h1_K(64), h1_V(64), ...]
#   Slicing W[:hidden] as W_Q is WRONG. The de-interleave is
#       W.view(num_heads, 3, head_dim, hidden)  ->  [:, 0] / [:, 1] / [:, 2]

import argparse
import hashlib
import json
import os
import shutil

import torch
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM

# Pythia was trained with a global batch of 1024 sequences x 2048 tokens.
TOKENS_PER_STEP = 1024 * 2048

PYTHIA_SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]

# The 154 public revisions: log-spaced over the first 512 steps, then every 1000.
ALL_REVISIONS = (
    ["step0"]
    + [f"step{2 ** i}" for i in range(10)]          # step1 .. step512
    + [f"step{1000 * i}" for i in range(1, 144)]    # step1000 .. step143000
)


def extract_attention_weights(model):
    """
    Extract W_Q, W_K, W_V, W_O (and biases) from a Pythia / GPTNeoX model.

    The fused attention.query_key_value.weight is [3 * hidden_size, hidden_size]
    with per-head interleaving, so it is reshaped to
    [num_heads, 3, head_dim, hidden_size] before Q/K/V are pulled apart.

    Returns
    -------
    dict : Dictionary containing:
        Fused matrices (in [in_features, out_features] convention):
        - W_Q, W_K, W_V: [num_layers, hidden_size, hidden_size]
        - W_O:           [num_layers, hidden_size, hidden_size]
        - b_Q, b_K, b_V: [num_layers, hidden_size]
        - b_O:           [num_layers, hidden_size]

        Per-head matrices:
        - W_Q_heads, W_K_heads, W_V_heads: [num_layers, num_heads, hidden_size, head_dim]
        - W_O_heads:                       [num_layers, num_heads, head_dim, hidden_size]
        - b_Q_heads, b_K_heads, b_V_heads: [num_layers, num_heads, head_dim]

        LayerNorm applied before attention:
        - attention_ln_weight, attention_ln_bias: [num_layers, hidden_size]
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    num_heads = cfg.num_attention_heads
    head_dim = hidden_size // num_heads

    print(f"Pythia model config:")
    print(f"  num_layers  = {num_layers}")
    print(f"  hidden_size = {hidden_size}")
    print(f"  num_heads   = {num_heads}")
    print(f"  head_dim    = {head_dim}")
    print()

    W_Q_all = torch.zeros(num_layers, hidden_size, hidden_size)
    W_K_all = torch.zeros(num_layers, hidden_size, hidden_size)
    W_V_all = torch.zeros(num_layers, hidden_size, hidden_size)
    W_O_all = torch.zeros(num_layers, hidden_size, hidden_size)

    b_Q_all = torch.zeros(num_layers, hidden_size)
    b_K_all = torch.zeros(num_layers, hidden_size)
    b_V_all = torch.zeros(num_layers, hidden_size)
    b_O_all = torch.zeros(num_layers, hidden_size)

    W_Q_heads = torch.zeros(num_layers, num_heads, hidden_size, head_dim)
    W_K_heads = torch.zeros(num_layers, num_heads, hidden_size, head_dim)
    W_V_heads = torch.zeros(num_layers, num_heads, hidden_size, head_dim)
    W_O_heads = torch.zeros(num_layers, num_heads, head_dim, hidden_size)

    b_Q_heads = torch.zeros(num_layers, num_heads, head_dim)
    b_K_heads = torch.zeros(num_layers, num_heads, head_dim)
    b_V_heads = torch.zeros(num_layers, num_heads, head_dim)

    ln_weight_all = torch.zeros(num_layers, hidden_size)
    ln_bias_all = torch.zeros(num_layers, hidden_size)

    print("Extracting attention weights from Pythia...")
    print("Note: fused QKV is de-interleaved from [num_heads, 3, head_dim, hidden]")

    for layer_idx in range(num_layers):
        layer = model.gpt_neox.layers[layer_idx]
        attn = layer.attention

        # De-interleave the fused QKV. Weight rows are [out_features, in_features],
        # grouped as [head][q|k|v][head_dim].
        qkv_w = attn.query_key_value.weight.cpu().float()
        qkv_w = qkv_w.view(num_heads, 3, head_dim, hidden_size)
        qkv_b = attn.query_key_value.bias.cpu().float()
        qkv_b = qkv_b.view(num_heads, 3, head_dim)

        # [num_heads, head_dim, hidden_size] -> [num_heads, hidden_size, head_dim]
        W_Q_heads[layer_idx] = qkv_w[:, 0].transpose(1, 2)
        W_K_heads[layer_idx] = qkv_w[:, 1].transpose(1, 2)
        W_V_heads[layer_idx] = qkv_w[:, 2].transpose(1, 2)

        b_Q_heads[layer_idx] = qkv_b[:, 0]
        b_K_heads[layer_idx] = qkv_b[:, 1]
        b_V_heads[layer_idx] = qkv_b[:, 2]

        # Fused view: concatenate the heads along the output axis, in head order.
        W_Q_all[layer_idx] = qkv_w[:, 0].reshape(hidden_size, hidden_size).T
        W_K_all[layer_idx] = qkv_w[:, 1].reshape(hidden_size, hidden_size).T
        W_V_all[layer_idx] = qkv_w[:, 2].reshape(hidden_size, hidden_size).T

        b_Q_all[layer_idx] = qkv_b[:, 0].reshape(hidden_size)
        b_K_all[layer_idx] = qkv_b[:, 1].reshape(hidden_size)
        b_V_all[layer_idx] = qkv_b[:, 2].reshape(hidden_size)

        # Output projection: [hidden, hidden] as [out, in] -> transpose to [in, out].
        W_O = attn.dense.weight.cpu().float().T
        W_O_all[layer_idx] = W_O
        b_O_all[layer_idx] = attn.dense.bias.cpu().float()

        # The attention output is the head-concatenation, so heads split W_O's *input*.
        for h in range(num_heads):
            W_O_heads[layer_idx, h] = W_O[h * head_dim:(h + 1) * head_dim, :]

        ln_weight_all[layer_idx] = layer.input_layernorm.weight.cpu().float()
        ln_bias_all[layer_idx] = layer.input_layernorm.bias.cpu().float()

        print(
            f"  Layer {layer_idx:2d}: W_Q={list(W_Q_all[layer_idx].shape)}, "
            f"W_Q^h={list(W_Q_heads[layer_idx, 0].shape)}, "
            f"W_O^h={list(W_O_heads[layer_idx, 0].shape)}"
        )

    return {
        "W_Q": W_Q_all, "W_K": W_K_all, "W_V": W_V_all, "W_O": W_O_all,
        "b_Q": b_Q_all, "b_K": b_K_all, "b_V": b_V_all, "b_O": b_O_all,
        "W_Q_heads": W_Q_heads, "W_K_heads": W_K_heads,
        "W_V_heads": W_V_heads, "W_O_heads": W_O_heads,
        "b_Q_heads": b_Q_heads, "b_K_heads": b_K_heads, "b_V_heads": b_V_heads,
        "attention_ln_weight": ln_weight_all,
        "attention_ln_bias": ln_bias_all,
    }


def extract_mlp_weights(model):
    """
    Extract MLP (FFN) weights from Pythia.

    - mlp.dense_h_to_4h.weight: [intermediate_size, hidden_size] (up projection, GELU)
    - mlp.dense_4h_to_h.weight: [hidden_size, intermediate_size] (down projection)

    Returns
    -------
    dict : Dictionary containing (in [in_features, out_features] convention):
        - W_ffn:        [num_layers, hidden_size, intermediate_size]
        - b_ffn:        [num_layers, intermediate_size]
        - W_ffn_output: [num_layers, intermediate_size, hidden_size]
        - b_ffn_output: [num_layers, hidden_size]
        - ffn_ln_weight, ffn_ln_bias: [num_layers, hidden_size]
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size

    print(f"\nPythia MLP config:")
    print(f"  hidden_size       = {hidden_size}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  activation        = {cfg.hidden_act}")
    print()

    W_ffn_all = torch.zeros(num_layers, hidden_size, intermediate_size)
    b_ffn_all = torch.zeros(num_layers, intermediate_size)
    W_ffn_output_all = torch.zeros(num_layers, intermediate_size, hidden_size)
    b_ffn_output_all = torch.zeros(num_layers, hidden_size)
    ln_weight_all = torch.zeros(num_layers, hidden_size)
    ln_bias_all = torch.zeros(num_layers, hidden_size)

    print("Extracting MLP (FFN) weights from Pythia...")

    for layer_idx in range(num_layers):
        layer = model.gpt_neox.layers[layer_idx]

        W_ffn_all[layer_idx] = layer.mlp.dense_h_to_4h.weight.cpu().float().T
        b_ffn_all[layer_idx] = layer.mlp.dense_h_to_4h.bias.cpu().float()
        W_ffn_output_all[layer_idx] = layer.mlp.dense_4h_to_h.weight.cpu().float().T
        b_ffn_output_all[layer_idx] = layer.mlp.dense_4h_to_h.bias.cpu().float()

        ln_weight_all[layer_idx] = layer.post_attention_layernorm.weight.cpu().float()
        ln_bias_all[layer_idx] = layer.post_attention_layernorm.bias.cpu().float()

        print(
            f"  Layer {layer_idx:2d}: W_ffn={list(W_ffn_all[layer_idx].shape)}, "
            f"W_ffn_output={list(W_ffn_output_all[layer_idx].shape)}"
        )

    return {
        "W_ffn": W_ffn_all,
        "b_ffn": b_ffn_all,
        "W_ffn_output": W_ffn_output_all,
        "b_ffn_output": b_ffn_output_all,
        "ffn_ln_weight": ln_weight_all,
        "ffn_ln_bias": ln_bias_all,
    }


def extract_embeddings(model):
    """
    Extract embeddings, the final LayerNorm and the unembedding matrix.

    embed_in.weight is [vocab_size, hidden_size], which is already
    [in_features, out_features] (a one-hot token index maps to a hidden vector),
    so it is stored as-is. embed_out is an nn.Linear and is transposed.

    Pythia has no learned position embeddings - position enters only through RoPE.
    """
    print("\nExtracting embeddings and final LayerNorm...")

    embed_in = model.gpt_neox.embed_in.weight.cpu().float()
    W_unembed = model.embed_out.weight.cpu().float().T
    final_ln_weight = model.gpt_neox.final_layer_norm.weight.cpu().float()
    final_ln_bias = model.gpt_neox.final_layer_norm.bias.cpu().float()

    print(f"  embed_in={list(embed_in.shape)}, W_unembed={list(W_unembed.shape)}")

    return {
        "embed_in": embed_in,
        "W_unembed": W_unembed,
        "final_ln_weight": final_ln_weight,
        "final_ln_bias": final_ln_bias,
    }


def verify_roundtrip(model, attn, mlp, emb):
    """
    Rebuild the original HuggingFace tensors from the [in, out] arrays and check
    they match exactly. This is what proves the QKV de-interleave went the right
    way round - a wrong de-interleave still produces plausible-looking shapes.
    """
    cfg = model.config
    num_heads = cfg.num_attention_heads
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // num_heads

    print("\nVerifying round-trip against the original state_dict...")

    for layer_idx in range(cfg.num_hidden_layers):
        layer = model.gpt_neox.layers[layer_idx]

        # Fused QKV: [in, out] -> [out, in] -> per-head -> re-interleave.
        q = attn["W_Q"][layer_idx].T.view(num_heads, head_dim, hidden_size)
        k = attn["W_K"][layer_idx].T.view(num_heads, head_dim, hidden_size)
        v = attn["W_V"][layer_idx].T.view(num_heads, head_dim, hidden_size)
        qkv_w = torch.stack([q, k, v], dim=1).reshape(3 * hidden_size, hidden_size)

        qb = attn["b_Q"][layer_idx].view(num_heads, head_dim)
        kb = attn["b_K"][layer_idx].view(num_heads, head_dim)
        vb = attn["b_V"][layer_idx].view(num_heads, head_dim)
        qkv_b = torch.stack([qb, kb, vb], dim=1).reshape(3 * hidden_size)

        checks = [
            ("query_key_value.weight", qkv_w, layer.attention.query_key_value.weight),
            ("query_key_value.bias", qkv_b, layer.attention.query_key_value.bias),
            ("attention.dense.weight", attn["W_O"][layer_idx].T, layer.attention.dense.weight),
            ("mlp.dense_h_to_4h.weight", mlp["W_ffn"][layer_idx].T, layer.mlp.dense_h_to_4h.weight),
            ("mlp.dense_4h_to_h.weight", mlp["W_ffn_output"][layer_idx].T, layer.mlp.dense_4h_to_h.weight),
        ]
        for name, rebuilt, original in checks:
            if not torch.equal(rebuilt, original.cpu().float()):
                raise AssertionError(f"round-trip mismatch: layer {layer_idx} {name}")

        # Per-head slices must agree with the fused matrices.
        q_heads = attn["W_Q_heads"][layer_idx].permute(1, 0, 2).reshape(hidden_size, hidden_size)
        if not torch.equal(q_heads, attn["W_Q"][layer_idx]):
            raise AssertionError(f"per-head / fused mismatch: layer {layer_idx} W_Q")
        o_heads = attn["W_O_heads"][layer_idx].reshape(hidden_size, hidden_size)
        if not torch.equal(o_heads, attn["W_O"][layer_idx]):
            raise AssertionError(f"per-head / fused mismatch: layer {layer_idx} W_O")

    if not torch.equal(emb["W_unembed"].T, model.embed_out.weight.cpu().float()):
        raise AssertionError("round-trip mismatch: embed_out.weight")

    print("  All layers: fused QKV, W_O, MLP and unembedding rebuild exactly.")

    # Functional check: project a random input with the [in, out] matrices and
    # compare against what the model's own attention module computes.
    torch.manual_seed(0)
    x = torch.randn(4, hidden_size)
    layer0 = model.gpt_neox.layers[0].attention
    ref = layer0.query_key_value(x.to(layer0.query_key_value.weight.dtype)).float()
    ref = ref.view(4, num_heads, 3, head_dim)

    ours_q = x @ attn["W_Q"][0] + attn["b_Q"][0]
    ours_k = x @ attn["W_K"][0] + attn["b_K"][0]
    ours_v = x @ attn["W_V"][0] + attn["b_V"][0]
    for name, ours, idx in [("Q", ours_q, 0), ("K", ours_k, 1), ("V", ours_v, 2)]:
        expected = ref[:, :, idx, :].reshape(4, hidden_size)
        if not torch.allclose(ours, expected, atol=1e-4):
            raise AssertionError(f"functional mismatch on {name}: de-interleave is wrong")
    print("  Functional check: x @ W_Q/W_K/W_V matches the model's own QKV split.")


def build_metadata(model, attn, mlp, emb, model_name, revision, commit_hash, dtype_str):
    """Assemble the metadata.json describing every saved file and convention."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    num_heads = cfg.num_attention_heads
    head_dim = hidden_size // num_heads
    intermediate_size = cfg.intermediate_size
    vocab_size = cfg.vocab_size

    rope = getattr(cfg, "rope_parameters", None) or {}
    partial_rotary_factor = rope.get("partial_rotary_factor", 1.0)
    rotary_ndims = int(head_dim * partial_rotary_factor)

    step = int(revision.replace("step", "")) if revision.startswith("step") else None

    def shapes(d, keys):
        return {f"{k}.pt": str(list(d[k].shape)) for k in keys}

    return {
        "model": model_name,
        "model_type": f"Pythia ({model_name.split('/')[-1]}) - GPTNeoXForCausalLM",
        "revision": revision,
        "training_step": step,
        "tokens_seen": step * TOKENS_PER_STEP if step is not None else None,
        "tokens_per_step": TOKENS_PER_STEP,
        "hf_commit_hash": commit_hash,
        "dtype": dtype_str,
        "transformers_version": cfg.transformers_version,

        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "hidden_act": cfg.hidden_act,
        "layer_norm_eps": cfg.layer_norm_eps,
        "attention_bias": cfg.attention_bias,
        "tie_word_embeddings": cfg.tie_word_embeddings,

        "parameter_sharing": "None - every layer has its own parameters (unlike ALBERT)",
        "position_encoding": {
            "type": "rotary (RoPE), applied to a prefix of each head",
            "partial_rotary_factor": partial_rotary_factor,
            "rotary_ndims": rotary_ndims,
            "note": (
                f"Only the first {rotary_ndims} of {head_dim} dims per head are rotated; "
                f"the remaining {head_dim - rotary_ndims} pass through unchanged. "
                "There are no learned position embeddings."
            ),
            "rope_theta": rope.get("rope_theta"),
        },
        "residual_structure": {
            "use_parallel_residual": cfg.use_parallel_residual,
            "formula": (
                "h_out = h + attn(input_layernorm(h)) + mlp(post_attention_layernorm(h))"
                if cfg.use_parallel_residual
                else "h1 = h + attn(input_layernorm(h)); h_out = h1 + mlp(post_attention_layernorm(h1))"
            ),
            "note": (
                "Parallel residual: the MLP reads the pre-attention residual stream, "
                "so attention and MLP of the same layer do not compose sequentially."
            ),
        },

        "directory_structure": {
            "per_head/": "Per-head attention matrices and biases",
            "fused/": "Full attention, MLP, LayerNorm, embedding and unembedding weights",
        },

        "per_head_files": shapes(attn, [
            "W_Q_heads", "W_K_heads", "W_V_heads", "W_O_heads",
            "b_Q_heads", "b_K_heads", "b_V_heads",
        ]),

        "fused_files": {
            "embeddings": {
                "embed_in.pt": f"{list(emb['embed_in'].shape)} - token embedding, NOT transposed (already [in, out])",
                "W_unembed.pt": f"{list(emb['W_unembed'].shape)} - hidden -> logits (untied from embed_in)",
                "final_ln_weight.pt": str(list(emb["final_ln_weight"].shape)),
                "final_ln_bias.pt": str(list(emb["final_ln_bias"].shape)),
            },
            "attention": shapes(attn, [
                "W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O",
                "attention_ln_weight", "attention_ln_bias",
            ]),
            "mlp": shapes(mlp, [
                "W_ffn", "b_ffn", "W_ffn_output", "b_ffn_output",
                "ffn_ln_weight", "ffn_ln_bias",
            ]),
        },

        "qkv_deinterleave": {
            "problem": (
                "HuggingFace stores attention.query_key_value.weight as "
                f"[{3 * hidden_size}, {hidden_size}], but it is NOT [Q; K; V] in three "
                "contiguous blocks - the output rows are interleaved per head as "
                "[h0_Q, h0_K, h0_V, h1_Q, h1_K, h1_V, ...]. Slicing the first "
                f"{hidden_size} rows as W_Q is incorrect."
            ),
            "source": (
                "transformers GPTNeoXAttention.forward: "
                "qkv = query_key_value(h).view(*shape, -1, 3 * head_size); "
                "q, k, v = qkv.chunk(3, dim=-1)"
            ),
            "extraction": (
                f"W.view({num_heads}, 3, {head_dim}, {hidden_size})[:, 0] -> Q "
                "(index 1 -> K, 2 -> V), then .transpose(1, 2) for per-head [in, out] "
                "or .reshape(hidden, hidden).T for the fused [in, out] matrix."
            ),
            "inverse": (
                "q = W_Q.T.view(num_heads, head_dim, hidden); same for k, v; then "
                "torch.stack([q, k, v], dim=1).reshape(3 * hidden, hidden)"
            ),
            "verified": "Round-trip equality against the original state_dict, plus a "
                        "functional check that x @ W_Q matches the model's own QKV split.",
        },

        "notes": {
            "weight_convention": "All weight matrices are transposed to [in_features, out_features]",
            "layer_axis": "Leading dimension of every stacked tensor is the layer index (0 = first layer)",
            "W_Q_heads": f"Each head: [{hidden_size}, {head_dim}], input -> query projection",
            "W_K_heads": f"Each head: [{hidden_size}, {head_dim}], input -> key projection",
            "W_V_heads": f"Each head: [{hidden_size}, {head_dim}], input -> value projection",
            "W_O_heads": f"Each head: [{head_dim}, {hidden_size}], attention output -> hidden",
            "b_Q/K/V_heads": f"Each head: [{head_dim}], bias for Q/K/V projection",
            "b_O": f"[{num_layers}, {hidden_size}], shared output bias (not split per head)",
            "W_ffn": f"Up projection: [{hidden_size}, {intermediate_size}], hidden -> intermediate ({cfg.hidden_act})",
            "W_ffn_output": f"Down projection: [{intermediate_size}, {hidden_size}], intermediate -> hidden",
            "embed_in": "Row v is the embedding of token id v; already [in, out] so it is stored untransposed",
            "W_unembed": "embed_out.weight transposed; NOT tied to embed_in",
            "attention_ln": "input_layernorm - applied before attention",
            "ffn_ln": "post_attention_layernorm - applied to the SAME residual as input_layernorm "
                      "because use_parallel_residual is True",
            "vocab_padding": f"vocab_size {vocab_size} is padded above the tokenizer's 50277 real tokens; "
                             "the trailing rows are unused",
        },
    }


def save_weights(attn, mlp, emb, metadata, out_dir):
    """Save extracted weights into fused/ and per_head/ under out_dir."""
    fused_dir = os.path.join(out_dir, "fused")
    per_head_dir = os.path.join(out_dir, "per_head")
    os.makedirs(fused_dir, exist_ok=True)
    os.makedirs(per_head_dir, exist_ok=True)

    per_head_keys = [
        "W_Q_heads", "W_K_heads", "W_V_heads", "W_O_heads",
        "b_Q_heads", "b_K_heads", "b_V_heads",
    ]
    fused_keys_attn = [
        "W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O",
        "attention_ln_weight", "attention_ln_bias",
    ]

    print(f"\nSaving weights to {out_dir} ...")

    print("  per_head/")
    for key in per_head_keys:
        torch.save(attn[key], os.path.join(per_head_dir, f"{key}.pt"))
        print(f"    {key}.pt - shape {list(attn[key].shape)}")

    print("  fused/")
    for source in (attn, mlp, emb):
        keys = fused_keys_attn if source is attn else list(source.keys())
        for key in keys:
            torch.save(source[key], os.path.join(fused_dir, f"{key}.pt"))
            print(f"    {key}.pt - shape {list(source[key].shape)}")

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json")


def purge_revision_cache(repo_id, commit_hash):
    """Delete the downloaded snapshot for one revision, keeping other revisions."""
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"))
    snapshot = os.path.join(repo_dir, "snapshots", commit_hash)
    blobs = set()
    if os.path.isdir(snapshot):
        for name in os.listdir(snapshot):
            link = os.path.join(snapshot, name)
            if os.path.islink(link):
                blobs.add(os.path.realpath(link))
        shutil.rmtree(snapshot)
    freed = 0
    for blob in blobs:
        # Only remove a blob if no other snapshot still points at it.
        if os.path.exists(blob) and os.stat(blob).st_nlink <= 1:
            freed += os.path.getsize(blob)
            os.remove(blob)
    print(f"  Purged HF cache for {commit_hash[:8]} ({freed / 1e6:.0f} MB freed)")


def process_revision(repo_id, revision, out_root, force, purge_cache):
    """Download one revision, extract, verify, save. Returns True if it ran."""
    out_dir = os.path.join(out_root, revision)
    if os.path.exists(os.path.join(out_dir, "metadata.json")) and not force:
        print(f"\n=== {revision}: already extracted, skipping (use --force to redo) ===")
        return False

    print(f"\n{'=' * 70}")
    print(f"=== {repo_id} @ {revision}")
    print(f"{'=' * 70}")

    commit_hash = HfApi().model_info(repo_id, revision=revision).sha

    model = AutoModelForCausalLM.from_pretrained(
        repo_id, revision=revision, dtype=torch.float32
    )
    model.eval()
    print("Model loaded.\n")

    with torch.no_grad():
        attn = extract_attention_weights(model)
        mlp = extract_mlp_weights(model)
        emb = extract_embeddings(model)
        verify_roundtrip(model, attn, mlp, emb)

    metadata = build_metadata(
        model, attn, mlp, emb, repo_id, revision, commit_hash, "float32"
    )
    save_weights(attn, mlp, emb, metadata, out_dir)

    if purge_cache:
        del model
        purge_revision_cache(repo_id, commit_hash)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract Pythia weights in [in_features, out_features] convention"
    )
    parser.add_argument("--size", type=str, default="70m", choices=PYTHIA_SIZES,
                        help="Pythia model size (default: 70m)")
    parser.add_argument("--deduped", action="store_true",
                        help="use the deduped-Pile variant (pythia-<size>-deduped)")
    parser.add_argument("--revision", type=str, nargs="+", default=["step143000"],
                        help="training checkpoint(s), e.g. step0 step1000 (default: step143000)")
    parser.add_argument("--all-steps", action="store_true",
                        help=f"extract all {len(ALL_REVISIONS)} public revisions")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="parent directory for the per-model output dirs (default: .), "
                             "so weights land in <out-dir>/<model-name>/<revision>/")
    parser.add_argument("--force", action="store_true",
                        help="re-extract revisions that are already on disk")
    parser.add_argument("--purge-cache", action="store_true",
                        help="delete each revision's HF cache snapshot after extraction")
    args = parser.parse_args()

    model_name = f"pythia-{args.size}" + ("-deduped" if args.deduped else "")
    repo_id = f"EleutherAI/{model_name}"
    out_root = os.path.join(args.out_dir, model_name)
    revisions = ALL_REVISIONS if args.all_steps else args.revision

    print(f"Model:     {repo_id}")
    print(f"Revisions: {len(revisions)} ({revisions[0]} ... {revisions[-1]})")
    print(f"Output:    {out_root}/")

    for revision in revisions:
        process_revision(repo_id, revision, out_root, args.force, args.purge_cache)

    print("\nDone!")


if __name__ == "__main__":
    main()
