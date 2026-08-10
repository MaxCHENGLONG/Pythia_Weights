# Hand-written Pythia (GPTNeoX) forward pass
# ------------------------------------------
# Runs a Pythia checkpoint from the [in_features, out_features] weights produced by
# get_Pythia_weights.py, without importing transformers. Everything about the model
# (depth, heads, rotary fraction, eps) is read from metadata.json, so the same code
# runs any size and any training step.
#
# Usage:
#   from Pythia_model_hand import PythiaHand
#   model = PythiaHand.from_dir("pythia-70m/step143000")
#   logits = model(ids)                              # [T, vocab_size]
#   logits, cache = model(ids, return_hidden=True)   # cache["resid"][l], cache["attn_probs"][l], ...
#
# Self-check against HuggingFace (needs transformers, downloads the reference model):
#   python Pythia_model_hand.py pythia-70m/step143000
#
# Because the weights are stored as [in, out], every projection is a plain `x @ W + b`.
# There is not a single transpose in this file - that is the whole point of the convention.
#
# Three things that are silently wrong if you implement them the obvious way:
#   1. Parallel residual. Pythia computes
#          h_out = h + attn(input_ln(h)) + mlp(post_attn_ln(h))
#      Both LayerNorms read the SAME pre-attention h. Writing the usual sequential
#      h1 = h + attn(ln(h)); h_out = h1 + mlp(ln(h1)) produces a plausible but wrong model.
#   2. Partial rotary. Only the first rotary_ndims (= head_dim // 4 for Pythia) dims of
#      each head are rotated; the rest pass through untouched.
#   3. Exact GELU. config.hidden_act is "gelu", which maps to the erf formulation,
#      not the tanh approximation. Measured: the tanh variant only doubles the logit
#      error (6.8e-3 -> 1.4e-2) and does not flip any argmax, so unlike (1) and (2) the
#      self-check below will NOT catch it. It is a quiet systematic bias, not a blowup.
#
# Sensitivity of the self-check, measured by deliberately breaking each of the above:
#   (1) serial residual  -> max logit diff 5.5e+01, argmax wrong   caught
#   (2) full rotary      -> max logit diff 2.1e+01, argmax wrong   caught
#   (3) tanh GELU        -> max logit diff 1.4e-02, argmax fine    NOT caught

import json
import os
import sys

import torch


class PythiaHand:
    """A Pythia checkpoint as plain tensors plus a forward pass."""

    def __init__(self, weights, meta):
        self.w = weights
        self.meta = meta
        self.num_layers = meta["num_layers"]
        self.num_heads = meta["num_heads"]
        self.head_dim = meta["head_dim"]
        self.hidden_size = meta["hidden_size"]
        self.eps = meta["layer_norm_eps"]
        self.rotary_ndims = meta["position_encoding"]["rotary_ndims"]
        self.rope_theta = meta["position_encoding"]["rope_theta"]
        self.dtype = weights["embed_in"].dtype

        if not meta["residual_structure"]["use_parallel_residual"]:
            raise ValueError("this implementation assumes use_parallel_residual=True")

    @classmethod
    def from_dir(cls, path, device="cpu", dtype=torch.float32):
        """Load metadata.json and every tensor in fused/ from an extracted checkpoint."""
        with open(os.path.join(path, "metadata.json")) as f:
            meta = json.load(f)

        fused = os.path.join(path, "fused")
        weights = {}
        for name in sorted(os.listdir(fused)):
            if name.endswith(".pt"):
                t = torch.load(os.path.join(fused, name), map_location=device)
                weights[name[:-3]] = t.to(dtype)
        return cls(weights, meta)

    def layer_norm(self, x, weight, bias):
        """LayerNorm over the last axis. Biased variance, matching PyTorch."""
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * weight + bias

    def rope_tables(self, positions):
        """
        Build the cos/sin tables for the rotated prefix of each head.

        Returns two [T, rotary_ndims] tensors. Only rotary_ndims of the head_dim
        columns are ever rotated - Pythia sets rotary_ndims = head_dim // 4.

        The table is built in float32 and cast at the end, mirroring transformers:
        inv_freq is a float32 buffer there and the rotary forward ends with
        cos.to(x.dtype). Building it natively in float64 would disagree with HF by
        ~1e-6 rad at the longer positions and would put a floor under the float64
        comparison in verify_hand_vs_hf.py. For float32 the cast is a no-op.
        """
        half = torch.arange(0, self.rotary_ndims, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.rope_theta ** (half / self.rotary_ndims))
        freqs = torch.outer(positions.float(), inv_freq)      # [T, rotary_ndims / 2]
        emb = torch.cat([freqs, freqs], dim=-1)               # [T, rotary_ndims]
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    def apply_rope(self, x, cos, sin):
        """
        Rotate the first rotary_ndims dims of x and leave the tail untouched.

        x is [T, num_heads, head_dim]; cos/sin are [T, rotary_ndims] and broadcast
        across the head axis.
        """
        rot, pass_through = x[..., :self.rotary_ndims], x[..., self.rotary_ndims:]
        half = self.rotary_ndims // 2
        # rotate_half: [x_hi, x_lo] -> [-x_hi_second_half, x_lo_first_half]
        rotated = torch.cat([-rot[..., half:], rot[..., :half]], dim=-1)
        rot = rot * cos[:, None, :] + rotated * sin[:, None, :]
        return torch.cat([rot, pass_through], dim=-1)

    def attention(self, x, layer, cos, sin, causal_mask):
        """Multi-head causal self-attention for one layer. x is the normalized [T, H] input."""
        w, T = self.w, x.shape[0]
        shape = (T, self.num_heads, self.head_dim)

        q = (x @ w["W_Q"][layer] + w["b_Q"][layer]).view(shape)
        k = (x @ w["W_K"][layer] + w["b_K"][layer]).view(shape)
        v = (x @ w["W_V"][layer] + w["b_V"][layer]).view(shape)

        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        scores = torch.einsum("qnd,knd->nqk", q, k) / self.head_dim ** 0.5
        probs = (scores + causal_mask).softmax(dim=-1)

        # Concatenating the heads is what makes W_O's input axis the head axis.
        out = torch.einsum("nqk,knd->qnd", probs, v).reshape(T, self.hidden_size)
        return out @ w["W_O"][layer] + w["b_O"][layer], probs

    def mlp(self, x, layer):
        """Two-layer FFN with exact GELU. x is the normalized [T, H] input."""
        w = self.w
        h = x @ w["W_ffn"][layer] + w["b_ffn"][layer]
        h = torch.nn.functional.gelu(h)              # exact (erf), not the tanh approximation
        return h @ w["W_ffn_output"][layer] + w["b_ffn_output"][layer]

    def forward(self, ids, return_hidden=False):
        """
        Run the model over a 1-D LongTensor of token ids.

        Returns logits [T, vocab_size], or (logits, cache) when return_hidden is set.
        cache holds per-layer residual stream, attention output, MLP output and
        attention probabilities - the quantities you actually want for weight dynamics.
        """
        w = self.w
        ids = torch.as_tensor(ids, dtype=torch.long).flatten()
        T = ids.shape[0]

        positions = torch.arange(T)
        cos, sin = self.rope_tables(positions)
        causal_mask = torch.full((T, T), float("-inf")).triu(1)

        h = w["embed_in"][ids]                                   # [T, H]
        cache = {"resid": [], "attn_out": [], "mlp_out": [], "attn_probs": []}

        for layer in range(self.num_layers):
            # Parallel residual: both branches read the same h.
            attn_in = self.layer_norm(h, w["attention_ln_weight"][layer], w["attention_ln_bias"][layer])
            mlp_in = self.layer_norm(h, w["ffn_ln_weight"][layer], w["ffn_ln_bias"][layer])

            attn_out, probs = self.attention(attn_in, layer, cos, sin, causal_mask)
            mlp_out = self.mlp(mlp_in, layer)

            if return_hidden:
                cache["resid"].append(h)
                cache["attn_out"].append(attn_out)
                cache["mlp_out"].append(mlp_out)
                cache["attn_probs"].append(probs)

            h = h + attn_out + mlp_out

        h = self.layer_norm(h, w["final_ln_weight"], w["final_ln_bias"])
        logits = h @ w["W_unembed"]

        if return_hidden:
            cache["resid"].append(h)
            return logits, cache
        return logits

    __call__ = forward


def _self_check(path):
    """Compare this implementation against HuggingFace on a short prompt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = PythiaHand.from_dir(path)
    repo = model.meta["model"]
    revision = model.meta["revision"]
    print(f"{repo} @ {revision}: L={model.num_layers} H={model.hidden_size} "
          f"heads={model.num_heads} rotary={model.rotary_ndims}/{model.head_dim}")

    tok = AutoTokenizer.from_pretrained(repo)
    ids = tok("The capital of France is Paris, and the capital of Germany is",
              return_tensors="pt").input_ids[0]

    ours = model(ids)
    ref = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.float32)
    with torch.no_grad():
        theirs = ref(ids[None]).logits[0]

    diff = (ours - theirs).abs().max().item()
    agree = bool((ours.argmax(-1) == theirs.argmax(-1)).all())
    print(f"  max abs diff on logits : {diff:.2e}  (logit scale {theirs.abs().max():.1f})")
    print(f"  argmax agrees at all {len(ids)} positions : {agree}")
    print(f"  next token             : {tok.decode(ours[-1].argmax())!r}")
    if not agree:
        raise SystemExit("MISMATCH against HuggingFace")


if __name__ == "__main__":
    _self_check(sys.argv[1] if len(sys.argv) > 1 else "pythia-70m/step143000")
