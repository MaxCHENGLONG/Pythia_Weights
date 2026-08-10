# Strict check: exported weights + hand-written forward == HuggingFace
# ---------------------------------------------------------------------
# Usage:
#   python verify_hand_vs_hf.py pythia-70m/step143000
#
# The self-check inside Pythia_model_hand.py only compares final logits on one
# prompt, so a sign error that the last LayerNorm mostly absorbs could survive it.
# This goes further:
#   1. per_head/ tensors on disk rebuild the fused/ ones bit-for-bit
#   2. every layer's residual stream matches HF's output_hidden_states
#   3. every layer's attention probabilities match HF's output_attentions
#   4. logits match and argmax agrees at every position
# on three lengths (1, 17, 128) - T=1 exercises the degenerate causal mask - in both
# float32 and float64.
#
# Why attention probabilities get their own, much looser tolerance:
# rel() divides by max|theirs|, but attention probabilities always max out at 1.0, so
# for them rel() is an absolute error - a far harsher test than the same number applied
# to the residual stream, whose scale is inflated by Pythia's outlier dimensions.
# Measured on pythia-70m/step143000 at T=128 (diagnose_mismatch.py), the per-layer
# relative error sits at a few float32 ulps through layer 2 and then jumps: layer 3's
# attention turns ~1e-6 of input error into 2.9e-3 of probability error, because
# LayerNorm's normaliser is dominated by the outlier dims and a sharp softmax amplifies
# the resulting coherent shift. Layer 0 is the one layer with nothing to hide behind -
# its input is the embeddings, bit-identical on both sides - so it stays tight, and it
# is the layer that would catch a rotary, mask or head-splitting bug.
#
# The float64 pass is the control: both sides widen the same float32 weight files, so
# if the float32 gaps are precision rather than a real difference in what is computed,
# they collapse. (HF's eager softmax may internally compute in float32, which would
# floor the float64 probability gap near 1e-7 instead of 1e-13 - it still collapses.)

import os
import sys

import torch
from transformers import AutoModelForCausalLM

from Pythia_model_hand import PythiaHand

TOL = 1e-4          # residual stream and logits, relative to tensor scale
PROB_TOL = 5e-2     # attention probabilities, absolute
PROB_TOL_L0 = 1e-5  # layer 0 carries no accumulated error


def rel(ours, theirs):
    """max|ours - theirs| divided by the scale of theirs."""
    scale = theirs.abs().max().clamp(min=1e-12)
    return ((ours - theirs).abs().max() / scale).item()


def check_per_head(path, hand):
    """The per_head/ files must be exact re-slicings of the fused/ ones."""
    per_head_dir = os.path.join(path, "per_head")
    ph = {
        name[:-3]: torch.load(os.path.join(per_head_dir, name), map_location="cpu")
        for name in os.listdir(per_head_dir) if name.endswith(".pt")
    }
    H = hand.hidden_size

    print("per_head/ vs fused/ (exact equality):")
    for name in ("Q", "K", "V"):
        # [L, heads, H, head_dim] -> [L, H, heads * head_dim]
        w = ph[f"W_{name}_heads"].permute(0, 2, 1, 3).reshape(-1, H, H)
        b = ph[f"b_{name}_heads"].reshape(-1, H)
        assert torch.equal(w, hand.w[f"W_{name}"]), f"W_{name}_heads != W_{name}"
        assert torch.equal(b, hand.w[f"b_{name}"]), f"b_{name}_heads != b_{name}"
        print(f"  W_{name}_heads, b_{name}_heads  ok")
    # [L, heads, head_dim, H] -> [L, heads * head_dim, H]
    w_o = ph["W_O_heads"].reshape(-1, H, H)
    assert torch.equal(w_o, hand.w["W_O"]), "W_O_heads != W_O"
    print("  W_O_heads             ok")


def check_forward(hand, ref, ids):
    """Compare residual stream, attention probs and logits at every layer."""
    logits, cache = hand(ids, return_hidden=True)
    with torch.no_grad():
        out = ref(ids[None], output_hidden_states=True, output_attentions=True)

    # hidden_states[i] is the input to layer i; the last entry is post-final-LN,
    # which is exactly what cache["resid"] holds.
    resid = max(rel(cache["resid"][i], h[0]) for i, h in enumerate(out.hidden_states))
    probs = [(cache["attn_probs"][i] - a[0]).abs().max().item()
             for i, a in enumerate(out.attentions)]
    logit = rel(logits, out.logits[0])
    agree = bool((logits.argmax(-1) == out.logits[0].argmax(-1)).all())

    ok = (resid < TOL and logit < TOL and agree
          and probs[0] < PROB_TOL_L0 and max(probs) < PROB_TOL)
    print(f"    T={len(ids):3d}  resid={resid:.2e}  probs_L0={probs[0]:.2e}  "
          f"probs_max={max(probs):.2e}  logits={logit:.2e}  "
          f"argmax_agrees={agree}  {'ok' if ok else 'FAIL'}")
    return ok


def main(path):
    hand = PythiaHand.from_dir(path)
    meta = hand.meta
    print(f"{meta['model']} @ {meta['revision']}: L={hand.num_layers} "
          f"H={hand.hidden_size} heads={hand.num_heads} "
          f"rotary={hand.rotary_ndims}/{hand.head_dim}\n")

    check_per_head(path, hand)

    # eager attention so that output_attentions actually returns probabilities.
    ref = AutoModelForCausalLM.from_pretrained(
        meta["model"], revision=meta["revision"],
        dtype=torch.float32, attn_implementation="eager",
    )
    ref.eval()

    print(f"\nhand-written vs HuggingFace (resid/logits relative, tol {TOL:.0e}; "
          f"probs absolute, tol {PROB_TOL_L0:.0e} at layer 0, {PROB_TOL:.0e} after):")
    ok = True
    for dtype in (torch.float32, torch.float64):
        # Both sides widen the same float32 weights, so the float64 pass isolates
        # arithmetic precision from any real difference in what is computed.
        theirs = ref.to(dtype)
        ours = hand if dtype is torch.float32 else PythiaHand.from_dir(path, dtype=dtype)
        print(f"  {str(dtype).rsplit('.', 1)[-1]}:")
        torch.manual_seed(0)
        # list, not a generator: all() would short-circuit and hide the later lengths.
        ok &= all([
            check_forward(ours, theirs, torch.randint(0, 50277, (T,)))
            for T in (1, 17, 128)
        ])

    print()
    if not ok:
        raise SystemExit("MISMATCH")
    print("All checks passed.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pythia-70m/step143000")
