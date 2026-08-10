# Localise the hand-written vs HuggingFace mismatch.
# ------------------------------------------------
# Usage:
#   python diagnose_mismatch.py pythia-70m/step143000 [T]
#
# verify_hand_vs_hf.py reports one number per length; this reports one per layer.
# Layer 0 sees the raw embeddings, which are identical on both sides, so its
# attention probabilities carry no accumulated error:
#   layer 0 already off by ~1e-2  -> a real implementation difference
#   layer 0 ~1e-7, growing with depth -> accumulation, amplified by near-tied softmax rows

import sys

import torch
from transformers import AutoModelForCausalLM

from Pythia_model_hand import PythiaHand


def main(path, T):
    hand = PythiaHand.from_dir(path)
    ref = AutoModelForCausalLM.from_pretrained(
        hand.meta["model"], revision=hand.meta["revision"],
        dtype=torch.float32, attn_implementation="eager",
    )
    ref.eval()

    torch.manual_seed(0)
    ids = torch.randint(0, 50277, (T,))
    _, cache = hand(ids, return_hidden=True)
    with torch.no_grad():
        out = ref(ids[None], output_hidden_states=True, output_attentions=True)

    print(f"T={T}  (resid scale = max|hf| on that layer, shows the outlier dims)")
    print("  layer   resid max|d|   resid scale   probs max|d|")
    for i, h in enumerate(out.hidden_states):
        d = (cache["resid"][i] - h[0]).abs().max().item()
        line = f"  {i:5d}   {d:10.2e}   {h[0].abs().max().item():10.2e}"
        if i < len(out.attentions):
            p = (cache["attn_probs"][i] - out.attentions[i][0]).abs().max().item()
            line += f"   {p:10.2e}"
        print(line)

    # Where the worst probability error lives, and whether that softmax row is tied.
    worst = max(range(len(out.attentions)),
                key=lambda i: (cache["attn_probs"][i] - out.attentions[i][0]).abs().max())
    ours, theirs = cache["attn_probs"][worst], out.attentions[worst][0]
    n, q, k = (x.item() for x in torch.unravel_index((ours - theirs).abs().argmax(), ours.shape))
    print(f"\nworst prob error: layer {worst} head {n} query {q} key {k}")
    print(f"  ours={ours[n, q, k]:.6f}  hf={theirs[n, q, k]:.6f}")
    top = ours[n, q].topk(min(4, q + 1))
    print(f"  top of that row: {[round(v, 6) for v in top.values.tolist()]} at {top.indices.tolist()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pythia-70m/step143000",
         int(sys.argv[2]) if len(sys.argv) > 2 else 128)
