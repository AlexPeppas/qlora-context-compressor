# Reading the training logs

Three columns, three independent stories. Read together = health check.

## Loss: 1.74 → 0.33 (5×↓ over 3 epochs)

Cross-entropy on next-token prediction over the assistant span only. Mental conversion: `prob ≈ exp(-loss)`.

| Loss | Approx. prob. on correct token |
|------|-------------------------------|
| 1.74 | ~18% |
| 0.94 | ~39% |
| 0.53 | ~59% |
| 0.33 | ~72% |

Phase-by-phase:
- **Epoch 0 (1.74 → 0.94):** rapid descent — model learns the *format* (chat template, compression style).
- **Epoch 1 (0.94 → 0.53):** flatter — learning the *substance* (what to keep vs drop per turn-age tier).
- **Epoch 2 (0.53 → 0.33):** refinement; mild overfitting risk on 999 examples but no spike.

## Gradient norm: 0.28 → 0.78 (rising — and that's good)

L2 norm of all LoRA grads each step, pre-clipping. Rising during training is normal:

1. **Early grads are noisy and self-cancelling.** 40M trainable params from random init produce conflicting directions that partially cancel in the L2 norm.
2. **Late grads are concentrated.** As the loss surface narrows, grads from different examples increasingly agree → less cancellation → larger total norm.
3. **Loss drops faster than grads grow.** Net signal-to-noise per step improves over training.

Threshold check: `max_grad_norm=1.0` is the default clip. Max observed = 0.78 → **clipping never engaged**. If you ever see 1.5+, clipping is silently truncating updates and the LR is too aggressive.

## Learning rate: warmup → peak → cosine to zero

Three regimes:

| Phase | Steps / Epoch | LR | Why |
|-------|---------------|-----|-----|
| Warmup | step 1-2 / 0.08-0.16 | 1.05e-4 → 2.0e-4 | Adam's m, v moments init to 0; warm them up before peak LR |
| Peak | ~epoch 0.16 | **2.0e-4** | Standard QLoRA recipe; 5-10× higher than full FT because only 0.5% of params are trainable |
| Cosine decay | epoch 0.16 → 3.0 | 2.0e-4 → 9e-7 | Spends budget at productive LRs, small tail at ~0 for stable basin convergence |

## Joint health checklist

A healthy SFT curve has all four. This run has all four:

- [x] Loss decreases monotonically, no spikes
- [x] Grad_norm stays well below clip threshold throughout
- [x] LR decay correlates with diminishing loss returns (biggest drops during high-LR window)
- [x] No grad_norm spike → loss spike sequence (which would indicate exploding gradient or bad batch)

## What it does NOT tell you

Loss is computed on training data only:

- **Generalization** — measured by the bake-off, not by training loss.
- **EOS supervision worked** — the whole point of this rerun. Loss including EOS tokens converged → EOS was in the loss, but the *behavioral* proof is inference output ending naturally with `<|im_end|>`.
- **Overfitting magnitude** — no held-out validation loss is computed. Would need to add `eval_dataset` + `eval_steps` to detect.

Final loss 0.33: in the typical range for SFT on a small specialized dataset. Not pure memorization (<0.1), not underfit (>1.0). Healthy.
