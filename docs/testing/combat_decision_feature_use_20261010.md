# Decision-head feature-use diagnostic (2026-10-10)

The decision-scope update improves some held-out combat results, but freezes
the base/attention encoder. Does that actor meaningfully use the newer
previous-target, navigation and remembered-threat inputs?

Receipt: `workspace/artifacts/combat-decision-input-ablation-v1-20261010.json`.
Entry script: `workspace/artifacts/audit-decision-input-ablation-v1-20261010.py`.
Actor: selected update9, SHA
`be4c135ed530269c22e99c63d32e465cfef99a14360e0f825e4b64577e479caa`.
Data: own-policy trio training histories,6409 eligible rows, from
`combat-decision-trio-training-recovery-v1-20261010/merged`.

CUDA reconstructs all history segments and checks captured likelihood/state
parity. Then each feature band is independently zeroed across the entire
context, keeping physical observations and current target/weapon availability
fixed. The actor is not trained, and no evaluation observations are used.

| Ablated band | Target mean total variation | Target argmax changes |
|---|---:|---:|
| Previous-target one-hot,845:854 | 0.000001129 | 0/6409 |
| Navigation,854:881 | 0.000000250 | 0/6409 |
| Remembered threats,881:1121 | 0.000002644 | 0/6409 |

For example, remembered-threat ablation changes target probabilities by only
0.000264 percentage points on average. Effects are nonzero; this is not proof
that the inputs are entirely ignored. Small first-encoder weights and these
sensitivities show that merely supplying new input columns does not establish
useful decision learning. The discrete-only scope cannot change the frozen
encoder's input representations. Temporal attention also still carries its
own history, so ablating threat-memory columns does not remove all memory.

This is a counterfactual model diagnostic on training histories, not a valid
gameplay trajectory, causal reward attribution or held-out quality score.
It motivates a separate learned decision representation rather than proving
which architecture will win.

Next candidate: a shared target scorer that reads observed per-enemy features
directly (type, relative position/velocity, bbox/visibility/clear-shot evidence,
distance, previous-target flag), with explicit group context and remembered
threats. Keep movement and aim weights intact; initialize the new target
residual to zero and verify exact old policy/checkpoint state on CUDA. Train
the new representation on fresh own-policy captures, then compare on new
sites/seeds against the unchanged reference and ordinary rules. No hidden
native monster health or test labels enter inference. Activation and live
promotion require actual paired game evidence; this candidate is not
implemented or accepted yet.
