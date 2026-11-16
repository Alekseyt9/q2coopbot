# Expanded controlled-action dataset and paired objective

Artifacts: `workspace/artifacts/hjepa-paired-actions-v1-20261116`.
The native collection is384 actual episodes:64 registered TRAIN worlds, six
constant proposed-command interventions per world. These are diagnostic policies,
not learned controller candidates. Only short-horizon native effects are intended
as training labels; whole300-frame outcomes are not one-action expert labels.

The six controls are identical to the verified TRAIN470 pilot. Their frozen
models are copied byte-for-byte with hashes checked; the new schedule reuses the
64 source worlds from offsets460..463. All branches of each world stay together
in the pre-existing48fit/16internal-holdout partition. This internal holdout has
already served earlier observational diagnostics, so independent external
validation remains mandatory. External VALIDATION442 stays unplayed.

The generalized admission script derives expected world coverage from the frozen
plan and checks every intervention's seed set. It reproduced every pilot row,
omission, paired contrast and rows digest exactly. Six focused pair-admission
tests passed: no cross-world reference borrowing, no changed initial state,
mixed horizon, duplicate arm or empty contrast set.

`fit-protocol.json` fixes the next comparison before reading the new outcomes:
frozen corresponding32latent plus26 observed equipment/request context, fit-only
latent normalization, full20-field action commands per tick,64-unit ReLU heads,
300 full-batch Adam1e-3 steps from the same initialization. Paired objective is
contrast MSE plus0.2absolute effect MSE; control uses only absolute effect MSE.
There is no parameter sweep or checkpoint selection. The native effects head
remains separate from the Go runtime, motor and weapon policy.

The source-bound contrast report must first validate matching initial client
observations and native fixtures/entities. Interrupted provider windows remain
explicit omissions. No hidden target information is supplied to prolong control.
The fitter and tests are `scripts/train_combat_hjepa_paired_effects.py` and
`scripts/test_train_combat_hjepa_paired_effects.py`. `handoff.json` tracks the
actual running collection handle and subsequent verified stages. Current U1 is
preserved; this preparation does not establish improvement or goal completion.

## Collection and fixed-budget fit completed

All384 native episodes completed. Initial871 client features and native initial
entities match in all64 worlds. The admitted dataset has1139 horizon rows:
13windows are explicitly omitted after provider ownership changes. The reporter
reconciles native effect ledgers and command prefixes; this does not turn later
rules-owned frames into constant-control labels.

Both objectives finished all300 predefined CUDA steps. The exported encoders
remain exactly equal to parent, and the frozen trainer digest matches the fit
report. The holdout includes16 worlds,96 arm rows and80 paired comparisons at
each horizon. Windows/comparisons share worlds and are not independent trials.

| Horizon | Paired-objective reward contrast MSE | Absolute-control MSE | Zero-contrast MSE |
|---|---:|---:|---:|
| 1 | 0.003290 | 0.002904 | 0.002732 |
| 4 | 0.004511 | 0.002985 | 0.003839 |
| 16 | 0.071582 | 0.050186 | 0.045406 |

At horizon16, paired training predicts the correct sign in36/58 nonzero contrasts
(62.1 percent), satisfying that one criterion. It fails the required magnitude
improvement and absolute-effect baseline criterion. Overall gate FAILED. No
runtime integration or additional native evaluation follows this rejected fit.

The larger controlled dataset establishes that the current fixed representation
and head/objective combination remains inadequate; it does not identify one
cause. The world encoders and motor still use845inputs, whereas only the separate
weapon/effect heads receive the observed equipped/request context. Next investigate
an optional871world/motor contract with exact zero-extension compatibility, then
joint representation learning from the retained trajectories. Preserve old845
models and U1 and avoid further tuning against this same16-world holdout.
Reserved external VALIDATION442 remains unplayed. All owned jobs have finished.
