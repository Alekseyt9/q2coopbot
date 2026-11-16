# Direct native-effect targets for H-JEPA

Producer: `scripts/prepare_combat_hjepa_effects.py`.
Data: `workspace/artifacts/hjepa-world-coverage-v1-20261116/effects`.
This is prepared supervised dynamics data, not a trained or deployed controller.
Current U1 and all H-JEPA checkpoints remain unchanged.

The existing Go rollout exporter reproduces all7383 choices from64 TRAIN worlds
and exports871 causal client features, including equipped weapon and own previous
request. The producer verifies every retained source digest and reconciles all64
first-life native reward ledgers before constructing targets. Registered world
identities remain unchanged. The previously frozen48/16 source-world partition
is preserved exactly; no world occurs in both parts.

Each sample contains initial871features and an actual proposed action sequence
over1,4 or16 contiguous frames. Each command has eight primitive fields plus a
12-way one-hot weapon request, including keep. Thus different weapon requests
are distinguishable even when movement is identical. Outcomes, world seed,
source digest, frame and step indices are labels/metadata, never neural inputs.

Six targets describe monster HP progress, received HP/100, self HP/100, teammate
HP/100, win and death. Their fixed dot product with
`[0.1,-0.1,-1,-0.5,1,-1]` exactly reproduces the established native reward/100
for every causal step. Health loss is clamped to actual remaining health;
monster progress saturates at initial monster HP. Events outside the first-life
endpoint receive no label. Terminal bonuses attach to their unique verified
observation transition. Full windows only: no future-action padding, crossing
control transfers or post-death examples. Terminal nonexclusive windows would
be explicitly marked; none occur in the retained data.

| Horizon | Fit rows (48 worlds) | Internal holdout rows (16 worlds) |
|---|---:|---:|
| 1 | 5463 | 1886 |
| 4 | 5316 | 1838 |
| 16 | 4740 | 1646 |

These overlapping windows are not independent trials. Horizon16 has2888 fit and
1109 holdout windows with a nonzero effect; terminal events remain sparse:7wins
and26deaths in fit windows,2wins and7deaths in holdout windows. These counts do
not describe new evaluation games or establish training sufficiency.

Seven focused effect tests plus nine existing ledger tests passed, covering
overkill, received+self penalties, unique win/death placement, post-endpoint
exclusion, inconsistent native counts and distinguishable weapon actions.
An independent full-data pass checked input/action shapes, disjoint world sets
and reward projection for every exported row; see `effects/validation.json`.

Next fit direct effect prediction with fit-only normalization and compare a
same-budget action-ablated model plus constant mean/zero baselines on the16
internal holdout worlds. A lower fitting error alone does not justify integrating
this head into planning. Fresh external validation remains necessary.
