# Shared target residual integration, 2026-10-11

Motivation: selected decision update9 barely changes target selection when
previous-target/navigation/threat-memory columns are ablated. A frozen main
encoder cannot learn a better representation of these new columns.

The optional `shared_target` weights contract is
`combat_shared_target_residual_v1`: shared219→64→32→1 ReLU scorer,16193
parameters, added to enemy target logits21:29 only. None logit20 is unchanged;
existing target availability masks remain authoritative. Remembered enemies
cannot be selected unless the original current-observation mask allows them.

Inputs:119 observed spatial/weapon/self/planned-movement features, one
previous-target flag per current slot,27 navigation features, masked mean and
max over eight30-dimensional observed threat-memory records, and12-dimensional
masked group mean. No native hidden positions, monster health, outcomes or
future labels are added. The scorer is independent of the temporal encoder;
the actor retains its existing TemporalAttention64 and spatial aim branches.

Go inference support lives in `internal/policy/shared_target.go`, after the
spatial residual. Python uses a `target_branch` attached to the existing
`SpatialActor`, keeping old actor/state parameter names and ordering intact.
The same wrapper is used by PPO, CUDA finalization and head diagnostics.
Target/decision scopes now train the shared branch while preserving inactive
head rows, spatial/encoder/std parameters and inactive Adam states. Existing
models without the field keep their former path.

`migrate_combat_shared_target_cuda.py` appends a zero final layer and migrates
the selected update9 checkpoint. Old actor Adam parameter IDs remain stable;
std state moves past the six newly appended branch tensors by explicit name
mapping. New branch Adam state starts empty. Critic, std, RNG, counters,
consumed rollout history, training configuration and scope are copied intact;
an explicit `model_migrations` lineage entry records the architecture addition.

Accepted migration:
`workspace/artifacts/combat-shared-target-migration-v2-20261011/complete.json`.
Weights SHA `a5839761bd7d25280962c18354d4914d30a98501394bff53bfb511c394858600`;
checkpoint SHA `033247ec9a6b290000b1a5f21b0c03940a72ff5bc86945038a2f93d31b2a7cec`.
CUDA checks confirm exact actor outputs and memory on a33-frame sequence,
old parameter equality and name-mapped old Adam state preservation. This is
a migration, not an additional PPO update. The earlier v1 directory contains
an interrupted report publication and has no completion seal; do not use it.

Validation so far:

- Go client/episode/export command builds passed;
- three shared-target CUDA tests passed, including integrated projected
  training with exact inactive-output/spatial preservation;
- three existing projected head-scope CUDA tests passed;
- architecture export/resume CUDA integration test passed for GRU, temporal
  attention and entity attention without the optional branch.

Native smoke pool `combat-shared-target-smoke-pool-v1-20261011` runs16 battles,
16 slots×2: eight with the zero migrated model and eight with an intentionally
nonzero diagnostic branch. It uses second-site base3/jail1 trio ordinary
weapon scenes and validation offsets40:44, matched across variants. The
nonzero model is a numerical parity probe, not trained/promotable weights.
Both streams must pass CUDA likelihood/history verification before new
own-policy training. Gameplay superiority and live coop acceptance are pending.

Smoke closed16/16 usable with unchanged source. Both CUDA finalizers passed,
1033 eligible rows each; no training on these validation cases. The nonzero
probe audit on its captured histories confirms only target distributions change
(other head KL exactly zero), so the live branch is not merely accepted as
unused JSON. The effect is deliberately small and is not a quality result.
Receipt: `combat-shared-target-nonzero-probe-audit-v1-20261011.json`.

`combat-shared-target-zero-real-history-audit-v1-20261011.json` also compares
the migrated zero actor against the unchanged update9 on all1033 actual smoke
history rows: every head KL and target total variation are exactly zero.
This confirms initialization preservation on captured context in addition to
the migration's synthetic33-frame output/memory test. Neither audit is fitting
the validation histories or proving combat superiority.

Fresh own-policy collection now runs64 train battles: eight registered second-
site trio recipes, four maps×two inventories, eight seeds each at train offsets
24:32. Pool `combat-shared-target-training-pool-v1-20261011`,16 slots×2,
weights pinned to the zero migrated model. Resume from the sealed v2 checkpoint
in the same decision scope/LR. No previous validation/test traces are reused
for training. Await accepted capture and CUDA finalization before PPO.

Training capture closed64/64 usable with unchanged source. CUDA processing and
the resumed decision-scope update are now running in
`combat-shared-target-training-processing-v1-20261011`.

A matched control collects its own64 battles with the unchanged update9 model
on the same eight recipes/train seeds, then resumes the original checkpoint
with the same objective, scope, LR and maximum actor/value proposal budget.
Pool: `combat-shared-target-control-training-pool-v1-20261011`. This separates
the representation experiment from simply giving the parent more training.
KL guards can accept different numbers of proposals; report those counts.
Captures are not relabeled or reused across model SHA identities, even though
the zero-residual initialization has matching policy distributions. Final
held-out comparison must include both trained variants and the frozen parent.
