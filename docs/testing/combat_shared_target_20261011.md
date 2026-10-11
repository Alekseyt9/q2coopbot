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

Shared-branch training completed on CUDA,8376 eligible rows,10 accepted actor
proposals,update10. Weights SHA
`9976bb8c9f4bdb3eb88c7687bd4e225603abad95fd75d8004ab7ec392ed346e4`;
checkpoint SHA `09ef34db88c95568b155b99d8152c118f9d3acda456b9695db304ae3e793bc5a`.
The checkpoint audit confirms110236 inactive actor scalars,std and inactive
Adam states preserved;10170 selected/shared scalars changed. New branch Adam
states and accepted-step clocks were checked explicitly. Consumed-rollout
history appended once. Receipt: `combat-shared-target-checkpoint-audit-v1-20261011.json`.

Full-history head diagnostic confirms movement/vertical/aim/mode KL exactly
zero; target probabilities change by1.734 percentage points in mean total
variation. This is a training-history diagnostic, not a held-out game gain.
Receipt: `combat-shared-target-head-learning-v1-20261011.json`.

The matched control also completed CUDA training:8369 eligible rows from its
own64 accepted captures,10 accepted proposals,update10. Weights SHA
`f191b0d2d0428f88ebb6d6529f4de5c6a4c101372db8ce6266eccb5ca886bdf8`;
checkpoint SHA `615c0d785b712fcb46e230dffe77761c387c56564d2cc7b4501f0f8a2f22c08f`.
Control checkpoint audit preserves the same110236 inactive scalars/Adam,std;
1153 selected scalars changed. Capture conditions and training proposal
budgets match, but native frame/eligible-row counts differ slightly. Do not
claim bitwise identical trajectories or replay the other's traces as own-policy
data. The held-out quality comparator intentionally rejects train splits;
training captures are not used as an architecture acceptance benchmark.

`audit_combat_input_use_cuda.py` now provides a reusable full-history ablation
with a separately verified behavior actor and an optional updated actor. It
checks captured likelihood/memory/source/stream bindings on CUDA, then removes
each band across context while holding current availability masks fixed.
Native omitted zero-valued weapon/target/aim-mode samples follow their JSON
contract defaults. An initial missing-default diagnostic failed before output;
the corrected run passed. Shared update10 on8376 training rows:

| Removed band | Target mean total variation | Changed argmax rows |
|---|---:|---:|
| Previous target | 0.000239012 | 6 |
| Navigation | 0.000050679 | 0 |
| Remembered threats | 0.000106818 | 2 |

Receipt: `combat-shared-target-input-use-v1-20261011.json`. These effects are
still small. The same-data parent diagnostic has completed on the same8376
rows: target total variation0.000001063 for previous target,0.000000193 for
navigation,0.000001666 for remembered threats; zero changed argmax rows for
all three. Receipt: `combat-shared-target-parent-input-use-v1-20261011.json`.
This comparison controls the diagnostic corpus; it does not separate shared
representation training from ordinary head training without the matched
trained control, or prove useful decisions. Different earlier corpora must
not be used to attribute an increase solely to the new branch.

Held-out pool `combat-shared-target-eval-pool-v1-20261011` now runs320 battles,
16 slots×2,20 recipes×four unseen test offsets8:12×four controllers. Index0
unchanged update9;1 equally trained control update10;2 shared-target update10;
3 ordinary rules. The private20-recipe rules-capable registry is frozen across
all four plans. This covers pair and trio scenes, both inventory families and
first/second sites on four maps. All controllers use identical recipe fixtures
and seed lists under the same source/runtime build. Whole-pool acceptance,
paired environment/binary checks and learned CUDA replay remain required;
results and architecture promotion are pending.

Held-out pool closed320/320 with unchanged source. Native paired comparisons
now show frozen33 goals,control30,shared28,rules23 per80 cases. Shared ware2
goals8 versus frozen16 and control13. The new candidate is not promoted;
frozen update9 stays the reference. Full results and remaining CUDA/behavior
verification are recorded in `combat_shared_target_quality_20261011.md`.

`audit_registered_target_choices.py` now accepts an explicit learned plan from
a fully accepted mixed pool and adds observed/requested movement, stationary
clear-enemy runs, first-provider-to-first-attack delay, large applied turns and
attack-without-selected-target frame counts. Native attack commands are not
bullet counts/hit accuracy, and these observations do not label an action
useless. Closed64-case training audit passed before running all three held-out
variant audits. It separates stationary frames with and without movement
requests, so blocked movement is not mistaken for a deliberate stop.

Registered processing now LZX-compacts native exports after exporter exit and
memory-contract verification, and compacts merged streams after producer exit.
The existing helper checks unchanged logical SHA before/after. Current
evaluation processors were already running the previous main body; a separate
receipt-controlled pass compresses only their finished native exports after
all80 native workers exited. Active CUDA output files are excluded.

The previous evaluation's three CUDA finalizers and behavior audits all
completed:240 cases,26686 eligible rows. Native export compaction also closed
with a complete receipt. Aggregate behavior shows stationary-provider fractions
23.95%/22.64%/20.61% for parent/control/shared despite worse candidate goals;
first attack-command delay averages0.9/1.1/1.25 provider frames, with no
never-attacked cases. These counts neither explain every loss nor measure
bullet accuracy. The regression is ordinary-weapon trios11→6 goals/32; pairs
stay10/16. A matched target-only scope comparison now runs on fresh seeds;
see `combat_target_scope_20261011.md`.
