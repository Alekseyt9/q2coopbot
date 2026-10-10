# Own-policy decision training on trios (2026-10-10)

## Parent and new training corpus

The previously confirmed experimental decision-scope reference wins38/96
against26/96 for joint training on the same cases. It is a training reference,
not live or campaign acceptance. See `combat_decision_scope_20261010.md`.

All artifacts below are under `workspace/artifacts/`.
Parent: `combat-decision-scope-pair-update-v1-20261010`, update9,
weights `be4c135ed530269c22e99c63d32e465cfef99a14360e0f825e4b64577e479caa`.
Own-policy capture pool: `combat-decision-trio-training-pool-v1-20261010`,
64/64 usable captures, source unchanged,16 instances×2. Eight new train seeds
per trio descriptor, offset12;32 SSG-inventory and32 MG/shotgun/blaster fights
on base3/ware2/jail1/city1. Native outcomes:20 goals,43 deaths,1 horizon;
SSG14/32 goals17 deaths, ordinary inventory6/32 goals26 deaths.

## Windows progress-file failure and recovery

The original processing root `combat-decision-trio-training-processing-v1-20261010`
failed at an atomic rename of CUDA-batch progress (`WinError5`). No training
had started. A concurrent Windows reader can briefly omit delete-sharing.
`process_combat_architecture_pool.save` now retries only atomic replacement
for Windows sharing/access errors5/32/33, bounded to five seconds. It keeps
the old complete JSON visible; a permanent error propagates and preserves
the partial file. Two focused metadata tests verify transient-lock publication
and permanent-error bounds/preservation. These tests do not run neural inference.

Original failed diagnostics are retained. The recovery script
`resume-decision-trio-finalize-v1-20261010.py` validates all56 closed CUDA
outputs against model/native/request/source/stream/audit hashes, then finishes
only the8 missing outputs in a new batch. Unsealed partial outputs are refused.
`combat-decision-trio-training-recovery-v1-20261010/verified.json` pins the
combined64 cases and original closed-pool/request hashes. No fights or closed
CUDA outputs are regenerated. The original batch entrypoint is unchanged,
so completed source bindings remain valid.

## GPU update

`combat-decision-trio-training-recovery-v1-20261010/update`:6409 eligible
rows,6 accepted actor steps, update10, sampled approximate KL0.0049971.
Weights SHA `696392efbb1813eaeee4f6b3dfec291e6711911073a82274077267990d1b23c1`.
Resume retains decision scope/LR0.003, lambda0.95, reward and optimizer state;
no new scope/config fork. Only attack/weapon/target actor outputs train;
the critic updates normally. All training and numerical verification use CUDA.
Checkpoint audit: `combat-decision-trio-checkpoint-audit-v1-20261010.json`,
1252 selected scalars change;110236 inactive actor scalars, log_std and inactive
Adam states are exactly preserved. The consumed rollout appends once.
Head audit `combat-decision-trio-head-audit-v1-20261010.json` passes CUDA
likelihood/state replay. Exact KL0.005139: target0.004921, weapon0.000217,
attack0.000000580; frozen heads exactly zero. Target mean total variation
is4.07 percentage points, argmax changes on2.26% of these training rows.
The second update mostly changes target probabilities; this does not prove
better target choices. Approximate sampled KL and exact full KL are different
measurements, so this diagnostic is not the trainer's acceptance threshold.

## Broader test geometry

Draft model-independent registry:
`combat-decision-expanded-site-registry-v1-20261010/index.json`,20 recipes.
It copies12 previous descriptors byte-for-byte and adds8 trio descriptors
at the original second encounter site on each of the four maps, with both
loadouts. New source entity IDs and origins differ from their first-site
counterparts. `combat-decision-expanded-site-registry-provenance-v1-20261010.json`
pins descriptor-generation and parent-plan
sources. Global registry is unchanged; these new recipes need native acceptance.

`combat-decision-expanded-pool-v1-20261010` compares parent update9 and
new update10 on80 identical test-split cases each:48 previous-site cases,
32 new trio-site cases, fresh test seeds offset0/count4. Pool16×2,
indices0 before/1 after. Plans passed static BSP/fixture validation.
The new geometry was absent from this training update. Report old/new sites
and composition/loadout separately; lower duration alone is not better combat.
No test-data fitting or live replacement. Evaluation is running.

## Closed expanded comparison

Pool closed160/160 usable, source unchanged. Receipt:
`combat-decision-expanded-quality-v1-20261010.json`.
The comparator now also verifies current native runtime files against the
recorded q2ded/game.dll/entity/AAS hashes and requires equal runtime bindings
and reset controls across each pair. A focused nonneural test detects missing
native modules, duplicate file bindings and changed bytes. This improves
binary evidence; complete world/AI-state equivalence is still not established.

| 80 identical test fights | Update9 | Update10 |
|---|---:|---:|
| Goals | 29 | 32 |
| Kills | 106 | 111 |
| Deaths | 42 | 41 |
| Dealt health damage | 22261 | 22726 |
| Received health damage | 5454 | 5589 |
| Game frames | 9614 | 10422 |

| Geometry group | Cases | Goals9 | Goals10 | Kills9 | Kills10 | Deaths9 | Deaths10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous sites | 48 | 20 | 21 | 68 | 64 | 24 | 25 |
| New trio sites | 32 | 9 | 11 | 38 | 47 | 18 | 16 |

Pair-SSG goals9→11; trio-SSG17→15; trio ordinary inventory3→6.
Base3 goals5→8, city1 13→12, jail1 0→0, ware2 11→12. Jail1 kills9→5.
The update has small gains and regressions, not convincing general superiority.
It remains a separate experiment; the previously selected update9 training
reference is unchanged. Eight new descriptors passed their native capture
acceptance; global registration can now preserve them for later batches.

CUDA finalization: `combat-decision-expanded-quality-processing-v1-20261010`,
80 candidate cases,10105 eligible rows, completed on CUDA. No training on test data.

## Ordinary rules baseline

`combat-decision-expanded-rules-pool-v1-20261010` captures the same80 fixtures
and test seeds under the ordinary rules controller,16 slots×2. The comparator
allows rules only without a trained provider and checks the same frozen
fixture, client/exporter/native-source, runtime-binary and reset bindings.
Both learned checkpoints will be compared with that baseline; no neural
verification or training runs on CPU. Initial baseline preflight rejected
rules because group descriptors allowed only learned, and multiweapon rules
were also forbidden by registry/harness/client configuration. No baseline
battle started in those rejected attempts.

The contract now permits synchronous rules with stock multiweapon inventory;
learned mode still requires its masked PPO weapon head. Ordinary rules weapon
selection resumes only after the release barrier, preserving the same reset
weapon phase and inventory before combat. Group generators now allow both
controllers. Focused registry/geometry/seed tests and rules-capture metadata
rejection tests pass; these do not run neural inference. The20-descriptor
rules draft changes only the controller allowlist; generated fixture equality
must be checked before the full comparison.

`combat-rules-multiweapon-smoke-pool-v1-20261010` is running8 live rules cases
on the second base3 site, both inventory types. Runtime source edits change
the client binary, so old learned captures cannot be paired with new rules
captures under the strict equality guard. After smoke acceptance, recapture
both frozen learned checkpoints and rules in one common new-source pool.

Smoke completed8/8 usable captures, unchanged source:1 goal,7 deaths,
9 kills and6 weapon requests. All manifests confirm ordinary rules, no
trained weights and zero provider-controlled frames. This validates the
new benchmark path, not rules quality on the full registry.

`combat-rules-ab-pool-v1-20261010` is now running240 cases in the same new
source/binary configuration,16 slots×2. Index0 rules,1 frozen update9,
2 frozen update10;80 test cases each, identical generated fixtures checked
against the previous test plan before capture. The old rejected rules plans
and learned-only results remain diagnostic evidence, not mixed into this
new-source paired comparison. Baseline and full-campaign superiority remain
unproven until the corresponding accepted comparisons complete.

2026-10-11: the common-source240-case pool and both80-case CUDA finalizers are
complete. Results and promotion limitations are recorded in
`combat_rules_paired_quality_20261011.md`: rules26 goals,update9 29,update10 32,
but fewer kills/more received damage for both learned models, and0/20 jail1
goals. Update10 remains unpromoted. CUDA verified9371 update9 and10111 update10
eligible rows; no evaluation training was performed.
