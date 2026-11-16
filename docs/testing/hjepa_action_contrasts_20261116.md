# Controlled native action contrasts

Artifacts: `workspace/artifacts/hjepa-action-contrasts-v1-20261116`.
This diagnostic ran96 actual native episodes:16 new TRAIN470 worlds across eight
maps, with six constant proposed-command interventions per world. No learned
controller promotion or full-combat quality claim follows from these policies.

The interventions are idle, stationary fire, forward/back/left/right at0.5 with
fire. Turn deltas remain zero, pose release, weapon keep. Constant motor outputs
were verified independently by Go with zero parity error. Original encoders and
world components remain present but planning is disabled. Stock execution guards
remain active, so a proposed motion is not necessarily a physical displacement.

Go exported96 initial branch states. All16 worlds have exactly matching initial
871 client features and identity across all six branches. Native generated
fixtures and initial entity records also match. These checks do not assert a
complete hidden engine-state restore. Reports and frozen model hashes are bound.

The first1/4/16-frame effect targets reconcile against the existing native
first-life ledger. Terminal states are absorbing for the diagnostic horizon.
Three16-frame windows are excluded because ownership changes from provider to
rules before the horizon ends; they are not treated as continued constant
commands. This leaves285 rows. All1/4-frame pairs remain; each planned16-frame
comparison has15 paired worlds. Guard and applied-command differences are retained
separately. Source rows preserve full declared action sequences known before
outcomes, avoiding the previous observational future-action ambiguity.

## Frozen predictor audit

The previously rejected native-effect predictor was evaluated without fitting.
All16 worlds are disjoint from its64 source worlds, verified against training
metadata and frozen fit/model hashes. Comparisons pair fire vs idle and each
direction vs stationary fire. A state-only or constant predictor implies zero
contrast because the initial features are identical.

| Horizon | Paired comparisons | Predicted reward contrast MSE | Zero contrast MSE | Correct sign on nonzero effects |
|---|---:|---:|---:|---:|
| 1 | 80 | 0.022633 | 0.012302 | 13/21 |
| 4 | 80 | 0.015808 | 0.014637 | 17/33 |
| 16 | 75 | 0.265773 | 0.226264 | 26/51 |

Effects differ across interventions, but this frozen predictor does not rank
them reliably. Comparisons share16 worlds and are not80 independent samples.
They are diagnostic TRAIN interventions, not external learned-combat validation.
This reinforces rejection of the current effect head; it does not identify a
single cause or establish a preferred constant action for a learned bot.

All native jobs have finished. No fitting on these diagnostic worlds or U1
replacement occurred; reserved external VALIDATION442 remains unplayed.
Next investigate ownership during lost contact and source-separated controlled
action data, rather than another loss sweep on the same small holdout cohort.

## Ownership diagnosis: local completion, not lost-contact timeout

All three omitted windows belong to base3 seed1826160470. Only soldier385 was
observed before handoff. In stationary-fire/left-fire the server records its kill
at frame105, the client observes its death animation and no live enemies at106,
and ownership returns to rules. Back-fire has the corresponding kill106 and
handoff107. Gunner376 had never been observed at that point. It first becomes
visible at121/122, and provider ownership resumes on exactly that frame. The
rules-owned interval is15 native frames in each case. This is not the30-frame
visibility grace expiring, stale observations or provider failure.

`combatEngagement.observe` removes observed defeated identities and ends the
local engagement when none remain. It preserves ownership during temporary
occlusion of a still-known enemy. The existing three engagement tests and the
direct-provider occlusion test passed with the real base1 BSP loaded. No runtime
code change is justified by these three traces. The initial test command used
the engine directory instead of baseq2 and failed to locate BSP; corrected asset
root verification passed all four tests, with no skips.

Native kill receipts are diagnostic evidence only; the unseen gunner must not
be injected into policy observations or used to prolong combat ownership. Rules
still own between-engagement travel/search, which remains an explicit gap toward
fully learned autonomy. It should be addressed as learned search with observable
inputs, not disguised as a visibility-timeout bug. Details and owner segments:
`ownership-diagnosis.json`. The earlier proposed lost-contact explanation is
superseded by this verified diagnosis.
