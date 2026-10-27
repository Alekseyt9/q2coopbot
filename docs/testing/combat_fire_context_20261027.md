# Observed context for conditional fire, 2026-10-27

Implemented versioned context extension; combat quality is not established. It follows the closed phase-weight/LR experiments in `combat_fire_phase_balance_20261027.md` and their verified fire-branch input limitation.

## Contract

`combat_action_conditioned_fire_context_v2` keeps the complete original 181 fire/action inputs, then appends existing legacy observation channels:

- 49 projectile channels (`169:218`): availability plus four slots with observed presence, distance, relative position and tracked velocity fields.
- 37 nearby-prop channels (`259:296`): availability plus four observed object slots, including the existing explosive-box flag.

The branch is 267 → 32 → 16 → 1, with 9,121 parameters versus 6,369 originally. The full actor feature width and other branches are unchanged. The legacy fire-input helper stays 181 wide for action geometry and the separate historical Shotgun residual. No hidden projectile state, future event, new visibility, forced firing rule or projectile-type channel is introduced.

Go runtime and validation: `internal/policy/conditional_fire.go`. CUDA runtime, validation and export: `scripts/combat_conditional_fire.py`. The old version continues to require 181 inputs; the new version requires 267. Go metadata/input-packing tests pass with `TestConditionalFireContextContract` and the existing conditional-fire contract test. They do not execute neural inference on CPU.

## Explicit checkpoint migration

`scripts/migrate_combat_fire_context_cuda.py` starts from the original full-policy update 09 checkpoint and validates its restored CUDA scope. It appends 86 zero columns to the first fire layer, retaining all old values. The corresponding Adam first/second moment columns are padded with zero; existing columns, moments and step counters are preserved. Every other actor, joint, critic, optimizer and consumed-rollout record is retained. The resulting checkpoint restores against its own exported JSON without bypassing strict shape/configuration checks.

Actual migration evidence: `workspace/artifacts/combat-fire-context-source-v1-20261027/`.

- Weights SHA-256: `5ebce1e4a9d6cd8007131eba3331cb3b6e35cf30c05de01f1a03eef15083543e`.
- Checkpoint SHA-256: `58bb71ee5c05bb1f81dd067982053e52b1fea7308d1dd5dc4c1906a02611f847`.
- Source global update count remains nine; no gameplay fitting has occurred during migration.

This is an explicit architecture migration, not an optimizer reset. Its seal alone does not prove native behavior identity or quality.

## CUDA proof

`workspace/artifacts/combat-fire-context-check-v1-20261027/report.json` passed:

- On source-matched recorded TRAIN observations, old/new branch residual difference is at most 7.45e-9; full recorded joint likelihood matches within the existing numerical tolerance.
- New input columns receive a nonzero derivative on real recorded context; gradients remain restricted to the fire branch in the narrow training scope.
- Original Adam history survives the scope fork; non-fire parameters and moments stay exact.
- Eight synthetic optimizer steps equal four/save/reload/four in exported weights and both optimizer states. These synthetic artifacts are marked technical-only and cannot train on gameplay data.

The historical TRAIN observations are used for derivative/numerical checks only; no optimizer update uses their combat rewards.

## Native identity check

`workspace/artifacts/combat-fire-context-native-zero-v1-20261027/run.py` selects four existing registered TRAIN worlds by initial Grenades, Machinegun, Grenade Launcher and Rocket Launcher, before new outcomes. The source and migrated policy run in identical worlds: eight native games and eight CUDA replays, pool maximum 16 and timescale 2.

The check is COMPLETE and PASSED: all eight native games and eight CUDA replays succeeded. Across the four paired worlds, first-life canonical context, policy choices, requested/applied commands, native damage and ending frames match exactly, excluding policy-version identity and post-boundary shutdown callbacks. Report SHA-256: `80c09bed8557dcc0539d0f10b1b324d118bc5b4e2faada8151a1d50bc98a4bf6`. This is initialization identity, not learned quality.

## Matched learning pilot

`workspace/artifacts/combat-fire-context-pilot-v1-20261027/` starts the actual comparison after the successful native proof. Before outcomes it selects 16 fresh TRAIN worlds (eight initially holding grenades, eight other weapons) and 16 fresh validation worlds, using candidate offsets 453–468 and 469–484 respectively. This small selection does not cover every weapon type or establish all-map performance.

Both old and migrated source policies play each TRAIN world, producing 32 separate source-matched captures. Their first-life observations, actions, likelihood/value records, rewards and native damage must match before either fit starts. Each model then trains from its own verified capture/model/checkpoint binding. Actor LR 0.003, eight steps, reward, inherited optimizer history and frozen other-policy behavior are matched; no phase balancing is used.

The validation arms are original U9, legacy fire fit, context fire fit and rules. Fixed budget: 96 native games and 80 CUDA verification tasks, dynamic pool maximum 16 at timescale 2. No automatic promotion. Gameplay quality remains unknown until sealed completion and independent audits.

The full-joint trainer now binds a TRAIN request to exactly one pool plan by plan hash and model hash, then selects seed jobs within that plan. This permits the same world seed across different models in a shared pool while retaining exact capture-root/model/fixture checks. It does not weaken the duplicate-seed or consumed-rollout checks within one model's training request.

## Actual matched fitting evidence

The 32 source captures and 32 CUDA tasks completed. `train/paired-identity.json` passed across all 16 worlds: first-life native damage, canonical context, actions, likelihood/value records and rewards are exact before learning, apart from policy-version identity.

Both CUDA fits completed eight actor steps on 2,382 admitted rows each:

- Legacy fit weights SHA-256: `50799c0f08a64a89dfa495e17cec9d1b12dd188c1a153252a96d763f15afc6f8`; final KL 0.0015643389.
- Context fit weights SHA-256: `1e50ab6e1b075026614ccde8313bd3410d4ed6796f511ca2aeb44b0075290987`; final KL 0.0015388095.

Independent `checkpoint-audit-legacy-fit.json` and `checkpoint-audit-fit.json` reopen the actual checkpoints on CUDA. Non-fire actor/joint/standard-deviation tensors and their Adam states remain exact versus each model's source; each appends exactly 16 fresh cases to its source consumption history. The new context columns have updated from zero. Both use the same configuration. Critic parameters agree within 1.82e-5, under the explicit 1e-4 numerical tolerance; they are not claimed bitwise identical because case order differs between own-capture requests.

`admission-audit.json` independently admits both actual 16-case requests and confirms that swapping in the other model's same-seed capture is rejected. This exercises the shared-pool plan binding with real data without neural inference.

The read-only CUDA `training-effect-audit.json` confirms functional use of the newly trained context inputs: on the same 2,382 TRAIN states/actions, removing projectile channels changes the context branch logit in the range −0.02009 to +0.01382; removing prop channels changes it in the range −0.01599 to −0.00122. Removing each block also removes its availability flag. These are input-dependence measurements, not evidence of correct hazard recognition or combat benefit.

## Closed first-update quality screen

The pilot and independent `validation-audit.json` are COMPLETE and PASSED: 96 native games and 80 CUDA tasks including TRAIN; exact paired fixtures and unchanged source pins.

| Arm | Goals / 16 | Deaths | Monster kills | Received health damage | Self health damage | Held-grenade damage |
|---|---:|---:|---:|---:|---:|---:|
| Original update 09 | 2 | 9 | 18 | 1,026 | 246 | 113 |
| Legacy fire fit | 3 | 9 | 20 | 1,059 | 304 | 113 |
| Context fire fit | 3 | 9 | 20 | 1,059 | 304 | 113 |
| Rules | 3 | 7 | 24 | 908 | 122 | 0 |

Adding the direct context has not demonstrated a benefit over the matched legacy fit after one eight-step update. Equal wins with rules do not establish parity: rules have fewer deaths, more kills and lower received/self damage. No promotion.

On 2,567 fixed original validation inputs, the learned context branch depends on the projectile block (logit removal effect −0.01819 to +0.01712) and prop block (−0.01622 to −0.00113), including availability flags. These are functional input effects, not correct hazard recognition. Hold-frame attack counts are 155/161 in the original and 131/137 in both fitted trajectories; held-grenade damage remains 113 HP.

## Fixed continuation dose test

`workspace/artifacts/combat-fire-context-series-v2-20261027/` resumes both fitted checkpoints for exactly two more on-policy updates, with 16 fresh paired-condition TRAIN worlds per update and separate own-model captures. Architecture, LR, reward and optimizer history remain fixed. Unlike the zero-initialization test, trajectories after learning may differ; the generated world fixtures must match.

TRAIN candidate offsets are 485–500 and 501–516. Sixteen new validation worlds, selected before update 2, use registered validation offsets 485–500 (a separate split and seed range). Final comparison uses both variants at fire updates 1 and 3 plus rules in those same worlds. Total additional budget: 144 native games and 128 CUDA tasks, pool 16 at timescale 2. No intermediate quality-based model selection or automatic promotion.

Series version 1 failed during metadata planning, before any native game, because validation offset 517 exceeded the registered validation cohort. Its failed receipt is retained. Version 2 uses the unused in-cohort validation range above; generator bounds are not expanded or bypassed. Continuation quality remains unknown.

The series subsequently stopped after completing update 2: update 3's TRAIN candidate range 501–516 also exceeded its registered 512-instance cohort. Existing update-2 checkpoints are retained; the third update and final quality comparison did not run. The user then identified resource-starved encounters as unsuitable for initial learning. Further broad-mixture continuation is superseded by the assessed beginner curriculum in `combat_encounter_difficulty_20261027.md`; the stopped series must not be reported as active or as a completed quality result.
