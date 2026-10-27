# Conditional fire: one-update campaign pilot, 2026-10-27

The pilot is closed. One GPU PPO update of the existing conditional fire branch did not demonstrate better combat. No model promotion or default change.

## Implemented boundary

`scripts/combat_conditional_fire_scope.py` narrows the restored full-joint actor optimizer to the existing 181 → 32 → 16 → 1 conditional fire branch (6,369 parameters). The scope retains original optimizer layout and moments. All other actor, joint aim and standard-deviation parameters and their Adam states remain exact; the critic continues learning. This is not the historical proposed Shotgun-only residual.

`scripts/train_combat_full_joint_ppo_cuda.py --fork-conditional-fire` explicitly forks the original full scope, allowing only actor scope and actor learning-rate changes. Subsequent updates use ordinary resume with the new strict configuration. Technical-only checkpoints and diagnostic intervention trajectories are rejected.

The actual source is update 09 of `workspace/artifacts/combat-stock-full-series-v3-20261026`, weights SHA-256 `386404bc51489525111f3d894deade3bb50bc3c90379ccb0ced7321af367c87c`. The movement-options experiments are not its parent.

## Protocol and validation

Authoritative evidence: `workspace/artifacts/combat-fire-cycle-pilot-v2-20261027/`.

- 32 fresh registered TRAIN worlds on 32 maps, initially 16 hand-grenade and 16 other-weapon cases; all 11 initial weapon types represented in TRAIN.
- 16 fresh registered validation worlds, initially eight grenade and eight other-weapon cases, selected before TRAIN outcomes. This small validation selection does not cover every initial weapon type and does not establish generalization to unseen maps.
- Every validation world compared before, after and rules: 48 native games. Total pilot: 80 native games and 64 CUDA verification tasks.
- Existing dynamic pool, maximum 16 instances, timescale 2, separate registered seeds. Goal/death stopping enabled.
- Diagnostic grenade-release intervention disabled in all learning and validation games.
- Candidate geometry failure for Biggun seed 1901182411 excluded its candidate cohort before games, using only strict generator `profile_actor_overlap:128` failure. Receipt: `validation/excluded-candidate-cohorts.json`. No combat losses were filtered.
- Source code, models, configuration and runner pinned throughout the experiment.

PPO actor learning rate 0.0003; eight actor and eight critic steps; unchanged reward v4, gamma 0.99, lambda 0.95, clip 0.2, target KL 0.01. Final approximate KL 0.0000472511. Global update count 10, fire-only update count one. 4,494 admitted TRAIN rows.

## Closed outcomes

| Arm | Goals / 16 | Deaths | Monster kills | Received health damage | Self health damage | Held-grenade damage |
|---|---:|---:|---:|---:|---:|---:|
| Original update 09 | 4 | 7 | 17 | 910 | 218 | 58 |
| Conditional fire update 01 | 4 | 7 | 17 | 907 | 218 | 58 |
| Rules | 3 | 8 | 25 | 895 | 43 | 0 |

Identical goal and death outcomes in each before/after pair. The slight damage difference is not an improvement claim. The rules comparison here is a small inventory-balanced pilot; it does not replace the separate 38-world movement-options evaluation where rules led.

## Independent checks

`checkpoint-audit.json` reopens the actual fitted checkpoint on CUDA, restores it against exported JSON and verifies that only conditional-fire actor tensors and moments changed. Joint aim and log standard deviation are exact. Critic and critic Adam changed. All original consumed tokens remain a prefix, and exactly 32 fresh TRAIN cases were added without source-seed reuse.

`behavior-audit.json` checks the closed summary seal, complete pool and 32 validation CUDA tasks, then compares paired policy samples and evaluates the two fire branches on identical original observations/actions on CUDA. Across 2,325 original validation rows, fire-logit delta was positive throughout: minimum 0.02140, mean 0.03349, maximum 0.04144. The corresponding probability change is bounded by 1.036 percentage points. Only three world pairs had any sampled-action difference.

For the 161 admitted hand-grenade gun-frame-11 rows, attack was requested in 154 rows before and after. Mean same-input fire-logit change there was +0.03432. Thus this update did not learn a useful release distinction: its observed effect was a small general increase in firing probability.

Technical CUDA proof before gameplay fitting: `workspace/artifacts/combat-conditional-fire-scope-check-v1-20261027/report.json`. It verifies fire-only likelihood gradients, frozen parameters/moments and exact synthetic eight steps versus four/save/reload/four. Synthetic artifacts are marked technical-only and were not deployed.

## TRAIN gradient diagnosis

`train-signal-audit.json` is closed and passed on CUDA without fitting. The 4,494 admitted TRAIN rows contain 235 hand-grenade gun-frame-11 rows from 14 worlds; attack is true in 219. The remaining 4,259 rows cover all 32 worlds.

At the source-policy boundary, the reward-gradient contribution favors decreasing the fire logit in the grenade hold phase, while the remaining rows favor increasing it. The two actual conditional-fire parameter gradients have cosine **−0.95781**. Their norms are 0.0007801 for the hold phase and 0.0117693 for other rows, using the same global row normalization. Dot products with the actual fitted parameter update are **+0.0000119646** for hold-phase loss (locally worsening) and **−0.0002133231** for other-row loss (locally improving).

This supports a specific shared-parameter gradient conflict in this batch, beyond merely counting rare frames. It does not prove that balancing phases will win more games. These figures describe the source-policy reward component, excluding entropy and optimizer momentum, and are not causal outcome estimates. The audit uses only admitted TRAIN observations/rewards; validation outcomes were not converted into training labels.
