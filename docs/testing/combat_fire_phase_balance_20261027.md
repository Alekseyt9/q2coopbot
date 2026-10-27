# Fire phase balancing, 2026-10-27

Implemented experiment; gameplay quality remains unproven. This is a TRAIN objective fork, not a runtime action override. The preceding uniform-fire pilot and its gradient conflict are recorded in `combat_conditional_fire_pilot_20261027.md`.

## Loss and checkpoint boundary

`scripts/combat_fire_phase_balance.py` defines `observed_hand_grenade_hold11_equal_mass_v1`. From already observed feature channels it separates equipped hand grenades at gun frame 11 from other states. No selected action, enemy outcome or future event enters the grouping.

For a batch of N rows with H hold-phase rows, the hold weight is N/(2H) and other-state weight N/(2(N−H)). Thus the groups have equal total mass. Both must be present; an absent group fails explicitly. All admitted rows and the original rewards remain. The globally normalized advantages are unchanged. This deliberately changes the state weighting of the actor objective; it is not claimed to be an unbiased estimate of the original state-distribution objective.

The existing fire-only checkpoint fork allows the explicit `fire_phase_balance` configuration key. Ordinary resume continues to require exact configuration/layout. Other actor/joint/standard-deviation tensors and moments stay frozen. Critic fitting remains unweighted. Both unweighted and weighted approximate KL are bounded by the existing guarded step through their maximum. Inference has no new phase gate and no forced grenade release.

## CUDA technical proof

`workspace/artifacts/combat-fire-phase-balance-check-v1-20261027/report.json` passed:

- Feature-derived phase membership equals the recorded native observation classification for every one of the prior pilot's 4,494 TRAIN rows: 235 hold-phase, 4,259 others.
- Each group has equal total weight; absent-group batches are rejected.
- Actual likelihood derivatives reach the fire branch only; other weights and Adam states stay exact.
- Synthetic weighted fitting produces exactly the same exported weights and optimizer states for eight steps versus four/save/reload/four. Technical checkpoints are marked synthetic-only and cannot be used for gameplay fitting.

The old TRAIN captures in this technical check supply observations for classification/derivative checks only. No optimizer update uses their game outcomes.

## Paired fresh pilot

Active evidence directory: `workspace/artifacts/combat-fire-phase-balanced-pilot-v2-20261027/`. Protocol fixes 32 fresh TRAIN worlds and 16 fresh validation worlds with candidate seed offsets 421–436, disjoint from the preceding pilot. Both selections precede gameplay outcomes.

Two separate original-update-09 checkpoint forks fit the identical 32 TRAIN captures with the same learning rate, optimizer history, reward and eight steps. The only objective difference is phase weighting. Fresh validation compares original, uniform fit, balanced fit and rules in the same 16 worlds. Budget: 96 native games and 80 CUDA tasks; dynamic pool 16, timescale 2. No automatic promotion.

Version 1 stopped before any native game: candidate construction exhausted geometry attempts on Biggun (actor overlap), then Hangar1 (119 actor-overlap and 9 player-teleport-overlap attempts). Version 2 preserves those failed receipts and allows only these two explicit pre-game geometry reasons, with counts totaling exactly 128; the whole affected candidate descriptor cohort is excluded before outcomes. Unknown generator errors still fail. This changes eligible geometric cohorts, not combat-outcome selection.

Final quality must be read from the sealed summary and independent checkpoint/behavior audits after the pilot completes. A better weighted training loss alone is not a combat-quality result.

## Actual fresh fits and independent reopening

Both real CUDA fits completed eight actor steps on the identical 4,558 admitted rows from 32 fresh TRAIN worlds. The balanced batch contained 336 hold-phase rows and 4,222 other rows, weighted 6.7827381 and 0.5397916 respectively.

- Uniform weights SHA-256: `c4e532c78a28f775623f7e47f1308482824153d5f6d689e8c77d756430a82ca4`; final approximate KL 0.0000160516.
- Balanced weights SHA-256: `54868c6f13f8f3c5b370ece2dd55428e816a4ef11e24556c3824e5b92d99094b`; maximum of weighted/unweighted approximate KL 0.0000233631.

`checkpoint-audit-uniform-fit.json` and `checkpoint-audit-fit.json` independently reopen both actual checkpoints on CUDA. All non-fire actor/joint/standard-deviation tensors and Adam states match the original source exactly. Both preserve the original consumed prefix and add the same 32 fresh cases. Critic tensors and critic optimizer states are exactly equal between the two fitted variants. Their configurations differ only by `fire_phase_balance`.

## Closed four-arm gameplay evaluation

The pilot and independent `validation-audit.json` are complete. All 64 validation native games and all 48 validation CUDA tasks passed; including TRAIN, the experiment used 96 native games and 80 CUDA tasks. Paired generated fixtures match exactly, and source pins remain unchanged.

| Arm | Goals / 16 | Deaths | Monster kills | Received health damage | Self health damage | Held-grenade damage |
|---|---:|---:|---:|---:|---:|---:|
| Original update 09 | 3 | 7 | 24 | 1,018 | 329 | 18 |
| Uniform fire fit | 3 | 7 | 25 | 1,018 | 351 | 18 |
| Phase-balanced fire fit | 3 | 7 | 25 | 1,018 | 351 | 18 |
| Rules | 5 | 8 | 24 | 818 | 87 | 0 |

Neither learned fork changes a goal or death outcome in any paired world. Both increase self damage relative to the source. No quality improvement or promotion.

On 2,448 identical original validation inputs, both fits decrease the fire logit slightly everywhere: uniform mean −0.019636, balanced −0.023738. At the grenade hold phase the means are −0.021223 and −0.025565. All three policies request attack in 130 of 135 admitted hold-phase rows. Thus the balanced update produces only a slightly larger broad fire decrease; it does not establish a useful learned release distinction.

`training-effect-audit.json` gives the corresponding read-only CUDA comparison on identical TRAIN inputs: hold-phase means −0.020842 (uniform) and −0.025161 (balanced), other-row means −0.018949 and −0.022996. These small logit changes and final KL near 0.00002 motivate a separate bounded larger-step fork, not a claim that balancing is sufficient. The follow-up preregisters actor LR 0.003 versus 0.0003, keeping eight steps and the 0.01 KL guards, using the same source-policy TRAIN captures in independent source-checkpoint forks and fresh evaluation worlds. It does not replay already consumed data into a resumed checkpoint.

## Larger-step follow-up

`workspace/artifacts/combat-fire-phase-step-pilot-v1-20261027/` records the separate follow-up. It reuses exactly the preceding 32 source-policy TRAIN captures and their CUDA verification, in two independent original-U9 forks. Candidate validation offsets 437–452 are selected before the new fits and disjoint from both preceding pilots. Planned additional work is 64 native games and 48 CUDA validation tasks, without recollecting TRAIN.

Both actual GPU fits completed eight steps, and their independent CUDA checkpoint reopening passed:

- Uniform weights: `38f4731e0af73b07f7b8d28c836e8d3f58669b893f68df565bbac779a0018e45`, final KL 0.0012769162.
- Balanced weights: `f79d13607d7fd8074f6dbebfb1061c3d743cfd8d491215b329de1500073cc4d7`, maximum weighted/unweighted KL 0.0013488402.

Non-fire actor/joint/standard-deviation weights and their optimizer states remain exact. Critic weights and optimizer states match between the two fits. Only `fire_phase_balance` differs between these matched configurations; both use actor LR 0.003.

The larger-step pilot is now CLOSED. Its independent validation audit passed: 64 native games, 48 CUDA checks, exact paired fixtures and unchanged source pins. It must not be confused with the preceding small-step table.

| Arm | Goals / 16 | Deaths | Monster kills | Received health damage | Self health damage | Held-grenade damage |
|---|---:|---:|---:|---:|---:|---:|
| Original update 09 | 5 | 11 | 23 | 1,315 | 486 | 118 |
| Uniform larger-step fit | 5 | 11 | 24 | 1,295 | 559 | 109 |
| Balanced larger-step fit | 5 | 11 | 24 | 1,295 | 559 | 109 |
| Rules | 4 | 10 | 20 | 1,086 | 78 | 0 |

This does not establish an improvement over the original model or general superiority to rules. Both forks gain jail4 seed 1850082445 and lose base1 seed 1820102443, preserving the total goal count. The increased dealt damage comes with greater self damage. No promotion.

`regression-audit.json` verifies identical admitted action frame sets, observations and sampled actions between uniform and balanced fits in all 16 worlds, ignoring policy-version, value and likelihood metadata. Thus phase balancing produced no distinct selected behavior in this evaluation. Original-source versus changed-policy first differences are attack-only on identical inputs: base1 frame 145 (equipped Grenade Launcher), jail4 frame 146 (equipped Machinegun), ware2 frame 105 (equipped hand grenades), each changing attack true to false.

Base1's changed-policy trajectory ends at native frame 291 with self damage `MOD_R_SPLASH` (9), health 40 to −4, after earlier grenade splash damage. The original trajectory completes its goal at frame 254. These are observed outcomes after the first changed action, not proof that skipping that one shot alone causes the later death. Projectile entity IDs are not treated as lifetime identity. GAME and SERVER frame fields remain separate in the event evidence.

## Direct fire-branch context limitation

`fire-context-audit.json` is a passed read-only CUDA sensitivity check on all 4,558 admitted TRAIN inputs. It zeroes only existing projectile channels 169:218 or nearby-prop channels 259:296 while retaining the selected action and other inputs. The conditional fire branch's complete 181-element input remains exactly unchanged in both tests. The TRAIN set includes 989 rows with visible projectile slots and 232 with visible prop slots.

This is a limitation of the trainable conditional branch in the fire-only experiment. The full actor receives those channels and its frozen raw fire logit can depend on them. It is not a missing-observation bug in the full system, nor proof that adding context will improve combat. Further learning-rate/phase-weight sweeps are stopped; the next distinct hypothesis is a versioned observed-context fire branch with zero-behavior initialization and separate fresh validation.
