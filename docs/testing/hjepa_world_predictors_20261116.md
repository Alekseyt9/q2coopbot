# H-JEPA: predictor input contract and native validation

The proposed-input repair is not accepted. Only world predictors were refitted;
encoders, utilities, actor, motor and weapon head remained frozen. Current U1
was not replaced. Artifacts: `workspace/artifacts/hjepa-world-proposed-v1-20261116`.

The original world-v3 training rows contain applied actions; runtime planning
scores proposed primitives. This is a verified representation mismatch, not
proof that fixing it improves combat. Both predictor fits used the same TRAIN437
provider clips and fixed 200 epochs. Validation did not select epochs.

## Native validation441

Six learned variants cover 16 fresh worlds on eight maps and two coupled weapon
sampling streams. Keep/U1/rules each cover 16 worlds: 144 native episodes total.

| Variant | Episodes | Wins | Deaths | Timeouts | Received HP | Self HP |
|---|---:|---:|---:|---:|---:|---:|
| Proposed-input refresh | 32 | 7 | 18 | 7 | 1928 | 278 |
| Applied-input refresh | 32 | 9 | 15 | 8 | 1909 | 118 |
| Parent | 32 | 5 | 19 | 8 | 1996 | 118 |
| Keep | 16 | 3 | 9 | 4 | 977 | 0 |
| U1 | 16 | 4 | 9 | 3 | 938 | 79 |
| Rules | 16 | 4 | 8 | 4 | 774 | 80 |

The proposed candidate fails combat and forecast gates. It does not gain two
wins over applied refresh, increases deaths, and adds 160 self HP. Its unclipped
forecast error regresses by more than the specified 10 percent despite better
overall and guarded prediction. See `native-summary.json` and `forecast-gate.json`.
Applied refresh looks promising on this comparison but was identified after
inspection; independent evidence is required before selection or promotion.

## Replay and localization

`q2hjepa-audit -require-replay` reproduced all 11,147 provider decisions from the
96 learned episodes exactly. Six inventory audits found no unavailable weapon
choices. These checks establish replay and availability, not combat quality.

| Variant | Decisions | Planner changes | Fraction |
|---|---:|---:|---:|
| Proposed refresh | 3543 | 173 | 4.88% |
| Applied refresh | 3902 | 97 | 2.49% |
| Parent | 3702 | 2078 | 56.13% |

The extra self damage occurs entirely on base1 seed 1820162441 with initial BFG:
80 HP in each proposed stream, zero in corresponding applied/parent streams.
Recorded incoming damage has MOD 13 and attacker class player. All three models
also incur the same 118 self HP total on hangar1, so the additional base1 case
accounts for the complete aggregate difference. Details and source report paths
are in `replay-diagnosis.json`.

Predictor refresh strongly reduces planner intervention frequency. Better combat
could therefore reflect less intervention rather than better action ranking.
Next isolate this with a motor-only ablation on the same diagnostic worlds;
then preregister an independent cohort before selecting a refreshed predictor.
Do not treat this validation cohort as independent confirmation after choosing
the applied candidate. No promotion, final TEST claim, or full-combat completion.

## Motor-only diagnostic completed

An additional 32 actual native episodes reused exactly validation441 fixtures and
the two weapon sampling streams. Only `planner` changed from `sequence` to `bc`;
all arrays and other model fields remained equal to parent. This is a diagnostic
baseline, not a replacement for the requested learned hierarchical system.

Motor-only achieved 8 wins, 17 deaths, 7 timeouts, 2019 received HP and 118 self HP.
Compared with it, original planning loses three wins and adds two deaths;
applied refresh gains only one win, avoids two deaths and saves 110 received HP.
Proposed refresh loses one win, adds one death and 160 self HP. Thus the original
planner's high intervention frequency has no demonstrated benefit on this cohort;
applied refresh's apparent gain over parent mostly disappears against motor-only.
This supports testing planner action ranking and utilities rather than declaring
prediction error sufficient evidence of improved control. It does not establish
the best controller on independent worlds.

Both motor-only replay audits passed: 3450 decisions, zero mismatches and zero
planner changes. Reports are under `motor-only/summary.json`, with eight-map
cluster intervals and paired world differences. The matrix wrapper exited on a
post-capture compression check for job17. All32 captures had already completed
with accepted command proofs and matching fixtures; no owned writer remained,
compression retry succeeded, and completion was reconstructed with an explicit
recovery record in `motor-only/native/complete.json`. No episodes were rerun.

## Utility transport diagnostic

`scripts/audit_combat_hjepa_utility.py` audits 2416 overlapping provider clips from
3856 verified transitions in validation440. Native progress labels are generated
by the existing terminal-aware preparer; all source hashes are checked. Outcomes
remain diagnostic targets, never runtime inputs. This reused cohort supplies
mechanism evidence only; overlapping clips are not independent trials.

At horizon16, frozen utility evaluated on actual encoded future observations has
Pearson correlation 0.615 with the realized native progress change (MSE0.541 versus
zero-change MSE0.875). Evaluated on predicted states, the correlation drops:

| Predictor | Raw-input correlation | Raw-input MSE | Applied-input oracle correlation |
|---|---:|---:|---:|
| Parent | -0.061 | 1.536 | -0.064 |
| Applied refresh | -0.023 | 2.686 | -0.003 |
| Proposed refresh | 0.109 | 0.925 | 0.036 |

Even supplying actual applied actions does not restore useful association. Thus
the action-contract mismatch alone does not explain the poor value forecast.
The readout carries some useful information at horizon16 on real observations,
but latent prediction does not preserve it sufficiently on this cohort. Shorter
horizons and absolute accumulated-value calibration also have substantial error;
the result does not establish that the utility head is generally accurate.

This audit follows the executed nonconstant action sequence, not held hypothetical
plans selected by search, and cannot establish counterfactual action rankings.
Accumulated progress is the historical training objective, not future return.
Next isolate task-sensitive predictor training against a matching latent-only
control, with TRAIN-only fitting and fresh preregistered native validation. Do not
infer combat improvement solely from this diagnostic or change live U1.
