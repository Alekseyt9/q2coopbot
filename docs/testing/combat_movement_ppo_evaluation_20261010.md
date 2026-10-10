# Fresh movement PPO update2 evaluation — 2026-10-10

480/480 native members verified. Six variants share validation28..31, twenty
families, current movement guard, 16 refill slots and timescale2. This is
development validation, not an untouched final-test or promotion result.

| Variant | Wins /80 | Deaths | Mean damage received | Blaster wins /40 | Machinegun wins /40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| spatial-instant-before | 52 | 20 | 33.70 | 28 | 24 |
| spatial-instant-after | 48 | 24 | 37.65 | 28 | 20 |
| spatial-postmove-before | 49 | 29 | 42.01 | 29 | 20 |
| spatial-postmove-after | 51 | 26 | 38.77 | 30 | 21 |
| firebc-baseline | 51 | 26 | 43.25 | 27 | 24 |
| rules-baseline | 64 | 9 | 15.61 | 34 | 30 |

Instant gained4 and lost8 paired wins (-4); Postmove gained9 and lost7 (+2).
Postmove update2 matches FireBC on total wins, while rules retain13 more wins
and substantially lower received damage. No learned branch is promoted.
The prior rules70/80 result used different seeds; it is not a paired decline.

Job116 initially had an empty command trace after native configstrings buffer
overflow during signon. The failed attempt and queue receipt were archived
under `recovery/signon-20261010T005330Z`; identical seed/model/recipe was rerun
once. Its valid loss is included. No gameplay losses were retried. The original
failed aggregate remains as evidence; strict individual-member proof validates
the recovered480 records.

Quality receipt: `workspace/artifacts/movement-ppo-eval-v2-20261010/quality-report.json`
SHA256: `bba51bdfbeae8c53fe02791b7e9234209ba3b3943b95d4b6c062a8a644b6a39a` (protocol).
Quality SHA256: `47df07bccce862aa542c48556d501c1af3d6a63be8e19f1369960c9fd0615d6c`.

Selected-target angular error while attack is held: Instant16.88→16.17degrees,
Postmove16.25→16.11degrees. These frames are not verified shot events; target
intent is unavailable for FireBC/rules in this diagnostic. Movement reporting
and native Blaster hit receipts provide complementary diagnostics.

## Movement and native Blaster contacts

| Variant | Stationary before / after | Stationary with held attack before / after |
| --- | ---: | ---: |
| spatial-instant | 65.4% / 65.7% | 48.6% / 50.6% |
| spatial-postmove | 57.1% / 60.6% | 45.1% / 48.3% |

Rules are stationary96.6% of matched alive frames, yet achieve64/80 wins.
Native live-monster Blaster contact fractions: Instant50.6→52.9%, Postmove
57.7→59.6%, FireBC41.1%, rules94.0%. These are observed projectile contacts
within the reporter scope, not a stationary-movement causal experiment.
The large aim/contact gap warrants priority over a blind movement bonus.
Movement cancellation remains64–70% for learned branches; most cancellations
are static hull blocks. `movement-outcome-strata.json` separates win/loss
command counts and guard reasons without neural execution.

## Next matched parent control

`postmove-parent-control-20261010` evaluates the sealed Postmove BC parent on
the same80 validation28..31 conditions and unchanged movement guard as the
completed480 comparison. It uses the shared binary bundle and16 refill slots.
Results pending; no new training or parent selection until native proof.
