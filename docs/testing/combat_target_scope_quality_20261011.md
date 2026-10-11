# Target-scope comparison results, 2026-10-11

The480-battle pool closed with480 usable captures, no job errors and unchanged
source:16 independent episode slots, timescale2, wall1313.09 seconds. All six
controllers saw20 recipes×4 matching fixtures/test seeds at offsets12:16.
All400 learned captures completed CUDA verification:44733 eligible rows.
Five native behavior audits and their matched-fixture summary also completed.
No evaluation rows were used for training.

| Controller | Goals/80 | Kills | Deaths | Damage dealt | Damage received |
|---|---:|---:|---:|---:|---:|
| Frozen parent9 | 31 | 109 | 46 | 22855 | 5560 |
| Decisions control10 | 34 | 115 | 43 | 23274 | 5340 |
| Decisions shared10 | 37 | 133 | 37 | 25035 | 5222 |
| Target-only control10 | 33 | 117 | 43 | 23802 | 5785 |
| Target-only shared10 | 35 | 116 | 41 | 23956 | 5619 |
| Rules | 29 | 115 | 47 | 23913 | 5349 |

| Map,20/controller | Parent | Decisions control | Decisions shared | Target control | Target shared | Rules |
|---|---:|---:|---:|---:|---:|---:|
| base3 | 8 | 9 | 9 | 5 | 8 | 5 |
| city1 | 11 | 11 | 12 | 14 | 15 | 13 |
| jail1 | 2 | 1 | 2 | 2 | 1 | 1 |
| ware2 | 10 | 13 | 14 | 12 | 11 | 10 |

Target-only does not outperform the equally trained decision scope in this
new sample:control34→33 goals, shared37→35. Shared target-only also loses17
kills and gains4 deaths relative to shared decisions. Freezing attack/weapon
learning has not established a better controller. This is one matched update
budget/sample, not proof that narrower updates can never help.

The new seed block reverses the previous result:shared decisions28/80 against
parent33/80 becomes37/80 against31/80. Two disjoint blocks were combined only
for controllers with identical frozen weights and matching native binaries,
reset settings and AAS. Generated map ENT hashes differ across seeds as
expected; full ENT/runtime hashes and fixture positions remain matched between
controllers within each seed. Target-only has no first-block sample and is
excluded from the160-case comparison.

| Same frozen controller,160 cases | Goals | Kills | Deaths |
|---|---:|---:|---:|
| Parent9 | 64 | 229 | 86 |
| Decisions control10 | 64 | 217 | 90 |
| Decisions shared10 | 65 | 238 | 85 |
| Rules | 52 | 227 | 96 |

Shared decisions gains17 previous losses but loses16 previous wins across both
blocks. A one-goal net difference does not justify replacing the reference or
claiming architectural superiority. Parent9 stays the experimental training
reference; published models and live coop are unchanged.

Post-release ownership was audited separately from the held setup phase using
each sealed manifest's release frame. All400 learned cases have zero unknown
observation rows; no winning case contains an observed clear-enemy rules frame.
However, post-release clear-enemy rules frames occur in losses:parent3 frames,
decisions control6, decisions shared58, both target-only variants0. Shared
decisions has one observed rules attack command in a lost case; this is not
native bullet/damage attribution.

The shared fallback is concrete:seed312540012 switches to
`models/weapons/v_handgr/tris.md2` (hand grenades), then remains in
`pilot_equip_not_ready` for47 observed frames;another lost seed1174067013 has11
such frames. The existing pilot admits only Blaster/Machinegun/Shotgun/SSG
combat ownership, while the masked weapon policy can select other available
inventory weapons. The gate then prevents the policy from choosing its own
replacement weapon. This runtime/control boundary should be repaired before
more training; it is not evidence that a larger network is needed.

Next work:

1. Keep a weapon-enabled policy in charge while an unsupported weapon is
   equipped in the explicitly supported pilot/campaign context, allowing it to
   request a replacement. Retain the existing unsupported-fire execution guard;
   do not force a replacement weapon or relax projectile checks.
2. Verify this boundary with fixed-provider contract tests, then native
   multi-enemy scenes containing the observed grenade pickup/switch cases and
   CUDA replay. Go numerical neural tests remain excluded.
3. Extend train-only sequence-retention preparation/validation to registered
   multi-model pools without fabricated series directory layouts. Preserve
   whole-pool acceptance, source/model/data hashes and split separation.
4. Run balanced fresh own-policy training across the20 families with the full
   actor trainable, including movement, aim and temporal/environment inputs.
   Compare reference and shared challenger with the same episode/update budget
   and train-history retention; target-only is a diagnostic, not the final
   action space.
5. Validate the corrected runtime and training on fresh held-out fixtures and
   independent policy RNG arms before changing the live/published controller.

Authoritative artifacts under
`workspace/artifacts/combat-target-scope-evaluation-analysis-v1-20261011/`:
`complete.json`, `quality-strata.json`, `scope-effect-{control,shared}.json`,
`combined-seed-blocks.json`, `behavior-summary.json`, and
`ownership-*-v2.json`. The first ownership audit counted all first-life frames;
v2 explicitly separates pre-release setup from post-release combat. Its
setup rule frames must not be labelled learned combat contamination.
