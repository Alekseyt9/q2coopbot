# One-frame grenade trigger-release acquisition — 2026-10-27

The movement-only five-update experiment did not improve validation over its
initial exploration control. A separate TRAIN audit found four hand-grenade
detonations while attack remained held on gun frame 11. The conditional fire
network is frozen in that movement experiment and cannot directly learn a new
trigger decision.

An opt-in execution diagnostic now releases only attack at one configured
network frame, only for a live first-life actor holding the stock hand-grenade
weapon at observed gun frame 11. The sampled policy request and every other
command field remain unchanged. It is restricted to captured loopback,
synchronous, isolated learned fixtures using `stock_weapons_fire_v1`; it rejects
combination with motion, nominal-aim and advice interventions. Default is zero/off.

Configuration: `test.grenade_release_probe_frame`, harness parameter
`-GrenadeReleaseProbeFrame`. The manifest, native PPO exporter and CUDA receipt
preserve `diagnostic_grenade_release_frame`. Release captures finalize as
`combat_diagnostic_grenade_release_rollout_v1`, `diagnostic_only=true`,
`ppo_training_ready=false`, ordinary training eligibility false. They are not
ordinary PPO data or preferred-action labels.

Four previously observed TRAIN cases were compared using their exact original
fixtures and source weights. Each source/release arm had two cold repeats:
16 native captures and 16 CUDA density checks, pool16, timescale2. All 48 native
checks passed: disabled behavior matches the original trace, both arms have
matching observed prefixes, root policy request/sample match, only attack bit
changes, and native gun frame advances 11→12. Pure metadata/command Go tests
passed; no Go CPU neural inference was tested.

The first raw-callback comparison found differences after first observed goal
or death. Those callbacks remain preserved, and its failed result is retained
as `native-checks-all-callbacks-v1.json`. Comparison was corrected to the actual
first-life boundary already established by native capture: pre-boundary actions
and observed effects match across cold repeats, including exact damage totals.
Post-boundary shutdown callbacks are explicitly listed in the final report.

The following totals count one outcome per world, not both correlated repeats:

| TRAIN scene | Source goal / death / self HP | Release goal / death / self HP |
| --- | --- | --- |
| bunk1, seed 1832050401 | no / no / 58 | yes / no / 0 |
| jail4, seed 1850080402 | no / yes / 24 | no / yes / 24 |
| waste3, seed 1898000402 | no / yes / 58 | no / yes / 37 |
| base2, seed 1823080403 | no / yes / 87 | no / no / 82 |

Across four known worlds: goals 0→1, deaths 3→2, self HP 227→143,
monster HP dealt 441→1106, kills 2→5. This establishes a useful mechanical
intervention in a small known cohort. It does not show a learned improvement,
general safety of releasing at that frame, or superiority to rules. Released
grenades can still hurt or kill the player, as jail4 demonstrates. Defaults and
model weights remain unchanged.

Evidence:

- `workspace/artifacts/combat-grenade-release-protocol-v1-20261027.json`
- `workspace/artifacts/combat-grenade-release-native-v1-20261027/report.json`
- `workspace/artifacts/combat-grenade-release-audit-v1-20261027/report.json`
- `workspace/artifacts/combat-grenade-release-audit-v1-20261027/complete.json`

The subsequent learning hypothesis is selective conditional-fire training on
fresh, registered TRAIN conditions with a broad weapon control group. No forced
release intervention may be present in that training or its validation.
