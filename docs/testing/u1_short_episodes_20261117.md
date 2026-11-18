# U1: many short combat episodes

User switched development focus from H-JEPA to U1. H-JEPA code, models and
artifacts remain preserved; no H-JEPA collection/training is active.

Starting U1 weights SHA256:
`21c3b680e0b6090d601d524f6da53d73ae6dd083c8e7a9a6da2e6d268ccadd26`.
Original sealed source/critic/full Adam/config:
`workspace/artifacts/combat-beginner-full-pilot-v3-20261027/fit`.
The live U1 file is unchanged; all new outputs are isolated.

Artifact root: `F:/src/quake2/q2coopbot-src/workspace/artifacts/u1-short-episodes-20261117-100322`.
Plan: 64 fresh TRAIN episodes,16descriptors/eight maps, one or two native
opponents, at most64control frames (about6.4game seconds), cold native resets,
16parallel instances at timescale2. Independent seed ranges avoid all493consumed
U1training seeds; no validation conditions enter optimizer inputs. Random stock
inventory and native monster classes retain the existing generator contracts.

Generated combat-baseline recipes and wrapper now allow20..500frame budgets;
non-generated historical recipes retain the150frame minimum. Focused registry
test and default full Go suite pass. PS parser passes. An initial preflight
found an omitted-field canonicalization issue in custom solo descriptors; it
was corrected without changing core validation. An initial native pool attempt
was rejected by the old wrapper150frame guard before any game started. Those
failed preflight artifacts are preserved; a new frozen plan/output root is used.

All64live short episodes completed with valid capture/provenance/frame budgets.
64native PPO exports passed. CUDA likelihood/history and exact next-state critic
bootstrap verification are running. A short horizon is a truncation: existing
GAE retains gamma*next_value, whereas native terminal states use zero bootstrap.
No artificial timeout-as-death labels or forced actions. Full U1 policy/value/Adam
will continue under its original PPO config, not a fire-only update.

Frozen independent validation:32ordinary300frame conditions/eight maps, original
U1, short-trained candidate and rules. No promotion before this comparison.


## First CUDA update complete

64native exports and all64independent CUDA checks pass. The batch contains2969
PPO transitions. Nineteen explicit nonterminal truncations include13ordinary
nonzero next-state critic evaluations and6zero bootstraps on independently
verified completed-goal handoffs (`goal_handoff_verified`); none of these six
is an unfinished battle cutoff.16native terminal rows also use zero bootstrap.
Raw boundary/goal proof and `short-bootstrap-audit.json` preserve this distinction.

Full policy fitting accepted8actor steps with final approximateKL0.0064713068.
Candidate weights SHA256:
`1c1b1e32016afed1e782b2529a40b7bb372a1714633941ab05b8b55e22a98d33`.
Original config and4Adam parameter groups preserved;64fresh seeds added to the
consumed set. All54actor tensors (including motor, sampled-motion aim, weapon
state, fire, fine aim and memory-route branches),12critic tensors,5shared joint
scorer tensors and log_std changed. This is a full-policy update, not another
H-JEPA fire-row-only fit. Original U1 and checkpoint remain unchanged.

Independent ordinary300frame validation is running on32conditions/eight maps:
original U1, new short-trained candidate and rules,96native games total. Training
outcomes and fitting metrics are not quality evidence. No model promotion.


## Ordinary-battle validation closed

| Variant | Wins/32 | Deaths | Timeouts | Received HP | Self HP |
|---|---:|---:|---:|---:|---:|
| Original U1 |12|12|8|1478|187|
| Short64,8actor steps |10|15|7|1794|281|
| Rules |10|12|10|1784|51|

All96native games and64independent U1 CUDA evaluation replays pass, source fence
unchanged. An incorrect stock execution flag prevented the initial rules launch
before games; standard rules legacy contract was then run on exactly the same
32conditions in a separate output. Original64neural games retained, no reruns or
changed inputs. `quality-recovery.json` and recovered completion manifest preserve
this history. Ordinary primary goal receipts and secondary native clearances are
kept separate; rules have one secondary-only clearance.

Ownership audit finds4/32original and3/32candidate games with nonprovider control
while opponents remain, including one winning game in each. These are existing
U1engagement boundaries; do not describe the entire cohort as autonomous learned
combat or discard mixed-control games to improve the score. Training/evaluation
conditions and all optimizer inputs remain separate.

The first short-episode update is a negative quality result: fewer wins, more
deaths and harm. Original U1 remains the baseline; candidate/checkpoint retained,
no promotion. User-requested development focus remains U1, H-JEPA work preserved.
Next compare a conservative explicit actor-step budget while preserving the
original Adam/config, using the same verified short corpus to isolate update
size and a fresh independent validation cohort. Follow with fresh on-policy
short rounds; no repeated training on viewed validation or consumed rollouts.

## Actor update budget comparison

Artifact subdirectory: `step-budget-20261117-104750` under the pilot root.
The full-joint CUDA trainer accepts optional `--actor-step-limit` in the range
1..configured actor steps. This limits attempts for the current update while
leaving checkpoint config, learning rates, optimizer layout, restoration and
guarded acceptance unchanged. The explicit schedule is recorded in both the
checkpoint and report; absent the flag the original budget applies.

Independent arms both start from the original U1 checkpoint, use the same
64-episode training corpus and retain eight critic steps. This is a scheduling
comparison, not sequential reuse of consumed rollouts. The one-attempt arm
accepted one step with KL 0.0017307914; weights SHA256:
`d76b23978c43ae347f012c233f17ec86e6d4c6c8ee9ff51b50a2749052f95ccb`.
CUDA audit verifies all 60 active actor Adam states advanced by one, all 12
critic states by eight, config/layout and consumed-input sets match, and the
critic and its Adam state exactly equal the eight-step arm. The output restores
through the ordinary exact checkpoint gate. Original weights remain unchanged.
An independent default-budget rerun reproduces the original eight-step weight
file byte for byte; this checks that the new optional scheduling limit leaves
default training behavior intact.

Fresh frozen validation uses offsets 504..507, 32 worlds on eight maps, 300-frame
battles and four variants: original U1, one-step, eight-step, and standard rules.
These conditions are disjoint from training and the previous 500..503 cohort.
An initial evaluation accidentally selected an older historical engine; it was
stopped and its partial captures are excluded, with `wrong-engine-attempt.json`
retained. The complete comparison runs in `native-quality-correct-engine` using
the same pinned engine binaries as training. `evaluation-engine.json` binds them.
No model promotion; ordinary battle quality and control ownership still need
assessment for this new cohort.


## Actor budget comparison closed

| Variant | Wins/32 | Deaths | Timeouts | Received HP | Self HP |
|---|---:|---:|---:|---:|---:|
| u1-original | 7 | 17 | 8 | 2136 | 128 |
| u1-step1 | 7 | 17 | 8 | 2120 | 119 |
| u1-step8 | 8 | 14 | 10 | 1972 | 174 |
| rules | 9 | 13 | 10 | 1932 | 206 |

All 128 corrected-engine native games and 96 neural CUDA evaluation replays passed.
The engine audit verifies identical game/server binaries across training and
all evaluation roles. The failed wrong-engine attempt remains excluded.
Source fence, complete optimizer restoration, independent critic equivalence,
and default-budget byte-identical regression checks pass.

First-life ownership audit (games with rules while native opponents remain):

- u1-original: 2/32 games; 0 wins include this boundary.
- u1-step1: 2/32 games; 0 wins include this boundary.
- u1-step8: 4/32 games; 0 wins include this boundary.
- rules: 32/32 games; 9 wins include this boundary.

These matched cohorts retain the original U1 engagement boundary. They are not
evidence of fully autonomous learned combat. No promotion; source U1 unchanged.

`paired-case-diagnostics.json` records viewed diagnostic cases 1 and 6; none
becomes training data. In base2 case 6 (seed 1823082506), original U1 wins at
server frame 363, whereas both short-trained variants reach the 300-command
limit without winning or dying. All three use Blaster throughout this case;
there are no weapon-switch requests. This is a lost completion, not an extra
death or evidence that a specific weapon switch would solve it. It motivates
testing late-fight coverage: current cold-reset training stops after at most
64 commands and cannot directly sample those later states.

No verified quality gain from reducing actor steps. Next test explicit short GAE segments covering early/middle/late states of fresh full-length on-policy TRAIN trajectories, with exact next-state bootstrap and original U1/Adam. First verify segment admission and boundary math; keep validation diagnostic-only. Do not silently rewrite raw rollouts, repeatedly fit this corpus, or reset every segment to the beginning of the fight.

## Short credit over full native trajectories

Artifact subdirectory: `gae-full-20261117-111401`.
`combat_gae_segments.py` makes an explicit, versioned GAE view over admitted raw
rows. A scheduled cut stops backward advantage recursion but retains the exact
next-state critic value. Native terminal/truncation flags remain authoritative;
raw files, sample identities, rewards, action probabilities and attention history
are unchanged. Fragments are correlated parts of their original fights, not
additional independent native episodes. This experiment does not reload saves.

`--gae-segment-steps 64` opts into this estimator. Checkpoints persist its contract
and continuation inherits it; a conflicting explicit horizon is rejected rather
than silently changing the estimator. Original config and full Adam restoration
remain exact. Without a segmentation contract, the default path is unchanged:
the old eight-step pilot weights reproduce byte for byte. Eight focused tests
verify hand-calculated returns, nonzero bootstrap, no reward leakage across a cut,
native terminal versus truncation behavior, context/seed gaps and immutable rows.

Both experiment arms start from the original U1 checkpoint and use the same new
32 native TRAIN fights (eight maps, random native groups of 2..4, randomized stock
inventories, up to 300 commands). One arm retains ordinary GAE; the other uses
64-transition segments while retaining full observation context. Both retain the
original eight-attempt actor budget and eight critic steps. This controls for the
change from the earlier one/two-opponent cold starts to longer, harder encounters.

TRAIN offset 504 and validation offset 508 are separate registered splits with
disjoint actual engine seeds. The new validation worlds are disjoint from the
previous 500..503 and 504..507 validation cohorts. Initial attempted TRAIN offsets
520/524/528 exceeded the registry's split capacity and were rejected before any
games; their logs remain. All final selections are condition-only, not based on
combat outcomes. Pool max 16 at timescale 2; CUDA-only neural audits and learning.
All source Go/PowerShell/Python files and exact training-engine binaries are
bound by the experiment fence. Original U1 and H-JEPA artifacts remain preserved.


## Full versus short GAE pilot closed

All 32 fresh TRAIN native/CUDA cases passed. Both full-policy arms accepted eight actor steps over 4,403 transitions; full GAE KL 0.0093586832, SEG64 KL 0.0099039429. SEG64 contains 85 correlated fragments (32 early, 20 middle, 33 late), with 53 added cuts. Raw rows, observation context, source U1/config/Adam were preserved. Exact checkpoint restoration and rejection of a conflicting continuation horizon passed.

| Variant | Wins/32 | Deaths | Timeouts | Received HP | Self HP |
|---|---:|---:|---:|---:|---:|
| u1-original | 7 | 16 | 9 | 1879 | 278 |
| u1-full-gae | 9 | 15 | 8 | 1817 | 187 |
| u1-seg64 | 6 | 16 | 10 | 1795 | 216 |
| rules | 11 | 13 | 8 | 1406 | 229 |

All 128 native evaluation games and 96 neural CUDA evaluation replays passed.
Engine binaries match training; Go/PowerShell/Python source fence passed.
Primary goal receipts and secondary clearances retain the same definitions.

First-life ownership with native opponents remaining:

- u1-original: 4/32 games include nonprovider control; 0 wins include it.
- u1-full-gae: 3/32 games include nonprovider control; 0 wins include it.
- u1-seg64: 3/32 games include nonprovider control; 0 wins include it.
- rules: 32/32 games include nonprovider control; 11 wins include it.

SEG64 did not improve this pilot and scored below ordinary GAE on the same fresh
training corpus. This does not rule out training from many short native
continuations; here only credit-assignment recursion was cut, while collection
used full trajectories. Ordinary GAE has a small observed gain over original U1,
which is insufficient evidence for general superiority or deployment. All variants
retain the original engagement boundary; this is not full autonomous combat proof.

Keep original live U1. SEG64 has not earned promotion. Next test a predeclared sequence of fresh on-policy updates (not another single-batch horizon tweak), with frozen reward/architecture and exact full Adam; use new isolated registered seed ranges because original TRAIN/validation tails are nearly exhausted. Retain ordinary GAE as the reference learning arm; any further short-fragment arm must be compared causally and independently. Native U1 save/load branching is still unimplemented and needs temporal-memory/RNG restore plus CUDA-context proof before training from restored states.
