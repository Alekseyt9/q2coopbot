# PPO retention bank v4

06.10.2026. Следующий фактор после [matched retention](learned_combat_matched_retention_v4.md).

## Протокол до результатов

Original obstacle parent/MLP810→64→64→8, reward maneuver-v4/horizon300.
Fresh parent bank captures: Mixed23200–23211, Solo23300–23307. Train bank
только23200–23207/23300–23303; validation только23208–23211/23304–23307.
Наблюдения выбираются детерминированно, максимум64 на bucket; bucket
по visible count0/1/2+, known standing clearance<40, nearby visible
barrel<320, visible projectile. Каждый непустой bucket получает равную
суммарную массу, unknown clearance не считается стеной. Features —
Go native verified observations, без action/reward/value. Coverage
диагностируется, наличие всех ситуаций не предполагается заранее.

Обе ветки constant on-policy-state retention beta1. Единственный фактор:
control без банка, bank дополнительно weighted KL(anchor||actor) на
fixed train bank с коэффициентом1 во всех actor heads/std. Validation
bank только no-grad diagnostics, никогда не objective/backtracking/
benchmark. Bank sha/weight pinned на resume, удаление/смена отклоняется;
после начала retention branch нельзя добавить банк. Source receipts
проверяются перед/после update. Seeds банка не пересекаются с PPO/eval;
все прежние seeds<23200 и новые train/eval объявлены forbidden.
Не примешивать bank actions/rewards в on-policy PPO surrogate/GAE.

Общий fresh initial PPO batch23400–23403, затем каждой branch own-policy
23404–23415.4 fixed CUDA updates, final update4 назначен до оценки.
Fresh paired Mixed23500–23503, Solo23600–23603; known20800/21300/21700/
22100 по4 seeds. Parent before повторяется для каждой ветки, такие копии
не независимые samples. Первый update теперь может отличаться из-за
bank loss; initial behavior/data/parent Adam/RNG должны совпасть.

36 batches/144 unique captures:20bank+4shared+24own train+96eval.
Максимум4 server/client instances x2, индивидуальные confirmed seeds,
обучение/gradient probes только CUDA. Sources frozen от collection до
eval. Тесты split/forbidden observations-only/pinning/weighted gradient;
native re-export/reward/Adam/RNG/source receipts и exact CUDA actor
replay. Отдельно train/validation bank KL before/after и sample coverage.
Принятие требует обе class kills без смерти/early rules; не выбирать
checkpoint или банк по eval. Нет live promotion, GRU/оружие не добавлять.

Root `workspace/artifacts/combat-bank-retention-v4-20261006`.

## Результаты

36 batches/144 unique captures завершены; source/native fingerprints
одинаковы,4 workers x2, confirmed independent seeds каждого эпизода.
72 Python tests прошли; реальный negative runner check запретил bank/PPO
seed overlap до collection/output. Первый bank capture23200 был сохранён
при исправлении отсутствующего родительского каталога экспортера; capture
не повторялся, native export затем прошёл. Sources frozen во всех captures.

### Банк и optimizer

Bank SHA256 `66f0fc7f34529dfde841eb8509a83db5849673decf41b6cd1ca2f20669ed9db7`.
Train2545 available→988 selected, validation1173→570. В обеих частях21
непустой bucket; mass1/21 каждого подтверждена независимо. Train:
238 без видимого врага,43 Gunner-only,269 Parasite-only,438 обе угрозы.
Validation83/13/208/266. Balancing по count/wall/barrel/projectile не
балансирует тип single enemy: Gunner-only лишь9.52% training bank mass,
Parasite-only28.58%. Это измеренный пробел в составе, не доказанная
причина конкретной неудачи. Features продолжают содержать типы/параметры
каждого видимого монстра; скрытая server truth не добавлялась.

Shared initial PPO input742 rows точно одинаков у branches, parent
resume SHA/config/Adam/RNG одинаковы. Update1 weights теперь различаются
ожидаемо из-за bank penalty, режим constant у обеих. Далее только свои
fresh behavior rollouts. Не применять прежний first-weight-parity gate
к этому фактору. Validation bank нигде не входит в gradient/objective.

Обе ветки4 CUDA updates/40 accepted actor steps, cumulative57/564.
Actor clocks10/20/30/40, critic40/80/120/160, RNG/history/source receipts,
reward/native re-export и exact CUDA actor+Adam replay прошли. Direction
repair не понадобился. Дополнительный независимый analytic KL audit
(с отдельными Normal/Bernoulli/Categorical формулами) подтвердил bank
train/validation diagnostics и точную связь bank features с native data.

| Training | Без банка | С банком |
|---|---:|---:|
| Fresh PPO rows, общий первый батч включён в обе sums | 3219 | 3822 |
| Empty-enemy rows | 1567 | 1460 |
| Kill reward rows | 0 | 6 |
| Max reward error | 8.74e-16 | 6.66e-16 |
| KL к parent на собственных states update4 | 0.01159 | 0.00573 |

Все6 kill rows включены в bank-branch PPO. Это награды от собственных
fresh captures, не из observation bank. Ноль kill rewards у control
показывает сильную зависимость короткого пилота от sample coverage.
Bank final independent KL train0.0062003/validation0.0065860; это mean
weighted KL, а не гарантия отсутствия ошибок на отдельных состояниях.

### Игровые результаты

Победа — оба native class kills без смерти/предшествующих rules;
Solo — один kill. Ownership audit прошёл, raw/strict totals совпали.

| Seeds / набор | Parent wins | Без банка | С банком | Deaths без/с банком |
|---|---:|---:|---:|---:|
| Fresh Mixed23500–23503 | 3/4 | 2/4 | 4/4 | 1 / 0 |
| Fresh Solo23600–23603 | 4/4 | 4/4 | 4/4 | 0 / 0 |
| Known20800–20803 | 3/4 | 2/4 | 3/4 | 2 / 1 |
| Known21300–21303 | 3/4 | 3/4 | 2/4 | 1 / 1 |
| Known21700–21703 | 3/4 | 3/4 | 3/4 | 0 / 1 |
| Known22100–22103 | 3/4 | 2/4 | 3/4 | 1 / 1 |

Описательно20 Mixed условий: control parent15 побед/3 смерти→12/5,
kills31→28,incoming1452→1581. Bank parent15/3→15/4,kills31→32,
incoming1488→1318. Parent copies не независимые samples;24 before
uninterrupted prefixes совпали, rules-assisted full outcomes нет:
known21300 parent incoming248/284 и некоторые held durations различны.
Один fresh holdout+4 известных regression quartets не общий statistical
acceptance. Не складывать повторные parent captures как новые seeds.

Fresh Mixed incoming267→188 с банком против267→303 без; Solo200→170
в обеих ветках. Bank новая неудача20803 компенсировалась победой20800;
на21300 появилась новая смерть,21302 закончился без обоих kills.
На21701 прежний невыигранный бой теперь смерть. Поэтому одинаковая
сумма15 побед не равна сохранению всех прежних навыков/жизней.

Blocked provider frames control1113/4336→985/4485 (25.7%→22.0%),
bank1128/4329→1047/4271 (26.1%→24.5%). Временные окна до death/final
kill различаются; улучшение guard metric не доказательство обхода.

Банк на этой выборке улучшил control и fresh holdout, но не прошёл
сохранение21300 и увеличил смерти относительно parent. Final branch
сохранена как экспериментальный candidate, не новая опорная политика;
original obstacle parent остаётся reference. Live не менялся.

## Следующий отдельный фактор

Проверить балансирование bank по visible monster class composition,
а не только count0/1/2: Gunner-only и Parasite-only отдельно, с прежними
wall/barrel/projectile strata. Сравнить нынешний и class-balanced bank
от того же parent с одинаковыми fresh bank inputs/initial PPO rollout,
коэффициентом1/4 updates и собственными дальнейшими batches. Validation
episodes не использовать для состава/веса после eval; known eval states
никогда не добавлять в bank. Зафиксировать protocol до новых оценок.
Отдельно логировать clear-shot/finish-state coverage и редкость full
successful stochastic episodes. Reward/horizon/GRU одновременно не менять.
Это следующий план, class-balanced bank в текущем коде ещё не реализован.

Root audits: `bank-audit.json`, `composition-audit.json`,
`comparison-audit.json`, `completion-status.json`; branch
`training/direction/consumption/evaluation/blockage-audit.json`,
`retention/native-reexport-audit.json`, `retention/shaping-audit.json`.
Final control weights
`6ce8a7fbfd55ca1c1a992b214c695a94ac268548df66b2bbad40b7f1af8113bc`,
checkpoint `93838fd3d607f01f5f5a9102a444eecd24786ccb71eb094d61d2c3a02f8628cc`.
Final bank weights
`a5c7daa23ba9fa32016ffa0287ed5f2eeee81a1a61f3d69cdefeb8bd410d6a99`,
checkpoint `3fc6561719ba6ef779aef0e9558ff2c441114319341512209a551cfc3b6ec455`.
