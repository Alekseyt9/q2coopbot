# Наблюдаемая геометрия: выход из упора

06.10.2026. Продолжение [Adam direction cycle](learned_combat_adam_direction_v4.md).

## Протокол до результата

Отдельный `--obstacle-escape` offline curriculum использует existing810
features:8 current-view BSP probes, observed enemies/types/velocities,
projectiles/velocities, observed barrels, grounded masks. Parent command
переводится в current-view горизонтальное направление с одновременными
yaw/pitch. Ближайший45-degree probe должен иметь known standing clearance
<40. Отсутствие clearance не считается препятствием. Это локальная
аппроксимация, не точный sweep команды. Exit candidates требуют known
standing clearance≥40, known ground/drop≤48;
выбор учитывает видимые угрозы и снаряды существующим local score.
Unknown input не заменяется server truth. При отсутствии safe candidate
метка не создаётся. Видимый одиночный Parasite также подходит; отсутствие
других видимых целей не считается доказательством их смерти.

Меняется только movement: parent aim/fire/vertical дистиллируются.
Все остальные Mixed states и отдельный Solo retention сохраняются.
Fixed2000 epochs/fit. До оценки первый fit с прежними barrel weights
не прошёл Solo retention gate, второй с усиленными Solo/aim/fire weights
не прошёл inactive side drift:0.05622 при пороге0.05 (forward0.03200).
Оба fit не экспортированы; eval ещё не запускался. Третий fit дополнительно
усиливает inactive movement до100; active movement20, Solo movement100,
aim weights400 (inactive20000), fire/vertical distillation100.
Gates неизменны: mean movement drift≤.05, aim≤2°, attack≤.05, vertical KL≤.02
на соответствующих training states; это не closed-loop safety guarantee.
64 Python tests проходят, gradient training/tests только CUDA.

Исходный barrel parent: weights91697086e5c57d21f85ad393e0674e044ccd08fb54f77e91485e98ee3df7fd6b,
checkpoint6a46f68ed40db4fbd2f9b5a5e1b6b34db9c814c4188eb00ec987c7b9c1f32151.
Последний не принятый PPO actor не наследуется. Два fresh Mixed batches
21900–21907 и один fresh stochastic Solo retention22000–22003, только
original policy. Один supervised fit, не PPO: critic output zero,
оба Adam states reset; std/RNG/history/PPO counters53/524 сохранены.
Никаких прошлых/eval states в fit. Архитектура/reward/horizon неизменны.

До fit закреплены парные deterministic fresh Mixed22100–22103,
fresh Solo22200–22203, regression20800–20803/21300–21303/21700–21703.
Known regression не новый holdout и не используется для выбора fit.
52 captures:12 training +40 evaluation, везде4 instances x2, confirmed
individual seeds, Blaster/stock HP/fixed100 synchronous/300 frames,
Solo175 HP. Go/native/PS sources неизменны во всех captures. Trainer
зафиксирован перед третьим fit и последующими eval; его зависимости
прикреплены hashes в trainer-inputs.json. Исправление known clearance
маски сделано до любого fit; первоначальный coordinator завершил12
сборов и отказался от fit из-за изменённого trainer hash. Сохранённые
rollouts проверены и использованы corrected trainer без повторного сбора.
Fixed checkpoint без выбора по eval. При training gate failure eval
не запускается, fit не экспортируется.
Нет live promotion. Runtime Go не выбирает обход/тактику вместо сети.

Root `workspace/artifacts/combat-obstacle-escape-v4-20261006`.

## Результат обучения и проверки

Все52 captures завершены:12 fresh training/retention и40 парных evaluation,
4 instances x2, отдельные подтверждённые seeds. Третий fit прошёл исходные
gates до запуска оценки. Подготовка labels только no-grad; fit/benchmark
только CUDA. 1637 Mixed rows (829+808),548 Solo retention,383 movement
labels:249 при нескольких видимых врагах,134 при одном;291 с видимым
Parasite. Ноль новых aim labels. Остальные1254 Mixed states distilled.

| Training metric | Forward | Side |
|---|---:|---:|
| Movement label MAE before | 0.31422 | 0.86757 |
| Movement label MAE after | 0.22586 | 0.33292 |
| Inactive movement drift | 0.02405 | 0.04007 |
| Solo movement drift | 0.01452 | 0.02325 |

Solo aim drift0.2264°/0.1456°, inactive aim0.1717°/0.2012°,
mean attack probability drift0.000385, vertical KL0.00000679.
Это mean metrics на training states, не гарантии для новых состояний.
64 Python tests проходят. Native re-export всех трёх rollout совпал
побайтово. Независимый аудит повторил feature-only labels, composition,
MAE/retention gates, pinned trainer/import hashes и input receipts,
model/checkpoint tensors, пустые Adam states и zero-output critic,
неизменные std/RNG/history/config/53 PPO updates/524 actor steps.
Новые supervised gradient steps не приписываются PPO counters.

Final weights SHA256
`3c5cd86df29e73b1709a02b5e969e35cfa262ff4464c6cded022d403653a8d7d`,
checkpoint `969b8a9b174e193081a444f1444f726148c08a2687d0f3e3cc9f4da6925ed39c`.
Trainer `779d6c1f4fcd1fbfbacd53d61361b1b9d6f13927437a74d790929ea0f2e08dd8`.

## Парная оценка

| Набор | Полные uninterrupted победы before→after | Kills | Deaths | Incoming health damage |
|---|---|---|---|---|
| Fresh Mixed22100–22103 | 1/4→3/4 | 3→6 | 1→1 | 295→305 |
| Fresh Solo22200–22203 | 4/4→4/4 | 4→4 | 0→0 | 138→152 |
| Known20800–20803 | 2/4→3/4 | 5→6 | 1→1 | 243→369 |
| Known21300–21303 | 2/4→3/4 | 6→6 | 1→1 | 288→317 |
| Known21700–21703 | 2/4→3/4 | 4→6 | 2→0 | 306→263 |

Все успешные Mixed captures убили и Gunner, и Parasite без смерти и
без rules до завершения. Kill windows принадлежат learned provider,
native dispatch/seed/world/model/source fingerprints проверены.
16 различных Mixed условий дали описательные7→12 побед,18→24 kills,
5→3 deaths,1132→1254 received damage. Это сумма одного fresh holdout и
трёх уже известных regression наборов, не16 новых независимых test seeds
и не статистическое доказательство общего превосходства.

Увеличение числа побед сопровождается потерями на отдельных условиях:
20800 parent выживал без убийств, final погиб;21302 parent выигрывал,
final не убил врагов и погиб после visibility timeout. На21701 смерть
заменилась незавершённым боем с5 timeouts и234 суммарными held frames;
это не победа/полноценное learned survival acceptance. На22103 также
смерть и2 timeouts. Ни одна из12 after побед не содержала held frames,
поэтому поиск врага за препятствиями не продемонстрирован.

## Проверка заявленной цели обхода

Учитываются уникальные provider-controlled alive first-life frames
до последнего убийства для завершённых боёв, иначе до capture/death.
Static hull limit — spatial guard при learned owner, не передача rules.
Разные длины боёв и изменившиеся исходы ограничивают причинное сравнение.

| Набор | Blocked frames / provider frames before→after | Доля before→after | Максимальный непрерывный blocked run |
|---|---|---|---|
| Fresh Mixed | 193/1009→203/817 | 19.1%→24.8% | 49→69 |
| Solo | 280/344→294/355 | 81.4%→82.8% | 70→75 |
| Known20800 | 96/938→281/763 | 10.2%→36.8% | 38→77 |
| Known21300 | 353/886→206/853 | 39.8%→24.2% | 256→68 |
| Known21700 | 118/659→200/974 | 17.9%→20.5% | 38→68 |

Общее освоение обхода стен не подтверждено: в4/5 наборах доля blocked
команд выросла. В21300 доля и длиннейшая серия снизились, но это локальный
результат. Не объяснять весь рост побед исправленным обходом: curriculum
меняет движение на множестве наблюдений и меняет траектории боя.

Ветка сохранена как кандидат для следующего fresh GPU PPO с actual
damage/kill rewards, закреплённым anchor и прежними regression наборами,
плюс новыми train/holdout seeds. Грубый nearest-probe label не является
engine trace; дальнейшую проверку геометрии выделить отдельно. Не менять
сразу reward/horizon/MLP. Общей приёмки и live promotion нет: пользовательские
live marker/config сохранены, временные server/client процессы завершены.
Go/native sources не менялись, Go suite повторно не запускался.

## Артефакты

`update/report.json`, `fork-audit.json`, `native-reexport-audit.json`,
`evaluation-audit.json`, `blockage-audit.json`, `trainer-inputs.json`;
неэкспортированные fit failures сохранены в `rejected-fit.log` и
`rejected-fit-2.log`. Runtime preparation использует кеш и cleanup PAK
ссылок; загрузок не было. `git diff --check` прошёл.
