# Barrel escape curriculum v4

06.10.2026. После [finish-only](learned_combat_finish_only_v4.md), который дал
две полные победы, но две смерти от MOD_BARREL, проверяется другая локальная
учебная гипотеза. Runtime guards/Go inference/reward v4 не меняются.

## Протокол до результата

Parent — finish-only weights `2eaf8ee9c9d6ba18bbf957cc473a05004e9005da40d9a8008ea67781927a7ccf`,
checkpoint `a51cc9666a1c2e999cb866dd4932f8f6e93ce167d36e042443a45042fa64b84b`.
Один supervised fit2000 epochs,6691 verified training states17800–17815,
18100–18115,18600–18615;1720 solo retention16800–16815. Eval states не входят.
GPU-only CUDA; недоступность GPU останавливает обучение без CPU fallback.
PPO counters53/524, std/RNG/history сохраняются; optimizer reset/critic zero.

`--barrel-escape` ставит только movement labels при наблюдаемом barrel ближе256
и хотя бы одном видимом враге, известной grounded BSP geometry. Выбор aim,
attack и vertical distill parent; остальные states тоже distill parent.
Существующий выбор восьми направлений дополнен штрафом
`8 * max(0, 1 - future_horizontal_barrel_distance/256)^2` на40-unit шаг.
Known props/barrel masks обязательны; неизвестная цель/объект не создаёт label.
Это консервативный горизонтальный margin: не точная модель blast и не знание
момента взрыва. Engine radius задаётся `self->dmg+40` в g_misc.c, но скрытый
damage/health/barrel activation не входят в наблюдения/labels.
Clearance/drop/barrel box и known projectile velocity masks сохранены.

Первый fit с movement100/inactive5 не прошёл inactive retention gate и не
экспортирован; до любого eval усилено сохранение прежнего поведения.
Финальный fixed fit: movement labels weight20, inactive distillation weight50, aim40,
solo movement10, attack/vertical10. Training-state gates solo и inactive:
movement mean≤0.05, aim mean≤2°, attack drift≤0.05, vertical KL≤0.02.
Эти средние не гарантируют сохранение траектории нового боя.

Новые парные deterministic Mixed19300–19303 и Solo19400–19403,
4 independent server/client instances x2,300 frames, Blaster/synchronous;
Mixed stock/release0, Solo175HP/release100. Reference — finish-only,
candidate — fixed единственный прошедший fit. Основной критерий — оба врага
убиты без смерти; отдельно barrel damage/deaths, total incoming и guards.
Нет выбора checkpoint по eval, новых live подключений или promotion.
18 focused tests прошли; новые tests покрывают retreat от известной бочки,
неизменность aim и отсутствие меток для неизвестных props/nonbarrel.

## Fit до live оценки

Финальный fit прошёл gates:1919 movement labels,0 aim labels,4772 inactive
states. Movement MAE0.171/0.631→0.140/0.0974. Inactive movement drift
0.0152/0.0200, aim0.165/0.166°; solo movement0.0128/0.0117,
aim0.258/0.178°, attack0.000130, vertical KL2.23e-6. CUDA1.86ms/step,
CPU не использовался для training/benchmark. Это учебные показатели.

Weights SHA256 `91697086e5c57d21f85ad393e0674e044ccd08fb54f77e91485e98ee3df7fd6b`,
checkpoint `6a46f68ed40db4fbd2f9b5a5e1b6b34db9c814c4188eb00ec987c7b9c1f32151`.
Lineage audit проверил exact input proofs/seed exclusion/tensor parity,
std/RNG/consumed history,53/524 counters и empty Adam/zero critic output.
55 focused Python tests прошли на текущем CUDA-only коде.
Root: `workspace/artifacts/combat-barrel-escape-v4-fork-20261006`,
eval: `workspace/artifacts/combat-barrel-escape-v4-eval-20261006`.

## Завершённая Mixed оценка

| Mixed19300–19303 | Finish-only reference | Barrel escape |
|---|---:|---:|
| Оба монстра убиты без смерти | 3/4 | 3/4 |
| Kills / deaths | 6 / 1 | 7 / 1 |
| Incoming health damage | 244 | 310 |
| Outgoing health damage | 1080 | 1365 |
| MOD_BARREL damage | 100 | 0 |
| MOD_GRENADE damage | 0 | 153 |

Парное сравнение относится только к новым19300–19303. Нельзя заявлять рост
2/4→3/4 относительно предыдущих19100: reference на новой четвёрке сам даёт3/4.
Новая policy выиграла19302, где reference погиб от бочки; на19301 новая
policy погибла после одного kill, тогда как reference выиграл. На19300/19303
оба варианта убили обоих. Уменьшение одного риска не означает общую победу.
Все kills принадлежат provider; handoff только после полного завершения.

Native attribution: reference Parasite144/Gunner100, новая Parasite142/Gunner168.
MOD и attacker class различаются: barrel может иметь attribution Gunner.
В новой ветке grenade damage153 входит в Gunner168, barrel damage отсутствует.
Runtime barrel suppression11→0, hull201→204, unsupported0→5;
applied attack644→798. Bbox error>15°4/644→50/798 — геометрическая метрика,
не точность попаданий. Полные paired/native proof/seed/kill audits пройдены.

Это четыре сида: нет статистического вывода о blast safety и нет live
promotion. Следующий шаг — fresh stochastic PPO на CUDA для совместного
риска barrel/гранат/Gunner/Parasite, с reward actual damage/kill и отдельными
новыми eval seeds, без обучения на этих оценочных states.

Solo19400–19403: оба варианта4 kills/0 deaths, outgoing700; incoming131→112.
Все kills принадлежат provider, handoff после убийства. Итоговые16 captures
валидны, парные условия и fingerprints совпадают. Независимый reward component
audit max error8.88e-16; checkpoints сохранены, фоновых игровых процессов
после оценки не осталось. Runtime пользовательской игры не менялся.
