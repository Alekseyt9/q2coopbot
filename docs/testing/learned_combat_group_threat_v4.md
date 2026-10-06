# Group threat curriculum v4

06.10.2026. Следующий опыт после отрицательной
[проверки angular noise](learned_combat_narrow_aim_ppo_v4.md).
Parent — `combat-ppo-maneuver-mixed-v4-20261006/iteration-4/update`,
53 PPO updates / 524 actor steps, weights SHA256
`f55325f08d1185bfe2f5998c658470b18feef2e38832c6f549e3fdad6b20cdfe`.

## Протокол до результата

Новая отдельная supervised ветка `scripts/fork_combat_group.py` совместно
обучает movement и bbox aim, включая выбор наблюдаемой цели. Runtime Go,
810→64→64→8 actor, наблюдения v4, std и guards сохраняются. Входы — только
имеющиеся features; seed/hidden health/AI/reward/server truth не входят в labels.

Метки применяются только при наблюдаемом составе≥2. Близкий Parasite получает
более высокий приоритет aim, чем более близкий Gunner: это проверяемая локальная
учебная гипотеза, не доказательство оптимального выбора цели. Неизвестный bbox
или clear-shot не дают aim label. Yaw/pitch ограничены±20° на шаг.

Movement рассматривает8 известных BSP-направлений: standing clearance≥40,
известный ground drop0–48, отсутствие padded barrel box на локальном шаге40.
Кандидат учитывает Parasite range, боковое движение относительно Gunner и
closest approach видимых снарядов с известной скоростью. Отдаляющиеся снаряды
не трактуются как приближающиеся. Линейный прогноз на0.25 game seconds — только
учебная подсказка, не engine rollout/модель динамики и не гарантия безопасности.
Неизвестная enemy velocity не используется в прогнозе. Команды компенсируют
одновременный новый yaw/pitch и знак Quake sidemove.

Initial fit:4168 проверенных training rows из17800–17815 и18100–18115;
1720 solo retention rows16800–16815. Все прежние held-out evaluation исключены.
Fixed2000 epochs. Gate на training states: solo movement mean drift≤0.05,
solo aim mean drift≤2°, attack probability drift≤0.05, vertical KL≤0.02.
Оба Adam reset, critic output zero; std/RNG/consumed history/PPO counters retained.
Это supervised fit, не новый PPO update; после изменения policy старые rollouts
не становятся её on-policy данными.

Initial fit gate пройден:3153 movement labels,3244 aim labels,2838 выборов
не ближайшей цели. Movement MAE0.365/0.430→0.150/0.217, aim MAE16.936/2.501→
1.300/0.733°. Solo movement drift0.00727/0.00681, aim0.838/0.424°, attack
0.0000810, vertical KL3.30e-7. CPU13.01ms против CUDA2.16ms — RTX5070 быстрее
примерно6 раз и использована для fit. Это training-state показатели, не бой.
Initial fork weights SHA256
`201718a3d9476bf648ad6fae9ed1106fe8e41e9402f45458a45b3799d2478e68`.

До live проверки зафиксировано: `run_combat_group_curriculum.ps1`,4 fresh Mixed
batch18600–18615, после каждого supervised group fit2000 epochs на исходных
учебных inputs и накопленных новых states, solo retention отдельно. Каждый
batch4 independent server/client instances x2,300 frames, stock Parasite+Gunner,
Blaster/release0. Native `q2ppo-data` проверяет свежую stochastic behavior,
но награда не используется в supervised loss; PPO counters не растут.

Фиксированный четвёртый результат сравнивается с parent на Mixed18700–18703
и solo175HP/release100 на18800–18803, обе policy deterministic, по4 instances x2.
Нет выбора best checkpoint или обучения по eval. Все captures проходят новый
PAK cleanup; исходные модели и пользовательский live сохраняются.
Root: `workspace/artifacts/combat-group-threat-v4-curriculum-20261006`.

15 focused tests прошли: masks/clearance/drop, threat target vs nearest,
incoming/receding/unknown projectile velocity, совместный yaw/sidemove,
finite/bbox validation, сохранение parent/std/history/counters/retention.
PowerShell driver проходит parser check.

## Завершённая оценка group curriculum

Четыре fresh batch дали2523 новых training states:646/292/721/864.
Второй batch прервался на sharing violation общего AAS при атомарной замене.
Он исключён, повторён как `iteration-2-retry` с теми же policy/сидами.
Immutable hash reader теперь открывает файл с ReadWrite|Delete sharing,
не изменяя hardlink contents; bounded retries сохранены. Оба focused storage
теста прошли. Итоговый fit содержит6691 group rows и1720 solo retention rows.
CUDA быстрее CPU примерно5–9 раз в четырёх fit. PPO counters остаются53/524;
новых PPO updates и training kills в этих16 валидных captures нет.

Final weights `c5b04e538238ad6be39ee8334d282241fb7f2d51f105f1ba1f6f34d9ae00d808`,
checkpoint `1aea95a92c54ab4ef255525eb3fcc0fd3f19a8723bfa612832561e4d8d8b9811`.
Независимые lineage/native provenance/tensor parity audits проверили все inputs,
std/RNG/consumed history, reset Adam/critic. Reward component max error8.88e-16.
Partial failed capture не включён. Frozen trainer лежит в experiment root.

| Новые парные условия | Parent | Group curriculum |
|---|---:|---:|
| Mixed18700–18703: kills | 3 | 3 |
| Mixed: deaths | 4 | 1 |
| Mixed: оба монстра убиты без смерти | 0/4 | 0/4 |
| Mixed: incoming health damage | 400 | 224 |
| Mixed: outgoing health damage | 1085 | 675 |
| Solo18800–18803: kills/deaths | 4/0 | 4/0 |
| Solo: incoming health damage | 174 | 149 |

В Mixed новая policy убила Gunner в трёх эпизодах, Parasite оставался живым
во всех четырёх. Три окончания по лимиту кадров — выживание, не победа.
Native incoming attribution: Parent Parasite364/Gunner36, новая Parasite0/Gunner224.
Это локальный результат четырёх сидов, не доказательство общего превосходства.
Все solo kills принадлежат provider, handoff происходит после убийства.

Mixed guards: hull167→522, unsupported4→13, barrel attack suppression0→31;
applied attack523→956. Ошибка aim более15° относительно хотя бы одного
наблюдаемого bbox:0/523→187/956. Эта метрика не является точностью попаданий;
nearest-only диагностика также может ошибочно штрафовать выбор другой цели.
Все16 парных eval captures имеют valid provenance/dispatch/seeds и одинаковые
условия каждой пары. Live promotion нет: основная цель полного завершения не достигнута.

## Зафиксированное продолжение: ограниченная дистанция

Гипотеза: только repulsion поощряет бесконечный отход и не обучает продолжать
бой после исчезновения одного из двух врагов. `--range-band` добавляет штраф
за дистанцию выше320 от видимого Parasite и выше512 от видимого Gunner.
Учебная метка разрешена также для одного видимого Gunner или одного Parasite
дальше256. Близкий одиночный Parasite остаётся под retention; невидимая цель
не создаёт меток. Геометрия/velocity masks и guards прежние. Это offline
supervised гипотеза, не новый runtime контроллер и не PPO.

Один fixed2000 epoch fit от итогового group checkpoint на тех же6691+1720
training states, без18700/18800 eval inputs. Range-band weights
`ab0288d86b26b63c97b0a1e006d4f32ce34524a6d39ba7dbb9532f9048cff21c`.
CUDA1.86ms/step против CPU20.78ms/step; выбран GPU. Training gate пройден:
solo movement drift0.00811/0.01595, aim0.671/0.174°, attack0.000188,
vertical KL3.17e-6.16 focused tests прошли, включая возврат к дальней
оставшейся цели и отсутствие метки на неизвестную/близкую solo цель.

До результата зафиксированы новые paired Mixed18900–18903 и Solo19000–19003,
4 independent instances x2,300 frames, Blaster; solo175HP/release100.
Reference — final group, candidate — единственный fixed range-band fit.
Нет выбора checkpoint по eval. Артефакты: `combat-group-range-v4-fork-20261006`
и `combat-group-range-v4-eval-20261006`.

## Результат range-band: ветка отклонена

| Новые парные условия | Group reference | Range-band |
|---|---:|---:|
| Mixed18900–18903: kills | 2 | 0 |
| Mixed: deaths | 1 | 4 |
| Mixed: оба монстра убиты без смерти | 0/4 | 0/4 |
| Mixed: incoming health damage | 151 | 400 |
| Mixed: outgoing health damage | 640 | 290 |
| Solo19000–19003: kills/deaths | 4/0 | 4/0 |
| Solo: incoming health damage | 123 | 119 |

Все16 captures валидны; paired fingerprint/условия/seed совпадают,
native attribution/kill ownership/guards независимо проверены.
Range-band checkpoint SHA256
`98cb1c1fee49b3543df1e7f5318fd31ce3470f89562d3c74e2c43500c6f41abc`;
std/RNG/history/53 updates/524 actor steps сохранены, input proofs/tensor parity
проверены. Reward component max error4.44e-16. В eval не обучались.

В Mixed весь incoming урон обоих вариантов принадлежит Gunner. Reference
убивает Gunner в двух эпизодах и оставляет Parasite; range-band не убивает
ни одного. Barrel suppression3→85, applied attack938→84; hull587→3.
Меньше столкновений со стеной не означает лучший бой. В этих captures
закрытие дистанции сопровождается большим риском и блокировкой выстрелов
у бочек. Причинный вклад каждого изменения отдельно ещё не проверен.
Ноль bbox errors>15° на84 оставшихся applied shots не компенсирует отсутствие
убийств и не доказывает точность стрельбы. Solo kills — provider, frame178,
handoff179; выигрыш4 health points не является общим улучшением.

Range-band не принят. Group reference также не принят вместо прежнего parent:
полных Mixed побед нет, перенос результата18700 на18900 не гарантирован.
Следующий ограниченный опыт должен отдельно обучать продолжение боя после
первого убийства и сохранять поведение в состоянии двух угроз; нужен контроль
риска Gunner и возможности стрелять вокруг наблюдаемых препятствий. Простое
сближение со всеми видимыми целями и уменьшение hull interventions не подходят
как критерии успеха. Следующие training/eval seeds должны быть новыми.
