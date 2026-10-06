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
PowerShell driver проходит parser check. Боевой результат ожидает оценки.
