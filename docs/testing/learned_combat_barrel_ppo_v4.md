# Joint threat PPO after barrel curriculum

06.10.2026. После [barrel escape](learned_combat_barrel_escape_v4.md), который
снизил blast damage, но повысил grenade/total damage, выполняется ограниченный
PPO опыт с реальной наградой боя вместо новых геометрических меток.

## Зафиксированный протокол

Parent weights `91697086e5c57d21f85ad393e0674e044ccd08fb54f77e91485e98ee3df7fd6b`,
checkpoint `6a46f68ed40db4fbd2f9b5a5e1b6b34db9c814c4188eb00ec987c7b9c1f32151`.
53 updates/524 actor steps; optimizer после supervised fork пуст, value output
zero, std/RNG/consumed history сохранены. Config соответствует checkpoint.

Четыре fresh stochastic Mixed batch19500–19515, stock health/release0,
300 frames, Blaster/synchronous;4 independent server/client instances x2.
После каждого batch native q2ppo-data проверяет behavior replay и provenance;
только текущий fresh rollout используется один раз, старые/eval states не входят.
PPO10 actor steps/40 value steps, target KL0.01, backtracking прежний.
GPU-only CUDA, без CPU training/benchmark/fallback. Go inference прежний.

Reward v4 неизменен: monster kill+5, death-5, damage0.01, received-0.02,
self extra-0.02, friendly-0.1, tick-0.001 и прежние aim/Parasite potential.
Grenade/barrel damage учтён в actual received; отдельное изменение reward
не смешивается с этим опытом. Server truth только offline reward/diagnostics.

Единственный фиксированный результат четвёртого update сравнивается с parent
на новых deterministic Mixed19600–19603 и Solo19700–19703,4 instances x2,
300 frames; solo175HP/release100. Нет best checkpoint selection.
Primary — оба Mixed монстра убиты без смерти; secondary — actual total,
barrel/grenade/self damage, guards и solo kill/handoff. Четыре seeds не дают
общей статистической приёмки; live promotion не выполняется.

Root: `workspace/artifacts/combat-ppo-barrel-joint-v4-20261006`.
Перед прогоном CPU/CUDA selector подтверждает только CUDA; источник runner —
существующий `scripts/run_combat_ppo.ps1`.

## Обнаруженный дефект и исправленный протокол

Первый цикл19500–19515 завершил4 updates/2543 rows/40 принятых actor steps,
но независимый audit выявил Adam step counters50/101/173/238 вместо10/20/30/40.
На CUDA `load_state_dict(saved_opt)` может сохранить ссылки на tensors того
же device; subsequent optimizer.step изменял baseline moments/step counters.
Отвергнутые backtracking attempts поэтому влияли на последующие попытки.
Value counters40/80/120/160 корректны. Native replay/rollout bytes совпали,
но это не проверяет optimizer rollback. Первый цикл не принят как корректный
PPO опыт; его checkpoints/captures сохранены для диагностики, solo eval отменён.

Исправление: отдельный `restore_optimizer` делает deepcopy snapshot при каждом
retry и final rejection restore. CUDA regression test проверяет неизменность
baseline moments/counters после нескольких проб и точное восстановление.

До нового eval фиксируется повтор всего опыта от первоначального barrel parent,
fresh training19800–19815, Mixed19900–19903, Solo20000–20003; прежние training
и eval seeds не повторяются и не попадают в новый PPO. Остальные параметры
те же, fixed4 updates. Root `combat-ppo-barrel-joint-v4-20261006-retry`.
Теперь независимый audit требует exact Adam accepted step counters и value
counters, помимо lineage/native proofs/KL/objective. Live promotion нет.
