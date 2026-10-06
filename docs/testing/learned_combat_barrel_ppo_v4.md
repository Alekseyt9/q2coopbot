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

56 focused Python tests прошли, включая actual CUDA retry regression.
Первые weights остаются наблюдаемым поведением, но их optimizer lineage
не подтверждает заявленный откат; такой же старый код мог влиять на прежние
PPO циклы. Новый опыт начинает с пустого optimizer после supervised parent,
а не с optimizer первого ошибочного цикла. Архитектура и reward прежние.

## Исправленный цикл: обучение и Mixed оценка

581/781/561/718 fresh rows, total2641. Accepted actor steps6/10/10/9,
actual Adam counters6/16/26/35; value counters40/80/120/160. Final57 updates,
559 total actor steps. Все четыре updates CUDA-only, KL0.0099993/0.0085831/
0.0095841/0.0099998; два update остановлены после rejected trial без его
применения. Baseline moments/counters теперь остаются неизменными.
Exact lineage/optimizer counters/objective/gamma/source proofs проверены;
native replay повторён, exported rollout bytes совпали во всех четырёх batch.

Final weights `ced33b062164cec90e253f13707b18a198090cf64bf611407e2d11ad81cd3b92`,
checkpoint `41ee1af1c24ab3783a79c9adbc2e2731491e5650da2b4ab34284b46c6a9c3282`.
Training first-life:3 kills/9 deaths, в том числе полное завершение19805.
Все3 kill reward rows (19805 steps164/274,19809 step196) вошли в PPO.
2641 consumed scores совпали с native source rewards; independent component
audit max error8.88e-16. Sparse успешные переходы есть, но их мало.

| Mixed19900–19903 | Barrel parent | Corrected PPO |
|---|---:|---:|
| Оба монстра убиты без смерти | 2/4 | 0/4 |
| Kills / deaths | 5 / 2 | 3 / 3 |
| Incoming health damage | 325 | 318 |
| Outgoing health damage | 1035 | 865 |
| Barrel damage | 0 | 0 |
| Grenade damage | 162 | 0 |

Native incoming attribution Gunner183→33, Parasite142→285. Это перенос риска,
не общее улучшение: основной критерий полного боя ухудшился. Before19902
kill перед смертью не считается победой; after19900 trace_end без kills
тоже не победа. Все kills provider; runtime handoff не создаёт улучшение.
Hull interventions131→21, unsupported37→24, no-ground1→2, applied attack
709→842; уменьшение movement guards не является приёмкой боя. Bbox error>15°
34/709→107/842, не hit accuracy. Парные условия/fingerprints/seeds проверены.

Ветка не принята и не включена в live. Следующий ограниченный опыт должен
проверять сохранение исходных навыков при PPO, например временную distribution
retention к исходному learned actor на fresh training states с заранее
фиксированным ослаблением веса. Это будущая гипотеза, пока не реализована;
не Go tactical override и не обещание улучшения. Sparse full-success episodes
и влияние discount/GAE на позднее добивание надо учитывать отдельно, не меняя
одновременно reward/horizon/архитектуру и retention. Eval states исключены.

Solo20000–20003: оба варианта4 kills/0 deaths, outgoing700, incoming192→150.
Kill ownership и handoff после kill подтверждены. Исправленный цикл содержит
32 валидных captures:16 training +16 paired Mixed/Solo eval. Парные source/native
fingerprints и условия совпали, source receipts и reward audit пройдены.
Первый дефектный цикл хранится отдельно с `validation-status.json=false`;
его states/checkpoints не использованы в повторе. Пользовательский live marker
сохранён, принадлежащих этому опыту игровых процессов после оценки нет.
