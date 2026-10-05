# Обучение обходу препятствий и бочек в смешанной группе

06.10.2026. Продолжение [совместного обучения прицела](learned_combat_aim_joint_v4.md). Reference: `combat-ppo-aim-joint-v4-20261006/iteration-4/update`, weights SHA `0ea1f80bf0c46bc56195f6fbf16a760a6e4d3b8c795b5a6d3e8b66a377202c55`, checkpoint SHA `7f5474f6d650d6305fa95d13797bbee0f419b829ac88ab73e4ae5743c8a99424`. Counters49 PPO updates/486 actor steps. Старые checkpoints сохраняются отдельно; пользовательский live не заменяется.

`scripts/fork_combat_maneuver.py` создаёт отдельный offline supervised fork. Runtime gameplay/protocol остаются в Go; новых Python runtime correction/rules нет. Actor810→64→64→8, features/actions/std прежние. Только observed mixed composition≥2, standing/on-ground и известная BSP geometry допускают новые movement labels. Из восьми view-relative probes выбирается известный проход: standing clearance≥40 из64 units, известный ground drop на40 units. Unknown probe/ground/clearance не считается безопасным. Наблюдаемые Parasite ближе320 и barrels ближе384 формируют локальное направление отхода; позиция шага дополнительно исключает padded horizontal barrel box. Это **локальные учебные геометрические метки**, не оптимальный маршрут и не гарантия безопасности на всём следующем шаге.

Скорость target0.75, forward ограничен±0.95. Учтены одновременный parent yaw command и Quake positive sidemove справа; ground pitch/3 уменьшает forward horizontal projection. Targets строятся только из имеющихся810 input features, без enemy hidden health/AI/seed/server state. Все прочие actor outputs distill из parent, raw MSE weight10; solo movement также distill. Fit gate: solo mean movement drift≤0.05, aim mean command drift≤2°, attack probability drift≤0.05, vertical KL≤0.02 на training observations. Critic output обнулён, оба Adam state reset; std/RNG/consumed history/PPO counters сохранены. Старые PPO rows здесь только supervised inputs; их нельзя повторно использовать как on-policy PPO после fork.

Training:1477 verified Mixed rows, seeds16300–16315; solo retention1720 verified rows, seeds16800–16815. Новые movement labels у1153 rows; остальные movement targets сохраняют parent. Fixed2000 epochs, full-batch Adam0.0003. Train movement MAE0.508/0.706→0.105/0.113; solo movement drift0.0166/0.0115, aim drift1.982/1.154°, attack probability drift0.0009305, vertical KL0.00006363. CPU7.61ms vs CUDA1.93ms: **RTX5070 CUDA быстрее3.9 раза**, использована GPU. Fork root `workspace/artifacts/combat-maneuver-geometry-v4-fork-20261006`, weights SHA `0ba5524111174486d494215efbfa2f16eb68d0612c0e953c42dc655dfb9b5ac2`.

Сначала заранее зафиксирована paired deterministic проверка reference/fork: Mixed stock Parasite+Gunner release0, seeds17300–17303; solo stock175HP Parasite release100, seeds17400–17403. Каждый batch4 независимых server/client instances, x2,300 game frames. Capture/native provenance/dispatch/seed подтверждены16/16. `workspace/artifacts/combat-maneuver-geometry-v4-eval-20261006/{mixed,solo}` содержит paired-audit/steps/proofs. Eval observations не используются для fit.

| Mixed first-life metric | Reference | Geometry fork |
|---|---:|---:|
|Kills / deaths|0 / 4|0 / 4|
|Outgoing health damage|170|80|
|Provider frames / handoffs|203 / 0|258 / 0|
|Requested / applied attack|203 / 87|258 / 258|
|Barrel-risk suppressed frames|116|0|
|Static hull blocked frames|116|0|
|Unsupported motion frames|31|6|
|Requested movement stationary frames|112|3|
|Observed provider path length|1029.11|2447.50|
|Provider attack bbox angle error>15°|5 / 87|228 / 258|

Mean visible horizontal Parasite range98.47→254.03. Обход препятствий улучшился, но при новых траекториях прицел выходит за область прежних training states. Урон20 Gunner в каждом after эпизоде сам по себе не доказывает успешный learned combat: все эпизоды закончились смертью и без убийств. Geometry warm-start **не принят как победа над группой**, защитные guards сохранены.

Solo17400–17403: kills4→4, deaths0→0, outgoing700→700, incoming128→140. Incoming по seed30/24/46/28→25/32/34/49. Таким образом на этой новой четвёрке solo kill capability сохранилась, хотя входящий урон суммарно немного вырос. Handoff один после убийства на каждой жизни; native kill ownership проверяется отдельно. Не заявлены универсальность или полный воспроизводимый world reset.

Следующий fixed-budget опыт: `scripts/run_combat_aim_curriculum.ps1`,4 итерации fresh Mixed captures/seeds17500–17515, каждая4 instances x2. После каждой итерации `q2ppo-data` проверяет stochastic behavior/native replay, затем существующий joint aim fitter1500 epochs получает накопленные **учебные** состояния и прежние solo retention16800–16815. Observed bbox angular labels исправляют прицел; движение/fire/vertical distill из текущей ветки. Это iterative supervised geometric curriculum, **не новые PPO updates**. Training counters не растут; critic/Adam каждый раз reset. Fixed fourth model сравнивается с исходным solo reference на новых Mixed17600–17603 и solo17700–17703. Никакого выбора best checkpoint по eval или добавления eval в fit. Root `workspace/artifacts/combat-maneuver-aim-dagger-v4-20261006`.

20 focused Python tests прошли: mask/clearance threshold, barrel/Parasite avoidance, simultaneous yaw/sidemove sign, finite input validation, fork/parent/checkpoint retention и прежние aim/exploration paths. PowerShell driver прошёл parser check. Go/native исходники в этом этапе не изменялись.

Fixed4-round curriculum завершён. Новые verified rows211/262/480/489, всего1442 уникальных; fit вместе с1720 retention rows на1931/2193/2673/3162 состояниях. CUDA выбрана во всех четырёх fits, последний CPU5.38ms vs GPU2.07ms. Последний train aim command MAE0.816/0.651→0.571/0.496°, movement drift0.000915/0.000777, attack probability drift0.00001166, vertical KL2.01e-8. Training kills0/1/1/2, deaths4/4 во всех итерациях; эти данные нельзя считать held-out success. Все4 native training kill rows под provider ownership присутствуют в curriculum data. Награды здесь проверены для provenance, **не используются в supervised loss**. Полные scalar scores некоторых kill frames близки нулю, потому что death-5 на том же frame компенсирует kill+5; отдельный kill component остаётся положительным.

Final `iteration-4/update/weights.json` SHA `34cf01aaeab249d9c36861bdfe2932620d3c67a17f0c6d495e5d35aaaf7003d7`, checkpoint SHA `5a890503bf3ec1f12d3728f98f9a1e4f82b1f9d464572f12bba87461dc014aba`. `training-lineage-audit.json` подтверждает каждый source SHA, JSON/checkpoint exact tensor parity, retained std/RNG/history и прежние49 PPO updates/486 actor steps. Это четыре supervised fits, а не ещё четыре PPO updates.

| New held-out Mixed seed | Outgoing reference → final | Kills reference → final | Incoming reference → final | Final outcome |
|---|---:|---:|---:|---|
|17600|50 → 160|0 → 0|100 → 100|Death|
|17601|50 → 350|0 → 2|100 → 89|Killed both, trace end|
|17602|40 → 175|0 → 1|100 → 100|Killed Gunner, then died|
|17603|40 → 120|0 → 0|100 → 100|Death|

Итог: **первая полная победа над stock Parasite+Gunner в1/4 held-out эпизодов**, total kills0→3, deaths4→3, outgoing180→805, incoming400→389. Все3 final kills native/provider-owned: seed17601 frames106/197, единственный handoffframe198 после обоих kills; seed17602 killframe111, handoff0. Нет приписывания rules-assisted kills. Whole Mixed criterion4/4 both kills+no death **не пройден**, автоматического live promotion нет.

Mixed provider frames278→366, requested/applied attack278/80→366/366; barrel-risk suppression198→0, hull blocked194→78, unsupported28→3; stationary190→77, zeroed movement222→81. Applied attack observed bbox angular error>15°4/80→0/366. Mean visible horizontal Parasite range94.58→216.18. Геометрическое предобучение и новая область aim states дали улучшение, но местами остаются блокировки движения и три смерти. Сравнение этих итогов с первой warm-start проверкой на17300 не является paired effect: это разные held-out сиды.

Final solo17700–17703: **kills4→4, deaths0→0**, outgoing700→700; incoming155→182 (41/50/16/48→48/50/34/50). Все4 final kills provider frame178, handoffframe179; это проверенный сохранённый kill skill, но не улучшение received damage. Native rewards/aim/spacing independently audited для всех32 curriculum train/eval captures, maximum component discrepancy8.88e-16; all1442 row scores match proof. В совокупности с16 warm-start eval captures проверено48 captures. Complete world reset/statistical generalization не доказаны.

Driver preflight проверен отдельно: overlapping training/eval seeds, отрицательный seed и retention/planned overlap отклонены **до создания output и запуска servers**. Пользовательский live marker сохранился; isolated33100–33103 после завершения свободны. Следующий этап — fresh Mixed PPO от сохранённого final checkpoint, с контролем kill/death outcomes и отдельной solo regression; типы/состав, препятствия и observed motion уже передаются сети. GRU/attention/выбор оружия ещё не реализованы.
