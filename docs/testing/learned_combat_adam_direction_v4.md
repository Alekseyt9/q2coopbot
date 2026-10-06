# PPO: проверка направления Adam

06.10.2026. Продолжение [fresh occlusion PPO](learned_combat_occlusion_final_ppo_v4.md).

## Протокол до результата

`guarded_actor_step` проверяет grad·delta фактически предложенного шага
после std projection. При положительном значении зануляет только exp_avg
для нового предложения от того же snapshot; exp_avg_sq и step сохранены.
Все предложения требуют finite loss/KL, grad·delta≤0, loss не выше
исходного+1e-7, KL≤0.01. Backtracking13 попыток. При окончательном
отказе параметры/std/optimizer/LR полностью восстановлены. При принятии
LR возвращается к configured base. Fallback/retry/direction записаны
в report, accepted optimizer state сохраняется в checkpoint.

61 Python tests прошёл; gradient tests только CUDA. Проверены uphill
repair с сохранёнными second moments/clock, ordinary downhill без repair,
и полное восстановление после отказа fallback. На прежнем consumed batch
отдельный CUDA probe принимает repaired step при retry2: loss0.004312→
−0.002344, KL0.006940, Adam clock30→31. Этот probe не экспортирует
policy/checkpoint и не считается новым update.

Один новый фиксированный цикл от original barrel parent (53 updates/
524 actor steps), без наследования предыдущего диагностического actor:
4 fresh batches21600–21615,4 CUDA-only updates, anchor original parent,
beta1,2/3,1/3,0. MLP810→64→64→8, std/reward/config/horizon неизменны.
Последний четвёртый checkpoint заранее назначен final; выбора по eval нет.

Парные deterministic fresh Mixed21700–21703 и Solo21800–21803, плюс
заранее закреплённые regression20800–20803 и21300–21303. Regression
seeds известны из прошлых оценок, не новый holdout и не PPO training.
Всего48 captures,4 server/client instances x2, Blaster, stock HP,
synchronous fixed100 world hold,300 controlled frames. Solo175 HP.
Нет live promotion. Source changes запрещены во время captures.

Root `workspace/artifacts/combat-adam-direction-v4-20261006`.
Обязательны fresh lineage/Adam counters/rewards/native re-export,
CUDA-only device и проверка побед без rules до завершения боя.

## Завершённый цикл

Все48 captures завершены и валидны. Учебные/evaluation seeds разделены,
каждый эпизод подтверждён native receipt. Training и benchmark только
CUDA (torch2.10.0+cu128). За4 updates потреблено3047 fresh transitions,
652 empty-enemy observations,5 kill reward rows. Старые данные не
переиспользованы; numerical regression probe отдельно от training.

| Update | Rows | Empty enemies | Kill reward rows | Accepted actor steps | KL | Direction fallback |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 640 | 221 | 0 | 10 | 0.006512 | 0 |
| 2 | 790 | 252 | 2 | 10 | 0.009822 | 0 |
| 3 | 740 | 0 | 2 | 10 | 0.009943 | 0 |
| 4 | 877 | 179 | 1 | 5 | 0.009995 | 0 |

Итого35 принятых actor steps; final57 updates/559 cumulative actor steps.
Adam actor clocks10/20/30/35, critic40/80/120/160 проверены. Последний
actor trial update4 отклонён действующими gates, без uphill направления.
На этих новых batches repair ни разу не применился: исходы данного
цикла не доказывают игровую пользу direction fallback. Не сравнивать
35 steps здесь и30 в прошлом цикле как эффект исправления: seeds разные.

Exact CUDA replay повторил все actor trials/веса/std/Adam tensors до
финальных checkpoints. Дополнительно независимо проверены recurrences
exp_avg/exp_avg_sq и clock каждого принятого шага. Первое выполнение
аудитора считало anchor на GPU вместо CPU no-grad подготовки trainer и
обнаружило небольшую разницу grad·delta; после согласования именно этой
операции replay совпал точно. Training/checkpoints не переделывались.
Проверены lineage/consumed hashes/frozen trainer/config/reward/anchor,
все native rollout bytes re-export совпали. Компоненты награды всех3047
rows независимо пересчитаны, max error8.88e-16;5 kill rows включены.

Trainer SHA256
`e5ecadc5241b9fec45c29ebf056085dec9e39a2bdab5536f3446e606579e3a98`.
Final weights `f2d4d3bb1299249ad5690a72e9b45b990e6efd1902b488d38aa95426d5c5b50d`,
checkpoint `d8b78d8c159cb917c54838762534552ef590cddba99828dbbc255a25cb419ba2`.
Final anchor KL на собственных training states0.020559; это не общий
quality metric. MLP/810 features/reward/horizon сохранены.

| Парная проверка | Полные uninterrupted победы before→after | Kills | Deaths | Incoming health damage |
|---|---|---|---|---|
| Fresh Mixed21700–21703 | 2/4→2/4 | 4→6 | 2→2 | 306→366 |
| Fresh Solo21800–21803 | 4/4→4/4 | 4→4 | 0→0 | 153→153 |
| Known regression20800–20803 | 2/4→2/4 | 5→4 | 1→1 | 243→251 |
| Previous Mixed21300–21303 | 2/4→2/4 | 6→5 | 1→2 | 288→317 |

Владение kill windows подтверждено. Успешные captures не содержат
передачи rules до завершения боя. На20800 final впервые в этой паре
убивает обоих и выживает, но20803 сменился незавершённым боем с timeout.
Parent20800 held168 суммарных empty-enemy frames/2 visibility timeouts;
final20803 held83/1 timeout. Fresh Mixed/Solo и21300 не имели held frames.
Поэтому улучшение20800 не доказывает поиск невидимого врага: новая policy
там вообще не потеряла видимости. Более редкие timeout не равны качеству.

На21702 появилась победа, но21703 сменился смертью. На21300 появился
полный успех, но21301 сменился смертью. Нельзя принимать отдельные
удачные seeds за улучшение на группе. Ветка не принята вместо parent,
live policy/config не изменены. Нет статистического обобщения по4 seeds.

## Следующая проверяемая проблема

Offline native attribution и last16 trace frames у пяти after deaths:

| Seed | Входящий health damage | Kills | Последние frames с static_hull_blocked |
|---|---|---:|---:|
| 21701 | Gunner35, Parasite65 | Gunner1, Parasite0 (170 damage) | 15/16 |
| 21703 | Gunner95, Parasite5 | Gunner1, Parasite0 (110 damage) | 15/16 |
| 20802 | Gunner100 | 0 | 12/16 |
| 21301 | Gunner100 | 0 | 0/16 |
| 21303 | Parasite100 | Gunner1, Parasite0 (170 damage) | 15/16 |

Это локальная диагностика хвоста до смерти, не причинный эксперимент
и не доля блокировок за весь бой. Два эпизода оставили Parasite почти
добитым и три — вообще живым после Gunner kill. Проверять следующий
ограниченный curriculum на выход из static blockage и завершение боя
с оставшимся Parasite, учитывая обе угрозы/наблюдаемую геометрию.
Training states должны быть свежими и отдельными от evaluation;
server truth используется только в offline reward/диагностике, никогда
как скрытые позиции/HP во входе policy. Go не выбирает обход вместо сети.
Сохранить pinned Mixed/Solo/regression наборы и добавить новый holdout;
не менять одновременно reward, horizon и архитектуру. GRU/attention и
обучаемый выбор оружия остаются последующими отдельными опытами.

## Артефакты и проверки

Root содержит `direction-probe.json`, `training-audit.json`,
`direction-audit.json`, `consumption-audit.json`, `evaluation-audit.json`,
`death-tail-diagnostic.json`; retention содержит `shaping-audit.json`,
`native-reexport-audit.json`. Checkpoint/rollout/audит scripts сохранены.
61 Python tests прошёл до captures; Go/native не менялись, новый Go suite
в этом цикле не запускался. `git diff --check` прошёл. Игровые ресурсы
используют существующий кеш/ссылки и cleanup; загрузок не было.
