# Fresh CUDA PPO от obstacle-escape fork

06.10.2026. Продолжение [obstacle curriculum](learned_combat_obstacle_escape_v4.md).

## Протокол до результата

Parent weights `3c5cd86df29e73b1709a02b5e969e35cfa262ff4464c6cded022d403653a8d7d`,
checkpoint `969b8a9b174e193081a444f1444f726148c08a2687d0f3e3cc9f4da6925ed39c`.
Inherited counters53 PPO updates/524 actor steps; supervised fork оставил
оба Adam states пустыми и zero-output critic, std/RNG/history сохранены.
Предыдущие failed PPO actors не наследуются. Actual runtime и checkpoint
проверяются до старта; источники frozen, новые source edits не планируются.

4 fresh on-policy Mixed batches22300–22315,4 CUDA-only PPO updates.
Каждый batch collected своей behavior policy, один раз consumed. Anchor
сам obstacle parent, pinned beta1,2/3,1/3,0. Final четвёртый checkpoint
назначен заранее, без выбора по eval. Сохранены MLP810→64→64→8,
reward maneuver-v4 (kill+5, death−5, received−0.02) и horizon300.
Initial std наследуется без нового exploration fork; PPO обучает log_std
в штатном actor update, checkpoint/model parity проверяется после каждого.
Guarded Adam direction repair/полный rollback включены как в предыдущем
cycle; fallback отдельно аудируется, не объявляется ростом качества.

Парная deterministic оценка parent→final: fresh Mixed22400–22403,
fresh Solo22500–22503, известные regression20800–20803/21300–21303/
21700–21703/22100–22103. Known regression не fresh holdout и не training.
64 captures:16 training +48 evaluation,4 server/client instances x2,
отдельные confirmed seeds, Blaster/stock HP/fixed100 synchronous,
Solo175 HP. Нет live promotion/изменения пользовательского config.

Основной показатель: оба fixture monster classes убиты без смерти и
без rules до завершения. Incoming damage, visibility timeouts/held frames,
static hull guard blockage и Solo retention отдельно. Улучшение количества
побед не доказывает обход стен; прошлый fork усилил blockage в4/5 наборов.
Проверить lineage/consumption/Adam/CUDA-only/native byte re-export/reward
components и exact CUDA actor/optimizer replay. Блокировки считаются на
уникальных provider alive frames до конца боя с учётом разной длины/исхода.

Root `workspace/artifacts/combat-obstacle-ppo-v4-20261006`.

## Завершённое обучение

64 captures завершены и валидны:16 training +48 evaluation,4 instances
x2, каждый seed подтверждён. Go/native/PS/trainer sources не менялись
во время captures. Training/benchmark и numerical actor replay только
CUDA; CPU использовался для no-grad preparation/проверок и Go inference.

| Update | Fresh rows | Empty-enemy rows | Kill reward rows | Accepted actor steps | Anchor beta | Approx KL |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1104 | 460 | 2 | 10 | 1 | 0.004625 |
| 2 | 1047 | 390 | 0 | 10 | 2/3 | 0.009467 |
| 3 | 997 | 680 | 0 | 10 | 1/3 | 0.010000 |
| 4 | 996 | 404 | 0 | 8 | 0 | 0.009994 |

Итого4144 fresh transitions,1934 без видимых enemies,2 actual kill reward
rows. Последние три batches не дали consumed kill reward. Это ограничивает
утверждение о достаточном обучении завершению боя. Пустые enemies сами по
себе не доказывают наличие живого скрытого врага в каждой такой строке.

38 accepted actor steps; final57 updates/562 cumulative actor steps.
Adam clocks10/20/30/38, critic40/80/120/160, unchanged CPU/CUDA RNG и
checkpoint consumption history проверены. Direction fallback не применялся;
последний actor trial update4 отклонён при downhill proposal действующими
gates. Exact CUDA replay повторил все trials/actor/std/Adam tensors;
независимо проверены moment recurrences, counters и rollback. Ни один
rollout не reused для PPO. Native re-export совпал побайтово во всех4
batches. Reward components всех4144 rows пересчитаны независимо:
max error8.81e-16, оба kill+5 rows включены и принадлежат provider.

Final weights `ae7eba5d4aaadfb95106074b434c2cb56ce21e738a4a4ff9f663f829d261db64`,
checkpoint `51d3c54f4d9fcac1bdce3c7b1b9d64c45e36b0d6fa1139778c454402be0e0001`.
Trainer SHA256 `e5ecadc5241b9fec45c29ebf056085dec9e39a2bdab5536f3446e606579e3a98`.
Final anchor KL на собственных training states0.024111; mean movement
drift0.01515/0.01668, aim0.2940°/0.3624°. Эти небольшие mean изменения
не гарантируют сохранение замкнутых игровых траекторий и старых побед.

## Парные результаты

| Набор | Полные uninterrupted победы parent→final | Kills | Deaths | Incoming health damage |
|---|---|---|---|---|
| Fresh Mixed22400–22403 | 2/4→3/4 | 4→7 | 1→1 | 313→321 |
| Fresh Solo22500–22503 | 4/4→4/4 | 4→4 | 0→0 | 177→153 |
| Known20800–20803 | 3/4→2/4 | 6→6 | 1→2 | 369→311 |
| Known21300–21303 | 3/4→0/4 | 6→1 | 1→3 | 317→308 |
| Known21700–21703 | 3/4→3/4 | 6→6 | 0→1 | 263→304 |
| Known22100–22103 | 3/4→2/4 | 6→5 | 1→1 | 305→242 |

Native kill windows/dispatch/model SHA/source conditions и seeds проверены;
успешные Mixed captures убивают обе fixture classes и не имеют rules до
завершения. Описательная сумма20 разных Mixed seeds:14→10 полных побед,
4→8 deaths,28→25 kills,received1567→1486. Это один fresh holdout и четыре
уже известных regression набора, не20 новых test seeds и не общий
статистический вывод. Снижение общего received не компенсирует рост deaths
и незавершённых боёв; длительность/траектории тоже изменились.

На21300 после PPO3 смерти и1 незавершённость; единственное убийство —
Gunner на21302, Parasite остался живым. Offline attacker attribution
полученного damage282 Gunner/26 Parasite. Это наблюдаемый риск, не
доказательство, что именно anchor decay вызвал забывание навыков.

## Blockage и потеря видимости

Те же unique provider alive first-life windows до final kill для побед,
до конца capture/death для остальных. Rules windows сюда не входят;
победы с предшествующей rules помощью не принимаются. Native first-life
outcomes для неудач могут включать rules после handoff и помечены отдельно.

| Набор | Blocked/provider frames parent→final | Доля parent→final | Max blocked run |
|---|---|---|---|
| Fresh Mixed | 190/916→174/864 | 20.7%→20.1% | 74→60 |
| Solo | 300/364→276/344 | 82.4%→80.2% | 75→69 |
| Known20800 | 281/763→116/829 | 36.8%→14.0% | 77→58 |
| Known21300 | 206/853→14/701 | 24.2%→2.0% | 68→5 |
| Known21700 | 198/972→183/884 | 20.4%→20.7% | 68→63 |
| Known22100 | 203/817→128/939 | 24.8%→13.6% | 69→64 |

Суммарно Mixed blocked1078/4321→615/4217 (24.9%→14.6%). Но на21300
резкое сокращение blockage сопровождается потерей всех полных побед.
Нельзя считать эту метрику самостоятельным освоением безопасного манёвра:
состав посещённых состояний и длины боёв разные, требуется kill/survival.

Final held frames:Mixed22400=0,20800=0,21300=207,21700=0,22100=235,
Solo=0. В победах final нет held frames: поиск невидимого противника не
доказан, длительное continuation/timeout присутствует только в неудачах.
Небольшие вариации parent traces после visibility timeout не меняют
указанные before outcomes; full rules-assisted trajectory не объявляется
полностью воспроизводимым learned rollout.

## Решение и следующий эксперимент

Final PPO ветка отклонена как замена obstacle parent. Все checkpoints
сохранены для диагностики; промежуточные checkpoints не выбирались по eval.
Obstacle fork остаётся учебной опорой, а не принятой live policy.
Пользовательские marker/config не изменены, временные server/client
процессы завершены, игровые assets используют кеш/PAK link cleanup.

Следующий контролируемый опыт — сохранение anchor при PPO: сравнить
constant retention с текущим затуханием на new fresh train/holdout seeds
и закреплённых regression наборах, при одинаковых starting parent,
reward/horizon/std initialization/update budget. Расписание должно быть
pinned в checkpoint и проверяться при resume. Не приписывать результат
сразу ёмкости сети, отсутствию GRU или reward; разделить проверяемые
гипотезы. Если нужен retention state bank, использовать только training
observations с отдельными hashes, не известные eval traces; это отдельный
фактор, не добавлять его незаметно в тот же comparison.

## Проверки и артефакты

Root: `training-audit.json`, `training-audit.log`, `direction-audit.json`,
`consumption-audit.json`, `evaluation-audit.json`, `blockage-audit.json`,
`previous-mixed-attribution.json`. Retention: `native-reexport-audit.json`,
`shaping-audit.json`, frozen trainer/config/reward и4 batches/checkpoints.
Новых source изменений не было, Python/Go suites повторно не запускались;
здесь выполнены independent training/live harness audits. `git diff --check`
прошёл, загрузок/commits/push/live promotion не было.
