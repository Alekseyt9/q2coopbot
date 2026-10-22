# Первый matched fit conditional fire

22.10.2026. Оба варианта обучены на GPU и прошли resume/export/optimizer audits. Fresh32native/24CUDA screen закрыт: **start/full/fire-only по3/8, rules4/8**. Нового победителя нет. Default и опубликованные веса не заменялись.

## Корпус и сравнение

`workspace/artifacts/combat-conditional-fire-training-v1-20261022`.

Сначала пройдены [checkpoint migration](combat_conditional_fire_migration_20261022.md) и [fresh native compatibility](combat_goal_boundary_repair_20261022.md). Новый own-policy corpus содержит **32 TRAIN боя, 5058 eligible rows**, 32 native/32 CUDA checks. Восемь family/site/loadout комбинаций base3/ware2: site01/site02, ordinary trio и grenade recovery. Реальные условия генерируются из реестра; пул16, ×2, отдельный seed каждого боя.

Оба fit стартуют с одного zero-migrated checkpoint и используют один корпус, полные causal histories, reward v13 без изменения, одинаковые40critic steps и до10actor steps. Actor update — прежний PPO со sequence retention weight1; conditional Bernoulli retention использует fixed complete behavior action. Для fire branch LR0.0003. Различается scope:

| Вариант | Обучаемые actor/std параметры | Принято actor steps | GPU fit, с | Approx KL | Sequence retention KL |
|---|---:|---:|---:|---:|---:|
| full | 194333 | 10 | 3.256 | 0.004893 | 0.005040 |
| fire-only | 6369 | 10 | 2.340 | 0.00000548 | 0.00000487 |

GPU fit time не включает native collection, загрузку/сериализацию данных и проверки. Это один ограниченный update, не доказательство сходимости новой ветки. Fire-only изменил политику значительно слабее full; причину и дальнейший бюджет нужно оценивать по native результатам и train-only diagnostics, а не объявлять архитектуру успешной или бесполезной по числу шагов.

## Проверки после fit

- `full-audit.json` и `fire-only-audit.json`: initial actor/value/std/Adam/RNG точно совпадают с исходным checkpoint; rollout не использован ранее и добавлен в consumed history ровно один раз; counters увеличены корректно; runtime JSON и checkpoint численно совпадают на CUDA.
- Новые fire parameters изменились; Adam clocks новой группы равны10.
- В fire-only прежние actor/std и численное состояние старых Adam groups сохранены точно, включая защиту после fallback.
- `matched-fit-audit.json`: critic и его optimizer у двух fit совпали точно. Различия actor scope не смешаны с разными critic updates или обучающими данными.
- `training-complete.json` закрывает обе ветки. Reproducible runner: `scripts/run_combat_conditional_fire_training.py`; допускает `--resume` только для закрытых фаз. Незавершённый fit не перезапускается поверх существующих артефактов.

## Native screen

Заранее записанный `evaluation-spec.json`: **32 native /24 CUDA**, четыре варианта по8 одинаковых fresh условий — start, full, fire-only, rules. Seed offset754;16×2. Сначала проверяются собственно goal wins, deaths и received damage; physical hits/misses и лишние действия служат объясняющими метриками.

Это малый механизм-screen. Он не доказывает общее превосходство над rules или причинный выигрыш архитектуры над старой моделью, обученной на равном бюджете. Автоматического promotion и выбора промежуточных checkpoints нет. Нельзя переносить результаты на полное прохождение кампании или coop с человеком без отдельной проверки.

## Закрытый результат

| Вариант | Победы | Убийства | Полученный урон | Смерти |
|---|---:|---:|---:|---:|
| start | 3/8 | 11 | 564 | 4 |
| full | 3/8 | 12 | 567 | 3 |
| fire-only | 3/8 | 11 | 549 | 4 |
| rules | 4/8 | 11 | 562 | 4 |

Full приобрёл одну победу и потерял одну относительно start: base3 2/4→1/4, ware2 1/4→2/4. Fire-only не изменил набор побед. Source и оба fit происходят из одного текущего checkpoint; сравнивать эти3/8 с прежними15/24 на других условиях как регрессию веса нельзя.

Native MG outcomes, только проверенные provider windows:

| Метрика | start | full | fire-only |
|---|---:|---:|---:|
| Выстрелы с уроном по живому монстру / все | 157/240 | 169/247 | 158/240 |
| Доля | 65.42% | 68.42% | 65.83% |
| Без выбранной цели и видимого bbox | 19 | 9 | 19 |

Все перечисленные выстрелы без цели попали в solid geometry. Это не все случаи выстрелов в стены и не показатель общей точности всех видов оружия. У full одновременно менялись movement/aim/target/base-fire, поэтому сокращение этих MG shots нельзя приписать новой fire branch отдельно.

По всем provider frames large-yaw fraction:4.39→5.09→4.73%; stationary-with-clear-enemy:22.27→22.44→21.84%. Избыточные повороты и стояние не устранены. No-selected attack-command fraction10.17→10.64→9.29% измеряет команды всех видов оружия, а не native MG rays; эти метрики нельзя смешивать.

`screen-summary.json`, `machinegun-aim.json`, `machinegun-outcomes.json` и `decision.json` сохраняют источники/хэши и решение **no promotion**. Полный pipeline закрыт `complete.json`.

## Train-only fire signal

`fire-signal-diagnostic.json`: на5058 actual behavior contexts средняя attack probability fire-only изменилась0.83357→0.83246, всего−0.11 процентного пункта. Для688 записей без выбранной цели —0.75145→0.75007. Это маленькое изменение, а не подтверждение сходимости новой6369-parameter branch.

Начальный local PPO score `normalized_advantage*(attack-p)` в группе без selected target в среднем **+0.00422**, а с selected target−0.00447. Это групповое направление для независимого fire logit на TRAIN, не причинная оценка каждого промаха. Текущая награда не даёт однозначного сигнала уменьшать слепой огонь в этой группе. Простое увеличение числа шагов пока не обосновано этим screen.

Следующий [план проверки награды](combat_conditional_fire_reward_signal_plan_20261022.md) сначала использует сохранённые TRAIN captures для relabel/gradient diagnostics. Новые игровые бои не нужны до подтверждения корректного сигнала. Исходной точкой остаётся predeclared zero migration, а не выбранный по этому малому screen checkpoint.

Последующее [уточнение по native misses и300-step budget probe](combat_conditional_fire_reward_signal_20261022.md): прямые68blindMGmiss ужеимеют negative signal;малыйcost почти не меняетinitialgradientdirection. Новыйunchanged-rewardfit обучен тольконаTRAIN;егоfreshnativequality ещёизмеряется.
