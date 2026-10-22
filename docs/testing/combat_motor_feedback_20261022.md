# Свежие циклы PPO с motor branch, 22.10.2026

Root `workspace/artifacts/combat-motor-feedback-v1-20261022`.

Предыдущий [96-case screen](combat_motor_architecture_ablation_20261022.md) дал full9/24, source9/24, base-only7/24, rules15/24. Новая ветка после10шагов имеет очень слабую зависимость поправки от сцены. Следующая гипотеза: несколько свежих циклов собственного взаимодействия дадут более полезное обучение, чем один update на одном корпусе. Это не установленная причина отставания.

## Зафиксированный бюджет

- Две ветки: full (весь actor и motor), base-only (старый actor, motor frozen zero).
- Стартуют с соответствующих уже закрытых matched fits; начальные веса различаются и обе ветки получили одинаковый предшествующий TRAIN corpus/budget. Контроль не является новым общим случайным стартом.
- Два раунда на ветку, по32 новых TRAIN боя: **128 новых боёв**, одинаковые8семейств/base3+ware2/site01+02, ordinary и recovery, разные свежие seeds.
- Каждый раунд использует собственные текущие веса. Число кадров/состояния и trajectories могут различаться между ветками; это равенство case/update budgets, а не побайтно одинаковые данные.
- После своего раунда каждая ветка обучается только на его свежем корпусе, до10actor/40critic шагов, CUDA. Actor/critic/Adam/RNG возобновляются; rollout нельзя потребить дважды.
- Наградаv13, learning rates и retention strength1 прежние. Retention teacher/rows обновляются вместе с own-policy corpus через явный sequence-retention fork.
- Никаких промежуточных оценок или выбора удачного checkpoint. После двух раундов fixed endpoints сравниваются с собственными начальными весами и rules: **40native/32CUDA**, по8свежих условий на вариант, четыре карты, offset740.
- Сбор и оценка: один пул16инстансов ×2. Обучение и численный аудит — только GPU. Явный резерв диска20GiB; при нарушении работа останавливается, данные автоматически не удаляются.

## Реализация и статус

`scripts/run_combat_motor_feedback.py` выполняет конечный протокол, сохраняет фазы и hashes. Возобновление `--resume` переиспользует только закрытые фазы; незавершённую фазу сохраняет и отвергает вместо скрытого перезапуска. Код Go/native/Python зафиксирован в `execution-source.json`.

`scripts/audit_combat_motor_feedback_cuda.py` после каждого update проверяет точное начальное actor/critic/Adam/RNG, одноразовое потребление текущего rollout, конфигурацию/counters и соответствие runtime JSON/checkpoint. Для base-only дополнительно проверяет неизменность motor params/moments. Равенство итоговых critic между ветками не требуется: их новые траектории различаются.

Оба раунда закрыты для обеих веток: **128новых native TRAIN боя/128CUDA replay**, четыре обновления и четыре аудита complete; `training-complete.json` опубликован. Полная ветка8471TRAIN переход, контроль9376; case/update budgets одинаковые, длины собственных траекторий различаются.

| Ветка и раунд | Свежие бои | TRAIN переходы | Actor steps | Critic steps |
|---|---:|---:|---:|---:|
| Full,1 | 32 | 3721 | 10 | 40 |
| Base-only,1 | 32 | 4500 | 10 | 40 |
| Full,2 | 32 | 4750 | 10 | 40 |
| Base-only,2 | 32 | 4876 | 10 | 40 |

Все initial actor/critic/Adam/RNG и runtime JSON/checkpoint checks прошли. У base-only motor params/moments неизменны. Разница числа переходов ожидаема: собственные траектории и seeds различаются. Final approximate KL по четырём fits0.004567/0.004688/0.004771/0.004464 находится внутри прежнего guard0.005. Обе ветки завершили23исторических updates/225actor steps; именно в новом feedback добавлены2updates/20actor steps на ветку. GPU fit занимал≈3.0–3.7с на update без сбора/парсинга/проверок; это не полное время цикла.

Final fixed40-case evaluation закрыта: **40native/32CUDA**, initial-full/initial-base/final-full/final-base/rules,8одинаковых свежих условий на вариант. Основной игровой model/default не заменён, promotion=false.

| Вариант | Победы | Убийства | Полученный урон | Смерти |
|---|---:|---:|---:|---:|
| Initial full | 3/8 | 11 | 508 | 3 |
| Initial base | 3/8 | 10 | 532 | 4 |
| Final full | 4/8 | 10 | 438 | 3 |
| Final base | 2/8 | 7 | 617 | 6 |
| Rules | 4/8 | 8 | 349 | 2 |

Full vsown initial1gained/0lost,vsfinal base2/0,vsrules2/2. Единственная новая победа кinitial — ware2site01recovery,seed1252026740. В этом малом probe fullMG live damage126/199→118/237,largeyaw6.58→10.38%;полного исправления прицеливания/движения нет.

## Независимый screen закрыт

Те же fixed weights без refit: **72native/48CUDA**,24свежих случая на вариант,offset744–746,12ordinary+12recovery,16 ×2. Все cases/audits complete, `screen/complete.json`, `screen-bundle/complete.json` опубликованы.

| Метрика | Initial full | Final full | Rules |
|---|---:|---:|---:|
| Победы | 11/24 | 15/24 | 17/24 |
| Ordinary | 5/12 | 6/12 | 5/12 |
| Recovery | 6/12 | 9/12 | 12/12 |
| Убийства | 28 | 37 | 39 |
| Полученный урон | 1360 | 982 | 918 |
| Смерти | 6 | 2 | 3 |

Full vsinitial **4gained/0lost**,vsrules1gained/3lost. Это положительный независимый сигнал продолжения обучения в данной lineage. Превосходства над rules нет; одна training lineage и24условия не доказывают общую устойчивость. Family bootstrap в bundle описательный, с родственными картами/условиями и без поправки на множественные проверки.

| Карта | Initial | Final | Rules |
|---|---:|---:|---:|
| base3 | 0/6 | 0/6 | 3/6 |
| city1 | 6/6 | 6/6 | 5/6 |
| jail1 | 1/6 | 3/6 | 3/6 |
| ware2 | 4/6 | 6/6 | 6/6 |

Рост дают jail1recovery2победы,ware2recovery1иware2site02trio1. Наbase3 fullво всех6случаях спускается на192–264units и подолгу теряет контакт с врагами наверху. Это остаётся отдельным геометрическим провалом, не исправленным общей прибавкой побед.

Native MG live damage315/567→327/604(55.56→54.14%); targetless/no-visible-bbox shots49→74. Largeyaw5.86→3.91%, смены цели10.56→9.31/100frames; attack without target19.53→28.09%. Команды, реальные выстрелы и попадания — разные измерения; поворот/стояние сами по себе не ошибка. Независимый screen не подтвердил сильного ухудшения yaw из малого smoke, но стрельба в геометрию остаётся.

Post-hoc TRAIN outcomes после всехfits:full base3site02trio0/8,base3site02recovery4/8;полного отсутствия успешного геометрического опыта нет,ноtrio труден. Это не validation quality. CUDA branch-input report:final mean forward offset−0.002249 противinitial−0.001299;среднее отклонение между состояниями выросло примерно0.000081→0.000196,ноостаётся малым. Base motor остаётся нулевым.

`screen-decision.json`: current endpoint допускается к следующему конечному **четырёхраундовому** fresh PPO learning-curve опыту,32TRAIN боя/round,128новых TRAIN,midpointпосле2иendpointпосле4. Без смены награды и промежуточного выбора checkpoint. Этот [следующий опыт](combat_ppo_learning_curve_20261022.md) теперь запущен отдельным протоколом;его128TRAIN и96native/72CUDA endpoint evaluation ещё не завершены. [Условный fire residual](combat_conditional_fire_plan_20261022.md) оставлен резервной гипотезой; runtime/training likelihood пока не менялись.

Команда продолжения после штатного закрытия фаз:

```powershell
& F:/src/strat/.venv-gpu/Scripts/python.exe scripts/run_combat_motor_feedback.py --protocol workspace/artifacts/combat-motor-feedback-v1-20261022/protocol.json --resume
```

Источник статуса — `progress.json`, завершение всех128TRAIN подтверждает только `training-complete.json`, итог всего опыта — `complete.json`. Не запускать второй runner параллельно.
Preflight: новый CUDA resume-auditor успешно проверил оба существующих закрытых matched updates (all иbase-only), включая исходные Adam/RNG и runtime/checkpoint; receipts preflight-full.json/preflight-base.json. Это проверка аудитора на реальных состояниях, не результат новых раундов.

Для сохранения20GiBрезерва дополнительно убраны24 точно восстановимых merged caches закрытых старых оценок: прирост свободного места≈1.12GiB; см.[retention record](combat_cold_cache_retention_20261022.md). Первичные case streams, текущие TRAIN/dev и модели сохранены.

Ещё12дубликатов старой112-case quality evaluation дали≈4.84GiB свободного места. Выполнен также отдельный [read-only аудит geometry/navigation inputs](combat_geometry_observation_audit_20261022.md). Текущий sealed feedback protocol не менялся.
