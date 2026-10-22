# AWR-style обновление actor, 22.10.2026

## Основание и границы

Новый корпус геометрии дал необходимые TRAIN примеры спусков, но обычный PPO update не улучшил fresh smoke:4/8 против source5/8/rules6/8. Аудит не доказал ошибку reward: важна фактическая нормировка advantages, а не знак сырой GAE.

Проверяется другая политика обновления: чаще повторять реально записанные действия, после которых возврат выше оценки критика. Основание — [Advantage-Weighted Regression, Peng et al.,2019](https://arxiv.org/abs/1910.00177). В исходной работе AWR сочетает value regression и взвешенное правдоподобие действий и поддерживает continuous/discrete actions. Здесь это ограниченный AWR-style pilot на одном on-policy батче, с нашими causal histories, composite action likelihood, sequence retention и KL guard. Off-policy reuse разных behavior models и общее превосходство ещё не реализованы/не доказаны.

В отличие от прежнего SAC actor не получает градиент через Q для новых действий вне записанной выборки. Это мотивирует опыт после проблемы Q extrapolation; результат SAC не доказывает заранее пользу AWR.

## Реализация

`scripts/ppo_recurrent.py`: явные `--policy-update ppo|bc|awr` и `--fork-policy-update`, spec/история переходов в checkpoint. Смена/отключение objective без fork запрещены. Default PPO сохранён.

Для BC/AWR выбран одинаковый `critic-credit=mc-before`:40 CUDA value updates на gamma=.99/lambda=1 observed returns. Baseline пересчитывается на фактических полных историях. Actor AWR: `w=exp((return-V)/3)`, raw cap20, затем средняя масса весов нормируется к1; loss — взвешенное negative log likelihood записанного composite action плюс прежние entropy/sequence retention. Returns/baselines/weights detached от actor graph. Температура3 и cap20 закреплены до новых оценочных боёв, без development sweep.

BC использует единичные веса. Проверочный AWR strength0 возвращает точные единицы без дополнительных статистических вычислений. AWR strength1 включает фактические веса. Runtime observation/action schema и архитектура не меняются; модель по-прежнему сама выбирает движение, прицел, цель, огонь, оружие и stance.

## Закрытая GPU-проверка

Root: `workspace/artifacts/combat-awr-pair-v1-20261022`.

Четыре fit: legacy PPO / BC / uniform AWR / weighted AWR. Все получают один source actor/critic/std/Adam/RNG и те же2496 TRAIN rows/16 боёв. У каждого10 принятых actor updates и40 critic steps. Development/final-evaluation данные не используются.

- Default PPO weights точно совпали с закрытым elevation-PPO update.
- Uniform AWR и BC runtime weights побайтно совпали; actor/critic/std/Adam/RNG/consumed-rollout state также совпали.
- У BC/uniform/weighted AWR итоговые MC critic weights и critic Adam точно совпали. Отличие weighted actor реально подтверждено.
- Weighted AWR KL0.004917, BC/uniform0.004995; оба в guard0.005.
- Веса AWR: min0.04096,max5.5334,mean1; raw clipping0rows; effective rows1807.84 из2496. Это распределение учебного веса, не измерение качества боя.
- `scripts/test_combat_advantage_regression_cuda.py`:3 CUDA tests на uniform/RNG, detached gradient routing и cap/common baseline shift. `scripts/audit_combat_advantage_regression_cuda.py`: независимые source boundary/zero-control/critic/lineage checks.

Legacy PPO имеет прежние GAE targets/critic-after, а BC/AWR общий MC-before critic. Поэтому причинная проверка именно advantage weighting — AWR vsBC; AWR vsPPO сравнивает весь вариант обновления, не один коэффициент.

## Оценка

Fresh five-way smoke закрыт: source / PPO /BC /AWR /rules,8 одинаковых условий и seeds на вариант,40 native /32 CUDA replay,16 ×2,offset716. Uniform отдельно не прогоняется, поскольку он точен BC. Геометрия/цели и budget frozen до native старта.

| Метрика | Source | PPO | BC | AWR | Rules |
|---|---:|---:|---:|---:|---:|
| Победы | 5/8 | 6/8 | 4/8 | 5/8 | 5/8 |
| Убийства | 13 | 13 | 12 | 13 | 12 |
| Нанесённый урон | 2125 | 2137 | 2116 | 2102 | 1997 |
| Полученный урон | 541 | 396 | 363 | 429 | 250 |
| Смерти | 3 | 2 | 1 | 2 | 1 |

Primary AWR vsBC:2 gained /1 lost. AWR vs source1/1, vsPPO0/1, vsrules1/1. Один net win над BC — слабый сигнал; выигрыш над source/PPO/rules не установлен. AWR не продвигается, большой screen не назначен, default сохранён. PPO из этого smoke — те же weights, которые ранее проиграли source4/8 против5/8 наoffset712; выбирать его по одной новой удачной восьмёрке нельзя. Суммарно по двум этим одинаково устроенным fresh probes PPO иsource имеют по10/16 побед.

## Диагностика распределения обновления

`scripts/diagnose_combat_action_groups_cuda.py` без обучения разложил conditional policy KL на TRAIN. В source self-comparison все группы0 с допуском1e-6; current actor/critic вычислялись только наCUDA. `action-group-diagnostic.json`:

- В PPO conditional aim даёт0.003371 из total0.004246 (79.39%); movement0.000824. Средняя абсолютная разница команды tanh(move) от source0.005348.
- В BC aim84.62% изменения, movement shift0.003092.
- В AWR aim88.70%, movement shift0.003091. Weight для340 attack-without-target TRAIN rows в среднем0.848;4 deep-departure rows получают0.764/1.081/0.751/0.831. Это не строгий запрет лишнего действия и не признак перепутанного reward.

Эти доли описывают итоговый update на TRAIN, а не доказывают, что gradient clipping или KL guard причинно блокируют движение. Spatial/motion branches читают movement means; изменение одного движения тоже может менять conditional aim при замороженных aim weights. Следующий [motor-scope probe](combat_movement_scope_20261022.md) проверяет эту связь и достаточный размер изменения на TRAIN до нового native запуска.

Предсказание или повторение удачных записей не гарантирует полезного поведения вне этих состояний. При положительном результате понадобится независимый screen и новый собственный TRAIN round; при отрицательном не продвигать модель по красивым training losses.
