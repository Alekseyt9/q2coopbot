# Первый motor-residual PPO fit, 22.10.2026

Моторная ветка прошла механическую проверку: она меняет распределение движения, сохраняя остальные outputs и conditional aim на фиксированной истории и actual movement sample. Сравнение в игре запущено; улучшение победности пока не установлено.

## Корпус и покрытие

`combat-motor-corpus-v1-20261022` закрыт:48 native /48 CUDA,32 TRAIN +16development,4903/2586 eligible rows. Две карты base3/ware2, обе площадки site01/site02, ordinary trio иgrenade recovery. Source — [zero motor migration](combat_motor_residual_20261022.md), reward совпадает с исходным checkpoint. Все training/evaluation seed ranges проверены по истории; development не использован для fit.

TRAIN departure audit:71 отрыв от земли,6 глубоких спусков, все после release. Все6 имеют precommand ground warning и сигнал в предыдущих5 кадрах; ни один не покрыт однозначным PMove prediction. Фактические события используются как проверка покрытия, а не как counterfactual unsafe labels; fall penalty не добавлялся.

## Matched fit

Root `workspace/artifacts/combat-motor-ppo-pair-v1-20261022`. Один source actor/value/std/Adam/RNG, одинаковые4903 TRAIN rows и полные causal histories. Обе ветки получили10 actor /40 critic updates наCUDA, тот же gamma/lambda/reward/sequence retention. Motor group LR0.0003 одинаков в обеих; base LR0.00003, weapon group0.0003.

- Full PPO обучает всю существующую модель вместе с новым motor branch:187964 trainable parameters.
- Motor-only обучает50594 новых параметра; остальной actor/std и его Adam states сохраняются.
- Source boundaries и JSON/checkpoint сверены; финальные critic и critic Adam обеих веток совпали точно. Новые motor Adam clocks продвинулись на10.
- Аудит считает отсутствующий Adam slot эквивалентным пустому словарю: оба означают uninitialized, не числовые moments. В source было31 такое поле. Проверка всех числовых состояний и parameter-ID groups остаётся строгой; форматная разница не потребовала переобучения.

Собственно GPU fit занял около3.43с full и2.03с motor-only; эти времена не включают сбор/канонизацию корпуса и проверки. Быстрый update не означает, что модель обучилась сложному бою за несколько секунд.

## Механический gate

До fit закреплено: среднее изменение tanh(move) не меньше0.005 и maximum conditional nonmovement KL не выше1e-6 на TRAIN histories. Это gate для целесообразности native smoke, не критерий победы.

| Метрика | Full PPO | Motor-only |
|---|---:|---:|
| Среднее абсолютное изменение mean команды движения | 0.003262 | 0.005101 |
| Полный conditional KL | 0.003708 | 0.000512 |
| Movement KL | 0.000271 | 0.000512 |
| Максимальное изменение прочих raw outputs | 0.072045 | 0 |

Motor-only gate пройден; conditional nonmovement KL ниже1e-6. Обновление средней команды движения примерно1.56× сильнее full. Это касается одной и той же истории/выбранного actual sample. На новых игровых траекториях меняются наблюдения и выбранное движение, поэтому aim/target/fire естественно могут реагировать иначе даже при сохранённых параметрах.

Проверка: `scripts/audit_combat_motor_pair_cuda.py`, receipt `pair-audit.json`.

## Native smoke

Source-zero /full-PPO /motor-only /rules smoke закрыт:8 одинаковых условий и seeds на вариант,32 native /24 CUDA,offset728,16 ×2. Четыре карты, ordinary/recovery; source-zero служит прежней политикой с проверенной нулевой добавкой. Новые веса fixed до старта, promotion=false.

| Метрика | Source | Full PPO | Motor-only | Rules |
|---|---:|---:|---:|---:|
| Победы | 4/8 | 5/8 | 4/8 | 5/8 |
| Убийства | 11 | 12 | 11 | 12 |
| Полученный урон | 397 | 363 | 441 | 221 |
| Смерти | 2 | 1 | 3 | 1 |

Motor-only к source1 gained/1 lost, к full0/1: улучшение механики не дало улучшения боя. Фактические MG damage rays99/162 source,101/160 full,122/184 motor. У motor также25 targetless MG shots против7 source и0 full. Изменение состояний на новой траектории может ухудшить бой, хотя защищённые параметры прицела не менялись.

Base3 провал не устранён: ordinary source/full спустились на264units и долго не видели запомненного врага наверху; motor на192units. Уменьшение количества таких кадров у motor не считается успехом, поскольку смерти участились. Эти наблюдения доступны в `elevation-report.json`; MG metrics — `machinegun-outcomes.json`.

Full имеет1 gained/0 lost к source. Чтобы отделить вклад новой ветки от общего PPO update, выполнен [base-only architecture control](combat_motor_architecture_ablation_20261022.md) на том же TRAIN. Motor-only не продвигается, default сохранён.

После положительного сигнала совместного full-варианта в отдельном architecture probe запущен независимый96-case screen. Механический gate сам по себе не является основанием продвижения.
