# Геометрия спусков: покрытие данных, 22.10.2026

## Основание

[Critic-credit screen](combat_critic_credit_20261022.md) показал0/6 побед learned на base3 против4/6 rules. В нескольких случаях бот оказался на192 units ниже начала и надолго потерял наблюдаемый контакт с противниками наверху. Это не объясняет все проигрыши и не означает, что любой спуск вреден.

Проверен именно существующий TRAIN корпус `combat-reactive-advice-corpus-v3-20261022`, без использования validation траекторий для меток:

- 32 случая, 4 карты, все 8 episode families имеют site-01.
- 52 наблюдаемых отрыва от земли; ни одного глубокого спуска >=96 units в следующие 12 последовательных кадров до приземления. Максимум 16.125 units в этом определении.
- Отдельная проверка полных native-exclusive alive траекторий относительно начальной позиции: максимальное снижение 8 units, глубоких случаев 0/32. Это подтверждает отсутствие нужного поведения, а не только пропуск его выбранным departure detector.

Артефакты в `combat-critic-credit-pair-v1-20261022`: `train-departure-audit.json`, архив использованного скрипта `train-departure-audit-script-v1.py`, `train-elevation-coverage.json`.

## Реализованный аудит

`scripts/audit_combat_ledge_departures.py` связывает native steps с исходным bot trace по точным observation, sent command и client sequence; проверяет полный набор TRAIN jobs. Для каждого grounded→air перехода сохраняет последующее фактическое снижение, приземление/цензуру, вертикальную команду, наличие/неоднозначность PMove prediction и предыдущие5 кадров ground probes.

Направление предупреждающей пробы берётся из наблюдаемой мировой скорости при speed>25; при малой скорости — из исполненного движения с учётом знака side и yaw_delta. Stale geometry маскируется. Это описательный сигнал отрыва, не точный симулятор действия и не label «этот манёвр запрещён».

Проверки `scripts/test_combat_ledge_departures.py`: поворот мировой скорости, знак side/поворот команды, stale geometry, причинность входного warning, отсутствие мутации, разрыв истории и смена жизни.3 pure-data tests прошли; нейросетевых CPU вычислений нет.

## Новый корпус

Подготовлен отдельный objective-aligned registry для4 семейств: base3/ware2 site-02, ordinary parasite+infantry+gunner с оружием и parasite+infantry grenade recovery. Условие генерации не менялось; reward SHA сохранён как в исходном checkpoint. Исходная модель — `combat-sequence-hitscan-pair-v2-20261022/control-source`, новая warm-critic ветка не продвигается.

`combat-elevation-corpus-v1-20261022` закрыт: 16 TRAIN +8 development native боёв и все 24 CUDA canonical replay, пул16 ×2. Получены 2496 TRAIN /1345 development rows. Fresh seeds проверены по предыдущим TRAIN/development/evaluation/screen планам. Новая проверка истории collector включает вложенные evaluation и screen plans. В fitting входят только TRAIN rows.

Эта выборка адаптируется к геометрии, на которой обнаружена проблема. Поэтому следующий успех на свежих seeds этих же площадок нельзя называть переносом на неизвестную геометрию. Более широкий контроль должен включать остальные карты/площадки; test/confirmation trajectories не используются для обучения.

## Следующий gate

Повторный `departure-audit.json` закрыт на новом TRAIN: 30 отрывов от земли, 4 глубоких спуска, 3 после release и1 после crouch. Все4 имеют precommand ground warning и предупреждение в предыдущих5 кадрах; ни один не имеет применимой однозначной dry-ground PMove prediction. Появилось нужное фактическое покрытие, но не counterfactual proof альтернативных движений.

Существующий PMove diagnostic не покрывает jump/air и не должен генерировать уверенные counterfactual labels для них. Даже без jump его applicability conditions исключили4 наблюдаемых глубоких спуска. Наблюдаемый исход после12 разных фактических команд не является результатом альтернативного одиночного действия. Не добавлены blanket fall penalty, runtime motion restriction или правило, исполняющее движение вместо модели.

## Первый actor fit

`combat-elevation-ppo-v1-20261022` содержит первый фиксированный полный PPO continuation: 2496 TRAIN rows, 10 actor /40 critic updates на CUDA, прежние reward/gamma/lambda/legacy critic timing, sequence retention=1 на текущих полных историях, actor scope all. Development/evaluation rows не используются. Архитектура и runtime action contract сохранены.

`scripts/audit_combat_elevation_ppo_cuda.py` подтвердил точные initial actor/critic/std/Adam/RNG относительно source, JSON/checkpoint parity на CUDA, неизменность reward/config и ранее не использованный rollout. Проверены Adam clocks и source/protocol hashes. Отдельный дополнительный loss за падения не добавлялся: это опыт обновления на новом покрытии, а не причинная изоляция одного признака геометрии.

Fresh source / elevation-PPO /rules smoke закрыт: 24 native /16 CUDA,8 случаев на вариант, четыре карты и ordinary/recovery, seed offset712,16 ×2. Веса заранее fixed, promotion=false. Итог: source5/8, update4/8, rules6/8; к source0 gained/1 lost. Полученный урон556→590, смерти3→4, убийства11→10. `decision.json`: first_elevation_ppo_not_advanced. Большой screen не назначен. На base3 ordinary оба learned проиграли; recovery оба выиграли. У source recovery был спуск62units, у update0, но этих двух случаев недостаточно, чтобы считать глубокие спуски исправленными. Успех на этих семействах не означал бы переноса на новые площадки, поскольку часть соответствующей геометрии теперь присутствует в TRAIN.

`outcome-credit-audit.json` отдельно проверил TRAIN:8 побед имеют среднюю суммарную награду17.456,5 смертей−0.689,3 случая без goal/death marker5.578. Перевёрнутого ранжирования reward не обнаружено. Из4 глубоких спусков3 имеют положительную **сырую** GAE, но после фактической batch-нормировки PPO положителен только1. Это не доказательство, что обучение вознаграждает падения. Награда по этому аудиту не изменена.

Следующий отдельный [AWR-style regression опыт](combat_advantage_regression_20261022.md) сравнивает взвешенное повторение наблюдавшихся действий с обычным копированием и PPO на том же новом TRAIN. Это проверка способа обновления actor, а не объявленное исправление reward или PMove.

## Место

Для небольшого 24-case сбора явно задан `--minimum-free-gib 24`; default collector остаётся 32 GiB, допустимый минимум 16 GiB, запас проверяется до подготовки и перед native запуском. Значение и исходное свободное место закреплены в protocol. Evaluator также поддерживает явный `minimum_free_gib` в sealed spec с тем же минимумом16/default32; в новом24-case smoke задан24. Контроль диска не отключён, предыдущие закрытые specs не менялись.
