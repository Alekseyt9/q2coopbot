# Общие головы цели и прицела, 2026-10-09

Реализованы выбор наблюдаемой цели, отдельный yaw/pitch для каждого слота, вход предыдущего намерения и offline-метки упреждения бластера. Общий контракт используется MLP, GRU и temporal attention. Новые ветки являются экспериментом, не принятой заменой FireBC/rules.

## Контракт и выполнение

- `combat_features_v7`: 854 входа; старый префикс 845 сохранён, добавлен one-hot предыдущей цели (none + 8 distance-sorted slots). ID/track нужны только для сопоставления видимой цели, числовой ID не подаётся сети. История сбрасывается при разрыве кадра/жизни/видимости/track.
- `combat_target_conditioned_aim_v1`: 45 выходов; прежние 20, 9 логитов цели и 8 пар yaw/pitch. Gaussian likelihood использует выбранную пару, PPO включает masked target categorical. None сохраняет глобальную пару. Недоступные bbox/clear-shot цели маскируются; команда отвергает устаревший ID/track.
- Runtime Go управляет боем непосредственно. Нет автоматического наведения из offline-меток. Старые V6/20-output веса остаются совместимыми.
- CUDA likelihood добавлен в оба PPO-тренера; target-aware retention KL пока отсутствует, поэтому такие настройки явно отвергаются. Новый BC checkpoint имеет отдельный формат: это новый optimizer, не продолжение Adam/PPO.

## Метки и ограниченное начальное обучение

518 train и 179 validation target-pairs с известной видимой скоростью бластера; 818/365 меток выбора ближайшей цели. Контекст 6987/6531 кадров из существующего native-verified own-policy корпуса. SHA256 корпуса проверены, train/validation seeds не пересекаются; существующие guards/query masks сохранены. Это unexecuted counterfactual annotations, без reward/hit-кредита за новый прицел.

Упреждение решает constant-velocity interception для скорости blaster 1000 units/s, горизонта до 2 s. Projectile не наследует скорость игрока. Начало луча приближено eye point, без точного muzzle offset/задержки выстрела/ускорения цели. Machinegun имеет свежий weapon kick; наблюдаемый camera kick содержит damage/bob и не является точной отдачей следующей пули. Его новые aim labels исключены, вместо выдуманной recoil compensation.

Все восемь моделей получили одинаковые 50 эпох на CUDA/RTX5070, lr=.001. Заморожены encoder, memory/attention, старые 20 actor rows, critic и std. Обучаются только новые selector/target-aim rows. Выбор ближайшего монстра — bootstrap-эвристика, не доказательство оптимального порядка убийств. Вход прошлого намерения уже доступен runtime, но в этом этапе замороженный encoder с нулевыми новыми колонками ещё не учится использовать его; обучение удержанию цели требует следующего own-policy этапа.

| Модель | Архитектура | Validation RMSE до, градусы | После |
| --- | --- | ---: | ---: |
| m0 | mlp | 14.65 | 11.02 |
| m1 | attention | 15.21 | 12.58 |
| m2 | attention | 11.59 | 11.53 |
| m3 | gru | 12.42 | 11.79 |
| m4 | mlp | 11.37 | 10.16 |
| m5 | attention | 14.85 | 12.09 |
| m6 | attention | 15.15 | 9.36 |
| m7 | gru | 11.21 | 8.98 |

RMSE усредняет yaw/pitch по наблюдаемым геометрическим запросам; это не hit rate и не победы. Значения разных архитектур имеют различный inherited training budget.

## Проверка

- CUDA migration: все 8 моделей, старые actor/value выходы и дублированные пары сохраняются с допуском 3e-5; unavailable categories отвергаются; gradients отсутствуют у скрытых и невыбранных aim rows.
- CUDA native-data training checkpoint audit: веса совпали с checkpoints; shared rows, critic и std сохранены точно. Intercept checks: stationary, lateral, unknown velocity, machinegun mask.
- Go build основных runtime/registry CLI прошёл; Go numerical model tests не запускались по пользовательскому требованию. Общая сборка ./... была остановлена при длительном обходе generated workspace; отдельные runtime CLI собраны.
- Live smoke: 64/64 captures, pool16/x2, solo/mixed Blaster; 16840 provider frames, 16288 selected-target frames, invalid_provider=0. Первоначально скопированные старые планы не прошли проверку устаревшей metadata; пересозданы текущим registry compiler, старые игровые instances/seeds сохранены.

## Текущий эксперимент и следующие проверки

Запущена парная оценка 288 боёв: 8×before/after + FireBC/rules, четыре pilot family (solo/mixed Blaster/Machinegun), по 16 боёв/вариант, pool16/x2, validation offset24. Before — мигрированная старая голова, after — новые BC rows; веса и текущие registry bindings заморожены. Это повторный validation для разработки, не final test. Результатов качества пока нет.

После оценки выполняются строгая native/source/runtime/model/seed/frame-budget проверка и selected-target yaw/pitch/3D-ray диагностика по фактически applied action. У старых контролов без declared target метрика остаётся unavailable; ближайший враг не подставляется вместо намерения. Firing frames не являются числом пуль, angular proximity не объявляется точностью попаданий.

Далее: fresh stochastic own-policy сбор на новых весах; DAgger-like corrective data с обучением previous-target intent; точная instrumented hitscan telemetry и fresh-kick labels для автомата; coarse/fine или angular-rate aim ablation; firing-timing curriculum; target-aware retention и согласованное движение/прицел. Публичное качество проверяется только отдельной native paired оценкой, улучшение сети по RMSE не достаточно.

## Артефакты

- `workspace/artifacts/target-head-v1-20261009/{cuda-audit,smoke-audit,target-bc-report}.json`
- `workspace/artifacts/target-head-v1-20261009/m*/target-bc/{weights.json,checkpoint.pt,report.json}`
- `workspace/artifacts/target-eval-v1-20261009/{protocol,plans,cuda-audit,driver-process}.json`
- `scripts/combat_target_head.py`, `train_combat_target_bc.py`, `prepare_combat_target_heads.py`, `prepare_combat_target_evaluation.py`, `report_combat_selected_target_aim.py`
- `internal/policy/target_head.go`, `internal/aimquery/targeted.go` (offline query API; native BC bootstrap currently derives annotations on CUDA from the same feature geometry).
