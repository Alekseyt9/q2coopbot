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

Запущен первый сбор `round-1-full-collect`; обучение раундов и итоговая оценка пока не завершены. Результатов нового опыта нет, promotion=false. Основной игровой model/default не заменён.

Команда продолжения после штатного закрытия фаз:

```powershell
& F:/src/strat/.venv-gpu/Scripts/python.exe scripts/run_combat_motor_feedback.py --protocol workspace/artifacts/combat-motor-feedback-v1-20261022/protocol.json --resume
```

Источник статуса — `progress.json`, завершение всех128TRAIN подтверждает только `training-complete.json`, итог всего опыта — `complete.json`. Не запускать второй runner параллельно.
Preflight: новый CUDA resume-auditor успешно проверил оба существующих закрытых matched updates (all иbase-only), включая исходные Adam/RNG и runtime/checkpoint; receipts preflight-full.json/preflight-base.json. Это проверка аудитора на реальных состояниях, не результат новых раундов.
