# Совместный выбор поправки прицела: runtime и начальная приёмка

23.10.2026. Реализован `combat_joint_aim_candidates_v1`: новая обучаемая categorical-голова 1359→64→25, 88 665 параметров. Это расширенное событие PPO: исходный Gaussian latent4 плюс категория поправки. Gaussian оценивает исходную выборку; conditional fire видит итоговый прицел. Награда прежняя, predictor остаётся frozen hint, не выбирает действие максимумом и не запрещает огонь.

Позднее первый head-only fit и native screen завершены: см. [результаты обучения](combat_joint_aim_training_20261023.md). Данный документ сохраняет этап начальной приёмки; превосходство обученной головы над source не установлено.

## Реализация

- `internal/policy/aim_candidates.go`: строгий формат, отдельный frame-hash RNG, pure transforms/packing, live sampling и Review. `ActionForSample` проверяет metadata и transform без NN. Старые модели не получают новых JSON-полей при canonical hashing.
- `cmd/q2ppo-data`: новый distribution разрешён только с `--cuda-only`, заголовок сохраняется в данных. Merge запрещает смешивать distributions.
- `scripts/combat_joint_aim_policy.py`: dedicated CUDA scorer, frozen motor/hit hints, original-latent Gaussian и final-aim fire. Стандартные precision scorers по-прежнему отвергают новый формат.
- CUDA finalizer проверяет полный joint LP, categorical LP, явный source latent, преобразование углов и frame-hash выбранную категорию; используемые helpers сохраняются в verifier-sources.

Инициализация `combat-joint-aim-initial-v1-20261023` создана на CUDA из `combat-conditional-fire-zero-v1-20261022`, predictor `combat-action-effect-shot-predictor-v1-20261023`, prototype v2. Исходные actor/value/std/Adam/RNG/history сохранены; отдельный optimizer головы пуст. Боевого обучения ещё нет. Initial head задаёт прежнее распределение 50% baseline и равномерные остальные 24 поправки.

## Проверки

Focused Go tests выполняли только metadata, hash sampler и чистую геометрию; Go NN выполнялись только в игре. Dedicated CUDA test v2 проверил ненулевую зависимость fire от final aim, ошибочность transformed-Gaussian likelihood, четыре повреждённых события, ненулевой градиент новой головы и отсутствие градиентов predictor. Тест не обучал боевую модель.

Native compatibility: `combat-joint-aim-compatibility-v1-20261023`, proof `combat-joint-aim-compatibility-proof-v1-20261023`.

- 8/8 native captures валидны, четыре условия ordinary/recovery на base3/ware2 с разной геометрией. Пул до 16 инстансов, ×2; offset 28, seeds проверены против 1254 исторических планов. Между вариантами условия и seeds одинаковы.
- 627 provider steps на вариант: source latent4, target/mode/pose/weapon/fire, конечные latent и proposed/applied commands совпали **точно**. 149 eligible candidate events, 24 категории встретились в этих четырёх боях; прототип ранее покрывал все 25.
- 4/4 CUDA case replay: 626 экспортированных transitions. Один дополнительный provider step относится к sequence context и не вошёл в набор transitions.
- Максимальная ошибка полной log probability 0,00006390, categorical LP — 0. Memory/value проверки также прошли. Это обычная допустимая численная погрешность, не прирост качества.
- Обе версии: 1/4 выполненных целей, 3 убийства, 621 нанесённого и 263 полученного урона, 2 смерти. Frames total различаются на три из-за служебной длительности захвата; provider commands совпали.

Проверка доказывает начальную совместимость на этих условиях. Голова ещё не обучена; её начальный последний слой нулевой, поэтому приёмка не доказывает качество learned nonuniform control. Следующий gate после head-only обучения обязательно включает живой CUDA replay уже ненулевой головы. Default/promotion не менялись.

## Следующий опыт

TRAIN уже собран и закрыт: `combat-joint-aim-corpus-v1-20261023`, **32 native / 32 CUDA**, 4061 transitions и 4076 context rows. Все восемь условий × четыре свежих seed, train offset 92; существующий collector подтвердил отсутствие пересечений с историей и соответствие reward SHA checkpoint. Merge сохранил явный action_distribution и cuda_verified_v1, SHA rollout/sequence проверены. Максимальная ошибка full joint LP по 32 case replay — 0,00016571. Ни validation compatibility, ни predictive-only корпуса в этот TRAIN не включались. Нейронное обучение головы пока не запускалось: требуется отдельный head-only trainer, стандартный PPO scorer новый формат по-прежнему отвергает.

Далее — отдельный head-only PPO на закрытом TRAIN (до 30 actor steps, LR 3e−4, 40 critic steps), с сохранением всех прежних actor tensors и optimizer moments. Frozen actor/hint inputs можно кэшировать один раз на CUDA; при этом likelihood остаётся полным joint, не transformed-Gaussian proxy. Старые predictive-only и validation-бои в TRAIN не подставлять. Затем заранее зафиксированный held-out screen с source, initial, trained и rules.
