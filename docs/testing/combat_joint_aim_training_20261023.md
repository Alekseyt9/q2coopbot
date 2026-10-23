# Обучение совместной categorical-поправки: первый head-only опыт

23.10.2026. Голова обучена на CUDA, принята живым runtime и CUDA replay. Преимущество над исходной боевой моделью не показано; default не меняется и расширенная confirmation этой версии не запускается.

## Обучение и техническая приёмка

`scripts/train_combat_joint_aim_cuda.py`, update `workspace/artifacts/combat-joint-aim-training-v1-20261023`.

Источник — `combat-joint-aim-initial-v1-20261023`; закрытый корпус `combat-joint-aim-corpus-v1-20261023`: 32 fresh TRAIN, 4061 transitions, 4076 context rows, 1078 eligible событий, все 25 категорий. Compatibility proof привязан к исходным весам. Проверены provenance/SHA, TRAIN split, reward SHA и отсутствие повторного потребления rollout.

30 принятых actor steps, LR 3e−4, 40 critic steps с прежними GAE/reward, clip=0,2 и target KL=0,005. Обновлялись только новая голова и critic. Исходные actor/std/Adam/RNG сохранены точно, predictor frozen. Новая scope/config/history отделены от исторических frozen actor metadata. Optimizer головы и critic возобновлялись из checkpoint; исходный actor optimizer не пересоздавался и не обновлялся.

Фактические frozen inputs/выходы actor и predictor кэшируются на GPU. Полный текущий scorer после обучения подтвердил кэшированный LP (max error менее 1e−6), отсутствие изменений frozen tensors/градиентов и корректную eligibility mask. JSON/checkpoint roundtrip точен. Полный совместный LP оценивает исходный Gaussian latent и fire по final aim; кэш не подменяет плотность transformed-Gaussian proxy.

Final approximate KL 0,002371; exact categorical KL по eligible rows 0,008509, по всем строкам 0,002259. Средняя baseline probability 0,5→0,47435; argmax остался baseline, но stochastic categorical probabilities изменились. Head entropy eligible 2,282→2,355. Value MSE 2,976→0,745. Оптимизационные циклы заняли около 0,88 с на GPU без учёта загрузки/хэширования/приёмки; это один update на небольшом корпусе, не объём игрового опыта и не признак улучшения боя.

## Заранее зафиксированный native screen

Spec `combat-joint-aim-screen-spec-v1-20261023.json`, root `combat-joint-aim-screen-v1-20261023`, decision `combat-joint-aim-screen-decision-v1-20261023`. Все восемь ordinary/recovery условий base3/ware2, offset 29, одинаковые новые seeds между вариантами, проверка истории 1260 plans. Пул 16, ×2. Все 32 native captures валидны, все 24 обязательных CUDA replay прошли.

| Вариант | Цель выполнена | Убийства | Нанесённый урон | Полученный урон | Смерти |
|---|---:|---:|---:|---:|---:|
| Исходная модель | 5/8 | 14 | 2363 | 467 | 3 |
| Initial joint, до обучения головы | 4/8 | 12 | 2020 | 515 | 4 |
| Head-only trained | 5/8 | 14 | 2292 | 499 | 3 |
| Rules | 4/8 | 12 | 2337 | 567 | 4 |

Trained против initial: 1 gained / 0 lost. Против source: 2 gained / 2 lost. Против rules: 2 gained / 1 lost. Выигрыша по числу целей против source нет, полученный урон больше. Восьми боёв недостаточно для утверждения превосходства.

Обученная ненулевая голова прошла CUDA replay на 848 transitions: full joint LP max error 0,00005931, categorical LP 0,000000477. Frame-hash assignments и source→final transforms приняты. Это положительная техническая приёмка trained distribution, а не доказательство качества управления.

## Выстрелы и лишние действия

Native MG telemetry, `machinegun-aim.json`, `machinegun-outcomes.json`; NN не запускались при подсчёте этих показателей.

| Метрика | Source | Initial joint | Head-only trained |
|---|---:|---:|---:|
| MG shots с уроном живому монстру / все MG shots | 149/229 | 113/165 | 111/173 |
| Доля таких shots | 65,07% | 68,48% | 64,16% |
| Solid contact shots | 76 | 48 | 57 |
| Corpse contact shots | 4 | 4 | 5 |
| Средняя native ошибка прицела после recoil, измеренные shots | 5,03° | 6,41° | 7,42° |
| Стоит при наблюдаемом clear enemy, доля provider frames | 38,51% | 35,12% | 43,63% |
| Applied yaw ≥30°, доля provider frames | 2,96% | 2,43% | 4,25% |

Shot metrics получены на разных траекториях: они не являются контрфактическим сравнением одинаковых выстрелов. В этом screen ни доля попаданий с уроном, ни stationary/large-yaw поведение не улучшились. Reward и правила исполнения не менялись ради этих validation результатов.

## Решение и следующий контроль

Promotion=false, expanded_confirmation=false. Эта версия отыграла одну потерянную при инициализации цель, но не превзошла source и не решила лишние движения/выстрелы. Предсказатель и корректная плотность сами по себе не обеспечили качественную стратегию.

Следующая гипотеза — совместное обучение исходного actor и новой головы. Проверить его против обновления только исходного actor на том же исходном joint behavior/том же TRAIN. Это новый matched control, а не расширенная quality confirmation неудачной head-only версии. Протокол: `../combat_joint_aim_coupled_plan_20261023.md`.
