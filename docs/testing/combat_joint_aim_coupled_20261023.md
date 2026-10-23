# Joint actor/candidate PPO: парный опыт и отрицательный результат

23.10.2026. Оба исправленных fit приняты технически; 48 native captures и 40 CUDA replay завершены без ошибок. Совместный вариант проиграл контролям. Promotion/expanded confirmation=false, default не меняется.

## Общие условия и два fit

План `../combat_joint_aim_coupled_plan_20261023.md`, тренер `scripts/train_combat_joint_aim_coupled_cuda.py`. Принятый update root: `workspace/artifacts/combat-joint-aim-coupled-training-v3-20261023`.

Начальные веса обоих fork — один `combat-joint-aim-initial-v1-20261023`. Тот же закрытый TRAIN, 4061 transitions / 1078 candidate events. Новые weights head-only не служили начальной policy; их critic использовался только для проверки совпадения независимого fit. Source actor LR3e−5, branch LR3e−4, candidate LR3e−4, прежние reward/GAE/clip/entropy, target KL0,005. Predictor frozen.

| Fit | Принятых actor steps, максимум30 | Critic steps | Full approx KL | Candidate conditional KL, eligible TRAIN |
|---|---:|---:|---:|---:|
| Только исходный actor; новая head frozen | 24 | 40 | 0,00499994 | 0 |
| Исходный actor + новая head | 21 | 40 | 0,00499911 | 0,00001163 |

Шаги остановил guarded Adam/KL safeguard. Critic tensors и Adam совпали точно между обоими fork и прежним head-only fit. Combined optimizer сохраняет старые parameter IDs; после fit его группы разделяются на исходный actor optimizer и отдельный candidate optimizer, roundtrip проверен CUDA. Frozen head и её optimizer у base-actor-only точны. Полные TRAIN likelihood после экспорта совпали с scorer; source checkpoint в памяти остался неизменным, RNG сохранён.

Два диагностических запуска не вошли в игровые сравнения:

- v1 остановлен RNG-проверкой: конструкторы временных проверочных сетей потребляли CUDA RNG. Проверка экспорта помещена в `torch.random.fork_rng`.
- v2 остановлен равенством critic: `load_state_dict` мог переиспользовать GPU tensors Adam из общего checkpoint, и первый fit менял начальные моменты второго. Все загрузки optimizer используют deep copy; дополнительно сверяется полный immutable source snapshot до и после каждого fit.

У v1/v2 нет успешного общего complete seal. Игры используют исключительно v3; обе исправленные проверки прошли. Наличие частичного base-only артефакта v2 не делает его принятым matched экспериментом.

## Native screen

Spec `combat-joint-aim-coupled-screen-spec-v2-20261023.json`, root `combat-joint-aim-coupled-screen-v2-20261023`. Восемь ordinary/recovery условий base3/ware2, validation offset30, новые парные seeds, история1268 plans. Один пул16 ×2. Предыдущий spec v1 не запускался: он ссылался на не прошедший paired fit v2.

| Вариант | Цели | Убийства | Нанесённый урон | Полученный урон | Смерти |
|---|---:|---:|---:|---:|---:|
| Source | 5/8 | 15 | 2577 | 531 | 2 |
| Initial joint | 2/8 | 7 | 1740 | 669 | 6 |
| Head-only | 2/8 | 10 | 1869 | 589 | 5 |
| Base actor only | 3/8 | 8 | 1990 | 612 | 4 |
| Joint | 0/8 | 2 | 1421 | 739 | 7 |
| Rules | 4/8 | 10 | 2084 | 563 | 4 |

Joint против base-actor:0 gained/3 lost; против source:0/5; против rules:0/4. Source и правила — те же веса/алгоритмы, что в первом screen, но другие seeds. Все пять learned variants прошли по8 CUDA cases; всего4912transitions. В том числе current joint785rows с обновлёнными исходными branches и ненулевой новой head принят полным scorer, frame-hash assignments и transform checks.

| Native MG metric | Source | Initial | Head-only | Base actor | Joint |
|---|---:|---:|---:|---:|---:|
| Урон живому монстру / shots | 181/280 | 141/276 | 148/276 | 155/276 | 137/271 |
| Доля | 64,64% | 51,09% | 53,62% | 56,16% | 50,55% |
| Solid contacts | 98 | 134 | 125 | 121 | 134 |
| Стоит при clear enemy, доля frames | 31,36% | 24,82% | 20,78% | 17,56% | 18,19% |
| Applied yaw≥30°, доля frames | 0,89% | 4,62% | 2,55% | 3,49% | 3,44% |
| Смен цели /100frames | 9,81 | 14,96 | 8,63 | 16,81 | 21,76 |

Меньше стояния не привело к лучшему бою. Shot/behavior metrics относятся к разным траекториям, не к одинаковым контрфактическим shots.

Описательное объединение двух screen для одинаковых моделей: source10/16, initial6/16, head-only7/16, rules8/16. Это posthoc summary, не заранее объявленная confirmation и не доказательство превосходства source над правилами.

## Диагностика изменения policy

`scripts/audit_combat_joint_aim_policy_shift_cuda.py`, root `combat-joint-aim-coupled-policy-shift-v1-20261023`. Без новых игр и fitting, все NN CUDA. Пересчитаны одинаковые сохранённые histories: TRAIN_initial4061, validation_source1233, validation_initial822, validation_joint785rows.

Сумма KL выбранных factors для joint: mean0,00493 на TRAIN;0,00445 на source histories;0,00562 на initial histories;0,00519 на joint histories. Большого скачка среднего в этой метрике не найдено. Это conditional diagnostic по recorded target/mode/movement/pose/action, а не full-policy KL, интегрированный по всем условным действиям. Не исключает другие distribution/control effects.

Почти всё изменение приходится на Gaussian: mean0,00481 из0,00493 на TRAIN. Candidate conditional KL всего0,00001163, meanP(baseline)0,5→0,49856. На validation initial category KL0,00001078. Исходный actor обновился заметнее новой головы при общем KL safeguard. У head-only base-factor KL равен0 на всех наборах — отрицательный контроль прошёл.

Среднее изменение mean aim angle joint на фиксированных histories:0,057–0,083°, максимумы0,98–1,38°. Эти данные не устанавливают причину0/8 и не измеряют замкнутую игровую реакцию на последовательность действий.

Дополнительный факт: initial head применяет случайную поправку на50% eligible frames. По grid до clipping это примерно3,61° RMS дополнительного yaw/pitch возмущения. Хорошая совместимость с exploratory sampler не означала сохранения исходного боевого поведения.

## Проверка reward export

`scripts/audit_combat_reward_export_coverage.py`, root `combat-joint-aim-reward-export-v1-20261023`. Все4061 доступные first-life provider labels вошли в CUDA rollout, scores совпали. Kill reward165 и death reward−95 сохранены целиком, потерянных доступных labels нет. Суммы: monster_damage74,1; spacing_potential43,36; aim_potential2,53; received_damage−46,98; остальные costs/bonus сохранены.

Это проверка label→export, не независимая сверка с engine events или доказательство правильного credit assignment. В экспортированных окончаниях встречаются observed_death, combat_goal_complete и control_handoff; соответствие handoff реальному окончанию боя и discounted potential/GAE требуется проверить отдельно перед дальнейшим fitting.

## Решение

Артефакт решения: `combat-joint-aim-coupled-screen-decision-v1-20261023`. Техническая корректность подтверждена; боевое преимущество не подтверждено. Эта версия отклонена. Следующий этап — [проверка credit и пересмотр инициализации/бюджета обновлений](../combat_joint_aim_reassessment_20261023.md), без использования validation как TRAIN и без изменения reward по отдельным проигранным seeds.
