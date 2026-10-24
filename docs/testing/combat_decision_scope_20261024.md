# Обучение решений при сохранённом прицеле — 24.10.2026

Предыдущий paired blocked-ground опыт изменил движение, но не повысил победность: cost/zero по4/8, rules5/8. Новые игровые прогоны в этой работе не запускались. Здесь завершены TRAIN диагностика входящего урона, CUDA разбор изменений политики и одноразовый эксперимент с областью обучения.

## Движение и входящий урон

`scripts/audit_combat_train_motion_exposure.py` обработал все64TRAIN записи двух веток/двух раундов:8148переходов, по4074на ветку. Это32парных условия, а не64независимых мира. Сопоставлены CUDA rollout indices, соседние native окна, наблюдения до команды, фактическое перемещение после неё, реальные MG shots и native effects.

Источник: `workspace/artifacts/combat-train-motion-exposure-v1-20261024/report.json`. Покадровые строки сохранены сжато в `frames.jsonl.gz`; полные старые observations/features не дублируются.

| Входящий урон | Zero | Cost |
|---|---:|---:|
| Всего HP | 1681 | 1715 |
| От паразита | 1064 (63.3%) | 1116 (65.1%) |
| От infantry | 300 | 300 |
| От gunner | 317 | 299 |
| От врага, наблюдаемого до команды | 1676 | 1710 |
| При ближайшем видимом враге ≤144units | 1185 | 1219 |
| В begin-frame окна | 66 | 66 |
| В end-frame окна | 1615 | 1649 |

Почти весь входящий урон можно связать с наблюдаемым противником. Это не доказывает достаточность всех входов, качество распознавания атаки или возможность уклониться. Положение скрытого противника в actor inputs не добавляется. End-frame damage может происходить от атаки, начатой раньше текущего решения.

Движение размечено как approach/retreat/lateral/still по фактическому следующему перемещению относительно предшествующего наблюдаемого направления к ближайшему врагу. Разные группы возникают в разных опасных состояниях. Их нельзя трактовать как причинное сравнение команд: например, больший урон при retreat не доказывает вредность отступления. Новый универсальный штраф/бонус за направление из этой таблицы не выведен.

## Куда изменялась политика при полном PPO

`scripts/audit_combat_policy_factor_changes_cuda.py` выполнил толькоCUDA forward/scoring для всех четырёх уже закрытых обновлений на их собственных TRAIN историях. Нового fitting не было. Проверены исходные likelihood против native samples, сумма компонент likelihood против полного scorer и совпадение полного K3 с отчётом тренера.

Источник: `workspace/artifacts/combat-policy-factor-changes-v1-20261024/report.json`.

| Обновление | Conditional movement KL | Conditional aim KL | Полный sampled K3 |
|---|---:|---:|---:|
| Zero1 | 0.0000691 | 0.0023744 | 0.0024953 |
| Zero2 | 0.0000805 | 0.0023483 | 0.0024989 |
| Cost1 | 0.0000689 | 0.0023721 | 0.0024928 |
| Cost2 | 0.0000956 | 0.0023386 | 0.0024977 |

У target/pose/weapon/fire/geometry изменения значительно меньше. Прицел меняется примерно в25–34раза сильнее движения по этим conditional KL. Это поддерживает гипотезу конкуренции за общий trust region, но само по себе не доказывает причину остановки оптимизатора или поражения. Component KL усреднены при записанных upstream actions; geometry — только по active rows. Их сумма не объявляется точным разложением sampled joint K3.

Отдельный motor-only уже исследовался22.10 (см. `combat_motor_architecture_ablation_20261022.md`) и не установил преимущества над правилами. Текущая гипотеза шире: обучать выходы решений вместе с существующими residual branches, сохраняя выученные условные углы.

## Одноразовый TRAIN эксперимент

`scripts/probe_combat_decision_scope_cuda.py` заново использовал оригинальный source checkpoint cost-round01 и его собственный ещё не consumed на этом source корпус round02 (1926строк). Это новая disposable ветка от исходного checkpoint, а не повторный PPO resume уже обученного endpoint. Validation данные в fit не входят.

Обучались:

- строки4:29 линейного head и attention residual: fire, vertical, weapon, target;
- существующие motor, weapon-state и conditional-fire residual branches.

Заморожены encoder/attention representation, базовые movement means0:2, условные coarse/fine углы, aim-mode, spatial/sample-motion aim, std и geometry scorer. Motor residual сохраняет прежнюю опору прицеливания. При смене выбранной цели фактическое направление взгляда может измениться — утверждение о неизменности относится к условному прицелу при фиксированных target/mode/action context.

Для обучаемых строк созданы отдельные Parameters и отдельные Adam clocks/moments, инициализированные срезами исходного Adam. Так frozen строки и их исходные состояния не обновляются через общий scalar step. Это диагностический adapter; он не сериализован как новая checkpoint схема.

LR прежние:3e-5 для выходных строк,3e-4 для трёх residual branches; cap30шагов; clip0.2,entropy0.001,gradclip0.5. Guard сохраняет прежний base KL≤0.0025; geometry frozen. Critic в этом probe не обучался.

Источник: `workspace/artifacts/combat-decision-scope-probe-v1-20261024/report.json`.

| Метрика на том же TRAIN корпусе | Full Cost2 | Decision scope probe |
|---|---:|---:|
| Принято actor шагов | 6 | 12 |
| Movement conditional KL | 0.0000956 | 0.0024185 |
| Aim conditional KL | 0.0023386 | 0 |
| Target conditional KL | ~0 | 0.0000145 |
| Weapon conditional KL | ~0 | 0.0000379 |
| Fire conditional KL | ~0 | 0.0000139 |
| Pose conditional KL | ~0 | 0.0000015 |
| Полный sampled K3 | 0.0024977 | 0.0024998 |

Movement change больше примерно в25.3раза при том же KL guard. Замороженные исходные параметры и source checkpoint сохранены точно. Это механический TRAIN результат. Полная surrogate loss у обычного PPO улучшилась сильнее; probe не объявляется более выгодным по loss или бою. Веса, checkpoint и resume-кандидат probe не записывались.

## Следующий проверяемый шаг

1. Оформить отдельный versioned тренер decision scope: материализовать обученные строки в обычные JSON веса для существующего Go runtime; сохранить dedicated row optimizer и состояния обучаемых residual branches без потери Adam clocks. Явно запрещать обычный full-resume нового scope без миграции.
2. Проверить CUDA decode/export, frozen координаты, source/RNG integrity и roundtrip/resume новой optimizer схемы. Для первого matched варианта допустимо повторно взять исходный cost-round01 и его собственный round02 TRAIN корпус; итоговый Cost2 checkpoint нельзя повторно обучать на consumed round02. Critic и его Adam должны совпадать с полным контрольным обновлением на тех же targets, либо обучаться одинаково.
3. До новых исходов зафиксировать короткий fresh screen с неизменным full-control endpoint, новой decision-scope policy, их общим parent и rules. Проверять победы/смерти/урон, движение и точность; TRAIN механический выигрыш не заменяет live quality. Пул≤16,×2; нейросетевое обучение и проверки толькоGPU. Широкое подтверждение — только при прохождении заранее записанного gate.

Эти три пункта пока не выполнены. Новый боевой эксперимент не запущен. Основные веса, runtime и исходная награда не заменены; общего преимущества над правилами всё ещё нет.

## Сохраняемый тренер и живое продолжение реализованы

`combat_decision_scope.py` содержит row adapter и canonical optimizer. `train_combat_decision_scope_cuda.py` поддерживает первый fork из общего source и последующие собственные TRAIN корпуса нового scope. Формат JSON inference weights прежний: обученные строки материализуются в исходные head/residual matrices. Изменение Go runtime не требуется.

Checkpoint сохраняет standard actor/value веса, отдельный `decision_scope_optimizer` со своими clocks/moments, прежний `actor_optimizer` как frozen anchor, историю, consumed rollout SHA и RNG. Поле `checkpoint_training_scope=combat_decision_scope_ppo_v1` требует явного нового тренера. В `ppo_combat.restore_checkpoint` и `ppo_recurrent` добавлен metadata admission guard. Реальный вызов обычного restore отверг новый checkpoint до восстановления сети; свидетельство `ordinary-resume-rejection.json`. Этот тест выполняет только CPU deserialization, без CPU neural forward.

Для первого обновления взяты source cost-round01 (`bf7d90b2...`) и его собственный round02 TRAIN,1926строк. Сравнимый full control — неизменный cost-round02 (`5048b6b7...`) на тех же данных. Это альтернативная ветка от исходного checkpoint, не повторный resume уже consumed endpoint.

Первое сохранённое обновление закрытоCUDA:12actor/40critic шагов, movement conditional KL0.00241852, aim conditional KL0, полный sampled K3≈0.00249977. Механический gate3x пройден, фактическое отношение25.31x. Frozen координаты, legacy optimizer, source checkpoint и RNG сохранены. Critic и его Adam точно совпали с full control. JSON decode/replay и checkpoint roundtrip пройдены. Проверка следующего Adam шага из памяти/с диска совпала; она disposable, без KL admission, и сама по себе не объявлялась свежим on-policy продолжением.

Первый endpoint: `workspace/artifacts/combat-decision-scope-workflow-v1-20261024/update`, weights SHA `0d7d847d25643e3f1f875c7a52f017a676414cee27b8aceca4845f115a2e90d2`.

Затем выполнено настоящее продолжение от него:16новых native/16CUDA TRAIN боёв,1670переходов,новый scope update2 с12actor/40critic шагами. Сохранены canonical Adam states, source/история и frozen coordinates; aim KL0, movement KL≈0.00166426. Выход: `continuation/update`, weights97e5cf9d... Это доказательство возможности дообучения новой схемы через реестр. В качестве quality endpoint второй checkpoint не выбирается.

Протокол `combat_decision_scope_protocol_20261024.json` и screen spec зафиксированы до новых оценочных исходов. Первый endpoint сравнивается на8новых validation условиях (offset52) с общим parent, matched full control и rules:32native/24CUDA. Всего новых игр в этой проверке48(16continuation+32evaluation); первые16TRAIN случаев переиспользованы от их исходного behavior policy для matched fork. Episode subset и глобальный баланс прежние; это не вся кампания.

Основной gate: decisions≥full+2победы и≥parent/rules, смертей не большеfull/rules, полученоHP≤110%full/rules. Частота движения сама по себе не является gate успеха. Promotion автоматически отсутствует.

Runner `run_combat_decision_scope_validation.py` фиксирует source fingerprint и phases/PID, переиспользует только закрытые шаги. Live duplicate --resume отвергнут без изменения исходного процесса (`resume-live-rejection.json`); восстановление произвольного partial output этим не доказывается. Пока текущий runner жив, существующие Go/Python/PS/native исходники не менять. На момент этой записи continuation полностью закрыт, выполняется quality evaluation. Итогов качества ещё нет.

## Итог live screen

Полностью закрыты16новых native/16CUDA TRAIN продолжения и32native/24CUDA оценки. Первая сохраняемая decision policy и её настоящее продолжение проверены в существующем Go runtime; для screen использован только заранее выбранный первый checkpoint. Итоговый sealed report: `workspace/artifacts/combat-decision-scope-screen-result-v1-20261024/report.json`, таблица `results.md`.

| Вариант | Победы / 8 | Смерти | Убийства | Получено HP |
|---|---:|---:|---:|---:|
| Общий parent cost-round01 | 4 | 4 | 10 | 464 |
| Matched full PPO | 4 | 4 | 10 | 469 |
| Decision scope | 5 | 3 | 12 | 450 |
| Rules | 7 | 0 | 17 | 284 |

Decision/full и decision/parent:1новая победа,0потерянных. Новый выигрыш — ordinary base3 с3противниками. Decision/rules:0новых,2потерянных. В группах из2противников все три нейрополитики выиграли4/4; в группах из3 — parent0/4,full0/4,decisions1/4,rules3/4. Страты малы и не доказывают общую устойчивость.

Gate=false: рост кfull только+1вместо+2, отставание отrules по победам, смертям и урону. Большое подтверждение24случаев для этих весов не запускается, promotion нет. Проверка управления наблюдаемым боем моделью пройдена, выигрыши не приписаны подмене на rules.

| Описательная метрика | Parent | Full | Decisions |
|---|---:|---:|---:|
| MG выстрелы с уроном живому монстру | 135/179=75.42% | 117/183=63.93% | 132/194=68.04% |
| Стояние при видимом враге на земле | 181/592=30.57% | 212/665=31.88% | 263/782=33.63% |
| Почти обнулённые большие запросы | 135/548=24.64% | 142/646=21.98% | 164/751=21.84% |
| Yaw≥30° / provider кадры | 20/635=3.15% | 21/739=2.84% | 32/859=3.73% |
| Attack без target / provider кадры | 1/635 | 28/739 | 25/859 |

Изменение политики не означает автоматического уменьшения стояния/поворотов. Часть показателей относительноfull лучше, часть хуже; относительноparent MG ниже. Это разные замкнутые траектории. Новый scope исправляет измеренное распределение изменений при оптимизации, но не объявляется исправлением всех лишних действий или достаточным условием победы.

## Следующая гипотеза

Проверить более длительное обучение решений на свежих боях с тройками противников при сопоставимом бюджете full и decisions. Причина выбора — TRAIN диагноз риска паразита и слабые результаты этой категории; оценочные отдельные проигрыши не превращать в TRAIN демонстрации. Все enemy classes/observations и reward пока прежние.

Перед новым опытом нужно:

1. Подготовить обычному full PPO явное продолжение с сохранением его собственного Adam/geometry state. Decision-scope нельзя передавать обычному full trainer. Выровнять количество update от общего parent: first shared TRAIN update уже есть у обеих веток; у decisions дополнительно есть проверочный own-policy update, full должен получить свой matched корпус на тех же физических условиях.
2. Зафиксировать новый curriculum/исследовательский протокол. При смене списка рецептов decision checkpoint требует явного metadata fork training contract SHA с CUDA проверкой неизменности весов/Adam/RNG/history; старый protocol не переписывать. Затем два свежих одинаково бюджетированных update на ветку, с независимыми собственными траекториями и парными TRAIN fixtures.
3. Заранее выбрать конечные checkpoint и новый маленький balanced screen ordinary/recovery,2/3врага, с rules. Увеличение корпуса или сложности сети само по себе не считать успехом. Широкое подтверждение — только после gate новой гипотезы.

Эти шаги ещё не выполнены, нового опыта сейчас нет. Текущий runner завершён; source fence больше не относится к живому процессу. Основные игровые веса и reward не заменены. Общая цель превзойти rules и проверка full campaign/human coop остаются открытыми.
