# Обучение решений при замороженном motor — 24.10.2026

Предыдущий опыт состава обучения не улучшил исходные веса: source/trio/balanced/rules получили 6/4/5/4 побед из 8. CUDA-разбор собственных TRAIN историй показал, что у balanced движение менялось примерно в 167 раз сильнее выбора оружия. Это основание проверить другой состав обучаемых параметров, а не доказательство причины проигрышей.

## Предварительная проверка на TRAIN

Протокол `combat_categorical_decision_probe_protocol_20261024.json` зафиксирован до fitting. Источник — исходный decision4 initializer с весами `4e31f0c1...`; его собственный balanced корпус: 24 боя, 2889 переходов. Этот rollout не consumed в исходном checkpoint; контрольная ветка использовала его отдельно.

Disposable actor-only CUDA probe обновляет rows4:29 и weapon/fire residuals, замораживает motor, encoder/attention, Gaussian движение/прицеливание, std/geometry/aim-mode. Сохранены исходные Adam moments, LR, PPO clip/entropy/gradient norm, cap30actor, fullK3≤0.005/baseK3≤0.0025.

Probe закрыт: 24 принятых actor шага; target KL0.000136, pose0.000751, weapon0.000753, fire0.000612. Сумма этих условных KL примерно в124.78раз больше, чем у motor+decisions контроля на том же корпусе. Все Gaussian means сохранились точно; source checkpoint и исходные optimizer states не изменены. Это не аддитивное разложение sampledK3 и не вывод о победах. Веса/checkpoint из probe не экспортировались. Seal: `combat-categorical-decision-probe-v1-20261024`.

## Рабочая реализация

`combat_categorical_scope.py` сохраняет отдельный canonical Adam из четырёх групп: два набора decision rows, weapon residual и conditional-fire residual. Наследуемые пятигрупповой decision optimizer и обычный actor optimizer остаются замороженными anchors. У motor заморожены и параметры, и его наследуемое состояние. Старый decision trainer и обычное восстановление отвергают новый checkpoint по `checkpoint_training_scope`.

`train_combat_categorical_scope_cuda.py` принимает только объявленный исходный decision4 checkpoint или собственные categorical checkpoints того же контракта. Сохраняет active optimizer, RNG/history/consumed rollouts, увеличивает счётчики ровно на один. JSON для игрового Go-инференса сохраняет прежнюю архитектуру и формат; новых runtime правил выбора оружия нет.

Первый рабочий update закрыт на GPU: 24actor/40critic, categorical scope1/global41, веса `cb4bc36406b470cde5f7199c2f2426ee65e87c65440d31cd3962e0c3273c1682`. Все Gaussian means и motor сохранены точно. Critic и его Adam точно совпали с motor+decisions контролем, который обучался на тех же переходах. JSON decode/replay, checkpoint roundtrip, frozen anchors и disposable resumed Adam проверены на CUDA. Metadata admission: categorical принят, ordinary и decision scope отвергнуты. Disposable Adam продолжение не считается самостоятельным игровым продолжением PPO.

Замороженные Gaussian средства движения не означают одинаковую физическую траекторию: поза обучается, выбор цели меняет дальнейшие наблюдения. Условный прицел при фиксированных target/mode/motion сохраняется; выбор другой цели может менять фактическое направление взгляда.

## Заранее выбранная игровая проверка

1. От первого categorical checkpoint собрать 24 новых собственных TRAIN боя на тех же 24 рецептах и выполнить второй update, проверив настоящее продолжение с сохранёнными optimizer/history. Это проверка возобновления; второй endpoint не выбирается для качества.
2. Оценить только первый categorical endpoint, motor+decisions контроль и исходный checkpoint, плюс правила. Первые два имеют одинаковый исходный checkpoint, одинаковый исходный TRAIN rollout и ровно один update; фактическое число принятых actor шагов различается из-за замороженного motor.
3. Fresh validation: 8 balanced рецептов, offset55, одинаковые условия/сиды между четырьмя вариантами. 32native/24CUDA; новые TRAIN и evaluation сиды не пересекаются.
4. Gate: categorical минимум на2победы больше контроля, не меньше source/rules; смертей не больше control/rules, полученный урон≤110%каждого, наблюдаемый бой принадлежит модели. После малого положительного экрана нужно независимое подтверждение; automatic promotion=false.

Новые игровые затраты: 24TRAIN +32evaluation native, соответствующие CUDA проверки; первые24TRAIN уже существовали и не переигрываются. Один пул≤16,×2; все нейросетевые вычисления толькоGPU. Текущая исправленная проверка границы goal применяется одинаково ко всем вариантам.

Runner `run_combat_categorical_scope_validation.py` сохраняет PID/child/stage и source/config hashes, повторно использует только закрытые шаги при `--resume`, не перезапускает живые процессы и не перезаписывает partial output. Итог формирует `report_combat_categorical_scope_screen.py`. До закрытия цепочки существующие Go/Python/PS/native исходники не менять.

Root: `workspace/artifacts/combat-categorical-scope-workflow-v1-20261024`. На момент создания документа первый update закрыт; fresh continuation и quality ещё не запущены. Основные игровые веса не заменены.


### Запуск живого продолжения

Runner запущен, этап `collect-continuation`: 24 новых TRAIN боя от первого categorical checkpoint. Далее автоматически выполняются CUDA replay, второй update и fixed32case quality. Это первые новые бои после изменения области обучения; первый fitting использовал уже существовавший собственный TRAIN корпус. Качества пока нет, основные веса неизменны. Progress: `combat-categorical-scope-workflow-v1-20261024/progress.json`.


### Настоящее продолжение закрыто, quality запущена

24 новых native/24CUDA TRAIN боя, 2809 переходов. Второй update на GPU:18actor/40critic, categorical scope2/global42. Веса `7bde54ebb3bbc2f3202fe2abe6c32adbd7f623dada9640a7fc4a2dc93a822424`; frozen motor/все Gaussian means и optimizer anchors, checkpoint/resumed Adam подтверждены. Это живое собственное продолжение, в отличие от disposable Adam проверки сохранения.

Для качества остаётся первый `cb4bc364...`, второй endpoint не оценивается и не выбирается по исходам. Начата native-оценка32боёв наfreshvalidationoffset55; CUDA24 и native audits следуют автоматически. Исходники заморожены. Итогов боевого качества пока нет.


## Оценка завершена: gate не пройден

Новые24TRAIN/24CUDA и32evaluation native/24CUDA закрыты штатно; source fence сохранён. Первый categorical endpoint не менялся между продолжением и оценкой.

| Вариант | Победы / 8 | Смерти | Убийства | Получено HP |
|---|---:|---:|---:|---:|
| Исходный decision4 | 6 | 2 | 15 | 379 |
| Контроль motor+decisions | 7 | 1 | 17 | 383 |
| Categorical-only | 6 | 2 | 14 | 363 |
| Правила | 5 | 1 | 13 | 276 |

У categorical нет новых/потерянных побед относительно source, одна потерянная относительно control. Gate=false: прироста побед над control нет, смертей больше control/rules, полученный урон выше110%rules. Подтверждение этих весов на большем наборе не назначено; default не заменён. Более сильное изменение categorical распределений не установило причину прежних поражений и не обеспечило улучшения боёв.

Control→categorical: стояние36.21→30.19%, большиеповороты7.42→5.06%, почтиобнулённое движение20.67→12.84%сильныхзапросов. MGвыстрелы с урономживомумонстру71.79→68.53%. Это описательные доли разных траекторий; уменьшение лишних действий не означает автоматически лучший исход. Итоговый seal: `combat-categorical-scope-workflow-v1-20261024/result/complete.json`.

## Следующая проверка исполнения движения

Закрыт read-only TRAIN audit `combat-train-motion-rejection-profiles-v1-20261024`, скрипт `audit_combat_train_motion_rejections.py`. Два собственных корпуса (исходный balanced и свежий categorical continuation):48боёв,5698переходов. Из5127сильных запросов1312(25.59%) почти обнулены контроллером. Причины: unsupported_motion_guard484, static_hull_blocked496, component_clipped105, no_ground_support227.

216почтиобнулённых запросов имеют on_ground=true и abs(vertical_velocity)>1. Нынешняя простая flat-ground модель исключает такие состояния. Это ещё не доказательство ошибочного запрета: тело может соприкасаться с монстром, ступенью или иной опасностью. На остальных случаях отказ также может быть обоснован.

В коде уже есть более подробный `predictPmoveGround` с наблюдаемыми pm_flags/gravity/entities и проверкой путей, который способен рассматривать часть состояний с ненулевой вертикальной скоростью. Следующий шаг — offline applicability audit на зафиксированных Snapshot и proposed_command из этих TRAIN traces: проверить восстановление уже записанного applied prediction, затем оценить поддержку предложенной команды и все существующие checks пути. Сначала только диагностика, без изменения guard, fitting или новых игровых исходов. Движок/предиктор не получает скрытых сведений в policy observation. Утверждать безопасность новых действий без native-проверки нельзя.


### Дополнительное рассогласование позы

Read-only ledger `combat-train-pose-interventions-v1-20261024` проверил те же5698TRAIN переходов. Из3019предложенийcrouch применены2381;638замененынаrelease. Изних224static_hull_blocked+163unsupported_motion_guard+87no_ground_support происходятприon_ground=true;164unsupported — ввоздухе. Ещё11измененийпозыотменяютjump. Суммарно474отменённыхcrouchзапроса имеютgroundflag.

Причина в реализации: `stopMove` одновременно обнуляетForward/Side/Up, поэтому горизонтальныйзапрет отменяетотрицательныйUp. Условный aim/fire branch учитывает sampledvertical, тогда как финальная поза может отличаться. Это наблюдаемое расхождение команд, но не установленная причина промахов или доказательство безопасности сохраненияcrouch.

Уточнение следующей проверки: сначала offline applicability подробногоPmove назафиксированныхSnapshot/proposed/appliedcommand, затем только при независимомnativeподтверждении отдельныйopt-in опыт сохранения исходногоcrouch при остановленныхForward/Side и проверенномпути. Не добавлятьприседание,если модельего невыбрала; не разрешатьновыйпрыжок/падение/движениеккраю. Default guard сейчас неизменён. Само сохранениеcrouch может менятьboundingboxиостаточноедвижение, поэтому простаязамена`Up=0`наотрицательныйUp безпроверки не назначена.
