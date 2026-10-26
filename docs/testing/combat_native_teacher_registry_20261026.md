# Реестр парных native-продолжений

Дата: 2026-10-26. Цель — учить предпочтение действия по его последствиям в реальном сервере. Это отдельный учительский корпус, не ordinary on-policy PPO.

## Несовместимость старых записей

Прежний `diagnostic_anchor_probe` подменяет углы в `applyAnchorReceipt` **после** исходного выбора attack. Он пригоден для проверки эффекта угла при фиксированном исходном attack. Рабочая wide-ветка сначала выбирает итоговый latent, затем пересчитывает conditional fire. Поэтому прежние anchor-пары нельзя незаметно использовать как метки wide-решения. Набор `combat-wide-gate-native-flips-v1-20261026` также не является single-action teacher: там менялись веса последующей политики.

## Новый диагностический контракт

Модель с `diagnostic_wide_gate_probe` версии `combat_one_frame_wide_gate_clamp_v1` содержит frame и choice. В одном указанном кадре Bernoulli logit новой ветки задаётся как −20 / +20. Сэмплирование использует прежний frame-hash uniform; фактически выбранная ветка проверяется и при редком несовпадении опыт завершается ошибкой. Это почти детерминированное диагностическое распределение, не обычное распределение source policy.

Далее выполняется **штатный conditional-fire sampler на выбранном итоговом latent с тем же fire uniform**. Все веса и алгоритм вне одного кадра остаются исходными. Source actor RNG не продвигается дополнительными draw. Нет нового постоянного наведения, изменений movement guards или ограничений оружия.

Go помечает такую модель диагностической и не допускает как обычную stochastic PPO policy. CUDA scorer допускает заголовок только с явным `allow_probe`; replay проверяет clamp, frame-hash CDF, фактический wide transform и final-fire CDF. Exports имеют `combat_diagnostic_wide_gate_probe_rollout_v1`, diagnostic_only=true, ppo_training_ready=false, ordinary_training_eligible=false. Их likelihood относится к диагностическому распределению, а не к обычной политике.

## Проверка в игре

`workspace/artifacts/combat-coupled-wide-teacher-pilot-v1-20261026/complete.json`: **6/6 native и 6/6 CUDA**, два состояния, три ветви на каждое — baseline / choice0 / choice1. Состояния выбраны первым SHA-порядком из пяти прежних roots, без фильтра по исходу.

Проверены точные observed/command prefixes, одинаковое текущее наблюдение, одинаковые исходные актёрские события и условия боя. Ветка с выбором, совпавшим с baseline, воспроизвела весь observed/command первый жизненный цикл и terminal receipt. Это подтверждает однокадровую изоляцию в проверенных случаях, не полную побитовую эквивалентность всех скрытых полей сервера.

Карты fact1 / base3, source weights `804ffd9662ebb7f20a67c5a11326342683993af4cf6af36032b712d31b089172`. Все составы native для исходной карты, случайные оружие/патроны и позиции сохранены. Rules-owned combat rows отсутствуют в обеих парах. На fact1 attack=False в обеих ветках root; эффект угла проявился в продолжении. На base3 attack=True в обеих. Новый код пересчитывает вероятность огня, но эти два случая сами по себе ещё не демонстрируют смену бинарного attack между ветвями.

| Карта | Choice0: победа / убийства / урон | Choice1 | Разница discounted observed return |
|---|---|---|---:|
| fact1 | нет / 1 / 355 HP | нет / 1 / 363 HP | +0,05642 |
| base3 | да / 4 / 500 HP | да / 4 / 500 HP | +0,51398 |

Ценность считается из фактически записанной shaped reward после root, gamma=0,99, без critic bootstrap. Это конечный наблюдаемый горизонт, не точная бесконечная Q. Одинаковые победы и урон могут иметь разные return из-за времени и промежуточных reward-компонентов. Preference margin=0,05: оба примера предпочитают choice1. Это две одиночные детерминированные пары, не оценка статистической надёжности; по ним модель не обучалась.

## Формат и границы

Малый индекс в Git: `scripts/scenarios/combat-teachers/index.json`. Большие записи и доказательства находятся вне Git:

- `workspace/artifacts/combat-native-teacher-registry-pilot-v1-20261026/manifest.json`.
- На каждую пару — отдельный JSON версии `combat_native_coupled_gate_teacher_v1`.
- Входы до решения отделены от outcomes/targets: feature version, текущие features, pre-gate context187, source latent и уже выбранные target/mode/vertical/weapon.
- Условия воспроизведения: fixture, seed, frame, source model SHA, ссылки и SHA native/CUDA evidence.
- Выходы: нативные победа/смерть/убийства/урон и observed return обеих ветвей, preference, margin и причины непригодности.
- Пара с rules-owned continuation не получает метку для supervised gate, но сохраняется в манифесте. Ties и неудачные ветви также сохраняются.
- Все записи имеют ordinary_ppo_eligible=false; supervised trainer пока не реализован. Две записи — проверка инфраструктуры, не достаточный датасет и не улучшение рабочей модели.

Построение: `scripts/build_combat_native_teacher_registry.py`, GPU-формирование pre-decision context и повторная проверка всех SHA/proofs. Native-контроль: `scripts/run_combat_wide_gate_teacher_pilot.py`. Метаданные и чистая геометрия проверены целевыми Go tests без neural forward; численные проверки выполнены на CUDA. Hindsight — единственная активная память по решению пользователя.

## Расширение на свежих исходных боях

`workspace/artifacts/combat-native-teacher-expansion-v2-20261026/complete.json` закрыт: **32 новых исходных native боя на 32 картах / 32 CUDA replay**, затем **48 контрольных native продолжений / 48 CUDA replay**. Исходные 5 013 переходов собраны текущими весами `804ffd…` без fitting. Режим `--collect-only` заканчивается до вызова PPO trainer; каталога `source32/fit` нет.

Стартовые инвентари перебраны по новым TRAIN offsets 320..447; исходные сиды исключены по истории всех физических планов. Геометрия и map-native составы проверены обычным planner. Из 32 исходных карт на 26 нашлось eligible wide-событие с frame>=110. Первое такое событие на карту выбирается без чтения исхода/урона, затем фиксированный SHA-порядок ограничивает набор 16 картами. Все 32 строки coverage сохранены. На каждый root собраны baseline/choice0/choice1 с одинаковыми весами продолжения.

Все 16 текущих observed prefixes и root observations совпали точно, совпадающий с baseline clamp воспроизвёл первую жизнь целиком. CUDA проверила диагностический clamp, transform и final-fire CDF во всех случаях. В этих roots binary attack по-прежнему одинаков между choice0/choice1; сам пересчёт conditional fire численно проверен, но смена бинарного решения стрелять в этой выборке не продемонстрирована.

Новых записей: **16**. Из них choice1 предпочитается в **9**, choice0 — в **2**, ничья — **2**, непригодны для supervision из-за rules-owned continuation — **3** (jail2, lab, city3). Все три сохранены, не заменены удобными случаями. Mine4 содержит смерть→победу при choice1; mine1 — выживание→смерть, поэтому choice0 лучше. Это результаты отдельных контролируемых состояний, не общий эффект применения новой политики.

Предпочтение определяется указанной discounted shaped reward, а не только health/damage. Например, на city2 выбор с большим полученным уроном всё равно имеет больший return; такие расхождения явно сохраняются для анализа цели обучения. Нельзя пересказывать каждую метку как «объективно лучший бой».

Общий малый индекс теперь содержит **18 записей**: 11 меток choice1, 2 choice0, 2 ничьи и 3 непригодных. Supervised trainer ещё не запускался. Двух отрицательных примеров недостаточно для проверки контекстного выбора; следующему расширению нужны другие состояния и более широкое покрытие, с разбиением по целым эпизодам/картам.

Admission reader: `scripts/combat_native_teacher_registry.py`. Он проверяет SHA индекса/записей, уникальность root, отделение входов от future/reward, согласованность предпочтения с utility/margin и запрет ordinary PPO. Metadata regression tests проверяют rejection несовместимой actuation, разных continuation policies, будущей reward во входах, неверной метки и rules-owned teacher label. Реальный индекс со всеми 18 записями прочитан успешно.

Первая попытка расширения v1 остановилась **до игр**: вспомогательный `teacher-index.json` ошибочно находился в каталоге обычных episode descriptors, и строгий Load вернул `unregistered episode file teacher-index.json`. Индекс перенесён в отдельный `scripts/scenarios/combat-teachers/index.json`, обычный `q2episode --list` снова проходит. Проверка основной episode registry не ослаблялась; исходная ошибка сохранена.

## Покрытие новых карт и более поздних состояний

`combat-native-teacher-temporal-v1-20261026` использует ту же закрытую source32 collection и те же веса `804ffd…`. Новых исходных миров здесь нет. Выбор выполняет `combat_teacher_root_selection.py`: первое eligible событие на ещё не разобранном source world, либо первое событие минимум через 30 кадров после последнего зарегистрированного root с теми же seed и source SHA. В выбор не входят метки, reward или исходы. Из всех 32 source worlds доступны 14 новых roots: 10 ещё не разобранных исходных карт и 4 более поздних состояния (mine3, hangar1, cool1, waste3). Корреляция состояний одного боя сохраняется; будущий split должен делить целые миры/карты.

На 14 roots выполнены **42 native продолжения и 42 CUDA replay**, пул 16 ×2. Проверены все варианты, без замены неудачных случаев. Первичная итоговая проверка остановилась на `cool1`: `Matching clamp altered terminal`. Весь admitted observed/command first-life trace совпал, как и исход, damage и terminal snapshots; отличался только `first_life.aim_by_class[0].aim_frames`: 47 против 48.

Причина подтверждена из исходных `bot.jsonl` и `analyze_combat_first_life.ps1`: монитор считает подготовительный период с кадра 40, а первый server-dispatched policy step находится на кадре 98. В кадре 97 один монитор уже содержит aim_entity gunner, другой ещё нет. В admitted интервале с кадра 98 arbitration и счётчики совпали. Разница появилась до вмешательства в кадре 246. Сами policy observations/commands не сортировались, не переписывались и не исключались из проверки.

Первичный запуск остаётся **failed**, его файлы не переписаны. Отдельный `combat-native-teacher-temporal-recovery-v1-20261026` содержит явную post-hoc проверку всех 42 прежних продолжений: `original_protocol_passed=false`, `posthoc_scope_correction=true`. Новый verifier сначала точно восстанавливает полный `aim_by_class` из monitor trace, затем вычитает только доказанный вклад до первого admitted step. Любое необъяснённое расхождение полного счётчика, пропуск кадра, отличие admitted counters, policy trajectory либо остальных terminal fields отклоняется. Все 14 пар прошли эту проверку. Старый collector сохранён как `collector-source-after-run.py` с SHA, совпавшим с исходным frozen protocol; snapshot создан после опыта, а не объявлен исходной предварительной фиксацией.

Будущий collector заранее декларирует `combat_teacher_admitted_terminal_v1` и применяет этот же verifier; исходное расхождение monitor terminal сохраняется в результатах. Численные проверки остаются CUDA-only. Metadata tests проверили, что нельзя скрыть неверный полный counter, пропущенный кадр или изменение attack внутри admitted интервала. Показанный recovery является живой проверкой verifier на всех 42 ветвях; повторный native запуск после изменения самого collector ещё не выполнялся.

Учительский корпус: `combat-native-teacher-temporal-corpus-v1-20261026`, manifest SHA `fcf03e7d11da13ca1d5f515d07c6cb0b49467ec5969e0592e4a0a9e0de494d3b`. Добавлены 14 записей: **4 choice0, 5 choice1, 2 ничьи, 3 непригодных** (mine3, command, power2). Все непригодные случаи сохранены без меток. Например, на cool1 поздняя доводка уменьшила убийства 2→1 и damage 655→488 HP; на hangar2 она изменила смерть на выживание и kills 0→2. Это отдельные контролируемые состояния, не качество новой политики.

Общий индекс теперь строго читает **32 записи, 28 уникальных source worlds, 27 карт**:

| Тип записи | Количество |
|---|---:|
| Предпочтение исходного прицела, choice0 | 6 |
| Предпочтение дополнительной доводки, choice1 | 16 |
| Ничья | 4 |
| Непригодная hybrid continuation | 6 |

Таким образом, supervised метка есть только у 22 roots. Это не 32 независимых обучающих боя и не достаточное доказательство обобщения. Новый trainer не запускался, веса и production policy не менялись.

## Чувствительность учительской цели

`audit_combat_native_teacher_objectives.py` проверяет исходные SHA и повторно считает observed returns на CPU как обработку чисел из файлов, без neural forward. Никакие веса, записанные rewards или метки не меняются. Итог: `combat-native-teacher-objective-audit32-v1-20261026/report.json`.

Среди размеченных пар есть одно различие goal и пять различий death; исходная preference согласуется с победой/выживанием во всех этих случаях. В двух случаях (biggun, city2) preference выбирает ветвь, уступающую по конечному вектору goal/death/kills/dealt/received. Это описательный Pareto check: он не учитывает время отдельных убийств, патроны или последующую безопасность и сам не определяет правильную цель.

У **трёх** размеченных пар preference меняется при gamma=1 вместо .99: base3/182, biggun/117, city2/123. На city2 смена происходит уже при gamma=.995. На biggun обе ветви побеждают с четырьмя убийствами и damage 180 HP; choice0 получает 13 HP урона за 177 оставшихся переходов, choice1 — 17 HP за 229. Тем не менее .99-discounted return предпочитает choice1; undiscounted сумма предпочитает choice0. Это показывает чувствительность к распределению наград во времени, а не ошибку CUDA scorer. Не следует молча переименовывать эти метки в «оптимальные действия» или менять discount по результату удобного отдельного боя.

Следующее решение: расширять независимые исходные миры и отрицательные контексты, сохраняя текущую reward version и отдельные альтернативные objective-аудиты. Перед fitting закрепить целевой критерий и разбиение по целым мирам/картам; после fitting нужны свежие paired native сравнения с source и rules. Последний корпус не доказывает превосходства модели над правилами.

## Независимые исходные миры: batch02 и extra16

`combat-native-teacher-expansion-batch02-v2-20261026/source32` закрыт: **32 новых source native / 32 CUDA, 3 675 переходов**. До чтения результатов парных ветвей зафиксирован дополнительный бюджет `extra16-predeclared.json`: `combat-native-teacher-extra16-v1-20261026/source16` добавил **16 source native / 16 CUDA, 2 393 перехода**. Три source cohorts — прежние 32, новые 32 и дополнительные 16 — не имеют пересечений engine seeds. Это свежие TRAIN seeds, не новые глобально удержанные карты.

Диапазон inventory draws — offsets 384..511 с обязательным исключением всей физической истории планов. Попытка batch02-v1 с offset512 корректно отвергнута до игр: TRAIN содержит 512 offsets. Добавлена ранняя проверка границ; зарегистрированные splits не расширялись. Неудачная попытка сохранена.

В batch02 eligible earliest wide roots нашлись на 22 из 32 карт. На все 22 собраны **66 native / 66 CUDA**. У source16 нашлись 11 roots, все **33 native / 33 CUDA** завершены. Исходные веса везде `804ffd…`, fitting в сборщиках отключён. Всего в этом расширении **48 независимых исходных миров, 99 парных продолжений, 147 native запусков и 147 CUDA replay**. Во время overlap новые native игры начинались только после подтверждённого завершения предыдущего native pool; лимит 16 сохранён.

Первичная batch02-проверка остановилась на `mine3`, seed1868070387, root110: monitor movement дал combat_retreat47/none12 против48/11, тогда как весь admitted observed/command trace совпал. Причина вновь в подготовительном периоде до первого dispatched шага98. `combat_teacher_movement_receipt.py` восстанавливает полный movement summary из `bot.jsonl`, включая реальные перемещения, и вычитает только подготовительные counts. Остальные native terminal fields и admitted счётчики остаются точными. `combat-native-teacher-batch02-recovery-v1-20261026` закрыл все 66 прежних продолжений с `original_protocol_passed=false`, `posthoc_scope_correction=true`, без новых игр или замены случаев. Исходный failed root сохранён. Extra16 прошёл исходную predeclared проверку без recovery.

До исправления active ownership новые teacher corpora содержали:

| Корпус | Записей | Choice0 | Choice1 | Ничья | Ineligible |
|---|---:|---:|---:|---:|---:|
| batch02 | 22 | 5 | 6 | 4 | 7 |
| extra16 | 11 | 5 | 1 | 3 | 2 |

Обе новые серии численно проверили final-fire CDF, однако binary root attack между choice0/choice1 вновь не различался. Не следует заявлять, что на этих примерах продемонстрирован flip решения стрелять.

## Исправление допуска: нулевая остановка после победы

При проверке ineligible случаев найдена отдельная ошибка: прежний счётчик включал `owner=rules` на последнем kill frame, даже если это уже нулевая terminal-команда `combat_goal_complete`, а следующий native objective подтверждает alive=0/pending=0. Например, city2/147, lab/255 и279, ware1/144 и142. Эти строки не означают вмешательство правил в ещё продолжающийся бой.

Новый `combat_teacher_ownership.py` допускает исключение только при одновременных доказательствах: living same-life transition, точное соответствие current/native/next frames, совпадающие actor/map/spawncount, goal-stop v2 и его native objective, matched exclusive dispatch без recovery, пустые damage indexes, нулевые Buttons/Forward/Side/Up/Impulse. Движение, выстрел, переключение оружия, несовпавший native frame, неполная цель или обычный control_handoff остаются вмешательством. Metadata regression test проверяет эти отказы. Go `MatchesVerifiedGoalHandoff` и `MarkGoalHandoffAtNativeObjective` уже различают реальную terminal-границу и раннюю потерю контроля; игровой контракт не менялся.

`correct_combat_teacher_goal_ownership.py` повторно проверил native/CUDA bindings и pinned source traces **всех 65 записей**. Он не меняет inputs, utilities, margin, source policy или условия генерации. Исключены **11 пассивных goal-stop строк**, исправлен допуск **8 записей**. Новое производное представление сохраняет все 65 IDs и исходные parent records с SHA; ни один неудобный бой не удалён. Старый индекс и пять корпусов сохранены в `combat-native-teacher-active-ownership-v2-20261026/source-index.json` и прежних directories.

Активный индекс теперь указывает на производный manifest SHA `96157faf393f98edffc1eb39a0ded53576d03c4d74db656bb3944787cee7909a`: **65 roots, 61 уникальный seed/source-model world, 29 карт**. В нём **22 choice0, 25 choice1, 11 ничьих, 7 настоящих hybrid ineligible**. Метки есть у 47 состояний на 25 картах.

Это явное post-hoc исправление metadata admission, а не тайная замена цели ради метрики. Защита от маленького корпуса сработала и на прежних 32 записях (22 метки), и на 65 до исправления (39 меток): оба guard reports имеют `training_performed=false`. Заранее выбранные требования 40 меток / 10 каждого класса / 16 карт не ослаблялись. Будущий collector заранее фиксирует active ownership v2 и оба reconciled monitor summaries. Его обновлённая orchestration после этих изменений ещё не имела нового native запуска; live evidence здесь — recovery на66 ветвях и проверка исходных pinned traces всех65 записей. Старый collector скопирован после опыта с проверкой исходного frozen SHA.

## CUDA learnability probe: 17 параметров против одного

Настройки и SHA `probe_combat_teacher_gate_cuda.py` зафиксированы в `learnability-predeclared.json` до получения новых парных outcomes; SHA после опыта совпал. Тест предсказывает исходное paired preference, не absolute GAE return прежних critic probes. Заморожен источник `804ffd…` и его представление 187→32→16. Сравниваются 17 trainable delta parameters последнего слоя и один bias delta. Изменений игрового actor export нет.

Четыре outer folds делят целые карты, все roots и seeds карты остаются вместе. Mean/std вычисляются только на TRAIN fold, веса выравнивают исходные миры, а не число кадров. Зафиксированы 300 Adam steps, LR.03, weight-delta ridge.1, std floor.05. Primary gate: OOF weighted log-loss не более95% bias-only, balanced accuracy≥.55. Это внутренний диагностический map split TRAIN-корпуса, не независимая проверка качества всей policy.

`combat-teacher-gate-learnability-v1-20261026/report.json`: **47 меток, 22 отрицательных / 25 положительных, 25 карт**. Всё обучение и neural numerical verification выполнено на CUDA. Реальные save/load checkpoint resume150+150 точно совпали с uninterrupted300 по parameters и Adam state для обеих моделей первого fold. Checkpoints имеют `actor_compatible=false`.

| Вариант | OOF weighted log-loss | Balanced accuracy |
|---|---:|---:|
| Исходный gate | 0,919576 | 50,00% |
| Только общий bias | 0,694740 | 48,81% |
| 17 contextual parameters | 0,705485 | 50,00% |

Contextual/bias loss ratio **1,015465**, то есть contextual вариант на1,55% хуже, вместо требуемого улучшения≥5%. **advance_to_native=false**, policy_training_performed=false, promotion=false. Исправление общей калибровки уменьшает log-loss относительно source, но не доказывает способность выбирать действие по контексту или улучшение боёв. Гипотеза о достаточности текущего замороженного представления для такого малого teacher readout не подтверждена. Причина не установлена: малый объём, представление и вариативность долгого продолжения остаются отдельными гипотезами. Не следует переносить этот вариант в игру или утверждать, что большей сети гарантированно поможет большее число параметров.

Metadata suite: 8 teacher tests + 2 strict-admission tests прошли. CPU neural tests не запускались. Источник игровых весов остаётся `804ffd…`; прежняя оценка его качества не заменяется результатами классификатора.
