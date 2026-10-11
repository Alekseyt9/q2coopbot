# Исследование оружейных действий: отдельная ветка

Исходный parent —закрытое equipped-balanced обновление; отклонённые recovery-веса не используются. На32 начальных recovery-парах CUDA-проба показала98,388% keep и1,141% суммарной вероятности доступных Blaster/SG/SSG/MG. Правила выиграли30/32, модель9/32. Задача нового эксперимента —получить больше собственного опыта выбора доступного оружия без принудительного выбора автомата.

`scripts/fork_combat_weapon_exploration_cuda.py` создаёт экспериментальную ветку: температура2 реализована делением строк8:20 обеих суммируемых выходных голов actor (основной и attention residual). Маски, набор действий, текущий инвентарь и исполнитель не меняются. Это намеренное изменение начальной политики, а не PPO-улучшение и не эквивалентное продолжение старых rollout. Ветка не продвигается в игру автоматически.

На5157 кадрах полных реальных историй проверено CUDA: все прочие выходы actor совпали точно, weapon logits совпали с исходными/2 точно, состояние воспроизведено с ошибкой<2e-5. Критик, encoder, attention, spatial/target branches и log_std сохранены. На32 начальных состояниях keep снизился до79,667%, сумма вероятностей поддерживаемого оружия выросла до14,824%, entropy0,10469→0,78627. Это увеличение исследования, а не доказательство улучшения побед.

Adam: обнулены только weapon-строки моментов четырёх изменённых параметров; остальные моменты и общие часы каждого параметра сохранены. Часы не сбрасываются по строкам, поскольку в Adam они общие для параметра. История корпусов, RNG и счётчики обновлений сохранены. Прежний sequence-retention objective архивирован: новая ветка требует свежего собственного train-корпуса. Восстановление опубликованного локального PT проверено через disposable CUDA Adam step; сохранённый PT не обучался при проверке. Receipt/seal: `workspace/artifacts/combat-weapon-exploration-parent-t2-v1-20261011/`.

Следующий учебный пул:224 боя,28 семейств × четыре новых train-сида × две политики. Каждая получает собственные112 боёв:20 обычных семейств и8 recovery-пар. Control —неизменённый исходный parent; exploration —его temperature2 fork. Episode/instances/seeds/split совпадают, seed-offset160 не пересекается с1548 сидами ранее подготовленных планов. Пул16 ×2, порт39700. Обучение обеих линий —full actor на CUDA, прежняя конфигурация PPO, по10 actor steps и40 critic steps, новые собственные retention specs. Control отделяет эффект нового curriculum от эффекта исследования.

Координаторы: `workspace/artifacts/prepare-weapon-exploration-training-20261011.py`, `capture-and-train-weapon-exploration-20261011.ps1`, `train-weapon-exploration-20261011.py`. После закрытия нужны отдельная парная confirmation и проверка прежних обычных боёв. Validation-диагностика32 не входит в обучение. До закрытия нового пула и GPU-обновлений результата качества нет.

## Подготовлена автоматическая оценка после обучения

`wait-and-evaluate-weapon-exploration-20261011.ps1` привязан к живому training coordinator PID31140 и времени его создания; он ждёт фактического завершения процесса, затем проверяет complete двух обновлений и CUDA-аудиты. Отсутствие complete после завершения считается ошибкой, а не поводом перезапускать сбор. При повторном использовании PID цепочка прекращается.

После успешного закрытия:560 confirmation-боёв на offset192,28 семейств × четыре сида × control-before/control-after/exploration-before/exploration-after/rules. Сверяются SHA весов/PT/отчётов и совпадение episode/instances/seeds/split; новые сиды проверяются против ранее подготовленных планов. Пул16 ×2, порт39800. Control-after сравнивается со своим control-before; exploration-after дополнительно со своим exploration-before. Общее сравнение с исходным control не заменяет собственный before/after экспериментальной ветки.

Далее448 CUDA-воспроизведений оценочных боёв, аудиты ownership/weapon/behavior/execution и первого нативного урона. Группировка использует фактический loadout замороженного дескриптора. Оценочные данные не участвуют в обучении. Промежуточный native-quality receipt не заменяет финальный complete; автоматическое продвижение весов отсутствует.

Контроль хранения: перед захватом требуется минимум20GiB свободного места. Read-only NTFS-замер учитывает compressed allocation и уникальные file IDs hardlink-файлов; logical byte totals не приравниваются к реально занятому месту. Идентификаторы читаются через полный os.stat, потому что DirEntry.stat на Windows может возвращать inode0. Данные активных прогонов не удаляются.

На проверке во время сбора записано168/224 job receipts, ошибок0,84 на каждую линию; training coordinator жив. Это промежуточное состояние, не закрытая сводка и не результат качества. Подготовленный анализ: `workspace/artifacts/analyze-weapon-exploration-confirmation-20261011.py`; замороженный группировщик: `summarize-weapon-exploration-quality-20261011.py`.

Учебный пул закрыт:224/224,640,305с,ошибок0,source_unchanged=true. Началась GPU-обработка control. В оценку добавлены прямое trained control/exploration comparison и отдельные strata; после complete будет CUDA-проба начального weapon head на32 recovery-случаях каждой линии. Новых закрытых весов пока нет.

Control update закрыт на13298 собственных кадрах:10 actor steps,sequence KL0,00275751487, веса SHA `9aa59ded0e2a43f875c2d0cc770b0927f1a946dc454ed317a4581ebbc1b4d924`. Full-checkpoint CUDA audit и equipped/own-request ablation прошли. GPU-цепочка перешла к обработке exploration. Это частичное закрытие обучения; сравнение качества ждёт вторую линию. Старые закрытые native/CUDA sequence-дубликаты объединены hardlinks без изменения байтов/путей, освобождён уникальный allocation1,734GiB; см. `storage_cleanup_20261011.md`.

## Оба обновления закрыты, оценка запущена

GPU-цепочка завершилась с кодом0. Control13298/exploration13632 собственных кадров, всего26930; по10 full-actor steps. Exploration sequence KL0,00305422222, SHA весов `c67b233f703168937efe2bc7b055b0965a9c59c4a2e6a69b43b5b055061bed7c`. SHA весов/PT/отчётов обеих линий дополнительно сверены с complete. Проверены encoder/movement/aim updates, optimizer clocks и собственный корпус: control изменил27287/234/2544 scalars, exploration27592/232/2539 соответственно.

Первая попытка подготовки confirmation остановилась до захвата:12 старых дескрипторов в копии основного реестра разрешали только learned. Их исторические дескрипторы из закрытой recovery confirmation разрешают rules+learned и совпали со всеми полями, кроме modes. В отдельном confirmation registry восстановлен только этот список режимов; frozen training registry, сиды, рецепты и веса не менялись. Proof: `workspace/artifacts/combat-weapon-exploration-confirmation-registry-proof-v1-20261011.json`. Незавершённые v1-планы сохранены как след неудачной подготовки; нативных боёв по ним не было. Пять новых v2-планов прошли matched fixture/seed checks, evaluation pool560 запущен,16 активных слотов. Оценочная chain: `evaluate-closed-weapon-exploration-20261011.ps1`.

## Диагностика свежих учебных данных до этого PPO-обновления

Отдельный train-only инструмент не ослабляет held-out evaluator: тот правильно запрещает train split. Эти224 боёв характеризуют сбор опыта исходной и temperature2 политиками; вывод о новых обученных весах следует только из confirmation. Receipt: `workspace/artifacts/combat-weapon-exploration-training-capture-diagnostics-v2-20261011/complete.json`.

| Метрика | control | exploration temperature2 |
|---|---:|---:|
| Победы /112 | 34 | 11 |
| Победы обычных /80 | 30 | 8 |
| Победы recovery-пар /32 | 4 | 3 |
| Нанесённый урон, обычные | 22132 | 12333 |
| Нанесённый урон, recovery | 2071 | 3581 |
| Weapon requests, прежний диагностический порог | 160 | 2452 |
| Уже экипированное оружие запрашивается | 51 | 631 |
| Кадры вне поддерживаемого набора огня | 2445 | 679 |
| Максимальная серия keep с видимым врагом вне поддерживаемого набора | 103 | 23 |

Temperature2 увеличила исследование и сократила ожидание с неподходящим оружием, но поведение при сборе стало хуже по победам и обычному урону. Это совместимо с избыточными переключениями, однако causal attribution требует анализа нативных переходов/готовности оружия: request не равен завершённому переключению, attack intent не равен выстрелу. Применённых attack intents стало8380→9832, а суммарный нативный урон уменьшился24203→15914: считать intents метрикой точности нельзя. Оружейный audit использует прежний cutoff, execution —correct-clock cutoff; их счётчики не смешиваются.

В CUDA-ablation новых весов влияние equipped/own-request именно на weapon-head KL —около1e-12 у control и3e-12 у exploration. Ненулевой total-policy KL ранее в основном означал изменение движения/прицела. Эти величины не подтверждают практического использования текущего оружия в решении о смене. Если confirmation подтвердит проблему, следующий эксперимент должен учить обусловленный состоянием weapon-head, а не повышать глобальную случайность: прямой residual по текущему оружию/запросу/инвентарю с собственными свежими корпусами и проверкой регрессии остальных действий. Это ещё план, не реализованный native модуль.

## Наблюдаемые фазы оружия в учебных боях

Добавлен `scripts/audit_registered_weapon_phases.py`; три теста проходят с ResourceWarning как ошибкой. Проверены границы фаз, Gunmodel aliases, цензурирование пропусков/смены владельца/замены запроса. Интеграционная проверка охватила112+112 закрытых учебных боёв; provider totals совпадают с correct-clock execution для каждого случая. Границы анимаций четырёх поддерживаемых оружий сверены с `yquake2/src/game/player/weapon.c`, SHA источника записан.

| Наблюдение | control | exploration до PPO |
|---|---:|---:|
| Provider-кадры после фактического сетевого отпускания | 13342 | 13643 |
| Наблюдаемые смены экипированного оружия | 127 | 820 |
| Activation animation frames | 739 | 4361 |
| Lowering animation frames | 395 | 2529 |
| Доля activation+lowering / provider | 8,50% | 50,50% |
| Firing animation frames | 6703 | 4112 |
| Requests observed equipped, correct-clock | 94 | 696 |
| Requests superseded by another target, correct-clock | 5 | 540 |

Это сопоставление наблюдаемых gun_frame и текущего оружия с stock animation ranges, не чтение скрытого WeaponState и не число выстрелов. Фаза firing не гарантирует выстрел/попадание; activation/lowering не доказывают причинность потери конкретного урона. Grenades и остальные неподдерживаемые weapon phase ranges не угадываются. Наблюдаемые смены могут отличаться от завершённых requests при замене ещё ожидающего запроса. Старые94/696 отличаются от legacy93/682 за счёт первых двух восстановленных сетевых кадров.

Артефакты: `workspace/artifacts/combat-weapon-exploration-training-capture-diagnostics-v2-20261011/phases-control-v2.json` и `phases-exploration.json`. Первоначальный `phases-control.json` помечен invalid: до исправления raw Gunmodel paths классифицировались без нормализации. Обучение/сырые записи не затронуты. В confirmation coordinator добавлены четыре phase audits и сверка каждого с execution ledger; это расширение диагностики, не изменение политики, награды или нативного захвата.

Native confirmation560 закрыта:1519,218с,ошибок0,source_unchanged=true. Control29→34 побед/112,exploration10→6,rules58. Обычные80:control23→29/exploration8→6/rules27; recovery32:control6→5/exploration2→0/rules31. Exploration отвергается по нативному качеству, control-after не продвигается; CUDA448 продолжается. Подробная сводка: combat_weapon_exploration_native_results_20261011.md.
