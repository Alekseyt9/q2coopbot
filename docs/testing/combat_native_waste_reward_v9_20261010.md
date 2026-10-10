# Native штраф за бесполезный выстрел: reward v9

Продолжение `combat_native_waste_audit_20261010.md`.
Версия v9 сохраняет selected-aim v8, kill/damage/death/time, spacing и
action-quality components. Добавлены `machinegun_miss=-0.02` и
`blaster_miss=-0.02`. Награда описывает экспериментальный objective;
качество поведения пока не установлено.

Для автомата учитывается настоящий native fire, а не attack на каждом кадре.
Проверяются полный формат telemetry, конечность координат, aim с отдачей,
разброс, ammo, firing state, ordered native generation/frame/sequence/actor,
дубликаты и следующий синхронный damage. Контакт с геометрией, небом,
отсутствие контакта или труп считаются бесполезными. Живой damageable контакт
без потери HP остается неоднозначным и не получает штрафа. Native takedamage
имеет enum 0/1/2; значение 2 не является поврежденной telemetry.

Бластер использует существующий delayed miss joiner: только geometry/sky,
только подтвержденный alive first-life launch и end, exclusive окно без
recovery. Freed/неразрешенные исходы не штрафуются. Стоимость относится к
шагу завершения. Ни выбранная цель, ни ее отсутствие не заменяют native исход.
Эти данные остаются в `server_outcomes.jsonl`, не входят в policy observation
и не меняют команды игры. Shot ID у автомата и бластера проверяется отдельно.

## Проверки

Go тесты parser/reward проверили malformed/nonfinite, aim/spread, дубликаты,
противоречивый damage, разные native окна, later-life/dead/recovery/exclusive
маски, corpse и неоднозначный living contact, exact-once, сохранение v8
без новых затрат и сохранение death/goal terminal. Также прошли существующие
ProjectileMiss/SelectedAimReward и выбранные registry/export/replay проверки.
Численные нейросетевые проверки на CPU не запускались.

`workspace/artifacts/native-waste-calibration-v3-20261010` завершен:
400/400 закрытых эпизодов, 23933 available reward steps. Все прежние
компоненты и selected-target references совпали с v8; изменение суммы
объясняется только двумя новыми компонентами (с обычной погрешностью
сложения float, без сравнения нейросетей). Native после-frame100 счетчики
совпали с независимыми MG/Blaster аудитами для всех пяти вариантов.

| Вариант | MG промахи | Blaster промахи с launch/end >100 | Ранний launch, end >100 | Стоимость на шагах >100 |
|---|---:|---:|---:|---:|
| parent-a | 403 | 217 | 11 | -12.62 |
| parent-b | 284 | 187 | 12 | -9.66 |
| update7-a | 378 | 216 | 11 | -12.10 |
| update7-b | 284 | 165 | 12 | -9.22 |
| rules | 3 | 9 | 0 | -0.24 |

Frame100 — отсечка отчетов качества, не универсальная граница обучения.
Ранние actual provider запуски, закончившиеся позже, корректно получают
delayed стоимость и отражены отдельно. Подготовительные v1/v2 calibration
артефакты сохраняют диагностику: v1 выявил native enum, v2 уточнил границы
denominator. V3 повторил весь экспорт последней реализацией и запечатал SHA.
Возобновление проверено после Windows кратковременно заблокировал atomic
rename progress.json при чтении: закрытые exports не повторялись, восстановлено
агрегирование; запись статуса теперь повторяет тот же atomic rename.

## Новый A/B

Запуск: `workspace/artifacts/native-waste-ab-v1-20261010`.
Общий родитель: selected-aim v8 attention64 update5,
`selected-aim-ab-v1-20261010/quality/processing/quality/update`.
В обеих ветках сохраняются actor/std; выход critic и Adam сбрасываются
одинаковой CUDA objective fork процедурой. Контроль v8, эксперимент v9.
Обе ветки получают по 80 свежих on-policy боев: 20 семейств ×4,
train offsets340..343, пул16 ×2; затем по80 общих validation боев28..31.
Validation reused development, не финальный test. Training GPU-only RTX5070.
Во время capture Go/PS1 заморожены. Основные опубликованные/live веса
не заменяются. Итоги нового обучения и качества пока отсутствуют.

Контрольная capture-ветка завершилась:80/80 закрытых записей без ошибок,
pool state=complete. CUDA batch finalization завершен; запущен
`ppo_recurrent.py` с запечатанным control fork checkpoint на GPU.
Новый update на момент этой записи еще не закрыт.

Итоговый A/B reporter дополнен native MG waste audit и delayed Blaster miss
credit audit с общей отсечкой100. Помимо исходов/движения/первого выстрела,
JSON и Markdown покажут настоящий расход MG впустую и реальные попадания
выстрелов без выбранной цели/видимого bbox. Это read-only отчеты; изменение
Python reporter не затрагивает замороженные Go/PS1 capture sources.

Обновление статуса: обе train-ветки закрыли по80/80 боев без ошибок.
Контрольный CUDA update6 завершен:3817 eligible transitions,10 actor steps.
Weights/checkpoint/report прочитаны заново и совпали с complete.json SHA256:
weights `bf3ad1a9e25bc2217af2390e2f6a03d591fb0ca252cffc7869bca1da3529d47b`,
checkpoint `3a3e18fc2f1143b994552e8cc58746887d853734d7ffc81e55a00246b1913376`.
CUDA checkpoint read-back audit запланирован драйвером после обучения обеих
веток; на этом этапе его результат еще не получен.
Quality-ветка находится в processing/CUDA подготовке; оценка160 еще не начата.

Итоговый reporter дополнен проверкой фактических sampling seeds, контролем
ownership команд, native movement и MG aim/recoil описательными метриками.
Visible-target rules fallback или pilot_equip_not_ready не допускаются для
закрытого A/B вывода. Все дополнения — анализ закрытых записей без NN inference.
Последний замер свободного места F:97.41GiB, не оценка размера самого проекта.

Обе CUDA ветки завершили update6:control3817 / quality3810 eligible transitions,
по10 actor steps. Actor/value/std и Adam после чтения checkpoint на CUDA
совпали точно у обеих веток; complete.json SHA проверены повторным чтением.
Quality weights `468fb45b9499b049c63ca08c5ff093a09a0b018678c4b29b243b56b01124b5ed`,
checkpoint `c8a1ad062653fbbfe509025823514f71e08883cc1e628123f79beb88c22ca7e9`.

`training-comparability.json` проверил80 пар условий: одинаковые первоначальные
fork weights SHA `11bbb789f6b2530e91171f0735629c4a7c6d4b14ca329f941c2e66ad8402469e`,
episode definitions кроме reward recipe, splits/seeds/modes, generated instances
(геометрия/позиции/монстры/loadout), native runner и hyperparameters кроме
objective hash. Разные фактические траектории и число eligible rows не объявляются
побайтно одинаковыми. Итоговый reporter связывает этот аудит SHA с исходными планами.
Началась общая160-бойная оценка; закрыто119 без ошибок на момент записи,
окончательных метрик качества еще нет.

## Завершенный A/B и проверка против сильного родителя

160/160 evaluation captures валидны, source_unchanged=true, frame gaps=0.
Сводка `native-waste-ab-v1-20261010/result.json` закрыта; native reports SHA,
sampling seeds, ownership и training-comparability проверены. Visible-target
rules fallback и pilot_equip_not_ready отсутствуют у обеих веток.

Восстановлен только отчет, без повторных игр: native movement reporter
ожидал финальную diagnostics фазу, хотя запускается после закрытого quality
report. Теперь он допускает подтвержденную quality_report_complete фазу,
дополнительно проверяя равенство полного числа quality/proof/protocol members.
`--resume-closed` A/B reporter требует законченный неизменившийся pool без ошибок.

| Метрика | Контроль v8 update6 | Native waste v9 update6 |
|---|---:|---:|
| Победы | 61/80 | 67/80 |
| Смерти | 11 | 11 |
| Полученный HP урон, средний | 19.99 | 19.75 |
| Нанесенный HP урон, средний | 72.98 | 79.33 |
| Выстрелы MG | 745 | 785 |
| Попадания MG в живого монстра | 327 (43.9%) | 371 (47.3%) |
| Подтвержденный бесполезный расход MG | 414 | 408 |
| MG без выбранной цели и видимого bbox | 274, 1 попадание | 220, 1 попадание |
| Выстрелы Blaster | 435 | 406 |
| Blaster попадания в живого монстра | 47.8% | 56.9% |
| Атака стоя, provider command metric | 19.64% | 18.18% |
| Native matched observations со скоростью <5, pooled | 41.05% | 33.77% |
| Поворот yaw, средний на provider кадр | 10.54° | 10.03° |
| Game frames на бой, средний | 77.94 | 63.83 |

Парно:8 приобретенных побед,2 потерянных. Медиана первого instrumented
MG/Blaster выстрела после видимой цели —0с у обеих; максимум4.1с у обеих.
Один эпизод на ветку без instrumented fire не получает искусственный ноль.
Standing и скорости включают ограничения геометрии/knockback и не доказывают
бесполезность движения. Результаты описывают этот development-набор.

V9 лучше симметрично дообученного контроля, однако число MG выстрелов выросло;
расход автомата еще не исправлен. Также сильный unchanged parent раньше
показывал70/80 в одной из development серий: преимущество над исходной
моделью не установлено этим A/B. Основные веса не заменяются.

Запущен `native-waste-parent-rules-eval-v1-20261010`:400 боев,
unchanged v8 update5 и v9 update6 ×два policy RNG arm плюс rules,
по80 общих условий28..31 на вариант, пул16 ×2. Полные generated instances
сверены между всеми пятью планами; reserved test не используется.
Общий objective v9 в labels не делает суммы наград сопоставимой метрикой
качества rules; сравниваются native исходы/урон/стрельба/движение.
Новый reusable driver принимает запечатанные parent/candidate update paths,
проверяет предыдущий закрытый A/B и source binding до dispatch. Выводов
нового400-бойного сравнения пока нет.
