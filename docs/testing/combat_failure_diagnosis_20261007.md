# Диагностика после Update29 и корпус прицела

Дата:07.10.2026. Выполнен разбор сохранённых native/Go traces и20 новых rules battles. Это диагностика исполнения/обучения, не подтверждение ёмкости архитектуры. Просмотр новых демо в Yamagi в этой партии не выполнялся.

## Точно парное сравнение

Пять семейств ×4 validation seeds (offset4), pool16/x2. Сверены полные generated instances с исходным eval-after-plan Update29: player, health, monsters, generation seed, engine seed, BSP SHA и loadout совпадают. Все20 rules captures прошли native/provenance checks. Старые learned captures взяты из завершённого mixed Update29.

| Семейство | Update28 | Update29 | Rules |
|---|---:|---:|---:|
| Parasite/Gunner Blaster |0/4|0/4|0/4|
| base1/site03 Blaster |3/4|1/4|4/4|
| base1/site03 Machinegun |0/4|0/4|0/4|
| base2/site02 Blaster |0/4|0/4|4/4|
| base2/site02 Machinegun |0/4|0/4|4/4|

На четырёх кампанийных семьях Update29:1/16, rules:12/16. Это диагностические малые cohorts; весь обычный контроллер сильнее на всём распределении здесь не доказано.

В Update29 для base1/site03/Machinegun161 из169 команд огня при видимом враге направлены горизонтально больше чем на20° в сторону от ближайшего clear enemy (95.3%). Для base2/site02/Machinegun73/79 (92.4%). Это yaw исполняемой команды с delta_angles относительно текущей видимой позиции; lead, pitch, spread и muzzle offset не оцениваются. Поэтому это не accuracy попаданий, но полезный признак неверного наведения. Дополнительная проверка по native damage:40 и16 health damage соответственно за4 first-life battles. Для base2/site02/Blaster:9 атакующих кадров и0 outgoing health damage.

В rules/base1/site03/Machinegun нет команд огня в первой жизни:127 кадров с barrel_blast_risk и85 с machinegun_burst_pause,212 живых боевых кадров суммарно. Присутствует guard от опасного попадания в бочку; успешного манёвра для освобождения линии стрельбы нет. Нельзя учить модель этим действиям как успешному бою или объявлять эту сцену чистой проблемой ёмкости/прицела. Защита от бочки не отключалась.

В Parasite/Gunner Blaster rules тоже0/4; native outgoing damage620 против370 у Update29. Хороший yaw сам по себе не обеспечивает победу. Нужен отдельный разбор дистанции, движения, выбора угрозы и разделения группы. Low-displacement proxy не подтверждает стену и не заменяет native hull/демо проверку.

## Изменения реестра и харнеса

Разрешены rules captures с фиксированным Machinegun в existing synchronous harness; multiweapon teacher остаётся отдельным неподдержанным адаптером. В восьми campaign Machinegun recipes добавлен rules, revision1→2. Условия оборудования/геометрии/reward не менялись.

Поскольку generation seed раньше зависел от episode revision, изменение списка controllers меняло позиции и HP. Добавлено optional generator.seed_revision: при отсутствии сохранено прежнее поведение; для этих recipes явно1. Проверены positive range и <=episode revision. Regression test подтверждает совпадение instances после capability revision при зафиксированной seed_revision. При реальном изменении распределения recipe необходимо отдельно решить, сохранять ли namespace генерации.

Первые diagnostic attempts v1/v2 сохранены отдельно. Вv1 Machinegun ещё блокировал solo guard. Вv2 две Machinegun сцены имели изменённые стартовые условия и не входят в парный итог. Итог основан только наv3 после сверки всех generated instances; неверная предварительная цифра14/16 заменена проверенной12/16.

## Корпус для BC

Три полезных семейства: base1/site03/Blaster, base2/site02/Blaster, base2/site02/Machinegun. Новый train offset32:8 battles/семейство,24/24 живых побед. Новый validation offset16:4 battles/семейство,12/12 живых побед. Все36 captures проверены, pool16/x2. Train и validation engine seeds раздельны; предыдущие диагностические validation offset4 не используются как train. Final-test cohort не запускалась.

Это сохранённые rules demonstrations с наблюдениями, реальными native commands и результатами. Позже в тот же день выполнены [sequence export и CUDA BC](combat_sequence_bc_20261007.md): 296 train/139 validation контекстных кадров, актуальные v6 features и отдельные masks. Парная игровая проверка full-actor BC завершена:36/80→28/80. Ветка не принята. Контроль с замороженными encoder/attention и остальными heads также завершён:29/80, не принят. Не принимать редкие соседние firing rows за полную последовательность. Для движения/вертикали этот короткий успешный corpus недостаточен; отсутствие класса действия не превращать в отрицательное доказательство полезности.

BC adapter актуальных aim/fire heads и sequence masks реализован. Обе paired оценки BC завершены. Следующее действие: расширить teacher labels на посещаемые learned states и проверить раздельные aim/fire вмешательства перед свежей mixed RL партией. При сравнении архитектур использовать одинаковый подтверждённый BC corpus/prior и собственный fresh PPO опыт; сначала attention64/128 и GRU128 с MLP-контролем. Увеличение сети ещё не начато и не выдаётся за исправление выявленных причин.

[Rules comparison](../../workspace/artifacts/combat-failure-diagnosis-v3-20261007/rules-pool/report.json), [диагностика traces](../../workspace/artifacts/combat-failure-diagnosis-v3-20261007/analysis.json), [teacher corpus](../../workspace/artifacts/combat-aim-teacher-corpus-v1-20261007/pool/report.json). Воспроизводимый offline анализ: scripts/analyze_registered_combat_failures.py; Python используется только для анализа сохранённых файлов, runtime/harness остаётся Go/native.

Проверки: go test ./cmd/... ./internal/... прошёл;20 парных diagnostic и36 teacher captures прошли. Runtime PAK hardlinks очищены штатным harness. Веса/checkpoints не добавлены в Git. Native physics/reward/guards не изменены.



