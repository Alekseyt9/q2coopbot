# Итог target-head pilot и следующий сбор, 2026-10-10

288/288 native captures проверены по source/native/runtime/model/seed/frame budget. Восемь новых BC-веток не приняты: лучший m0-after2/16, FireBC9/16, rules8/16. Это четыре сложных pilot family и повторный validation, не вся кампания и не final test.

| Вариант | Победы | Смерти | Средний урон монстрам | Полученный урон |
| --- | ---: | ---: | ---: | ---: |
| m0-before | 0/16 | 16 | 22.5 | 91.2 |
| m0-after | 2/16 | 14 | 42.9 | 82.1 |
| m1-before | 0/16 | 16 | 0.0 | 91.2 |
| m1-after | 0/16 | 16 | 3.5 | 91.2 |
| m2-before | 0/16 | 15 | 9.6 | 89.7 |
| m2-after | 0/16 | 16 | 22.4 | 91.2 |
| m3-before | 0/16 | 15 | 20.1 | 86.2 |
| m3-after | 0/16 | 16 | 19.5 | 91.5 |
| m4-before | 0/16 | 16 | 22.4 | 91.5 |
| m4-after | 0/16 | 16 | 10.1 | 91.2 |
| m5-before | 0/16 | 16 | 0.0 | 91.2 |
| m5-after | 0/16 | 16 | 4.5 | 91.2 |
| m6-before | 1/16 | 13 | 34.2 | 80.1 |
| m6-after | 0/16 | 16 | 9.2 | 91.2 |
| m7-before | 0/16 | 14 | 7.6 | 86.0 |
| m7-after | 0/16 | 16 | 0.0 | 91.2 |
| firebc-baseline | 9/16 | 6 | 129.9 | 59.1 |
| rules-baseline | 8/16 | 7 | 141.9 | 58.2 |

## Проверка происхождения слабых priors

Before — миграция старых архитектурных m0..m7, а не поздние сильные Attention128-v2b/FireBC. CUDA показала сохранение старых actor/value outputs при миграции. Ниже те же четыре family/seed в старой aevaluation-v2 (m*-after) и новой (m*-before):

| Модель | Ранее побед | После миграции | Совпало исходов / 16 |
| --- | ---: | ---: | ---: |
| m0 | 0 | 0 | 16 |
| m1 | 0 | 0 | 16 |
| m2 | 0 | 0 | 16 |
| m3 | 0 | 0 | 16 |
| m4 | 0 | 0 | 16 |
| m5 | 0 | 0 | 16 |
| m6 | 1 | 1 | 16 |
| m7 | 0 | 0 | 16 |

Низкие результаты before не следует объявлять провалом новой архитектуры target head: начальные priors уже слабы на этих случаях. Улучшение геометрического RMSE на небольшом корпусе не перенеслось в полезный бой. Причины не разделены: visitation distribution, смена прицела меняет направление движения, эвристика выбора цели, неполные weapon-phase/recoil labels.

## Прицел до явно выбранной цели

| Вариант | Yaw, среднее° | Pitch, среднее° | 3D ray, среднее° | Firing frames |
| --- | ---: | ---: | ---: | ---: |
| m0-before | 23.7 | 59.1 | 63.6 | 1171 |
| m0-after | 81.1 | 59.2 | 79.7 | 1141 |
| m1-before | 46.1 | 85.1 | 87.0 | 1290 |
| m1-after | 96.9 | 79.0 | 86.3 | 1232 |
| m2-before | 100.4 | 67.4 | 84.5 | 1614 |
| m2-after | 98.5 | 47.3 | 95.3 | 1316 |
| m3-before | 89.9 | 67.7 | 80.4 | 1434 |
| m3-after | 95.2 | 47.3 | 92.5 | 1468 |
| m4-before | 32.9 | 48.5 | 59.7 | 1195 |
| m4-after | 83.1 | 61.4 | 76.2 | 1014 |
| m5-before | 56.5 | 74.2 | 80.6 | 1330 |
| m5-after | 92.9 | 52.4 | 97.8 | 1158 |
| m6-before | 34.3 | 78.1 | 79.5 | 1168 |
| m6-after | 74.6 | 73.5 | 79.8 | 1289 |
| m7-before | 42.4 | 67.1 | 74.7 | 1498 |
| m7-after | 90.6 | 74.9 | 83.1 | 1180 |

Это applied-action ray к наблюдаемому bbox явно выбранного ID/track. Без lead/recoil коррекции. Frame count не равен bullet count, геометрическая близость не является hit rate. Legacy FireBC/rules не объявляют цель; ближайший монстр не подставлен вместо их намерения.

## Следующий реализованный этап

- Повторный callback одного кадра сохраняет тот же enriched observation/track/previous intent; age обновляется. Это устраняет несовпадение между cached recurrent action и новой history. Сборка Go runtime прошла; отдельная асинхронная повторная callback acceptance не заявлена.
- Новый `cmd/q2target-data` экспортирует V7 context и unexecuted intercept queries в `.jsonl.gz`, проверяя pinned steps SHA256 и native execution masks. Неподтверждённый terminal context сохраняется без меток. Не передаёт reward/server positions в сеть.
- BC принимает V7/gzip. Дополнительно можно обучать только восемь колонок actual previous target (846..853), сохраняя старые 845+none колонки и старые head rows. Это даёт возможность учиться удержанию цели. Encoder теперь может изменить общие выходы: loss дополнительно содержит raw-output MSE×100 к frozen parent. Это ограничение обучения, не гарантия поведения.
- Target labels: удержать прошлую observed identity/track, пока её slot валиден, иначе nearest visible. Это bootstrap-эвристика, не оптимальный порядок убийства. Fire/movement/weapon weights остаются замороженными. Machinegun aim labels с неизвестной fresh recoil по-прежнему исключены.
- CUDA smoke: 2 epochs на 1171 historical context frames с настоящим previous intent (не quality experiment); проверены старые weight columns/rows, формат gzip и checkpoint export.

Pipeline на сильном FireBC завершён:80 fresh stochastic own-policy train battles во всех20семействах (train offset104),16 validation battles (offset28), pool16/x2. Затем gzip corrective export,50 epochs CUDA с intent columns и64 paired native battles before/after/legacy FireBC/rules. Validation используется для разработки, final test не тронут.

## Итог fresh FireBC refresh

96/96 capture и64/64 paired evaluation завершены; native/source/binary proof проверен до изменений precision runtime. CUDA query audit:6268/1499 context frames,3844/1063 blaster aim pairs, максимальное расхождение serialized queries и CUDA геометрии0.0000153°. Validation query RMSE24.85→21.43°, но бой ухудшился.

| Вариант | Победы | Смерти | Средний урон нанесён / получен |
| --- | ---: | ---: | ---: |
| Target before | 7/16 | 9 | 176.8 / 67.8 |
| Target after intent BC | 6/16 | 6 | 146.1 / 70.6 |
| FireBC legacy | 7/16 | 9 | 184.3 / 67.2 |
| Rules | 8/16 | 5 | 117.8 / 43.8 |

After выиграл один прежде проигранный бой и проиграл два прежде выигранных. Firing-frame applied-ray error11.37°→14.97°; переключения18→69 при разных длинах траекторий. Это не bullet hit rate. **After отклонён; baseline не заменён.**

Следующая абляция начинает с сильного before45:81-output coarse/fine head с learned mode и диапазонами±180°/±15°. Режим записывается в sample; PPO учитывает target+mode+conditional Gaussian. Общий исполнитель и likelihood реализованы для MLP/GRU/temporal attention. Первые45 actor rows и encoder заморожены; обучаются только новые36 rows.100 CUDA epochs на том же сжатом corpus: fine-query validation RMSE6.88→6.05°, mode accuracy49.6→55.1%. Это подмножество536 fine queries; его RMSE не сравнивается напрямую с общим target RMSE. Machinegun recoil labels отсутствуют; перенос ещё требует native evaluation.

Артефакты: `workspace/artifacts/target-eval-v1-20261009/{quality-report,selected-target-aim}.json`; `workspace/artifacts/target-refresh-v2-20261010/`; launcher/process receipt в `workspace/build/target-refresh-v2-20261010-*`; код `scripts/run_combat_target_refresh.py`, `cmd/q2target-data/main.go`.
