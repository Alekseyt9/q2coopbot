# Что модель видит перед попаданиями гранат

24.10.2026,21:40МСК. Продолжение толькоTRAIN разбора после отрицательного neutral confirmation. Новых игр, fitting, neural CPU execution и изменений runtime guard/default нет.

## Метод

Новый `scripts/audit_combat_grenade_preimpact.py` читает96собственныхCUDA-verifiedTRAINбоёвкаждойветкииззакрытогоenemy-projectileопыта. Источники привязаны к SHA sealed risk v2/native windows/rollout. Все gunner события MOD_GRENADE6/MOD_G_SPLASH7 с потерейHP сверяются по числу, HP и смертельным событиям с независимым предыдущим разбором. Приняты119событий,3504HP,28смертельных.

Предшествующие observations выбираются строго по network frame за1/3/5/10кадровдоnativeevent.frame. Game frame не используется вместо network frame. Нужны точное совпадение кадра и map/spawncount/connection/life/actor; пропуски не интерполируются. В этом корпусе все заданные контексты найдены. Сортировка projectile slots — прежняя стабильная по distance, первые4.

Нативные future attacker/inflictor/mod служат только диагностикой. Совпадение inflictor ID с видимой grenade обозначено **same-ID candidate**, не доказывает lifetime и не исключает reuse. Ни эти метки, ни будущие исходы не становятся teacher/input.

Четыре focused scalar теста прошли: network/game clock и пропуски; жизнь/соединение/generation/actor; visible-but-outside-top4 и unknown velocity; стабильный порядок равных дистанций и известная нулевая скорость. Команда: GPU Python `-m unittest discover -s scripts -p test_combat_grenade_preimpact.py -v`. Тесты не исполняют нейронную сеть.

## Перед28смертельнымипопаданиями

Сводка двух разных own-policy TRAIN корпусов, не причинное сравнение архитектур. Один network frame соответствует 0.1 игровой секунды; ×2 сокращает реальное время прогона, а не эти игровые интервалы.

| До попадания | Любая grenade в top4 | Same-ID candidate в top4 | У candidate известна скорость | Модель просит движение | Applied движение>=0.25 | Request>=0.25,applied<0.1 |
|---|---:|---:|---:|---:|---:|---:|
| 1 кадр | 27/28 | 25/28 | 21/28 | 26/28 | 15/28 | 11/28 |
| 3 кадра | 26/28 | 21/28 | 21/28 | 27/28 | 11/28 | 15/28 |
| 5 кадров | 22/28 | 18/28 | 11/28 | 24/28 | 14/28 | 10/28 |
| 10 кадров | 20/28 | 8/28 | 6/28 | 23/28 | 17/28 | 6/28 |

Во всех119событиях и проверенных lead-контекстах нет same-ID grenade, видимой внеtop4: в этом разборе нет доказательства потери кандидата из-за лимита4слотов. Это не утверждение о всех снарядах/состояниях или времени до запуска гранаты.

На lead3 у смертельных событий записаны movement reasons: unsupported_motion_guard7, no_ground_support4, static_hull_blocked4, static_hull_component_clipped3. Причины могут пересекаться, а частичная коррекция не всегда полностью обнуляет движение. Это **не доказательство**, что выполненный запрос спас бы бота или что guard ошибочен.

## Геометрия четырёх состояний

Новый `scripts/audit_combat_grenade_motion_geometry.py` фиксирует первый подходящий контекст на каждую карту. Порядок: входной neutral,затемtyped;внутри round/seed/step/eventordinal. Всего13кандидатов:TRAIN lethal grenade,lead3,grounded,known same-ID velocity,request reduced nearzero. Отбор не использует новые игровые исходы.

Проверены исходные trace observation/proposal/integer sent command, position/velocity, shared PAK hashes и metadata; PAK/AAS не копировались. Существующие `q2motion-audit` и `q2support-audit` собраны и выполнены: они не вызывают NN. У всех4состояний записанные applied prediction/ground surface воспроизведены точно.

| Карта / ветка / seed / index / frame | Причина stop | Подробный запрос |
|---|---|---|
| base3 / neutral /1564140005/115/212 | no_ground_support | unknown,uneven_or_missing_support |
| city1 / neutral /1468045009/37/134 | unsupported_motion_guard | dry_flat,proposed prediction и path check=true |
| ware2 / neutral /1492145009/38/135 | static_hull_blocked | unknown prediction приdry_flat |
| jail1 / typed /1494145008/103/200 | no_ground_support | unknown,uneven_or_missing_support |

City1: proposed Forward156/Side289/Up0, applied0/0/0; взгляд/attack/light должны сохраняться отдельно. Control prediction position[-2895.42,-199.655,-279.875], proposed[-2874.0945057,-199.7608843,-279.875] — примерно21.33unitsразницыпоследующегокадра. Это существующий detailed static/path прогноз, не подтверждённое native исполнение и не уклонение от гранаты. Остальные unknown не разрешены к восстановлению.

Важное ограничение native проверки: на city1 следующий исходный кадр уже имеет health45→43,armor0→0. Старый checked-ground mechanical protocol требовал отсутствия урона. Его нельзя объявить применимым здесь или молча убрать проверку после неудачного запуска. Для under-fire пробы нужно заранее отдельно определить проверки исполнения/prefix/cold repeats, физические ошибки с возможным knockback и native HP/исходы на последующих кадрах. Нативная проба этого нового состояния **ещё не запускалась**.

## Артефакты

- Neutral: `workspace/artifacts/enemy-projectile-neutral-grenade-preimpact-v1-20261024/report.json`, SHA `791f83aeb2e107a86c3522ce3eb42e25c34be51718a737f23c5d3c9f7b037e73`.
- Typed: `workspace/artifacts/enemy-projectile-typed-grenade-preimpact-v1-20261024/report.json`, SHA `854f67fb90b7956792d9777121d341f4c2703bde7b86edbbb95d8a5bf42b1485`.
- Geometry: `workspace/artifacts/enemy-projectile-grenade-motion-geometry-v1-20261024/report.json`, SHA `9f955bacc1cd9d4ba8ec561cd8d0d19601a551e3a639cbeb76390bf6a2be9964`.

Все три sealed report SHA сверены после завершения. Вывод: объяснение «модель просто не видела гранату» недостаточно для этого TRAIN корпуса. Следующий шаг — ограниченная настоящая проверка единственного detailed-supported запроса city1 под огнём; неизвестные случаи, награда, веса и default остаются прежними. Превосходство над rules не установлено.
