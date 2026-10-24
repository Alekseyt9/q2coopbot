# Проверка отменённых наземных команд — 24.10.2026

Предыдущий full-episode stopped-crouch screen дал control/crouch/rules5/6/5побед из8,3/2/1смертей,405/341/280HP получено. Expansion gate=false; default не заменён. Следующий вопрос — отменённые горизонтальные команды, а не дальнейшее продвижение неподтверждённого результата.

## Основание

`audit_combat_checked_ground_candidates.py` проверил seal, SHA отчёта/request/geometry и все входные SHA закрытой `combat-recorded-crouch-motion-v2-20261024`. Из638старых TRAIN записей отмены crouch63proposed команды проходят прежний подробный Pmove и path checks: city1=25,jail1=31,ware2=7. Это подмножество crouch cancellations, а не все отказы движения. Все63на земле, хотят горизонтальное перемещение+crouch, прежний guard обнуляет Forward/Side/Up с `unsupported_motion_guard`. Маленькая вертикальная скорость при grounded=true может не поддерживаться упрощённой моделью, но допускаться подробной.

Сам факт offline path check не означает, что реальное движение безопасно или физически точно предсказано. Ground flags, packet quantization, динамические сущности и native contact resolution требуют проверки в движке.

Артефакт `combat-checked-ground-candidates-v1-20261024` заранее выбирает первые по frame/seed подходящие состояния common-initializer TRAIN в трёх strata: city1 без контакта, jail1 без контакта, city1 с контактом. Дополнительные условия: соседний control frame сопоставим по здоровье/броне/grounded, расстояние между прогнозами proposed/control≥1unit. Исходы новых боёв при выборе не используются.

## Диагностический режим

`test.checked_ground_request_probe_frame` включается только на заданном кадре первого life после release в captured/local/synchronous/learned/combat-only fixture. Default0 ничего не меняет. Совмещение с stopped-crouch probe запрещено.

Helper требует обычного полного stop с единственной причиной movement=`unsupported_motion_guard`, grounded/alive state, выбранного моделью ненулевого горизонтального запроса и отсутствия jump. Копирует из исходного запроса только Forward/Side/Up в уже проверенную команду; yaw/pitch/fire/impulse/msec/light сохраняет. Затем проверяет detailed Pmove и прежние `checkedCombatMotionPaths`: static hull, опасные contents/lasers, опора, альтернативные контакты, двери, movers и observed bodies. Другие причины отмены не обходятся, воздушное управление не разрешается, новое направление не придумывается. Интервенция: `diagnostic_checked_ground_request`.

Capture/native/CUDA сохраняют diagnostic flag и отдельную версию `combat_diagnostic_checked_ground_rollout_v1`. Эти данные не допускаются к обычным PPO fitting/merge. Никакого обучения или замены весов в опыте нет.

## Зафиксированный native тест

Протокол `../combat_checked_ground_native_probe_protocol_20261024.json`:

- city1-open, seed1228165303, frame104, прогноз перемещения относительно control≈10units;
- jail1-open, seed1174065303, frame104, ≈10units;
- city1-contact, seed1308165303, frame138, два контакта в predicted path, ≈2.39units.

Те же исходные веса4e31f0c1…, из них не выбирается новый endpoint. Control и request, по два cold повтора:12native/12CUDA, три технических исходных состояния. Пул≤16,×2, существующий baseline и общий binary bundle. Подключение после native signon hold40, release100. Количество независимых проверок качества не равно12; качество боя здесь не оценивается.

Заданный mechanical gate: исходный prefix control совпадает с исторической записью; все варианты совпадают до root observations/proposals/samples и integer commands до root. На root восстанавливаются только заданные Forward/Side/Up. На следующем кадре pose/grounded соответствуют ожиданиям, здоровье/броня не меняются, ошибка позиции≤0.25, скорости≤1.0 относительно отдельного прогноза своего режима. Cold repeats совпадают до следующего кадра. Float tolerance1e−9, нормализуется только spawncount; entity IDs не нормализуются.

Прошли целевые Go проверки двух записанных геометрических состояний, сохранения attack veto/light, отказа при другом safety reason/airborne/dead/unknown pmove/jump и границ local diagnostic config. Прежние crouch/recorded-motion/native-door проверки также прошли. Python/PowerShell AST и diff check пройдены. Go NN-тесты не выполнялись.

Runner `run_combat_checked_ground_probe.ps1`, auditor `audit_combat_checked_ground_native_probe.py`. Root `combat-checked-ground-native-probe-v1-20261024`. Default остаётся прежним, превосходство над rules не доказано.

## Закрытый результат

**12 native /12 CUDA, mechanical_gate_passed=true,46/46checks.** Native source_unchanged=true; seal и SHA всех входов проверены после завершения. Исходный prefix и оба cold повтора каждого режима совпадают, entity IDs не нормализуются.

| Состояние | Max ошибка позиции request, units | Max ошибка скорости request, units/s | Ошибка позиции control |
|---|---:|---:|---:|
| city1-open104 | 0.02325 | 0.06935 | 0 |
| jail1-open104 | 0.03782 | 0.01800 | 0 |
| city1-contact138 | 0.09497 | 0 | 0.08660 |

Все ошибки в заданных до исходов пределах0.25/1.0. Native grounded/ducked соответствуют ожиданиям, health/armor следующего кадра не меняются. Протоколом не добавлены aim/fire/weapon вмешательства: восстанавливаются только Forward/Side/Up выбранного моделью запроса. В двух open-floor состояниях вместо остановки исполнено примерно10units перемещения; в contact состоянии исполнено предсказанное движение с контактом. Ненулевая погрешность control на contact напоминает, что пакетная позиция не равна скрытой точной native позиции.

Это свидетельство излишнего отказа упрощённого guard в трёх конкретных наземных состояниях, а не разрешение всех63кандидатов или движения на любых картах. Целые бои с новым режимом пока не оценивались. Никаких новых весов/default изменений нет, диагностические rollout не обучают обычный PPO.

## Следующий опыт

Подготовить first-life диагностический режим, который сначала пробует подробную проверку исходного горизонтального запроса после unsupported-motion stop, затем использует уже проверенное сохранение crouch, если горизонтальное движение осталось отменённым. Так можно сравнить его именно с crouch-only контролем, сохранив одинаковую обработку позы, и отдельно показать обычный guard/rules.

До новых исходов зафиксировать свежий paired8recipe screen: ordinary/crouch-only/ground-request+crouch/rules,32native/24CUDA, те же веса и сцены/сиды внутри пары. Порог расширения заранее: +2победы к crouch-only, побед не меньше ordinary/rules, deaths/received не хуже заданных границ controls/rules. Нужны реальные вмешательства, точный prefix до первого различия и повторная проверка физики каждого restored движения. Неподдержанные состояния остаются остановленными. Без автоматического продвижения и без fitting на held-out данных.
