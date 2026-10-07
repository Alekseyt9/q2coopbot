# Sequence BC: прицел и огонь, 2026-10-07

Первый этап плана превосходства обычного контроллера: дообучена отдельная ветка Temporal Attention Update29 на проверенных rules demonstrations. Модель не назначена основной. Ошибка воспроизведения учителя уменьшилась; игровое сравнение выполняется отдельно и является обязательным условием оценки результата.

## Данные и контракт

Корпус: 36 успешных native-verified боёв в трёх семействах — campaign-base1-site-03-blaster, campaign-base2-site-02-blaster и campaign-base2-site-02-machinegun. Train: 24 боя, offset32; validation: 12 боёв, offset16. Финальный test не собирался. Seeds разделены; это проверка новых seeds в известных точках, не перенос на новую геометрию.

Новый Go exporter `cmd/q2bc-sequence-data` повторно проверяет receipts исходных captures и экспортирует combat_features_v6, 845 признаков. Временная история сохраняется вместе с кадрами без supervised метки. Masks исключают такие кадры из loss. Python trainer разрывает контекст при разрыве кадров или смене жизни/карты/подключения.

| Набор | Контекстные кадры | Aim labels | Attack true | Attack false |
|---|---:|---:|---:|---:|
| Train |296|187|187|6|
| Validation |139|80|80|5|

Стационарные rules-команды с `no_movement_goal` допускаются только в новой версии teacher_ranged_tracking_candidates_v4 для прицела/огня. Они не являются демонстрацией движения. Команды после safety correction, с неподтверждённым исполнением или несовпадающим оружием не становятся положительными метками. Бочка в base1/site03/Machinegun остаётся отдельной проблемой манёвра; её защиту не отключали.

Native goal receipts теперь повторно проверяются и при сборке demonstrations: manifest, configured enemy classes, события смерти и совпадение кадра границы. Полный шаг сверяется после восстановления verified goal boundary. Это устраняет прежний отказ проверки на успешном завершении боя.

## Обучение

`scripts/train_combat_sequence_bc.py`, конфигурация `scripts/scenarios/combat-sequence-bc-aim-v1.json`: CUDA RTX5070, 150 epochs, lr0.0001; aim loss weight4, balanced attack BCE weight1, retention weight0.1. Выбирается заранее заданная последняя эпоха; validation не выбирает checkpoint.

Архитектура сохранена: Temporal Attention, hidden64/64, 4 heads, context32. Critic и log_std сохранены. Actor PPO optimizer сброшен после изменения actor; consumed rollout hashes и счётчики опыта сохранены. BC не объявляется дополнительным PPO update: completed_updates29, actor_steps279. Другие action heads удерживаются через retention loss, но их фактическое поведение необходимо проверять в игре.

| Метрика | Train до | Train после | Validation до | Validation после |
|---|---:|---:|---:|---:|
| Teacher aim RMSE, градусы |11.406|1.485|11.403|1.516|
| Balanced attack BCE |1.0723|0.03129|1.1398|0.03403|

Это ошибки воспроизведения teacher-команд, не точность попаданий и не доля побед. Небольшой корпус объясняет короткое время GPU обучения (~2.84s); сбор native боёв и игровая оценка занимают гораздо больше времени.

Артефакты: `workspace/artifacts/combat-aim-teacher-corpus-v1-20261007/bc-sequences-v1` и `bc-update-v1`. Parent weights SHA256: `3cacf9adb1c29c3b19ec5b78b29328bb7c9fad1fb675879591b695fb695e896f`. BC weights: `f5d8afbab7460f11bfb1aa35c8ff138a00ff51fc936125df5b145d33909bc434`; checkpoint: `ea24704013e95f64d417f13cca48735ec8c5f7b1adb4e9a4511b8156f2018699`.

## Проверка в игре

Запущена отдельная paired оценка всех 20 семейств реестра: исходный Update29 против BC-ветки, по 4 одинаковых новых validation seeds, offset20. Всего 160 боёв, pool16, x2. Frozen evaluation models deterministic; физические условия генерации и seed_revision сохранены. Root: `workspace/artifacts/combat-sequence-bc-eval-v1-20261007`.

Оценка завершена: все160 captures проверены, pool/source receipts сохранены, стартовые generated fixtures всех80 пар совпали. BC full-actor ветка регрессировала и не назначена основной. Финальный test не запускался. Сравнение GRU/большей attention сети остаётся следующим этапом после устранения этого регресса.

Проверки к запуску оценки: `go test ./cmd/... ./internal/...` прошёл; exporter повторно проверил все 36 captures; CUDA trainer завершился и сохранил веса/checkpoint с SHA. Веса/checkpoints остаются вне Git.

## Полная игровая оценка BC full-actor

| Метрика | Update29 | BC full-actor |
|---|---:|---:|
| Победы |36/80|28/80|
| Смерти в первой жизни |42|45|
| Убийства в первой жизни |39|28|
| Полученный health damage |4619|5226|

| Семейство | Update29 wins/4 | BC wins/4 |
|---|---:|---:|
|parasite-blaster-generated|4|3|
|parasite-gunner-blaster-generated|0|0|
|parasite-machinegun-recoil|3|0|
|parasite-gunner-machinegun-recoil|0|0|
|campaign-base1-site-01-blaster|2|0|
|campaign-base1-site-01-machinegun|2|0|
|campaign-base1-site-02-blaster|3|3|
|campaign-base1-site-02-machinegun|4|2|
|campaign-base1-site-03-blaster|1|2|
|campaign-base1-site-03-machinegun|0|3|
|campaign-base1-site-04-blaster|2|0|
|campaign-base1-site-04-machinegun|0|3|
|campaign-base2-site-01-blaster|3|3|
|campaign-base2-site-01-machinegun|4|3|
|campaign-base2-site-02-blaster|0|0|
|campaign-base2-site-02-machinegun|1|1|
|campaign-base2-site-03-blaster|2|0|
|campaign-base2-site-03-machinegun|4|1|
|campaign-base2-site-04-blaster|1|2|
|campaign-base2-site-04-machinegun|0|2|

В traces доля firing commands с yaw error >20 градусов к ближайшему clear observed enemy выросла с1218/4192 (29.1%) до3209/6855 (46.8%). Native outgoing monster damage снизился с4798 до3261. Это горизонтальный command proxy без pitch/lead, не измерение hit accuracy. Частота движения и firing также изменилась; full-actor retention loss не обеспечил сохранение поведения. Изолированный cause без дополнительного эксперимента не установлен.

## Контроль: обновление только aim/fire heads

Добавлен train_scope=aim_fire_heads: encoder/MHA заморожены, gradient masks разрешают только строки2:5 линейных head и attention residual. После обучения проверяется точное равенство остальных параметров и action outputs на train/validation; critic/log_std сохранены. Для PPO параметры вновь доступны, actor optimizer создаётся свежим. CUDA training завершён: validation aim RMSE3.243°, balanced attack BCE0.6595, retention MSE строго0. Это отдельный контроль причин регресса, не доказанное игровое улучшение.

Root модели: workspace/artifacts/combat-aim-teacher-corpus-v1-20261007/bc-heads-update-v1. Завершены80 captures этой ветки на той же validation offset20; исходные80 Update29 captures переиспользованы только для оценки, не для PPO. Root: workspace/artifacts/combat-sequence-bc-heads-eval-v1-20261007. Стартовые generated fixtures всех80 пар повторно сверены и совпали. Этот cohort уже validation, не новый независимый final-test.


## Итог ограниченного BC контроля

| Метрика | Update29 | BC full-actor | BC aim/fire heads |
|---|---:|---:|---:|
| Победы |36/80|28/80|29/80|
| Смерти в первой жизни |42|45|39|
| Убийства в первой жизни |39|28|30|
| Полученный health damage |4619|5226|4284|

Heads-only контроль тоже не принят: меньший урон и меньше смертей не компенсируют снижение числа выполненных целей. Изменение других параметров не является единственной причиной регресса; aim/fire сами меняют последующие наблюдения и траекторию движения. Dataset shift между успешными stationary teacher trajectories и посещаемыми learned states — проверяемая гипотеза, не установленный факт.

Следующий этап плана уточнён: собрать teacher aim/fire метки на состояниях, посещаемых learned policy, включая большие ошибки прицела, отсутствие clear shot, движение и несколько противников. Проверить contract/следование цели на этих states; увеличить разнообразие отрицательных firing labels. Отдельно проверить aim-only и fire-only вмешательства. Затем BC+собственный свежий mixed PPO опыт и парная validation. Не переносить rollout исходной policy как on-policy опыт BC. Основной reference остаётся Update29; архитектурное сравнение64→128/GRU сохраняется после проверки данных и action heads.

Обе оценки завершены без ошибок source/provenance:160+80=240 captures, pool16/x2. Head-only trainer проверил точную сохранность остальных параметров и outputs, SHA checkpoint/model, PPO config и fresh actor optimizer; полноценный PPO resume этой ветки пока не запускался. Финальный test не затронут.


## Раздельные aim/fire контроли

Добавлены train_scope=aim_heads и fire_head: encoder/MHA и все невыбранные строки head/residual точно сохранены, включая неизменённый attack head в aim-only ветке и неизменённые yaw/pitch heads в fire-only ветке. Две ветки обучены на CUDA с прежним corpus и заранее заданными150 эпохами. Выполняется парная оценка20 семейств, по4 seeds на ветку, offset20; исходная Update29 оценка переиспользуется только как reference. Root: workspace/artifacts/combat-sequence-bc-ablation-v1-20261007. Final test не запускался.

Для следующего corpus потребуется отдельно маркированная counterfactual aim query на verified learned states. Query-команда не исполнена в исходном бою; её нельзя смешивать с applied action/reward или представлять native evidence будущего попадания. Наблюдения, actual command и эффекты должны сохранить исходную проверку, а desired aim храниться отдельным полем. Геометрическая query — обучающий ориентир, не доказанная оптимальная тактика.


## Итог раздельных контролей

Все160 новых captures проверены; generated fixtures каждой пары совпали с исходным reference. В сумме четыре BC варианта потребовали400 captures (160+80+160), reference80 одни и те же.

| Метрика | Update29 | BC aim-only | BC fire-only |
|---|---:|---:|---:|
| Победы |36/80|24/80|50/80|
| Смерти в первой жизни |42|44|29|
| Убийства в первой жизни |39|25|53|
| Полученный health damage |4619|4704|3422|

На этом validation cohort aim-only снижает число побед, fire-only улучшает его на14. Это локализует отрицательное вмешательство в aim BC, а не доказывает превосходство новой architecture. Выбор ветки выполнен по validation; её результат не считать независимым final-test и не переносить на неизвестные точки карт/составы.

Запущен следующий этап: mixed PPO из fire-only BC checkpoint, pool16/x2, CUDA-only. Все20 семейств,80 новых stochastic train battles, train offset40; модель получает только собственный on-policy опыт. Затем before/after160 deterministic validation battles на новом offset24. Suite: scripts/scenarios/combat-training-suites/mixed-fire-bc-resume-v1.json. Root: workspace/artifacts/combat-mixed-fire-bc-resume-v1-20261007. Reference Update29 сохранён; fire-only пока экспериментальная ветка. Counterfactual aim corpus остаётся отдельной задачей устранения выявленного aim-регресса.


| Семейство | Update29 | Aim-only | Fire-only |
|---|---:|---:|---:|
|parasite-blaster-generated|4/4|0/4|4/4|
|parasite-gunner-blaster-generated|0/4|0/4|0/4|
|parasite-machinegun-recoil|3/4|0/4|3/4|
|parasite-gunner-machinegun-recoil|0/4|0/4|0/4|
|campaign-base1-site-01-blaster|2/4|0/4|3/4|
|campaign-base1-site-01-machinegun|2/4|0/4|3/4|
|campaign-base1-site-02-blaster|3/4|4/4|3/4|
|campaign-base1-site-02-machinegun|4/4|2/4|4/4|
|campaign-base1-site-03-blaster|1/4|1/4|0/4|
|campaign-base1-site-03-machinegun|0/4|0/4|0/4|
|campaign-base1-site-04-blaster|2/4|2/4|4/4|
|campaign-base1-site-04-machinegun|0/4|0/4|2/4|
|campaign-base2-site-01-blaster|3/4|4/4|4/4|
|campaign-base2-site-01-machinegun|4/4|4/4|4/4|
|campaign-base2-site-02-blaster|0/4|0/4|0/4|
|campaign-base2-site-02-machinegun|1/4|0/4|0/4|
|campaign-base2-site-03-blaster|2/4|0/4|4/4|
|campaign-base2-site-03-machinegun|4/4|1/4|4/4|
|campaign-base2-site-04-blaster|1/4|4/4|4/4|
|campaign-base2-site-04-machinegun|0/4|2/4|4/4|

В парных исходах fire-only сохранила33 исходных победы, добавила17 побед и потеряла3;27 проигрышей остались проигрышами. Blaster18/40→26/40, Machinegun18/40→24/40. Улучшение охватывает оба loadout, но группы Parasite/Gunner остаются сложными. Native outgoing monster damage4798→5336. Большее число firing commands не является отдельным критерием качества: оценка основана на native goals и первой жизни.


## Проверка покрытия aim labels

Скрипт scripts/analyze_combat_aim_coverage.py проверил hashes данных, capture bindings и unchanged traces. Из187 train aim labels ни одна не требует yaw_delta >5 градусов, средний модуль0.114°. В80 сохранённых learned TRAIN captures:7059 уникальных first-life provider observations с clear observed enemy;6078 требуют номинальной горизонтальной коррекции >5°,3225 — >20°,333 — >90°, средняя25.919°. Геометрический ориентир — ближайший clear observed enemy; это не оптимальный выбор угрозы и не hit accuracy. Validation/test observations для этой диагностики не превращались в train.

Следовательно, в исходных BC aim labels подтверждён недостаток примеров больших поворотов. Гипотеза механизма регресса: policy учится малой коррекции уже наведённого teacher, а на собственных состояниях недостаточно восстанавливает aim. Раздельный aim-only контроль подтверждает отрицательный эффект этого вмешательства на выбранном validation cohort, но сам по себе не доказывает единственную причину. Query corpus должен сохранять исходные actual commands/rewards и отдельную неисполненную desired-aim annotation.

Доказательства: workspace/artifacts/combat-sequence-bc-ablation-v1-20261007/aim-coverage.json, source SHA receipts внутри файла. Дополнительный query exporter ещё не реализован.

