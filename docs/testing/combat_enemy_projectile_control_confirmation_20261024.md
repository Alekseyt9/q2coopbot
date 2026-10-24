# Независимая проверка neutral final — 24.10.2026

## Итог на 21:23 МСК

Авторитетный результат: `workspace/artifacts/enemy-projectile-control-confirmation-recovery-v1-20261024/result/report.json`, SHA `bd1b2c469653ac653cde2bd154f73aba4819c27347e8f7fba9b2bbea5021d5e6`. Derived pool закрыт72native/48CUDA:64исходныхvalid retained,8пустыхstartupповтореныбеззаменыкакого-либобоевогоисхода. Новый source fence неизменен; original_source_unchanged=false и startup recovery SHA сохранены отдельно. Все записи и неудачные попытки сохранены.

| Вариант | Цели / 24 | Смерти | Убийства | Получено HP |
|---|---:|---:|---:|---:|
| Initial | 14 | 7 | 44 | 1216 |
| Candidate | 13 | 9 | 40 | 1416 |
| Rules | 14 | 6 | 39 | 1132 |

Gate=false по всем критериям качества: нет нужной прибавки целей, смертей и входящего урона больше обоих сравнений. Model ownership подтверждён у обеих neural веток; live monster infighting=0. Promotion=false, default прежний. По парным целям кандидат получил2новые и потерял3относительнокаждогосравнения; descriptive sign p=1.0, это не доказательство статистически значимого ухудшения или улучшения.

Разрезы initial/candidate/rules: blaster2/3/3победыиз8, machinegun6/4/6из8, weapons6/6/5из8; пары9/9/10из12,тройки5/4/4из12. На machinegun у candidate4смерти против1initial/0rules. Эти маленькие разрезы описательные, не причинная диагностика и не основание дообучать на confirmation.

Native MG live-damage fraction initial56.65%→candidate55.92%, solid contacts260/699→261/667. Standing среди grounded-visible кадров895/2040(43.87%)→944/2060(45.83%);yaw>=30 градусов148/2842(5.21%)→142/2809(5.06%). Меньшее число поворотов не означает эффективный бой; лишние действия не устранены.

Следующий шаг — толькоTRAIN разбор собственных192боёв закрытого опыта: источники входящего урона с native mod, loadout, состав и запрошенное/исполненное движение. Confirmation не используется как training corpus или демонстрации. Отрицательные typed и neutral confirmation решения сохраняются; автоматического нового fitting нет.

## Условия до запуска

Предыдущий enemy-projectile опыт закрыт: typed и neutral final по5/8 целей, полученный урон386/328HP; rules2/8,5смертей,468HP. Исходный typed gate=false. Выбор neutral выполнен **после просмотра малого screen**, это отдельная гипотеза, не доказанное преимущество и не пересмотр отрицательного решения по типизации снарядов.

Фиксированные endpoints: neutral initial `f99f49f28d2d6b0b566d259484fb58f15a4b181e5477f9d32e9a2a9aed961d77`; neutral round04 `1b96f67cbe5f4d1d89ac3d79f3c425ee8178b942b7e9b8a8c11aaf2d903cf6a7`; rules. Нового fitting нет, веса не выбираются по новым исходам.

Все24 balanced рецепта текущего exact soldier_light реестра: base3/city1/jail1/ware2,по2BSPпозиции;12пар/12троек;по8blaster/machinegun/weapons. Confirmation split,offset0; перед играми общий evaluator проверяет всю физическую историю планов, включая прежние TRAIN/evaluation. При пересечении — остановка до игр, без скрытого переназначения сидов. Поодному новому seed на рецепт, одинаковому для трёх вариантов. Бюджет72native/48CUDA,пулдо16,×2,резервдиска32GiB. Численные neural checks толькоCUDA.

Порог до новых игр: candidate минимум+3целиrules и+1initial; смертейнебольшеобоих,полученоHPнебольше110%каждого; видимый бой принадлежит модели, live monster infighting=0. Corpse contacts считаются отдельно. Порог — инженерный критерий расширения проверки, не гарантия статистической значимости. Passing не означает принятия всей кампании/кооператива, автоматического promotion нет. Публикуются парные исходы и разрезы map/site/duo-trio/loadout, а также native MG/standing/yaw диагностика. Неудачная типизация остаётся неудачной при любом результате этого подтверждения.

Протокол `docs/combat_enemy_projectile_control_confirmation_protocol_20261024.json`, SHA `e111877061eb62c1db26bec40bc122886b1aa72ad376d0503a4d644acc1a8d4d`; spec `docs/combat_enemy_projectile_control_confirmation_spec_20261024.json`, SHA `56b813350fd907719085988a9534ba2e10b5896c867a6eb8aae412e023aa24b2`. Runner `scripts/run_combat_enemy_projectile_control_confirmation.py` фиксирует Go/Python/PowerShell/native sources и оба JSON до игр, проверяет provenance исходного закрытого опыта и endpoints, затем evaluation/native-quality/result. Output `workspace/artifacts/enemy-projectile-control-confirmation-v1-20261024`.

CLI/AST и баланс24рецептов проверены без neural execution. Живой запуск и результаты фиксируются ниже.

Запуск21:05:59МСК:actualsupervisor31892,evaluator18724живы. НеверныйweightsSHAотклонёндоoutput/игр;negative fixture/reportсохраненывenemy-projectile-confirmation-negative-v1-20261024. На21:06стадияevaluation/preflighthistory,stderrпуст,свободно66GiB. Источники/обаJSONзаморожены,результатовещёнет.

Около21:13 исходный workflow завершился failed:64valid captures,8empty startup failures. У6 Windows paging-file error, у1 Go runtime cannot allocate memory, у1 native Z_TagMalloc; trace нет, combat release не было. Другие сервисы не останавливали. Исходный pool/всеattemptsсохранены. classify в recover_registered_combat_startup.py расширен только на подтверждённый resource startup; compressed trace/released combat/unrelated Bot failed/accepted outcome отрицательные проверки отклонены,все8реальныхпустыхслучаевприняты. Сохраняются64исходныхбоя,повторяютсяпоследовательнотолько8с теми же frozen fixture/seed/model/binary bundle. Reward/gate/protocol/spec/веса не менялись. Admission SHA4fd91023c809ce7d48b9fb79791dc0c440f126c80b07fff4d8baa6f6bbcc5dd3,old/new source snapshots и тесты вstartup-recovery-admission-v1. Runner принимает ровно2объявленныеsource revision,проверяетостальнуюоригинальнуюsourcefence,далеесновазамораживаетисходники. Новыйoutput enemy-projectile-control-confirmation-recovery-v1-20261024; результаты качества пока не приняты.

21:15МСК:recovery supervisor20332/child34764живы,стадияrecover-native;первые2из8retry captures complete/provenance=true,по1usable. Исходные64validнепереигрываются. Новыйsourcefenceпрошёл;stderrпуст. Остальныеретраи/CUDA48/итогкачестваещёнеподтверждены.
