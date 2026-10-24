# Native enemy-bolt lifetimes — 24.10.2026

Предыдущий exact soldier_light пилот технически закрыт16native/16CUDA,1961rows. Состав корректен и взаимного урона монстров нет; есть51visible bolt slot и6incoming light-soldier hits. Но старый журнал писал lifecycle только player bolts. Один visible-inflictor window witness перед вражеским попаданием не доказывает владельца каждого наблюдения или достаточное число известных вражеских скоростей. Нового fitting по этим данным не было.

## Изменение native

В `yquake2` добавлен отдельный `g_test_enemy_bolt` stream, активный только при существующем диагностическом `g_test_damage`. В `fire_blaster` monster-owned bolts получают отдельный serial; spawn содержит entity, attacker/class и native initial origin/velocity. `G_FreeEdict` закрывает lifetime; номер сбрасывается вместе с картой. Отдельный массив не меняет player shot counters, reward telemetry или edict layout. Никакие RNG/physics/action решения не добавлены. Обычная игра с g_test_damage=0 не пишет эти события.

Сервер оборачивает сообщения `sv_test_enemy_bolt spawncount=... server_frame=...`; inner game frame сохраняется отдельно. Диагностические owner/velocity labels не передаются клиенту и не добавляются в policy features. Исходные native binaries сохранены в `workspace/artifacts/combat-enemy-bolt-lifetimes-v1-20261024/before-native`. Сборка `cmake --build .../yquake2/build/codex-speed-test --target q2ded game --parallel 8` завершена успешно.

`scripts/combat_enemy_bolt_lifetimes.py` связывает player/enemy streams по spawncount, serial и entity lifetime, использует network server_frame, отвергает malformed/overlapping/unready records. Reuse в одном network frame остаётся неоднозначным. Три scalar теста прошли: часы/поколения/повторное использование entity, same-frame ambiguity и повреждённые сообщения. Нейронных CPU вычислений нет.

## Зафиксированная живая проверка

Протокол `docs/combat_enemy_bolt_validation_protocol_20261024.json`, SHA `97a74f68c33de5715c7d6ee57ea200e7092204f82ce185ecf226daa1f98ffb8a`. Runner `scripts/run_combat_enemy_bolt_validation.py`. Root `workspace/artifacts/combat-enemy-bolt-lifetimes-v1-20261024/workflow`.

Всего20native/20CUDA: четыре technical same-world повтора (поsite01наbase3/city1/jail1/ware2) для сравнения команд/мира с предыдущими сохранёнными боями и16свежихTRAIN случаев на8рецептах,offset2. Повторные сиды явно technical, не считаются свежими TRAIN и не смешиваются с новым корпусом. Те же typed initializer, v14reward,loadoutweapons. Два плана исполняются общим динамическим пулом16,×2,порт41300..41315; fitting отключён.

Runner проверяет совпадение старых seed/instances четырёх repeats, свежесть новых16черезобычныйcollector, затем отдельно обрабатывает каждую часть наCUDA. Исходники Go/Python/PowerShell/native и native binaries заморожены до завершения. Даже successful collection seal не доказывает parity/coverage: эти отдельные аудиты обязательны после сбора. Модель/default не заменяются. Запуск состоялся, результатов пока нет.

## Закрытый технический результат — около19:15МСК

Workflow закрыт20native/20CUDA,source unchanged,безfitting. Из них4same-world repeats и16свежихTRAIN. Отдельный `validation-report.json` проверяет все входные SHA, CUDA replay receipts и native starts. Root `complete.json` связывает collection seal и отдельный validation report; исходный collection seal с `parity_audit_pending=true` не переписан.

Все4пары совпали на356игровыхкадрах(132/95/97/32): полные observation dictionaries, sent_command, owner, action, applied_action и наборы кадров до kill/death. Shutdown tail исключён одинаково. Это сильнее сравнения только итоговых побед, но остаётся проверкой четырёх условий, не всей игры. Диагностический журнал в этих условиях не изменил поведение.

Свежая часть16native/16CUDA содержит1991принятыйпереход. Все16start receipts подтверждают точный состав и20HP light soldier; положительного monster→monster damage нет. Наблюдаемые первые4слота:225grenade и28blaster_bolt. Native journal содержит13enemy bolt spawns. Все28видимыхbolt slots сопоставлены с действующими lifecycle и attacker_class=monster_soldier_light; self/other_player/unmatched/same-frame-reuse —0. Таким образом, в отличие от предыдущего typed опыта, реально наблюдаемые вражеские болты подтверждены.

Из28enemy slots на19есть observed_velocity. Максимальная L2ошибка к native initial velocity1.253875ед/с,средняя0.666556ед/с; ошибоквыше10нет. Это скалярная проверка наблюдаемой физики, не NN CPU test. Native initial velocity используется только для диагностики, не заменяет измеренную клиентом скорость в признаках. У184grenade slots также известна наблюдаемая скорость; их ownership этим enemy-bolt журналом не проверяется.

Аудит `audit-validation.py` сохранён внутри artifact root. Он исполняет scalar metadata/physics сравнения, а не neural forward; численная нейронная проверка всех20новыхcaptures выполнена отдельным CUDA processor. Никакого обучения на repeats или fresh cases не было, веса не менялись.

Ограничения: подтверждённое наличие разных угроз не означает достаточный учебный бюджет; enemy bolt slots составляют малую долю1991перехода. Четыре повторных мира не доказывают отсутствие любых timing effects на всех картах. Перед новым typed/neutral fitting следует расширить exact group curriculum по составам и оружию и заранее зафиксировать собственные TRAIN budgets и свежую оценку. Дальнейший quality gate нельзя подменять этим техническим результатом.
