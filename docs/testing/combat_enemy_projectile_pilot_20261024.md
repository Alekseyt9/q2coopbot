# Реальные вражеские болты и гранаты — TRAIN pilot 24.10.2026

Предыдущий typed/neutral опыт закрыт с непройденным expansion gate: typed 6/8 целей, neutral 5/8, rules 4/8. Все 198 наблюдаемых bolt slots его TRAIN оказались собственными выстрелами бота. Утверждать обучение различению нескольких вражеских projectile классов по нему нельзя.

## Изменение реестра

Добавлен отдельный `campaign-ground-projectile-group-v1`, сохраняющий exact `monster_soldier_light`. Старые одиночные campaign рецепты и их историческая нормализация не меняются. Старый group kind не принимает новый класс. Канонический порядок старых четырёх классов и их битовые маски сохранены; light получает отдельный bit16 и стоит перед gunner, чтобы даже группа light/gunner имела поддерживаемый primary.

Семейство новых split seeds начинается с 1400000000: отдельные domains для loadout, map, site, composition и split; count512. Тест перебирает все поддерживаемые карты,12sites,5loadouts,составы2..4 и4splits: новые диапазоны не пересекаются между собой или со старыми group диапазонами, engine int32 не переполняется. Исторический base3/site01 parasite-infantry-gunner weapons train start1164065000 закреплён отдельной проверкой.

Генератор проверяет placement по реальному BSP. `generated_combat_fixture.ps1` принимает light soldier и требует его штатные20HP в native start receipt. Отдельная scalar проверка подтвердила exact class/HP и отвергла30HP/shotgun alias. Это ещё не живая проверка.

`go test ./internal/trainingepisodes -count=1` пройден, включая новый geometry test на base3 без пропуска. Нейронные CPU вычисления/тесты не запускались. На первом запуске новый ID с underscore был отклонён; label исправлен на soldier-light при сохранении exact native class soldier_light. Старые ID не меняются.

## Зафиксированный пилот

Builder `scripts/prepare_combat_projectile_group_pilot.py`, root `workspace/artifacts/combat-projectile-enemy-pilot-v1-20261024`. Восемь рецептов: base3/city1/jail1/ware2, по две точки. В каждой группе parasite/soldier_light/gunner, loadout weapons, v14 cost reward без изменения. Два свежих TRAIN seed на рецепт,16native/16CUDA,общийпул16,×2. История сидов проверяется обычным рекурсивным collector.

Источник — typed initializer из предыдущего опыта, не выбранные по его оценке финальные веса. Native поведение initializer эквивалентно предыдущему parent до нового fitting. Сбор только TRAIN; fitting и качество в сравнении с rules здесь не проверяются, default не заменяется.

До нового обучения нужны фактические native start receipts exact light soldier20HP, корректные позиции/состав, отсутствие infighting, наблюдаемые вражеские bolts и grenades, доступность оценок скорости. Родство болта нельзя выводить только из class или отсутствия текущего собственного выстрела: требуется native свидетельство либо диагностическое дополнение телеметрии. Пилот должен проверить достаточную экспозицию разных угроз; слабый охват не считается положительным результатом архитектуры.

16 боёв запущены скрытым collector, порт41200..41215. Результаты ещё не проверены. Во время сбора исходники не меняются.

## Игровая часть закрыта, исправление offline goal admission

Около18:55 native pool закрыт16/16 usable,безошибок,source_unchanged=true. Collector завершился ошибкой до CUDA проверки: `q2ppo-data` отклонил `unverified combat goal stop`. В `VerifyGoalStopForClasses` остался список четырёх старых классов, без soldier_light. Данные и весь исходный failed processing сохранены; игр при исправлении не повторяли.

После подтверждения завершения collector добавлен exact soldier_light в offline verifier. Его scalar goal tests прошли; доказательство убийств по native events, совпадение expected classes, actor, map, spawncount и frame не ослаблялись. Дополнительно ограничен light primary только новым projectile group kind; старые одиночные generator версии его не принимают. `go test ./internal/trainingepisodes -count=1` и `go test ./internal/learningenv -run 'Test.*Goal' -count=1` прошли, нейронного CPU исполнения нет.

Запущена отдельная `recovery-v1/processing` через существующий CUDA processor на тех же16captures. Исходный `corpus/progress.json` остаётся failed, частичный `processing-train` не перезаписывается. Recovery receipt указывает исходный pool SHA, изменённый offline verifier и нулевое число новых native игр. Результат CUDA обработки ещё не подтверждён.

## Технический результат пилота — около19:00МСК

Recovery processor завершён:16native/16CUDA,1961принятыйпереход,training_performed=false. Игровой пул занял61.98секунды. Повторной native генерации при восстановлении нет. Все16native start receipts подтверждают exact parasite175HP/light soldier20HP/gunner175HP и штатного игрока на проверенных позициях. По всей сохранённой native damage telemetry нет событий положительного урона monster→monster. Это подтверждение данного пилота, а не всех будущих рецептов.

Read-only `exposure.json` связывает принятые CUDA rows с исходными steps и native log по SHA. Из1961строки150содержат снаряд в первых4слотах. Grenade161slots в7боях, скорость известна135; blaster_bolt51slots в8боях, скорость известна35. Одновременных mixed-class rows нет.

В принятых окнах зарегистрированы6попаданий soldier_light по игроку сMOD1,7прямых grenade попаданий gunner сMOD6 и81gunner bullet event MOD0. Есть один непосредственный observation→native enemy-bolt impact witness и шесть grenade witnesses: видимый inflictor ID перед ударом совпал с native event, network frame согласован, attacker class подтверждён. У единственного bolt witness оценка скорости ещё отсутствует. Такой window witness не доказывает полный lifecycle каждого видимого entity ID.

Пилот подтверждает работоспособность exact light-soldier рецептов и наличие реальных вражеских blaster атак, но **готовность данных для нового fitting ещё не подтверждена**. Текущий `g_test_projectile` пишет spawn/end только для player bolts. Нельзя объявить все51видимыйboltвражеским или считать35известныхскоростей доказанно вражескими без дополнительного lifetime учёта. Следующий шаг — отдельная диагностическая native telemetry вражеских bolt spawn/end, затем малый свежий TRAIN повтор с проверкой владельца, наблюдаемой скорости и частоты угроз. Native labels не должны попадать в policy inputs, награда и боевые действия не меняются.

Root `complete.json` закрывает техническую обработку и явно указывает `original_collector_failed=true`, `dataset_admitted_for_new_training=false`, fitting/promotion=false. Исходный failed collector не объявлен успешным. Общая цель превзойти rules остаётся открытой.
