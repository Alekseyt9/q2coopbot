# Первый curriculum: Parasite с 20 HP

05.10.2026. Цель — получить первые positive kill windows самой политики после отсутствия убийств в обычном PPO reward-v2 пилоте. Это изменение тренировочного fixture, не обучение всего боя и не скрытая помощь контроллера правил.

## Инициализация и границы

В native `g_main.c` добавлен `g_test_combat_monster_health`, default=0. При cheats и одиночном combat barrier заданное HP применяется к живому Parasite ровно в release frame перед первым tick его AI. Допустимый native диапазон 1–175; harness явно разрешает curriculum 10/20/30. Во время боя HP не переписывается. Обычный режим default=0 ничего не меняет.

Изменение касается только тестовой инициализации существующего движка. Go остаётся владельцем policy/action/protocol/harness. Не меняются оружие, damage, collision guards, geometry, шаг physics или reward v2. Сеть не получает HP монстра, номер уровня curriculum или seed как новые features. Ожидаемые HP и native event остаются отдельными метаданными.

Baseline runner передаёт `-TrainingMonsterHealth`, записывает `training_monster_health` в manifest и подтверждает ровно одно событие `before=175 after=20` с entity ID в release game frame. `q2ppo-data` повторно проверяет native release frame/seed/target; initialization target должен присутствовать в начальном observation. Для обычных эпизодов curriculum override запрещён. Это дополняет проверку наблюдаемого reset и не объявляет полный world reset доказанным.

Native CMake build game/q2ded прошёл с существующим MinGW runtime на F. Первый запуск compiler без его bin в PATH завершился без диагностики; после настройки PATH штатная сборка прошла. Никаких установок или скачиваний. Общий `go test ./...` прошёл, включая проверки missing/duplicate/wrong HP/frame/seed и скрытого override в обычном fixture. Синтаксис трёх PowerShell runner проверен. Пользовательский live runtime не перезапускался.

## Замороженный цикл

Root: `workspace/artifacts/combat-ppo-curriculum-hp20-v1-20261005`, protocol `curriculum.json` в artifact. Start weights/checkpoint — фиксированный четвёртый update reward-v2 цикла, SHA `3516e5f7050996310bc645ab0216f63989a0d5e9f3d60018b31e1c2cb751b3e1`. Reward/config v2 остаются прежними, optimizer/RNG resume сохраняется; старые rollout не используются повторно.

Четыре batch × 4 server/client instances, x2, 300 кадров, **training HP=20**, seeds 14500–14515. Парная deterministic evaluation до/после на **обычных 175 HP**, seeds 14600–14603. Evaluation не входит в training, fixed fourth update выбран заранее. Увеличение числа kills на 20 HP не является доказательством переноса на 175 HP. И curriculum, и обычная оценка сохраняют реальные движения, повороты и выстрелы policy.
# Результат завершённого цикла

Четыре batch дали 551/687/687/424 перехода, всего 2349. В экспортированных PPO rollout есть семь окон с kill reward +5: 2/2/1/2 по batch. Все 16 тренировочных capture complete и provenance valid. Финальные KL: 0.009389/0.007628/0.007825/0.009513. Checkpoint содержит 21 update и 210 actor steps с учётом предыдущих циклов. CPU выбран по измерению: для этих небольших batch он быстрее CUDA.

Парная оценка на обычных 175 HP, четыре отдельных seed:

| Seed | Урон монстру до → после | Полученный урон до → после | Убийства до → после |
| --- | --- | --- | --- |
| 14600 | 30 → 30 | 50 → 0 | 0 → 0 |
| 14601 | 30 → 30 | 50 → 11 | 0 → 0 |
| 14602 | 30 → 30 | 29 → 13 | 0 → 0 |
| 14603 | 30 → 30 | 50 → 0 | 0 → 0 |

Итого полученный урон снизился 179 → 24, урон монстру остался 120, смертей 0 → 0. Семь тренировочных убийств подтверждают появление положительного опыта добивания, но перенос на полное HP пока не достигнут. Четыре пары в одной геометрии не доказывают статистическое преимущество или обобщение; автоматической боевой приёмки нет.

Финальные веса: `workspace/artifacts/combat-ppo-curriculum-hp20-v1-20261005/iteration-4/update/weights.json`, SHA256 `9920296b170acaa1f184f61ea900d6579af88b4999978ce1e76d9bbbd76cb01b`. Результаты и diagnostics сохранены в корне этого цикла. Пользовательский live runtime не перезапускался.
