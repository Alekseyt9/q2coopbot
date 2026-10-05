# RNG и абсолютное время начала combat fixture

06.10.2026. Продолжение диагностики после width pilot: одинаковые seed и командные префиксы не обеспечивали одинаковый входящий урон.

Native release прежде сбрасывал game-DLL PRNG перед `ClientBeginServerFrame`, обработкой entities и `ClientEndServerFrames`. Idle weapon pause frames вызывают `randk()`, поэтому остаток release-кадра способен расходовать уже сброшенный генератор. Добавлен test-only post-frame seed reset и диагностический cursor receipt. Cursor — индекс в PRNG, не полный fingerprint/snapshot. Go offline replay и harness проверяют ровно один receipt с seed и release frame; legacy manifest без `post_frame_rng_reset` сохраняет старый путь проверки.

Пилот `workspace/artifacts/combat-reset-postframe-v1-20261006`: два batch по четыре инстанса, x2, seeds 14900–14903, одинаковые deterministic weights SHA исходного stochastic parent `9920296b170acaa1f184f61ea900d6579af88b4999978ce1e76d9bbbd76cb01b`, обычные 175 HP, game-frame cap 300. Логи показали cursor_before 258/259 в части release кадров, вместо исходного 256; post-frame reset везде вернул 256. Два seed повторились по командам/здоровью, два разошлись на step index 80. Входящий урон первой жизни: 92/0/67/0 против 100/0/74/0. Утечка RNG внутри release frame подтверждена, но она не была единственным ограничением.

Следующая гипотеза — разные абсолютные времена начала боя. Release происходил на game_frame 32/33/37. Native pain/attack/idle timers сравниваются с абсолютным `level.time`, включая строгие float сравнения. Точная причина позднего расхождения не локализована, поэтому перенос всего мира во времени не применяется.

Добавлен опциональный `g_test_combat_release_frame`: default 0 сохраняет прежний старт по готовности; `-ReleaseGameFrame 100` ждёт естественного достижения frame 100 при замороженном monster AI. Поздняя готовность после указанного кадра отклоняется, чтобы не выдавать разные времена за одинаковый старт. Native флаг разрешён только single-client cheats fixture. Harness ограничивает его isolated synchronous сценариями и game-frame cap >=150; контрольный receipt проверяется также Go exporter. Тест не меняет clocks, physics, AI во время боя или пользовательский live.

Второй пилот `workspace/artifacts/combat-reset-fixedframe-v1-20261006`: те же seed/веса/HP/cap, два batch, фиксированный release frame 100 и post-frame RNG reset. Общий game-frame cap включает подготовку, поэтому длительность боевого участка отличается от раннего-start пилота; между двумя повторениями условия совпадают. Оба изменения — границы тестовой инициализации, не помощь learned policy и не доказательство полного world snapshot.

Сборка native game/q2ded прошла существующим MinGW/CMake; `go test ./...` и PowerShell parser прошли. Проверки receipt отвергают missing/duplicate/wrong frame/seed/cursor и numeric overflow. Никаких скачиваний, пользовательский live runtime сохранён.

## Открытая фаза оружия и бюджет

Второй пилот не обеспечил равные первые команды: первый provider observation отличался только `gun_frame` (например, 9 vs 16), а этот UDP-visible параметр входит в feature vector. Значит, «разные результаты при одинаковых seed» нельзя было полностью относить к скрытому engine state: сама сеть получала разные входы. Также общий cap включал разную длительность подготовки, и provider участки имели 206–208 кадров.

Для opt-in fixed-frame isolated Blaster fixture добавлена однократная нормализация **готового** оружия к stock idle frame 9 после release `ClientEndServerFrames`. Оружие не переключается, патроны/урон/прицел/кнопки не меняются; во время боя gunframe не переписывается. Native receipt `g_test_weapon_start` и Go PPO replay проверяют стартовую фазу. Другие loadout fixed-frame harness не принимает. Default release_frame=0 не нормализует gunframe.

В synchronous combat barrier клиент сбрасывает начало game-frame budget при `test_combat_go`; следующий отправленный command начинает новый бюджет. Prep больше не сокращает боевой участок. Это test-only изменение Go harness path; ordinary/live/non-synchronous clock accounting сохраняется. Накопленные прежние отчёты не переписываются, длительности старых и новых циклов нельзя считать одинаковыми по одному значению cap.

Третий пилот `workspace/artifacts/combat-reset-canonical-v1-20261006`: fixed release 100, initial Blaster gunframe 9, post-frame seed reset и 300 post-barrier кадров; те же четыре seed/веса и два повторения. Проверка — равные командные потоки и UDP-visible motion/health/enemies на общих provider кадрах, с отдельным учётом границ и frame count. Даже успешный результат ограничен этим fixture и не доказывает полную воспроизводимость движка или snapshot-MCTS.

## Результат canonical repeat

Все восемь capture valid/provenance valid. Для seeds 14900/14901/14903 команды совпали на всех 301 provider observations; одинаковые границы server_frame 98–398 и received damage 98/5/0 соответственно в обоих повторениях. 301 observation включает последнюю границу окна; лимит движения — 300 post-barrier кадров.

Seed 14902 остаётся невоспроизводимым: здоровье расходится в observation server_frame 121 (100 vs 95), и команды становятся разными после получения этого observation. До этого командные префиксы совпадают. Полученный урон 64 vs 73; owner guard дополнительно сокращал provider участок одного повторения до 251 кадров. Начальная видимость projectile имеет разные entity IDs при одинаковых позиции/скорости; в других двух seeds IDs тоже различаются без расхождения команд. Прямой причинной связи ID и позднего урона этим аудитом не установлено. Имена/номера entities не входят в feature vector, но используются для наблюдаемого tracking; полная entity-generation/AI/world equivalence по-прежнему не подтверждена.

Итог — исправлены подтверждённые источники разных **начальных условий** и разного горизонта; 3/4 командных повторения совпали, полного deterministic engine reset ещё нет. Файл `repeat-audit.json` сохраняет исходные границы и первые различия. Сравнения результатов обучения должны сохранять это ограничение, а не объявлять одинаковые seed достаточным доказательством контрфактуальной воспроизводимости.
