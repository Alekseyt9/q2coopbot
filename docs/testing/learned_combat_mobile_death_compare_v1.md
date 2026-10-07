# Mixed: контроль штрафа за смерть, 07.10.2026

**Последующая проверка:** [32 новых paired seeds](learned_combat_mobile_control_validation_v1.md) дала3/32 победы checkpoint16 против2/32 control и28 против30 смертей. Предварительный выбор control ниже относится только к этому pilot и отменён новым selection.json; текущий training baseline — checkpoint16. Death−5 сохраняется, live/default не переключён.

Сравнение проверяет одну гипотезу: увеличение death penalty с −5 до −10 помогает завершить Mixed бой и сохранить жизнь. Parent — Temporal attention checkpoint16 из `combat-mobile-machinegun-curriculum-v1r2-20261007`; обе ветки начинают с одинаковых actor/critic weights и fresh Adam. Контроль со свежим Adam отделяет смену награды от сброса optimizer. Kill reward остаётся +5, остальные reward/PPO параметры неизменны; objective SHA закреплён в отдельном config.

На ветку: четыре CUDA updates, 32 учебных эпизода Mixed, по четыре сервера/клиента x2, два эпизода на worker в batch. Оценка до/после: 16 общих сидов, по четыре эпизода на worker. Machinegun со stock 100 bullets, skill1, stock HP, z24.125, native no-infighting; ранняя остановка после двух подтверждённых убийств на первой жизни. Ветки выполняются последовательно, максимум четыре активных инстанса. Каждый эпизод имеет отдельный seed; совпадение сидов между ветками намеренное для сравнения. Полная детерминированность AI/мира не заявляется.

## Найденный дефект delayed splash

Первый опыт `combat-mobile-death-compare-v1-20261007` остановился на обязательной no-infighting проверке после контрольного обучения. Seed37403, network server_frame251: граната убитого Gunner нанесла живому Parasite 5 HP (`MOD_G_SPLASH=7`, 175→170). Gunner был убит telefrag на frame236 после смерти бота; при смерти потерял `SVF_MONSTER`. Проверка двух текущих SVF flags пропустила оставшуюся гранату. Это реальное нарушение fixture, хотя событие произошло после первой жизни и не является новым обучающим переходом.

В `yquake2/src/game/g_combat.c` запрет учитывает classname `monster_` у владельца, потерявшего SVF. У живой цели по-прежнему требуется `SVF_MONSTER`; player damage и stock mode сохраняются. Проверка `scripts/test_native_no_infighting.ps1` компилирует действующее условие из native source и проверяет live/dead owner, corpse target, player/world/null classname и границы fixture. Game module собран CMake. Старый runtime DLL сохранён отдельным файлом перед линковкой, чтобы не перезаписать hardlinked исторические captures или активный playback.

Первый опыт не принят и не используется для выбора: `invalidated.json` фиксирует причину. Его четыре контрольных updates сохранены только диагностически; ветка death−10 не начиналась.

Повтор `combat-mobile-death-compare-v1r2-20261007` использует новые training seeds38000–38031 и evaluation seeds38400–38415. Обе ветки полностью повторяются на исправленной native сборке; rollout из первого опыта не переиспользуется. Checkpoint16 сохранён как reference; итоговый выбор следующего training candidate описан ниже, live/default policy не переключена.

## Протокол выбора

Полные Mixed победы первичны. При равенстве оцениваются deaths и убийства вместе; damage, ammo proxy и reward score сами по себе не основание для замены. Reward score между разными death penalties напрямую несопоставим. Аудит сверяет исходные веса, fresh Adam/последующие checkpoint seals, CUDA reports, seed provenance, мобильность обоих классов, native damage и goal receipts. Before evaluations сохраняются для проверки воспроизводимости baseline.

Повтор завершён, обе ветки и сравнительный аудит приняты.


## Итог повторного опыта

128/128 native captures приняты, восемь CUDA updates завершены. Control:3900 eligible rows,40 actor steps; death−10:3747 rows,39 actor steps (последний update9 из-за KL). Максимальный бюджет одинаковый, фактически принятые шаги различаются вследствие ограничителя KL. Fresh Adam проверен в обеих ветках; optimizer resume выполнялся только между updates внутри своей ветки. Config отличается только objective SHA, reward — только death. Политика исполнялась Go-клиентом.

| Модель,16 Mixed эпизодов | Победы | Убийства | Monster HP damage | Получено HP damage | Смерти |
|---|---:|---:|---:|---:|---:|
| Parent16, before обеих веток | 1 | 8 | 2488 | 1451 | 13 |
| Control −5, after | 3 | 9 | 2759 | 1376 | 12 |
| Death −10, after | 2 | 8 | 2910 | 1497 | 13 |

Before боевые показатели совпадают для каждого из16 сидов, включая class kills, damage, received, death и win. Actual frames ранней остановки38400 отличаются75/72; старт/закрытие client trace также могут отличаться. Это подтверждает повторение измеренных боевых итогов, а не полную engine/AI state equivalence.

### Все16 оценки

Каждая ячейка: damage / kills / win / death. Before боевые итоги одинаковы между ветками. Победа требует убийства Parasite и Gunner до первой смерти с положительным HP; props и corpse damage не учитываются.

| Seed | Parent16 | Control −5 | Death −10 |
|---|---|---|---|
| 38400 | 350 / 2 / да / нет | 350 / 2 / да / нет | 200 / 0 / нет / да |
| 38401 | 120 / 0 / нет / нет | 96 / 0 / нет / да | 136 / 0 / нет / да |
| 38402 | 175 / 1 / нет / да | 223 / 1 / нет / да | 120 / 0 / нет / да |
| 38403 | 64 / 0 / нет / да | 80 / 0 / нет / да | 80 / 0 / нет / да |
| 38404 | 48 / 0 / нет / да | 48 / 0 / нет / да | 88 / 0 / нет / да |
| 38405 | 48 / 0 / нет / да | 128 / 0 / нет / да | 152 / 0 / нет / да |
| 38406 | 144 / 0 / нет / нет | 48 / 0 / нет / да | 88 / 0 / нет / нет |
| 38407 | 191 / 1 / нет / да | 175 / 1 / нет / да | 231 / 1 / нет / да |
| 38408 | 223 / 1 / нет / да | 112 / 0 / нет / да | 350 / 2 / да / нет |
| 38409 | 184 / 0 / нет / да | 104 / 0 / нет / нет | 128 / 0 / нет / да |
| 38410 | 239 / 1 / нет / да | 350 / 2 / да / нет | 350 / 2 / да / нет |
| 38411 | 175 / 1 / нет / да | 335 / 1 / нет / да | 96 / 0 / нет / да |
| 38412 | 56 / 0 / нет / да | 350 / 2 / да / нет | 293 / 1 / нет / да |
| 38413 | 64 / 0 / нет / да | 192 / 0 / нет / да | 255 / 1 / нет / да |
| 38414 | 144 / 0 / нет / да | 72 / 0 / нет / да | 168 / 0 / нет / да |
| 38415 | 263 / 1 / нет / да | 96 / 0 / нет / да | 175 / 1 / нет / да |

Control сохранил победу38400 и добавил38410/38412. Death−10 потерял38400, выиграл38408/38410. Рост damage до2910 в treatment не компенсирует меньшее число полных побед и больший received damage.

### Диагностика

control: native incoming Gunner 738→597, Parasite 713→779. Observed alive MG rounds 760→846; damage/round proxy 3.274→3.261. Nearest clear-shot angle с observed camera kick 15.34°→19.79°, min-angle any clear-shot 14.26°→17.60° на 677→805 attack observations.

death10: native incoming Gunner 738→815, Parasite 713→682. Observed alive MG rounds 760→1019; damage/round proxy 3.274→2.856. Nearest clear-shot angle с observed camera kick 15.34°→13.09°, min-angle any clear-shot 14.26°→12.17° на 677→954 attack observations.

Treatment лучше по angle proxy, но хуже по победам/выживанию и ammo ratio. Разные длительности жизней/наборы attack samples не позволяют считать эти proxy точным hit rate или доказательством causal dodge/aim skill. HUD ammo delta excludes death-tick ambiguity. Политика здесь использует фиксированный MG; learned выбор оружия не проверяется.

Восемь ранних goal stops сэкономили 1709 active game frames из максимальных38400 (4.45%); это доля frame budget, не измеренный wall-time speedup. Failed episodes пока доходят до максимума; stop-on-first-death остаётся отдельной оптимизацией, не добавленной внутри сравнения.

## Выбор и следующий этап

Death−10 не выбран: его выигрыш относительно parent меньше контрольного, deaths не улучшились. Death penalty остаётся−5. Control после четырёх fresh-Adam updates выбран кандидатом для продолжения обучения; checkpoint16 сохранён reference. Это16-seed pilot, не статистически доказанное обобщение. Live/default policy не переключена. Перед следующими updates приоритет — более широкая независимая оценка контрольного кандидата против16; затем mixed training на ошибках, особенно выживание и завершение пары.

Selected weights SHA256 `68012b61cbba699f119d9f6216565fd2ee326651eb2529e219ebd7db513c43cc`; checkpoint SHA256 `911742d44f483a9233688ca9c287c320c60ef706b62e9d1a168cf3593c5b419b`. Счётчик optimizer branch=4 updates/40 actor steps, parent genealogy16+4; это новая ветка с fresh Adam, а не возобновлённый optimizer checkpoint20 из отклонённого Mixed-only опыта.

Evidence: `workspace/artifacts/combat-mobile-death-compare-v1r2-20261007/`: comparison-protocol.json, comparison.json, diagnostics.json, selection.json; arm audit.json, ammo-before/after.json, complete.json и goal receipts. Source fingerprint `b62ac54f6e00759a013e07942da000ca0856b20e18c47da6df3252e95464e0f0`; native fingerprint `a6eb88e75484c1aadd47dc07ba69257d2c003f75bc77b15138e438c707fdd53a`.
