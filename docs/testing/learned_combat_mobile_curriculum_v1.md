# Mixed/Solo curriculum: Temporal attention, Machinegun, early goal stop

07.10.2026. Завершён ограниченный GPU pilot; это не общая боевая приёмка.

Parent — recoil checkpoint12 с aim_potential0.5; итог — checkpoint16. Adam восстановлен из проверенного matching checkpoint. Четыре CUDA PPO updates на RTX5070,4 инстанса x2,48 независимых training seeds35100–35147. В каждом batch:8 Mixed и4 Solo Parasite, порядок Mixed→Solo→Mixed на каждом инстансе. Оружие — stock Machinegun с100 bullets; stock monster health/skill1, исправленные позиции z24.125, training-only no-infighting. Architecture Temporal attention и features v5 неизменны. Это обучение движению/прицеливанию/огню с фиксированным оружием; обучение выбору оружия этим опытом не подтверждается.

Early goal stop действует на Solo после убийства Parasite, на Mixed после убийства Parasite и Gunner. Следующий наблюдённый кадр сохраняет последнюю reward/terminal boundary;300 frames — максимум. Native evidence хранится отдельно от policy observations. Проверка PPO replay исправлена: повторно подтверждает goal receipt по native events и release, воспроизводит goal boundary и сравнивает SHA steps/rewards/outcomes. Первый диагностический batch35000–35011 не использован optimizer.

## Обучение и integrity

| Update | Eligible rows | Actor steps | Report seconds |
| --- | --- | --- | --- |
| 1 | 1329 | 9 | 1.76 |
| 2 | 1643 | 10 | 1.43 |
| 3 | 1646 | 10 | 1.36 |
| 4 | 1651 | 10 | 8.52 |

Всего6269 rows и39 accepted actor steps; cumulative16 recoil updates/150 actor steps. Проверены80 native captures (48 training+32 evaluation), единые source/native fingerprints,48 уникальных training seeds,16 других paired evaluation seeds35400–35415, движение нужных классов, отсутствие live monster infighting, Go/Torch logprob/value parity, KL, matching optimizer step counts и complete.json SHA. Reference retention/bank losses0; их provenance/seed split проверен, старые stationary rollouts не являются training loss. Проверки Go product packages, supervisor и native goal verifier прошли.

Training:10/16 Solo goals и3/32 Mixed goals. Среди всех80 эпизодов14 early goal stops, сохранено2623 активных кадра относительно максимума300 на каждом эпизоде. Это10.9% максимального суммарного frame budget; wall-time ускорение не измеряется этим числом. Неудачные эпизоды пока продолжаются до прежнего максимума.

## Paired evaluation на16 свежих Mixed seeds

| Метрика | Parent12 | Candidate16 |
| --- | --- | --- |
| Полные победы | 0 | 1 |
| Убийства монстров | 1 | 6 |
| Урон монстрам | 2039 | 2538 |
| Полученный health damage | 1569 | 1578 |
| Смерти | 14 | 15 |
| Наблюдаемый alive расход MG | 858 | 885 |
| Урон / наблюдаемый alive расход | 2.38 | 2.87 |

Урон вырос на24.5%; отношение урона к наблюдаемому alive расходу — на20.7%. Это диагностическое отношение, не exact hit rate: death-tick ammo неизвестен, выстрелы по props/world включены, длительности жизней различаются. Победа требует обеих native monster kills и положительного здоровья без первой смерти; kills props и corpse damage исключены.

| Seed | Урон до/после | Убийства до/после | Победа до/после | Смерть до/после |
| --- | --- | --- | --- | --- |
| 35400 | 248/239 | 0/1 | нет/нет | да/да |
| 35401 | 176/176 | 0/0 | нет/нет | да/да |
| 35402 | 88/112 | 0/0 | нет/нет | да/да |
| 35403 | 80/72 | 0/0 | нет/нет | да/да |
| 35404 | 80/215 | 0/1 | нет/нет | да/да |
| 35405 | 136/64 | 0/0 | нет/нет | да/да |
| 35406 | 104/192 | 0/0 | нет/нет | да/да |
| 35407 | 80/64 | 0/0 | нет/нет | да/да |
| 35408 | 88/136 | 0/0 | нет/нет | да/да |
| 35409 | 80/303 | 0/1 | нет/нет | нет/да |
| 35410 | 223/263 | 1/1 | нет/нет | да/да |
| 35411 | 144/350 | 0/2 | нет/да | да/нет |
| 35412 | 152/96 | 0/0 | нет/нет | да/да |
| 35413 | 72/72 | 0/0 | нет/нет | да/да |
| 35414 | 48/48 | 0/0 | нет/нет | да/да |
| 35415 | 240/136 | 0/0 | нет/нет | нет/да |

Candidate выиграл35411: native Parasite kill network frame169, Gunner205, supervisor observed207; конечное здоровье22. Клиент завершился после110 активных game frames вместо300. Это единственная полная победа в heldout batch.

## Решение и следующий приоритет

Checkpoint16 сохраняется как кандидат для дальнейшего обучения: kills1→6 и wins0→1 дают полезный сигнал. Live/default policy не переключена:1/16 побед, смертей14→15, поэтому устойчивость группы и выживание не доказаны. Сравнение со старыми16 seeds34000–34015 не является paired сравнением — текущие seeds другие. Следующий приоритет — маневрирование/выживание в Mixed и проверка на дополнительных свежих сидах; хороший Solo результат не заменяет group combat.

Evidence: ignored `workspace/artifacts/combat-mobile-machinegun-curriculum-v1r2-20261007/`: audit.json, training-audit.json, per-update complete.json, ammo-before/after.json, batches/goal-stop.json, selection.json. Native fingerprint29c039d7deed79f90271b87c7c9257f982355688a05fffa9d8c20990027c4027; source fingerprint e1506bf0983a9693e5040c224f457b2d5a97084b3e1e085d0f23a09383c62fb3.


07.10.2026: следующий controlled pilot `combat-mobile-machinegun-mixed-v1-20261007` продолжает checkpoint16/Adam. Четыре CUDA updates,4 инстанса x2,2 независимых эпизода на инстанс в каждом batch:32 training seeds36000–36031, все Mixed. Это сохраняет число Mixed эпизодов предыдущего curriculum, убирая16 Solo эпизодов. Архитектура, objective aim_potential0.5, stock equipment/positions/skill и early goal stop прежние. Paired before/after evaluation на16 новых Mixed seeds36400–36415. Приоритет — полные победы и выживание; live/default promotion не выполняется по одному удачному эпизоду.


Дополнительная offline aim диагностика того же paired batch35400–35415:817→857 alive first-life Machinegun attack-command observations. Sample-weighted угол до nearest known clear-shot target с observed camera kick19.24°→16.66°, минимальный угол до любой known clear-shot цели16.38°→15.39°. Exact geometry используется из Go policy. Не является bullet hit rate: камера показывает также damage/fall punch, выборка frames различается по времени жизни/числу выстрелов, bbox center proxy не равен попаданию в объём. Evidence: aim-paired.json; diagnostic helper сохранён в recoil-aim pilot.


Offline survival diagnostics той же paired оценки: native incoming health damage от Parasite844→556, от Gunner725→1022. Число policy movement command frames1773→1409; actual next-frame movement steps1056→695. Длительности жизней/достижение целей различаются; эти totals не доказывают причинность или отдельный навык уклонения. Evidence: survival-diagnostics.json. Смена фокуса на Mixed в следующем опыте проверяется свежим paired batch.


Mixed-only pilot завершён:4 updates,2755 rows,64 captures; paired36400–36415 wins2→2, kills6→8, damage2490→2656, deaths13→14. Checkpoint20 не выбран как новый training baseline; остаётся checkpoint16. Полная таблица и diagnostics: [Mixed-only pilot](learned_combat_mobile_mixed_v1.md).
