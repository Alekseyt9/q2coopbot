# Mixed-only GPU pilot после curriculum

07.10.2026. Завершён controlled pilot на исправленных mobile fixtures; общая боевая приёмка не получена.

Parent — checkpoint16 из Mixed/Solo curriculum. Четыре CUDA PPO updates с matching Adam resume,4 инстанса x2,2 независимых эпизода на инстанс в каждом batch. Training seeds36000–36031:32 боя, все Mixed Parasite+Gunner. Это то же число Mixed эпизодов, что в предыдущем curriculum, без16 Solo эпизодов. Stock Machinegun100 bullets, stock monster health/skill1, z24.125, training-only no-infighting. Temporal attention/features v5 и reward aim_potential0.5 прежние; early goal stop включён,300 frames — максимум.

## Training и проверка целостности

| Update | Eligible rows | Actor steps | Report seconds |
| --- | --- | --- | --- |
| 1 | 713 | 10 | 1.18 |
| 2 | 707 | 10 | 1.18 |
| 3 | 576 | 10 | 1.67 |
| 4 | 759 | 10 | 8.60 |

Использованы2755 rows,40 actor steps; cumulative20 recoil updates/190 actor steps. Проверены64 captures (32 training+32 paired evaluation), уникальные seed assignments, общий source/native fingerprint, command execution/reset/mobile proof, native no-infighting, goal terminal rewards, Go/Torch logprob/value parity, KL, matching optimizer steps/consumed-rollout chain и durable complete receipts. Retention/bank losses0, старые stationary rollouts не используются как losses. Product code в этом цикле не менялся; тот же source fingerprint, что в предыдущем curriculum.

Training goals по batch:0/8,2/8,1/8,1/8. Всего8 early goal stops среди64 captures,1588 активных кадров сохранено относительно максимального frame budget19200 (8.3%); это не измерение wall-time ускорения.

## Paired evaluation на16 новых сидах36400–36415

| Метрика | Parent16 | Candidate20 |
| --- | --- | --- |
| Полные победы | 2 | 2 |
| Убийства монстров | 6 | 8 |
| Урон монстрам | 2490 | 2656 |
| Полученный health damage | 1456 | 1489 |
| Смерти | 13 | 14 |
| Наблюдаемый alive расход MG | 730 | 653 |
| Урон / наблюдаемый alive расход | 3.41 | 4.07 |

Урон вырос на6.7%, наблюдаемый расход снизился на10.5%; damage/ammo proxy вырос3.41→4.07. Это не точный hit rate: death tick расход неизвестен, выстрелы по props/world включены, длительности жизней различаются. Полная победа определяется native kills обоих классов, первой жизнью и положительным minimum health, не исчезновением из PVS.

| Seed | Урон до/после | Убийства до/после | Победа до/после | Смерть до/после |
| --- | --- | --- | --- | --- |
| 36400 | 64/168 | 0/0 | нет/нет | да/да |
| 36401 | 96/80 | 0/0 | нет/нет | да/да |
| 36402 | 192/239 | 0/1 | нет/нет | да/да |
| 36403 | 64/72 | 0/0 | нет/нет | да/да |
| 36404 | 64/64 | 0/0 | нет/нет | да/да |
| 36405 | 112/240 | 0/0 | нет/нет | да/да |
| 36406 | 64/199 | 0/1 | нет/нет | да/да |
| 36407 | 80/192 | 0/0 | нет/нет | нет/да |
| 36408 | 72/80 | 0/0 | нет/нет | да/да |
| 36409 | 168/120 | 0/0 | нет/нет | да/да |
| 36410 | 350/350 | 2/2 | да/да | нет/нет |
| 36411 | 295/350 | 1/2 | нет/да | да/нет |
| 36412 | 72/56 | 0/0 | нет/нет | да/да |
| 36413 | 350/80 | 2/0 | да/нет | нет/да |
| 36414 | 272/191 | 0/1 | нет/нет | да/да |
| 36415 | 175/175 | 1/1 | нет/нет | да/да |

Обе модели выигрывают36410. Candidate получил победу36411, но потерял победу36413; рост общего числа побед не подтверждён. Native incoming health damage: Gunner717→810, Parasite739→679. Это totals для разной длительности жизней; отдельный causal dodge skill не выводится.

Дополнительный aim proxy на666→642 alive first-life MG attack-command observations ухудшился: sample-weighted nearest clear-shot angle с observed kick17.12°→19.03°, минимальный угол до любой known clear-shot цели15.10°→17.15°. Это geometry по Go policy, не bullet hit rate или намеренно выбранная цель. Смена выборки и camera punch ограничивают интерпретацию; лучший damage/ammo ratio не означает улучшение каждой aim метрики.

## Решение

Основной training baseline остаётся checkpoint16. Checkpoint20 сохранён для анализа, live/default policy не переключена. Selection protocol был записан до получения after results: полные Mixed wins первичны; при равенстве учитываются kills вместе со deaths, damage/aim shaping сами по себе недостаточны. При2→2 wins,13→14 deaths и потере36413 кандидат не заменяет parent. Текущий priority — выживание под Gunner огнём и завершение всей группы.

Позднейший этап07.10: [death-penalty compare](learned_combat_mobile_death_compare_v1.md) завершён отдельно. Первый compare остановился на delayed grenade splash от мёртвого Gunner и исключён из выбора; после native исправления обе ветки полностью повторены на новых сидах. Control−5 дал3/16 побед против1/16 parent и2/16 death−10, выбран следующим training candidate. Это отдельная fresh-Adam ветка; отклонённый checkpoint20 этого Mixed-only опыта не стал её parent.

Следующий отдельный single-factor опыт можно проводить с checkpoint16 и более сильным death penalty (сейчас−5 при monster_kill+5), сохранив stock бой и свежую paired оценку. Это гипотеза, не реализованный update и не обещание улучшения. Смена objective потребует отдельного pinned reward/config и свежего Adam; менять её внутри этого завершённого цикла нельзя.

Evidence: `workspace/artifacts/combat-mobile-machinegun-mixed-v1-20261007/`: audit.json, training-audit.json, selection-protocol.json, selection.json, diagnostics.json, ammo-before/after.json, goal-stop receipts и per-update complete.json. Source fingerprint e1506bf0983a9693e5040c224f457b2d5a97084b3e1e085d0f23a09383c62fb3; native fingerprint29c039d7deed79f90271b87c7c9257f982355688a05fffa9d8c20990027c4027.


07.10.2026: запущен death-penalty compare `combat-mobile-death-compare-v1-20261007`. Две ветки — control−5 и death−10, обе warm-start weights/critic от checkpoint16 и fresh Adam,4 CUDA updates на ветку. Число обновлений/episode budgets одинаковы; все параметры reward кроме death и objective SHA неизменны.32 training seeds37000–37031 и16 paired evaluation seeds37400–37415 общие между ветками, внутри каждого batch — отдельный seed на независимый эпизод. Ветки выполняются последовательно с4 инстансами x2. Fresh Adam control нужен, чтобы не смешивать изменение objective со сбросом optimizer. Перед/после evaluation каждой ветки сохраняется полностью; её одинаковый baseline также проверяет повторение prepared reset. Полная world/AI equivalence по-прежнему не заявляется.
