# Продолжение PPO на смешанной группе с entity inputs v4

06.10.2026. Продолжение [entity types v4](learned_combat_entity_types_v4.md). Цель этого ограниченного опыта — проверить первые обновления на Parasite+Gunner, используя individual type/motion/facing/bbox и observed group composition. MLP810→64→64/fixed Blaster; reward v4 сохраняет Parasite-specific spacing potential, общий damage/kill/death и bbox aim potential. Полной тактики для произвольных групп пока нет.

Первое ранее выполненное mixed обновление353 transitions фиксировано, SHA `1a3c07712f93d7e1980a99bba15d1fb39862cca470535b9138d30b67514d9e44`. Его отложенная deterministic оценка против parent zero-input fork использует новые seeds16200–16203,4 independent instances x2. Eval не обучает, промежуточные модели по нему не выбираются. Root `workspace/artifacts/combat-entity-types-v4-first-eval-20261006`.

Первый paired eval завершён: valid capture/native dispatch/provenance8/8, совпадение manifest conditions/source fingerprints. Incoming400→400, outgoing120→120, deaths4→4/kills0→0, first-life handoff0 во всех8 эпизодах. Provider first-life frames по seeds16200–16203:43→54,49→70,50→91,111→81, суммарно253→296. Это ограниченное изменение длины эпизода, не улучшение побед/добивания. Independent `paired-audit.json` сохраняет effects по target class, owner counts и SHA step records; `diagnostics.json` содержит provider-only motion/aim. Available provider monster-damage reward составляет0.3 в каждом эпизоде, соответствует30 native health damage; урон не приписывается скрытому учителю.

Provider-only diagnostics первого eval: all253→296 Parasite range samples внутри288, mean horizontal range175.50→169.20. Bbox angle error>15° у215/227→284/296 visible attack samples; это геометрическая ошибка после applied angles, не измерение hit accuracy. First-life native received health damage от Gunner268→183, от Parasite132→217. Следовательно увеличение суммарных frames не означает удержание дистанции/лучший прицел; наблюдается перераспределение урона между типами.

До дальнейшего запуска задан budget **4 дополнительных PPO updates**, training seeds16300–16315, по4 fresh synchronous server/client instances x2;300 game frames, stock HP, Mixed Parasite+Gunner. Начальный model/checkpoint — указанное первое mixed обновление. Fixed fourth update сравнивается с этим начальным model на held-out seeds16400–16403. Root `workspace/artifacts/combat-ppo-mixed-types-v4-20261006`. Подбор checkpoint по eval не выполняется. Eval seeds не пересекаются с текущими training seeds16100–16103/16300–16315 или предыдущими одиночными15700–16003.

`scripts/run_combat_ppo.ps1` теперь передаёт `-Mixed` в training и evaluation через существующий harness. Mixed запрещает curriculum HP/fixed release, которые поддерживаются только isolated fixture. Manifest pair validation сохраняется. Для Mixed fixture criterion требует kill обоих классов в каждой из4 after-eval first lives без observed death; это диагностический criterion, live promotion автоматически не выполняется. Rules-specific Shotgun acceptance не используется как критерий learned Blaster.

Проверки до сбора: PowerShell parser; unchanged Go actor/decoder уже прошли `go test ./...`,20 Python checks и4 mixed live capture в предыдущем этапе. Objective SHA `3fb87a9f771a2cca578417fe17530bd3bcd70576540a5fa65dfdb8166b2ea005`; оба optimizer/RNG/history/counters продолжаются из checkpoint без повторного использования rollout. CPU/CUDA выбираются по замеру каждого batch. Source fingerprints фиксируются harness; исходники не редактируются во время capture. Full engine reset не доказан, карта/расстановки ограничены fixture; first-life damage включает подготовку/rules handoff, которые проверяются отдельно.

Четыре дополнительных training updates завершены:16 independent episodes,1477 fresh transitions (360/299/390/428),36 accepted actor steps (10/6/10/10). Во втором batch KL guard отклонил следующий шаг и остановил actor после6 accepted steps; optimizer rollback включён, final KL0.00997. Остальные final KL0.00637/0.00958/0.00963 (точные значения сохранены в progress reports). Все training first lives завершились observed death, kills0; успешное обучение пайплайна не означает победу в fixture.

Fixed final weights SHA `7bbd24cd1fb8e1697d7c8231088bb7dc4deed373d5ece65224ae4c9e44abb297`; cumulative50 updates/492 actor steps. CPU был быстрее CUDA на каждом batch. Checkpoint продолжает policy/value/optimizer/RNG/consumed history; исходные model/checkpoint сохранены.

Final checkpoint SHA `09bd96924e371ba5fa805ee77c48339b9c9952135867116a12b8a2ffc839264c`, путь `workspace/artifacts/combat-ppo-mixed-types-v4-20261006/iteration-4/update/checkpoint.pt`.

| Held-out seed16400–16403 | Outgoing до → после | Incoming до → после | Provider first-life frames до → после |
|---|---:|---:|---:|
|16400|30 → 20|100 → 100|62 → 77|
|16401|30 → 20|100 → 100|95 → 74|
|16402|30 → 20|100 → 100|80 → 60|
|16403|30 → 20|100 → 100|85 → 59|

Итог второго paired eval: deaths4→4/kills0→0, outgoing120→80, incoming400→400; все8 first lives без handoff. Summed provider frames322→270. Mean Parasite range170.48→236.96, но все322→270 observed samples всё ещё внутри288; bbox angular error>15° у310/322→207/215 attack samples. Увеличение дистанции не привело к победам и сопровождалось ухудшением нанесённого урона. Fixed final model не удовлетворяет Mixed criterion и не назначен для live.

По native attacker class incoming от Parasite241→182, от Gunner159→218. Следовательно улучшение одного spacing prior в группе может сопровождаться большим воздействием другого противника; простого суммирования индивидуальной безопасной дистанции как готовой тактики этот опыт не подтверждает.

Independent `paired-audit.json` подтверждает manifest matching/provenance/dispatch/seed pairs и0 first-life handoff. `shaping-audit.json` проверяет native provenance всех24 training/eval episodes, aim+spacing available reward components и точный consumed score всех1477 rows, max error8.9e-16; training kill rewards0. Нельзя считать положительный shaping победой или качество actor достаточным из-за ненулевых type weights.

Практический следующий шаг: адресовать удержание прицела и sparse kill progress отдельным коротким упражнением/контрольным curriculum, затем вернуться к смешанным составам и отдельно проверить сохранение одиночного маневрирования. Повторение той же короткой Mixed серии пока не дало оснований ждать побед; больше ёмкости/GRU сами по себе этой проверкой не обоснованы. Веса/rollout сохранены для продолжения. PS parser и negative Mixed+HP guard прошли; runtime четыре training batch и два eval batch подтвердили передачу Mixed. Пользовательские live marker/config не изменены, порты33100–33103 после прогона свободны.
