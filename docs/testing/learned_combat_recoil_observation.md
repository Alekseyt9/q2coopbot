# Наблюдаемая отдача пулемёта

2026-10-06. При просмотре подвижного Mixed выяснилось, что UDP decoder пропускал `PS_KICKANGLES`, а policy и aim potential использовали только базовые view angles. В cooperative Machinegun stock `Machinegun_Fire` добавляет `machinegun_shots * -1.5` к pitch, с максимумом9 shots, то есть подъёмом13.5°. Основной aim delta политики не отключает эту механику.

Decoder теперь читает signed quarter-degree kick pitch/yaw/roll, сохраняет delta inheritance и передаёт их в Snapshot, trace и Observation. В full frame отсутствующий ненулевой delta означает наблюдаемый ноль; старые JSON observations без этих полей остаются unavailable. Поле описывает **общую UDP camera punch**: отдачу оружия, встряску от урона и падения. Это не скрытый счётчик очереди и не точное направление каждой пули; hits/damage/kill остаются native ground truth для reward.

Новый `combat_features_v5` сохраняет v4 prefix810 значений и добавляет4: availability mask, kick pitch/yaw/roll в degrees/32. Старые v4 модели работают с прежними признаками. Миграция Temporal mobile update4 расширяет первые Linear layers actor/value четырьмя нулевыми столбцами: исходная функция не получает вручную заданной компенсации. Новый optimizer свежий; сравнение качества проводится только после обучения на новом контракте. Entity attention v1 остаётся v4-only, поскольку его token schema ещё не включает отдачу.

Reward config `combat-reward-recoil-v5.json` сохраняет reward version4 и включает отдельный opt-in `aim_kick_angles:true`: potential использует heading view+наблюдаемый kick. Legacy configs без flag сохраняют прежнюю формулу. Новая конфигурация PPO закрепляет точный reward SHA. Камера является наблюдаемым proxy для aim shaping; доступные точные native damage/kill rewards имеют прежний смысл. Команды policy не исправляются ручным aim controller; recoil в движке не отключён.

Новый цикл `combat-mobile-recoil-v1-20261006`: Temporal attention, Machinegun100 bullets, подвижный Parasite+Gunner, skill1, no-infighting fixture,4 workers x2. Training seeds31000–31015, evaluation31400–31403,4 CUDA updates, без stationary retention loss. Исходные веса и migration receipt лежат в `combat-mobile-recoil-inputs-20261006`; прежний mobile block сохранён отдельно.

Проверки: все product Go packages (`go test ./cmd/... ./internal/...`) прошли, включая signed kick decode/delta inheritance, masked v5 prefix и aim potential с компенсацией. `go test ./...` остановлен во время обхода больших generated artifacts; product scope проверен полностью. CUDA resume CLI integration passed для трёх архитектур.

Цикл завершён: 4 CUDA updates, 1759 eligible rows, 35 accepted actor steps, 160 critic steps. `audit.json` проверяет checkpoint/optimizer/resume chain, уникальность consumed rollouts, native provenance и mobility/no-infighting для 24 принятых captures. `recoil-proof.json` подтверждает v5 во всех весах, закреплённый reward SHA и ненулевой наблюдаемый kick во всех 24 эпизодах (4243 строки с ненулевым kick). Это подтверждение целостности эксперимента, не готовности политики.

Первоначальная evaluation-after была неполной: seed31402 завершился на неподдерживаемом `TE_BUBBLETRAIL` (выстрел в воде). Decoder исправлен по stock protocol: две packed positions, 12 bytes; regression test проверяет следующий message и truncated payload. Обе оценки полностью повторены одной новой сборкой, исходные файлы сохранены. Веса не переобучались. Training source fingerprint `bca6db8538a58b117263ba0bb9c32a0fc029505e5950ce6deae93e91b2e776b9`, evaluation source `301012d1f06211e4505851c66acb4797530dc1d281c2c8bf94f4fe53180be656`; native fingerprint общий `29c039d7deed79f90271b87c7c9257f982355688a05fffa9d8c20990027c4027`. Audit проверяет эти области отдельно, обе evaluation arms используют одинаковый источник.

Детерминированная paired evaluation, только первая жизнь, Machinegun100, подвижные Gunner+Parasite:

| Seed | Урон до | Урон после | Убийства до/после | Смерть до/после |
| --- | ---: | ---: | --- | --- |
| 31400 | 64 | 80 | 0/0 | да/да |
| 31401 | 32 | 32 | 0/0 | да/да |
| 31402 | 112 | 106 | 0/0 | да/нет |
| 31403 | 72 | 216 | 0/0 | нет/да |
| Всего | 280 | 434 | 0/0 | 3/3 |

Победы: 0/4 до и после. Нанесённый урон вырос на55%, но четыре сида недостаточны для устойчивого вывода. Выживший до горизонта бот не считается победившим. Это не метрика точности выстрелов; компенсация recoil пока не доказана отдельно. Следующий этап — больше свежих GPU batches и paired seeds, контроль aim error с учётом kick и расхода патронов, затем отдельное сравнение управления очередями.

Артефакты: `workspace/artifacts/combat-mobile-recoil-v1-20261006`; принятые оценки `evaluation-before-retry` и `evaluation-after-retry`, финальные веса `iteration-4/update/weights.json`. Старый player demo пока показывает предыдущий mobile Machinegun block.

Дополнительная диагностика патронов (`scripts/report_combat_ammo.py`): наблюдаемые уменьшения Machinegun HUD ammo между живыми кадрами составили103→230, урон на такое уменьшение2.72→1.89. Обнуление ammo HUD при смерти исключено — это не расход патронов; regression test подтверждает это. Первый ненаблюдаемый расход и death tick не восстановлены, поэтому отношение диагностическое, не точная hit accuracy. Рост общего урона сам по себе не доказывает улучшение меткости.

Продолжение `combat-mobile-recoil-v2-20261007`: 8 дополнительных CUDA updates,4 workers x2, свежие training seeds32000–32031, paired evaluation32400–32403. Runner теперь принимает `InitialCheckpoint`, проверяет соответствие checkpoint SHA весам и нулевые retention weights, сохраняет копию checkpoint и возобновляет Adam. Первый update подтверждён: updates_completed5, total_actor_steps45 (35 прежних+10 новых), device CUDA, resume SHA совпадает с входным checkpoint. Цикл запущен; итоговые оценки ещё не завершены.

Во втором batch этот runner остановился до optimizer update: строгая проверка log class names сочла corpse hit взаимным уроном. Seed32006: Parasite352 погиб после первой жизни бота (`mod21`, health79→-99921); затем crusher243 вызвал `mod7`, health_before-99921, с Gunner343 в attacker credit. Это не damage живого монстра. Native no-infighting guard не изменён. Проверка runner теперь отвергает monster→monster damage при положительном health_before; положительный и отрицательный случаи regex проверены отдельно. Отклонённый batch32004–32007 сохранён и не потреблён PPO.

Продолжение перенесено в `combat-mobile-recoil-v2r2-20261007`: resume из единственного принятого v2 update, 7 оставшихся updates, training32008–32035, evaluation32400–32403. Первый новый CUDA update завершён (cumulative updates6, actor steps51); полный бюджет продолжения остаётся8 принятых updates с учётом первого v2. Финальная оценка будет сравнивать модель после принятого v2 update с моделью после ещё7 updates; это не прямое сравнение с началом всех8 updates.

Дополнительный offline Go helper `combat-mobile-recoil-v2r2-20261007/aim_report.go` измеряет угол до ближайшего наблюдаемого clear-shot target через тот же `policy.ObservedAimDirection`, что reward. Только живые first-life MG frames с attack command; heading base+общий observed camera kick, не точное направление пуль. Для прежних seeds31400–31403 средний угол с kick до:15.61°,21.01°,19.98°,13.30°; после:13.43°,25.67°,15.86°,16.18°. Результат неоднороден, устойчивое улучшение не доказано. Helper остаётся ignored диагностическим артефактом, gameplay inference не меняет.

07.10.2026: компьютер перезагрузился во время второго CUDA update v2r2. Процесс отсутствует; progress.json и dataset reward-config второго batch заполнены нулями. `verify_reboot.py` проверил последние завершённые weights/checkpoint (updates6, actor steps51) и40 referenced source files их native rollout. Второй rollout541 rows не числится consumed; его provenance SHA не сходится, поэтому весь batch32012–32015 исключён без исправления исходных evidence files. Receipt `v2r2/reboot-integrity.json` относится к последнему целому checkpoint, не к повреждённому batch.

Возобновление `combat-mobile-recoil-v2r3-20261007`: ещё6 CUDA updates,4 инстанса x2, seeds32016–32039, evaluation32400–32403. С учётом одного принятого v2 и одного принятого v2r2 это сохраняет бюджет8 дополнительных updates, итог12 recoil updates. Checkpoint и состояние Adam восстановлены из принятого v2r2 update1. Полная оценка исходных весов перед всеми8 updates потребует отдельного baseline; встроенная paired evaluation v2r3 сравнивает только последние6 updates.

Сохранение новых результатов усилено: `ppo_recurrent.py` пишет partial files, вызывает flush+fsync и публикует окончательные имена, затем создаёт `complete.json` с SHA weights/checkpoint/report. Runner проверяет receipt перед переходом к следующему batch. Это снижает риск принять незавершённые результаты; абсолютная устойчивость файловой системы при отключении питания не заявляется. CUDA resume integration для трёх архитектур прошла после изменения. Portable [PowerShell7.5.3](https://github.com/PowerShell/PowerShell/releases/tag/v7.5.3) для харнеса размещён только наF в ignored `workspace/tools/dev_tools/powershell-7.5.3`, ZIP SHA31588931DFCB752D1943F5E633A55337E2F12AF0803D670DB6D90C5937222818 сверён с upstream.
