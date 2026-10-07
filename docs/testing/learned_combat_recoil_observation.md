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

## Итог восьми дополнительных обновлений

07.10.2026: завершены все8 принятых дополнительных CUDA updates (76 actor steps,3195 eligible rows). Итого recoil branch:12 updates,111 actor steps. `ancestry-proof.json` проверяет цепочку weights/checkpoint/consumed rollouts через v2→v2r2→v2r3. `audit.json` проверяет6 updates v2r3, seal каждого update, optimizer closure, одинаковые source/native fingerprints и32 принятых captures; `recoil-proof.json` подтверждает наблюдаемый ненулевой kick во всех32 эпизодах. Acceptance этих receipts относится к целостности опыта, не к боевому качеству.

Coordinator завершился перед оценкой. Sandbox retry не смог создать shared PAK links, игровые captures отсутствовали; пакет сохранён как исключённый. После разрешённого запуска вне песочницы принята `evaluation-before-native`. Смена режима доступа прервала after workers до готового report; принят отдельный повтор `evaluation-after-final`. Все неполные каталоги сохранены и исключены. Проверенные обе оценки имеют тот же source fingerprint, что training, и одинаковый native fingerprint. До — модель перед всеми8 additional updates (`v2/initial.json`), после — `v2r3/iteration-6/update/weights.json`; это не сравнение только последних6 updates.

| Seed | Урон до/после | Наблюдаемый расход MG до/после | Убийства до/после | Смерть до/после |
| --- | --- | --- | --- | --- |
| 32400 | 128/168 | 100/100 | 0/0 | нет/нет |
| 32401 | 56/56 | 19/22 | 0/0 | да/да |
| 32402 | 80/72 | 33/19 | 0/0 | да/да |
| 32403 | 104/64 | 100/12 | 0/0 | да/да |
| Всего | 368/360 | 252/153 | 0/0 | 3/3 |

Победы остаются0/4. Урон на наблюдаемый alive расход1.46→2.35; это диагностическое отношение, а не hit rate (death tick не восстановлен, разные длительности жизней и цели). Угол heading+camera kick до ближайшего clear-shot target на alive MG attack frames по этим сидами: до11.46°,15.69°,11.69°,18.42°; после14.02°,15.60°,13.13°,10.69°. Улучшение угла неоднородно, успешный бой не доказан. Все оценки лежат в `combat-mobile-recoil-v2r3-20261007`, ammo/aim JSON рядом с audit receipts.

В1900 действительно использованных PPO rows последних6 updates средний абсолютный вклад aim shaping0.01023, spacing0.06737, damage0.01719. `reward-diagnostics.json` соединяет reward и rollout по seed/index, исключает неиспользованные строки. Это показывает относительный размер компонентов, не доказывает причину ошибок стрельбы.

Следующий single-factor pilot `combat-mobile-recoil-aim-v1-20261007`: aim_potential0.5→1.0 в допустимом диапазоне существующего reward contract. Native damage/kill/death, spacing, gamma, оружие, состав, positions, horizon, architecture и feature schema неизменны. Отдельные pinned reward/PPO configs `combat-reward-recoil-aim-v5.json` / `combat-ppo-recoil-aim-v5.json`; reward schema остаётся combat_reward_v4, features v5. Parent — проверенный recoil update12; Adam свежий из-за смены objective,4 CUDA updates,4 инстанса x2, training33000–33015, paired evaluation33400–33403. Условие продвижения: native kills/wins и диагностика стрельбы; рост shaping score без улучшения боя не считается успехом. Pilot запущен, результатов пока нет.

## Усиление aim shaping: завершённый короткий опыт

07.10.2026: pilot завершён,4 CUDA updates,31 принятый actor step,1490 eligible rows. Проверены seals, optimizer/resume chain, одинаковые source/native fingerprints,24 captures, движение обоих классов и запрет live monster friendly damage. `audit.json` и `recoil-proof.json` accepted относятся к целостности эксперимента.

| Seed | Урон до/после | Убийства до/после | Смерть до/после |
| --- | --- | --- | --- |
| 33400 | 144/56 | 0/0 | да/да |
| 33401 | 72/72 | 0/0 | да/да |
| 33402 | 350/350 | 2/2 | нет/нет |
| 33403 | 48/64 | 0/0 | да/да |
| Всего | 614/542 | 2/2 | 3/3 |

Победы1/4 у обеих моделей. На33402 parent update12 уже выиграл; более сильная награда не создала эту победу. Candidate получил44 входящего health damage против56 у parent. Обе цели подвижны, native Machinegun damage mod4 подтверждает убийства Parasite наframe149 и Gunner наframe171 у candidate. Первый завершённый полный Mixed на новых mobile fixtures подтверждён для этого seed; устойчивость по составам/сценам и общая боеготовность не доказаны.

Наблюдаемый alive расход230→126, урон на этот расход2.67→4.30. Средний угол с kick до ближайшего clear-shot target по четырём сидами:14.12°,11.31°,24.46°,18.31°→12.46°,9.66°,12.17°,11.22°. Выборки attack frames различаются по размеру и длительности жизни, это не causal hit rate. Дополнительный helper `aim_targets.go` сравнивает с минимальным углом до любой известной visible clear-shot цели:12.26°,11.08°,24.23°,17.10°→11.76°,8.75°,11.79°,11.12°. Переключение цели влияет на метрику, но не объясняет весь угол; центр bbox и camera punch являются proxy, попадание в объём монстра возможно при отклонении от его центра.

Candidate не продвинут в live: число побед не выросло, общий урон снизился, оценка короткая. Запущена отдельная fresh paired validation на16 seeds34000–34015, по4 эпизода на каждый из4 инстансов x2,32 captures для двух моделей. Optimization в validation не выполняется; сравниваются те же parent/candidate deterministic weights. Протокол и progress: `combat-mobile-recoil-aim-v1-20261007/validation-16/`.

Демка выигранного candidate seed33402 нормализована:392 уникальных game frames,503 повторных frame messages удалены; decoder сравнил все конечные per-tick states до/после. `winner-seed33402.dm2`157490 bytes, не включает смерть/respawn. Открыта в отдельном Yamagi playback runtime, timescale1, оконный client1920×1080; screenshot `playback-winning-mixed.png` показывает Machinegun, health80, ammo38, armor2 после подбора Armor Shard и60.02fps. Два убийства подтверждаются native damage/death events, не числом2 на HUD. Нормальная скорость относится к replay, collection остаётсяx2. Отдельный успешный seed показан для просмотра, не заменяет paired validation.


## Итог fresh validation на16 сидах

07.10.2026: обе оценки завершены,32 captures приняты, одинаковые source/native fingerprints, независимые seeds34000–34015. GPU optimization во время оценки не выполнялась.

| Seed | Урон до/после | Убийства до/после | Победа до/после | Смерть до/после |
| --- | --- | --- | --- | --- |
| 34000 | 263/64 | 1/0 | нет/нет | да/да |
| 34001 | 80/128 | 0/0 | нет/нет | да/да |
| 34002 | 128/175 | 0/1 | нет/нет | да/да |
| 34003 | 64/256 | 0/0 | нет/нет | да/да |
| 34004 | 160/136 | 0/0 | нет/нет | да/да |
| 34005 | 56/335 | 0/1 | нет/нет | да/да |
| 34006 | 328/64 | 0/0 | нет/нет | да/да |
| 34007 | 263/128 | 1/0 | нет/нет | да/да |
| 34008 | 350/320 | 2/0 | да/нет | нет/да |
| 34009 | 56/216 | 0/0 | нет/нет | да/да |
| 34010 | 112/128 | 0/0 | нет/нет | нет/да |
| 34011 | 104/48 | 0/0 | нет/нет | да/да |
| 34012 | 208/239 | 0/1 | нет/нет | да/нет |
| 34013 | 350/160 | 2/0 | да/нет | нет/да |
| 34014 | 88/136 | 0/0 | нет/нет | да/да |
| 34015 | 112/88 | 0/0 | нет/нет | да/да |

Parent:2/16 побед,6 убийств,13 смертей,2722 урона,1430 полученного урона. Candidate:0/16 побед,3 убийства,15 смертей,2621 урона,1595 полученного урона. Усиление aim_potential до1.0 отклонено. Для следующего обучения выбран прежний recoil update12 с aim_potential0.5 (`combat-mobile-recoil-v2r3-20261007/iteration-6/update`); показанная удачная демка candidate не является основанием для продвижения. Evidence: `validation-16/audit.json`.


## Завершение эпизода после боевой цели

07.10.2026: `-StopOnGoal` добавлен в existing baseline/solo harness. Temporal runner включает его для обучения и оценки. Solo требует убийства Parasite, Mixed — Parasite и Gunner. Supervisor отдельно читает native damage events: fatal damage из положительного здоровья, разные IDs нужных классов, player attribution, тот же spawncount, первая жизнь после combat release. Props, corpse damage, preparation kills, исчезновение из PVS и победа после respawn не засчитываются. Эти данные не передаются policy observations.

После последнего kill ожидается следующий наблюдённый кадр. Existing stop_file штатно завершает клиента, сервер останавливается в finally. Capture acceptance разрешает меньше максимума только после повторной проверки goal-stop.json/native evidence и полной обычной command/execution/reset proof. Exporter отмечает подтверждённую границу terminal `combat_goal_complete`, без death penalty. Команды за boundary сохраняются для dispatch audit, но исключаются из PPO; последняя незавершённая команда остаётся truncated.

Первый smoke `combat-goal-stop-validation-20261007` выявил старую проверку terminal=death: terminal reward был masked. Reward проверка исправлена; исходный пакет не используется для обучения. Финальный live пакет `combat-goal-stop-validation-v2-20261007`:4 инстанса x2, отдельные seeds33400–33403,4/4 accepted captures с полной provenance. Победный33402 завершился за73 game frames после barrier вместо300: экономия75.7% активных кадров этого эпизода. Обе kill rewards по5 сохранены, terminal reward available, death component0. Остальные3 эпизода достигли300. Это не75.7% ускорения всего пакета: cold setup и неудачные эпизоды остаются. Evidence `goal-stop-audit.json`. Go product tests и supervisor regressions прошли.

Winning demo обрезана по network kill_frame169 плюс closing observed frame170:163 уникальных frames вместо392, последнее здоровье56,62619 bytes. `winner-seed33402-goal.dm2` сохраняет исходные пакеты до cutoff. Это обрезка старой записи; ранняя остановка проверена отдельным live пакетом. Playback запущен через quake2 launcher, config timescale1/fixedtime0, оконный client1920×1080 подтверждён API окна. Новый screenshot оказался закрыт другим окном, сохранён как playback-window-covered.png и не считается визуальной проверкой игры.


## Mixed/Solo curriculum с ранней остановкой

07.10.2026: запущен `combat-mobile-machinegun-curriculum-v1-20261007`. Parent — принятой recoil update12 (aim_potential0.5), matching Adam checkpoint восстановлен. План4 CUDA updates:4 инстанса x2,3 независимых эпизода на инстанс в каждом batch (Mixed→Solo Parasite→Mixed), всего48 training seeds35000–35047. Stock Machinegun100 bullets, stock health/skill1, corrected z24.125, training-only no-infighting. Оценка до/после на16 новых Mixed seeds35400–35415, без optimization в оценке. Early goal stop включён в training/evaluation,300 frames остаются максимумом. Длительность training и число terminal outcomes могут различаться между эпизодами.

Retention/bank losses0; reference bank проверяется на пересечение со всеми48 training и16 evaluation seeds, включая расширенные EpisodesPerWorker. Все новые updates должны пройти Go/Torch state/logprob/value parity, KL и durable complete receipt до следующего batch. Нет продвижения в live до итоговой paired оценки и integrity audit.


Первый curriculum batch35000–35011 собран:12 accepted captures,4/4 Solo goals и1/8 Mixed goal. До optimization обнаружено несоответствие offline replay: q2ppo-data воспроизводил старый end_reason=game_frame_limit без goal boundary. Цикл остановлен, Adam/model не обновлялись. Исправленный экспортёр независимо проверяет receipt по native damage events, release и observed first-life frame, хранит SHA receipt и воспроизводит тот же goal terminal. Unit tests отвергают props, corpses, preparation, missing target, player death, другой world/life, подменённый target и незакрытый кадр. Диагностический re-export старого batch прошёл (898 rows), но этот пакет не употреблён optimizer.

После исправления запущен свежий `combat-mobile-machinegun-curriculum-v1r2-20261007` с тем же parent12,48 новыми training seeds35100–35147 и16 пока не оценёнными35400–35415. Первый каталог остаётся диагностическим и исключён из optimizer chain. Последующие batches/evaluations обязаны сохранить единый source/native fingerprint исправленного экспортёра.


07.10.2026: curriculum завершён и audited:4 updates,6269 rows,39 actor steps,80 accepted captures. На16 paired Mixed seeds35400–35415 kills1→6, wins0→1, damage2039→2538, deaths14→15. Candidate checkpoint16 сохранён для дальнейшего обучения, live/default policy не переключена. Полная таблица и ранние остановки: [Mixed/Solo curriculum](learned_combat_mobile_curriculum_v1.md).
