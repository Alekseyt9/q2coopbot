# Независимая проверка expanded Mixed кандидата, 07.10.2026

Кандидат подтвердил улучшение на новой32-seed выборке и выбран основой дальнейшего обучения. Live/default policy не переключена.

## Условия

32 fresh seeds41000–41031 на модель,64 native captures,4 инстанса x2, восемь независимых эпизодов на worker. Эти seeds не использованы в training40000–40063 и предыдущей evaluation40400–40431. Сначала candidate, затем parent; предыдущий pilot имел обратный порядок. Использован тот же Go-клиент и `run_learned_combat_baseline.ps1`. Веса/critic не менялись, optimizer не запускался. GPU-only требование к обучению сохраняется; это evaluation.

Mixed Parasite+Gunner, skill1, stock monster HP, Machinegun100 bullets, z24.125, release100,300 max active frames. Native no-infighting включает delayed splash от dead owner; early goal stop требует обоих убийств и живого бота на первой жизни. Общий campaign успех не подменяет отдельный capture_valid и first-life combat result.

Сравнивается resumed-Adam expanded checkpoint20 с checkpoint16; это другой checkpoint20, чем отклонённые прежние ветки. 64/64 captures прошли аудит: seeds/reset/dispatch provenance, frozen model hashes, source/native fingerprints, наличие/движение обоих monster classes, отсутствие взаимного урона живым монстрам. Goal terminals имеют доступный reward без death penalty и не менее+10 kill reward. Полная engine/AI determinism или general combat acceptance не заявляется.

Selection protocol записан до запусков: fresh32 wins должны вырасти без роста deaths. Старый pilot отражён отдельно; объединённые totals не используются, чтобы отменить провал fresh проверки.

## Fresh32 результат

| Модель | Победы | Убийства | Monster HP damage | Получено HP damage | Смерти |
|---|---:|---:|---:|---:|---:|
| Parent16 | 6 | 22 | 5870 | 3035 | 26 |
| Expanded20 | 8 | 24 | 6544 | 2845 | 23 |

Добавленные победы: [41003, 41006, 41012, 41019, 41021, 41029]. Потерянные победы: [41002, 41009, 41017, 41023].

### Все32 оценки

Ячейка: damage / kills / win / death. Native first-life kills только по monster classes; props/corpses исключены.

| Seed | Parent16 | Expanded20 |
|---|---|---|
| 41000 | 48 / 0 / нет / да | 56 / 0 / нет / да |
| 41001 | 295 / 1 / нет / да | 191 / 1 / нет / да |
| 41002 | 350 / 2 / да / нет | 96 / 0 / нет / да |
| 41003 | 192 / 0 / нет / да | 350 / 2 / да / нет |
| 41004 | 72 / 0 / нет / да | 72 / 0 / нет / да |
| 41005 | 48 / 0 / нет / да | 200 / 0 / нет / да |
| 41006 | 175 / 1 / нет / да | 350 / 2 / да / нет |
| 41007 | 48 / 0 / нет / да | 64 / 0 / нет / да |
| 41008 | 350 / 2 / да / нет | 350 / 2 / да / нет |
| 41009 | 350 / 2 / да / нет | 255 / 1 / нет / да |
| 41010 | 175 / 1 / нет / да | 343 / 1 / нет / да |
| 41011 | 88 / 0 / нет / да | 80 / 0 / нет / да |
| 41012 | 88 / 0 / нет / да | 350 / 2 / да / нет |
| 41013 | 327 / 1 / нет / да | 175 / 1 / нет / да |
| 41014 | 350 / 2 / да / нет | 350 / 2 / да / нет |
| 41015 | 303 / 1 / нет / да | 327 / 1 / нет / да |
| 41016 | 175 / 1 / нет / да | 303 / 1 / нет / нет |
| 41017 | 350 / 2 / да / нет | 96 / 0 / нет / да |
| 41018 | 48 / 0 / нет / да | 56 / 0 / нет / да |
| 41019 | 120 / 0 / нет / да | 350 / 2 / да / нет |
| 41020 | 319 / 1 / нет / да | 319 / 1 / нет / да |
| 41021 | 64 / 0 / нет / да | 350 / 2 / да / нет |
| 41022 | 56 / 0 / нет / да | 120 / 0 / нет / да |
| 41023 | 350 / 2 / да / нет | 192 / 0 / нет / да |
| 41024 | 72 / 0 / нет / да | 120 / 0 / нет / да |
| 41025 | 48 / 0 / нет / да | 112 / 0 / нет / да |
| 41026 | 48 / 0 / нет / да | 48 / 0 / нет / да |
| 41027 | 327 / 1 / нет / да | 183 / 1 / нет / да |
| 41028 | 88 / 0 / нет / да | 152 / 0 / нет / да |
| 41029 | 175 / 1 / нет / да | 350 / 2 / да / нет |
| 41030 | 96 / 0 / нет / да | 56 / 0 / нет / да |
| 41031 | 275 / 1 / нет / да | 128 / 0 / нет / да |

## Два раздельных cohorts

| Cohort | Parent wins/deaths | Candidate wins/deaths |
|---|---:|---:|
| Pilot40400–40431 | 4/26 | 10/22 |
| Fresh41000–41031 | 6/26 | 8/23 |

Описательные totals на64 оценочных seeds: parent 10 wins/52 deaths, candidate 18 wins/45 deaths. Pilot уже использовался для отбора кандидата, поэтому эти64 нельзя назвать полностью новым независимым confirmation set. Решение применяется по fresh32.

## Диагностика

parent: incoming Gunner 1438, Parasite 1597; observed alive MG rounds 1580; damage/round proxy 3.715; nearest clear-shot angle с observed kick 17.05°, min-angle any clear-shot 15.84° на 1540 alive first-life attack observations.

candidate: incoming Gunner 1347, Parasite 1498; observed alive MG rounds 1981; damage/round proxy 3.303; nearest clear-shot angle с observed kick 21.00°, min-angle any clear-shot 19.86° на 1972 alive first-life attack observations.

Geometry и HUD proxies не равны bullet hit rate или causal dodge/aim skill; наборы наблюдений и длины жизней различаются. Fixed MG не проверяет learned weapon selection.

Goal stops: 14; сэкономлено 2617 active frames из максимум19200 (13.63%). Это frame budget, не measured wall-time speedup.

## Дальше

Продолжить GPU-only обучение от подтверждённого expanded20 checkpoint. Reference16 сохраняется. Следующий опыт должен отдельно проверять проблему выживания/завершения группы; новый evaluation cohort сохраняется вне loss. Стойкой боевой приёмки и переключения live пока нет.

На64 оценочных seeds обеих выборок candidate добавил14 побед и потерял6 прежних; сохранил4 из10 выигрышей parent. Это заметная смена набора решаемых боёв, а не сохранение всей способности parent. Новое обучение должно получать разнообразные свежие training episodes и проверяться независимо. Если добавлять retention успешных mobile trajectories, брать их из training captures, а не превращать40400–40431 или41000–41031 в обучающие примеры с последующей оценкой на них же.

В свежей проверке angle proxy ухудшился17.05°→21.00°, damage/observed-alive-round3.715→3.303. Поэтому подтвердилось улучшение завершения боя и deaths по протоколу, но не улучшение точности стрельбы. Это ограниченные geometry/HUD measurements, не bullet accuracy или доказательство механизма прироста.

Selected weights SHA256 `f2add9c44e17a0ffd64050b3c66f02461ad2962375ff0dcfd9328e4eaaba92bc`; selected checkpoint `8704905decb585098fd3aa5da471b2a6592730d11f77dda219651a891c7c36e8`. Source fingerprint `b62ac54f6e00759a013e07942da000ca0856b20e18c47da6df3252e95464e0f0`; native fingerprint `a6eb88e75484c1aadd47dc07ba69257d2c003f75bc77b15138e438c707fdd53a`.

Evidence: `workspace/artifacts/combat-mobile-expanded-validation-v1-20261007/`: protocol.json, audit.json, diagnostics.json, selection.json, ammo-parent/candidate.json, manifests, goal receipts и reports.
