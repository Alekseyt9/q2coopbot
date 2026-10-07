# Независимая проверка Curriculum24 кандидата, 07.10.2026

Кандидат подтвердил улучшение на новой32-seed выборке и выбран основой дальнейшего обучения. Live/default policy не переключена.

## Условия

32 fresh seeds43000–43031 на модель,64 native captures,4 инстанса x2, восемь независимых эпизодов на worker. Эти seeds не использованы в training42000–42047 и предыдущей evaluation42400–42431. Сначала candidate, затем parent; предыдущий pilot имел обратный порядок. Использован тот же Go-клиент и `run_learned_combat_baseline.ps1`. Веса/critic не менялись, optimizer не запускался. GPU-only требование к обучению сохраняется; это evaluation.

Mixed Parasite+Gunner, skill1, stock monster HP, Machinegun100 bullets, z24.125, release100,300 max active frames. Native no-infighting включает delayed splash от dead owner; early goal stop требует обоих убийств и живого бота на первой жизни. Общий campaign успех не подменяет отдельный capture_valid и first-life combat result.

Сравнивается Curriculum24 Mixed/Solo curriculum с подтверждённым Expanded20. 64/64 captures прошли аудит: seeds/reset/dispatch provenance, frozen model hashes, source/native fingerprints, наличие/движение обоих monster classes, отсутствие взаимного урона живым монстрам. Goal terminals имеют доступный reward без death penalty и не менее+10 kill reward. Полная engine/AI determinism или general combat acceptance не заявляется.

Selection protocol записан до запусков: fresh32 wins должны вырасти без роста deaths. Старый pilot отражён отдельно; объединённые totals не используются, чтобы отменить провал fresh проверки.

## Fresh32 результат

| Модель | Победы | Убийства | Monster HP damage | Получено HP damage | Смерти |
|---|---:|---:|---:|---:|---:|
| Expanded20 | 7 | 25 | 6801 | 2991 | 25 |
| Curriculum24 | 9 | 31 | 7875 | 2665 | 22 |

Добавленные победы: [43001, 43011, 43017, 43020, 43021, 43024, 43031]. Потерянные победы: [43000, 43004, 43009, 43010, 43027].

### Все32 оценки

Ячейка: damage / kills / win / death. Native first-life kills только по monster classes; props/corpses исключены.

| Seed | Expanded20 | Curriculum24 |
|---|---|---|
| 43000 | 350 / 2 / да / нет | 327 / 1 / нет / да |
| 43001 | 255 / 1 / нет / да | 350 / 2 / да / нет |
| 43002 | 72 / 0 / нет / да | 287 / 1 / нет / да |
| 43003 | 168 / 0 / нет / да | 40 / 0 / нет / да |
| 43004 | 350 / 2 / да / нет | 191 / 1 / нет / да |
| 43005 | 215 / 1 / нет / да | 256 / 0 / нет / да |
| 43006 | 350 / 2 / да / нет | 350 / 2 / да / нет |
| 43007 | 350 / 2 / да / нет | 350 / 2 / да / нет |
| 43008 | 48 / 0 / нет / да | 216 / 0 / нет / да |
| 43009 | 350 / 2 / да / нет | 319 / 1 / нет / да |
| 43010 | 350 / 2 / да / нет | 321 / 1 / нет / да |
| 43011 | 207 / 1 / нет / да | 350 / 2 / да / нет |
| 43012 | 255 / 1 / нет / да | 183 / 1 / нет / да |
| 43013 | 88 / 0 / нет / да | 191 / 1 / нет / да |
| 43014 | 96 / 0 / нет / да | 183 / 1 / нет / да |
| 43015 | 80 / 0 / нет / да | 136 / 0 / нет / да |
| 43016 | 96 / 0 / нет / да | 64 / 0 / нет / нет |
| 43017 | 120 / 0 / нет / да | 350 / 2 / да / нет |
| 43018 | 335 / 1 / нет / да | 112 / 0 / нет / да |
| 43019 | 279 / 1 / нет / да | 303 / 1 / нет / да |
| 43020 | 303 / 1 / нет / да | 350 / 2 / да / нет |
| 43021 | 136 / 0 / нет / да | 350 / 2 / да / нет |
| 43022 | 184 / 0 / нет / да | 287 / 1 / нет / да |
| 43023 | 120 / 0 / нет / да | 128 / 0 / нет / да |
| 43024 | 255 / 1 / нет / да | 350 / 2 / да / нет |
| 43025 | 56 / 0 / нет / да | 239 / 1 / нет / да |
| 43026 | 265 / 1 / нет / да | 287 / 1 / нет / да |
| 43027 | 350 / 2 / да / нет | 264 / 0 / нет / да |
| 43028 | 303 / 1 / нет / да | 88 / 0 / нет / да |
| 43029 | 223 / 1 / нет / да | 175 / 1 / нет / да |
| 43030 | 80 / 0 / нет / да | 128 / 0 / нет / да |
| 43031 | 112 / 0 / нет / да | 350 / 2 / да / нет |

## Два раздельных cohorts

| Cohort | Parent wins/deaths | Candidate wins/deaths |
|---|---:|---:|
| Pilot42400–42431 | 5/24 | 13/18 |
| Fresh43000–43031 | 7/25 | 9/22 |

Описательные totals на64 оценочных seeds: parent 12 wins/49 deaths, candidate 22 wins/40 deaths. Pilot уже использовался для отбора кандидата, поэтому эти64 нельзя назвать полностью новым независимым confirmation set. Решение применяется по fresh32.

## Диагностика

parent: incoming Gunner 1795, Parasite 1196; observed alive MG rounds 1928; damage/round proxy 3.527; nearest clear-shot angle с observed kick 19.33°, min-angle any clear-shot 18.38° на 1906 alive first-life attack observations.

candidate: incoming Gunner 1207, Parasite 1458; observed alive MG rounds 2091; damage/round proxy 3.766; nearest clear-shot angle с observed kick 16.58°, min-angle any clear-shot 15.69° на 2061 alive first-life attack observations.

Geometry и HUD proxies не равны bullet hit rate или causal dodge/aim skill; наборы наблюдений и длины жизней различаются. Fixed MG не проверяет learned weapon selection.

Goal stops: 16; сэкономлено 3359 active frames из максимум19200 (17.49%). Это frame budget, не measured wall-time speedup.

## Дальше

На второй выборке также улучшились геометрические aim proxies: nearest с observed kick19.33°→16.58°, min-any-clear18.38°→15.69°. Damage/observed-alive-round3.527→3.766. Это не точный hit rate. Из семи прежних побед сохранены только две; устойчивость поведения по отдельным сценариям остаётся проблемой.

Работа по плану продолжается: [сверка незакрытых требований](learned_combat_remaining_gates_20261007.md). Расширение inventory/weapon action contract нужно делать отдельным изменением с Go/Python parity и native dispatch evidence, сохранив проверенный checkpoint как reference. Текущие32 сида не превращать в новый training loss.

Продолжить GPU-only обучение от подтверждённого curriculum24 checkpoint. Reference20 сохраняется. Следующий опыт должен отдельно проверять проблему выживания/завершения группы; новый evaluation cohort сохраняется вне loss. Стойкой боевой приёмки и переключения live пока нет.

Selected weights SHA256 `7b01b4999df979aaef85f9b7c908246edd74f5fb68c583b2083800de24931c40`; selected checkpoint `536c1f04d8f50760dce29165d2e7ba1eff888f69b0a3be1b3321480b6410da16`. Source fingerprint `b62ac54f6e00759a013e07942da000ca0856b20e18c47da6df3252e95464e0f0`; native fingerprint `a6eb88e75484c1aadd47dc07ba69257d2c003f75bc77b15138e438c707fdd53a`.

Evidence: `workspace/artifacts/combat-mobile-curriculum24-validation-v1-20261007/`: protocol.json, audit.json, diagnostics.json, selection.json, ammo-parent/candidate.json, manifests, goal receipts и reports.
