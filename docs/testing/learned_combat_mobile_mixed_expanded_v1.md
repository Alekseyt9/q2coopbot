# Mixed: увеличенный учебный batch, 07.10.2026

Новый цикл начинается от checkpoint16 после отрицательной расширенной проверки fresh-Adam control. Цель — проверить больший объём свежих Mixed данных на update. Веса и Adam возобновляются из checkpoint16; death penalty остаётся−5, все PPO/reward параметры прежние. Это расширение batch8→16 эпизодов относительно предыдущего Mixed-only resumed-Adam pilot, а не controlled fresh-Adam death-penalty comparison. Сиды другие, поэтому причинный эффект размера batch сам по себе не изолирован.

Четыре CUDA updates, четыре инстанса x2, четыре независимых эпизода на worker:64 training seeds40000–40063. Separate paired before/after evaluation:32 fresh seeds40400–40431, восемь эпизодов на worker. Максимум четыре одновременно работающих сервера/клиента. Бои Mixed Parasite+Gunner, fixed stock MG100 bullets, skill1, stock HP, z24.125, native no-infighting с dead-owner splash fix. Ранняя остановка после двух подтверждённых убийств первой жизни; не после одной цели.

Selection protocol записан до получения результатов: больше полных побед без роста deaths, либо равные победы при меньшем числе deaths и без потери kills. Иначе reference16 остаётся baseline. Damage/aim/ammo proxy не могут заменить завершение боя и выживание. Live/default policy не переключается автоматически.

GPU-only касается оптимизации сети; сбор и игровое исполнение остаются в Go. Reference bank/anchor нужны для provenance, retention/bank weights0; старые стационарные captures не входят в loss. Оценочные39000–39031 предыдущей проверки не использованы для нового обучения.

Полный цикл завершён:128 native captures приняты, четыре CUDA updates и paired32 evaluation прошли аудит.

Evidence: `workspace/artifacts/combat-mobile-mixed-expanded-v1-20261007/selection-protocol.json`; captures, rollout, CUDA updates и evaluations сохраняются в `cycle/`. Parent weights SHA256 `345582fae80ad32ec8b5f4062fc65ffb496d0293ddf4be8c6e313b14fc7bedaa`; checkpoint `fa5cefb8ebe8812cdd3d4dd5b262712aa8c6baa2b91f05ef7ea8739c96d40f41`.


## Завершённый цикл

| Update | Eligible rows | Actor steps | CUDA seconds |
|---|---:|---:|---:|
| 1 | 1875 | 10 | 1.045 |
| 2 | 1666 | 10 | 1.186 |
| 3 | 1356 | 10 | 1.469 |
| 4 | 1701 | 10 | 10.415 |

Всего 6598 eligible transitions, 40 новых actor steps. Adam/checkpoint продолжен от16 до20 cumulative updates, 190 cumulative actor steps. Это новая ветка expanded batch, а не один из прежних отклонённых checkpoint20. Frozen trainer/config hashes, old-logprob/value parity, KL limits и durable update seals проверены.

Training: 5/64 goal stops. Эти stochastic training эпизоды не являются результатом deterministic evaluation.

| Модель,32 новых Mixed эпизода | Победы | Убийства | Monster HP damage | Получено HP damage | Смерти |
|---|---:|---:|---:|---:|---:|
| Parent16 | 4 | 12 | 5412 | 3064 | 26 |
| Expanded candidate | 10 | 26 | 6746 | 2775 | 22 |

Новые победы: [40404, 40405, 40411, 40413, 40417, 40421, 40425, 40428]. Потерянные победы: [40414, 40419]. Учитывается первая жизнь и убийство Parasite+Gunner с положительным HP. Props/corpse kills не входят в эти результаты. Все captures пригодны, даже когда общий campaign harness не считает боевой провал успехом.

### Все32 оценки

Ячейка: damage / kills / win / death.

| Seed | Parent16 | Expanded candidate |
|---|---|---|
| 40400 | 128 / 0 / нет / да | 327 / 1 / нет / да |
| 40401 | 48 / 0 / нет / да | 48 / 0 / нет / да |
| 40402 | 168 / 0 / нет / нет | 128 / 0 / нет / да |
| 40403 | 56 / 0 / нет / да | 80 / 0 / нет / да |
| 40404 | 160 / 0 / нет / да | 350 / 2 / да / нет |
| 40405 | 144 / 0 / нет / да | 350 / 2 / да / нет |
| 40406 | 56 / 0 / нет / да | 296 / 0 / нет / да |
| 40407 | 112 / 0 / нет / да | 56 / 0 / нет / да |
| 40408 | 104 / 0 / нет / да | 72 / 0 / нет / да |
| 40409 | 72 / 0 / нет / да | 136 / 0 / нет / да |
| 40410 | 175 / 1 / нет / да | 259 / 1 / нет / да |
| 40411 | 200 / 0 / нет / да | 350 / 2 / да / нет |
| 40412 | 72 / 0 / нет / да | 88 / 0 / нет / да |
| 40413 | 319 / 1 / нет / да | 350 / 2 / да / нет |
| 40414 | 350 / 2 / да / нет | 311 / 1 / нет / да |
| 40415 | 56 / 0 / нет / да | 40 / 0 / нет / да |
| 40416 | 192 / 0 / нет / да | 247 / 1 / нет / да |
| 40417 | 263 / 1 / нет / нет | 350 / 2 / да / нет |
| 40418 | 96 / 0 / нет / да | 255 / 1 / нет / да |
| 40419 | 350 / 2 / да / нет | 104 / 0 / нет / да |
| 40420 | 303 / 1 / нет / да | 208 / 0 / нет / да |
| 40421 | 216 / 0 / нет / да | 350 / 2 / да / нет |
| 40422 | 350 / 2 / да / нет | 350 / 2 / да / нет |
| 40423 | 48 / 0 / нет / да | 80 / 0 / нет / да |
| 40424 | 112 / 0 / нет / да | 56 / 0 / нет / да |
| 40425 | 248 / 0 / нет / да | 350 / 2 / да / нет |
| 40426 | 72 / 0 / нет / да | 207 / 1 / нет / да |
| 40427 | 136 / 0 / нет / да | 112 / 0 / нет / да |
| 40428 | 264 / 0 / нет / да | 350 / 2 / да / нет |
| 40429 | 120 / 0 / нет / да | 72 / 0 / нет / да |
| 40430 | 72 / 0 / нет / да | 64 / 0 / нет / да |
| 40431 | 350 / 2 / да / нет | 350 / 2 / да / нет |

## Диагностика

before: incoming Gunner 1830, Parasite 1234; observed alive MG rounds 1553; damage/round proxy 3.485. Nearest clear-shot angle с observed kick 13.79°, min-angle any clear-shot 11.96° на 1503 alive first-life MG attack observations.

after: incoming Gunner 1187, Parasite 1588; observed alive MG rounds 1743; damage/round proxy 3.870. Nearest clear-shot angle с observed kick 18.24°, min-angle any clear-shot 17.21° на 1710 alive first-life MG attack observations.

Это geometry/HUD proxies, не точный hit rate и не причинная проверка dodge/aim. Длительности первых жизней и наборы observations различаются. Learned выбор оружия не проверялся; fixed MG.

Goal stops: 19, сэкономлено 3787 active frames из максимум38400 (9.86%); не measured wall-time speedup.

## Решение

Кандидат прошёл зафиксированное правило и выбран для дальнейшего обучения. Checkpoint16 сохраняется reference; результат ограничен32 fresh seeds. Перед live promotion нужна дальнейшая проверка. Live/default policy не переключена.

Следующий шаг — независимая расширенная оценка этого expanded кандидата против16 на новых сидах перед следующими updates. Не смешивать его с отклонённым fresh-Adam control: ancestry и checkpoint разные. Несмотря на10/32 побед,22/32 first-life deaths остаются; устойчивой боевой приёмки пока нет. Выживание против Parasite и сохранение прежних выигрышных сценариев остаются проблемами: в этой оценке Gunner damage уменьшился1830→1187, Parasite вырос1234→1588. Это totals разной длительности жизней, не causal skill attribution.

Selection: `candidate_for_further_training`. Selected weights SHA256 `f2add9c44e17a0ffd64050b3c66f02461ad2962375ff0dcfd9328e4eaaba92bc`; candidate SHA256 `f2add9c44e17a0ffd64050b3c66f02461ad2962375ff0dcfd9328e4eaaba92bc`. Source fingerprint `b62ac54f6e00759a013e07942da000ca0856b20e18c47da6df3252e95464e0f0`; native fingerprint `a6eb88e75484c1aadd47dc07ba69257d2c003f75bc77b15138e438c707fdd53a`.

Evidence: `workspace/artifacts/combat-mobile-mixed-expanded-v1-20261007/`: selection-protocol.json, selection.json; `cycle/` — audit.json, diagnostics.json, ammo-before/after.json, receipts, manifests, rollouts и update seals.
