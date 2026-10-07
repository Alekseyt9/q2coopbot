# Parasite curriculum от подтверждённого Expanded20

**Fresh confirmation завершён:** [ещё32 paired seeds](learned_combat_mobile_curriculum24_validation_v1.md): wins7→9, deaths25→22, kills25→31. Curriculum24 выбран основой дальнейшей работы; Expanded20 сохранён reference. Live/default не переключён. Остальные требования свёрены в [remaining gates](learned_combat_remaining_gates_20261007.md); learned weapon choice и общая боевая приёмка остаются незакрытыми.

07.10.2026. Цель текущего опыта — добавить отдельный опыт боя с мобильным Parasite, сохранив два Mixed боя на один Solo. Parent — подтверждённый Expanded20 (fresh confirmation41000–41031: wins6→8, deaths26→23). Награда death−5, kill+5, PPO config, наблюдения v5 и Temporal attention не меняются; Adam продолжается из этого checkpoint. Гипотеза о пользе Solo не является подтверждённым улучшением прицела или устойчивости.

Разбор четырёх потерянных побед41002/41009/41017/41023 не обнаружил movement-limit вмешательств в alive combat observations. Это не проверяет все причины потерь: скорость/направление команд и качество стрельбы могут быть неудачными. Candidate incoming по этим четырём боям: Gunner50/3/75/50, Parasite50/97/25/50. Parent завершал оба убийства в каждом. Получать обучающие примеры из этих eval captures нельзя, если продолжать оценивать на них же; текущий цикл использует новые seeds.

Четыре GPU-only updates, четыре инстанса x2, три эпизода на worker в порядке Mixed → Solo Parasite → Mixed.48 training seeds42000–42047:32 Mixed,16 Solo. Paired before/after evaluation —32 новых Mixed seeds42400–42431 на модель. Fixed MG100 bullets, skill1, stock HP, z24.125, no-infighting с dead-owner fix, release100, максимум300 active frames, stop после подтверждённой цели первой жизни. Для Solo цель — один Parasite; для Mixed — оба класса.

Выбор записан до результата: больше Mixed wins без роста deaths либо равные wins при меньших deaths без потери kills. Иначе основой остаётся Expanded20. Live/default policy не переключается. Old bank/anchor сохранены для provenance, loss weights0; стационарные captures не используются как новый loss. Следующая общая боевая приёмка и learned weapon choice всё ещё требуют отдельной работы по активному плану.

Статус: полный цикл завершён и прошёл аудит112/112 native captures. Evidence: `workspace/artifacts/combat-mobile-parasite-curriculum-v2-20261007/selection-protocol.json`, `cycle/`; диагностический разбор — `workspace/artifacts/combat-mobile-expanded-validation-v1-20261007/loss-diagnostics.json`. Parent SHA weights `f2add9c44e17a0ffd64050b3c66f02461ad2962375ff0dcfd9328e4eaaba92bc`, checkpoint `8704905decb585098fd3aa5da471b2a6592730d11f77dda219651a891c7c36e8`.


## Результаты

| CUDA update | Eligible rows | Actor steps | Seconds |
|---|---:|---:|---:|
| 1 | 1339 | 10 | 1.388 |
| 2 | 1037 | 10 | 1.176 |
| 3 | 1188 | 10 | 2.203 |
| 4 | 1272 | 10 | 11.087 |

Всего4836 transitions, 40 новых actor steps, Adam updates20→24, cumulative actor steps230. Checkpoint seals, CUDA reports, pinned config/trainer, rollout replay parity, source fingerprints и отсутствие повторного consumed rollout проверены.

Training goals: Mixed2/32, Solo15/16. Это stochastic training, не deterministic evaluation.

| Модель,32 Mixed seeds | Победы | Убийства | Monster HP damage | Получено HP damage | Смерти |
|---|---:|---:|---:|---:|---:|
| Expanded20 | 5 | 20 | 5854 | 2817 | 24 |
| Curriculum24 | 13 | 35 | 7801 | 2495 | 18 |

Добавлены победы[42401, 42402, 42403, 42405, 42407, 42410, 42413, 42414, 42418, 42428, 42431], потеряны[42408, 42409, 42416].

### Все32 оценки

Ячейка: damage / kills / win / death.

| Seed | Expanded20 | Curriculum24 |
|---|---|---|
| 42400 | 136 / 0 / нет / да | 175 / 1 / нет / да |
| 42401 | 64 / 0 / нет / да | 350 / 2 / да / нет |
| 42402 | 233 / 1 / нет / да | 350 / 2 / да / нет |
| 42403 | 283 / 1 / нет / да | 350 / 2 / да / нет |
| 42404 | 231 / 1 / нет / да | 48 / 0 / нет / да |
| 42405 | 112 / 0 / нет / да | 350 / 2 / да / нет |
| 42406 | 168 / 0 / нет / нет | 48 / 0 / нет / да |
| 42407 | 136 / 0 / нет / да | 350 / 2 / да / нет |
| 42408 | 350 / 2 / да / нет | 285 / 1 / нет / да |
| 42409 | 350 / 2 / да / нет | 80 / 0 / нет / да |
| 42410 | 320 / 0 / нет / да | 350 / 2 / да / нет |
| 42411 | 350 / 2 / да / нет | 350 / 2 / да / нет |
| 42412 | 72 / 0 / нет / да | 240 / 0 / нет / да |
| 42413 | 205 / 1 / нет / да | 350 / 2 / да / нет |
| 42414 | 277 / 1 / нет / да | 350 / 2 / да / нет |
| 42415 | 40 / 0 / нет / да | 104 / 0 / нет / да |
| 42416 | 350 / 2 / да / нет | 265 / 1 / нет / да |
| 42417 | 48 / 0 / нет / да | 56 / 0 / нет / да |
| 42418 | 271 / 1 / нет / да | 350 / 2 / да / нет |
| 42419 | 239 / 1 / нет / да | 215 / 1 / нет / да |
| 42420 | 350 / 2 / да / нет | 350 / 2 / да / нет |
| 42421 | 72 / 0 / нет / да | 239 / 1 / нет / да |
| 42422 | 64 / 0 / нет / нет | 311 / 1 / нет / да |
| 42423 | 136 / 0 / нет / да | 267 / 1 / нет / да |
| 42424 | 64 / 0 / нет / да | 80 / 0 / нет / нет |
| 42425 | 255 / 1 / нет / да | 48 / 0 / нет / да |
| 42426 | 96 / 0 / нет / да | 120 / 0 / нет / да |
| 42427 | 175 / 1 / нет / да | 285 / 1 / нет / да |
| 42428 | 72 / 0 / нет / да | 350 / 2 / да / нет |
| 42429 | 88 / 0 / нет / да | 72 / 0 / нет / да |
| 42430 | 64 / 0 / нет / нет | 313 / 1 / нет / да |
| 42431 | 183 / 1 / нет / да | 350 / 2 / да / нет |

## Диагностика и выбор

Основной результат: Mixed wins5→13, deaths24→18, kills20→35. Next — независимая проверка24 против20 на новых сидах до следующих обновлений. Для общих требований плана недостаточно текущего fixed MG/P+G fixture: ещё не доказаны learned weapon selection, перенос на другие составы/геометрию и устойчивая боевая приёмка. Более низкий angle proxy не даёт права утверждать точный hit rate.

before: incoming Gunner1377, Parasite1440; damage/observed-alive-round3.086; nearest observed-kick angle19.98°, min-any-clear angle18.89° на1854 alive MG attack observations.

after: incoming Gunner975, Parasite1520; damage/observed-alive-round3.603; nearest observed-kick angle15.41°, min-any-clear angle14.85° на2141 alive MG attack observations.

Это proxies разной длительности/sample sets, не точный hit rate или causal skill attribution. Оценивались native kills первой жизни, только monster classes; corpse/props kills исключены. Goal stop требует положительного HP и kill reward при death reward0. Наблюдения обоих типов монстров и horizontal mobility подтверждены.

Curriculum24 прошёл правило выбора и выбран для дальнейшей независимой проверки. Ни один кандидат не включён в live/default. Новый цельный бой, learned weapon choice, более широкие monster compositions и общая приёмка остаются последующими требованиями активного плана; этот pilot их не закрывает.

Selection `candidate_for_further_training`, selected weights SHA `7b01b4999df979aaef85f9b7c908246edd74f5fb68c583b2083800de24931c40`. Evidence: `workspace/artifacts/combat-mobile-parasite-curriculum-v2-20261007/`: protocol, selection; `cycle/` — audit.json, diagnostics.json, manifests, rollouts, native traces, goal receipts и complete.json.
