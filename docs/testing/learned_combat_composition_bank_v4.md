# Composition-balanced retention bank v4

06.10.2026. Следующий отдельный фактор после [count bank](learned_combat_bank_retention_v4.md).

## Протокол до результатов

Те же verified parent observation inputs23200–23211/23300–23307,
тот же train/validation episode split, cap64 на cell, parent/reward/
horizon/MLP/Adam/RNG. Старые bank captures не новые samples и не eval
traces. Два banks пересобираются из одинаковых raw native inputs;
count rows/features и float32 weights должны совпасть с прежним банком, кроме
receipt/forbidden metadata. Новый `--balance composition` учитывает
все наблюдаемые class names и их количества (sorted multiset), включая
unknown. Четыре представленных состава получают mass1/4 каждый;
контексты wall/barrel/projectile внутри каждого делят массу поровну.
Редкий Gunner больше не смешивается с Parasite. Отдельно учитывать,
что total both-threat mass меняется38.1%→25%, это часть этого balancing
factor, не скрытое сохранение прежних count-group weights.

Both branches constant on-policy retention1 + bank coefficient1,
4 fixed CUDA updates от original obstacle parent. Единственный фактор
bank balancing (и детерминированный отбор из тех же наблюдений по новым
cells). Общий fresh initial PPO23700–23703, затем собственные23704–23715.
Fresh paired Mixed23800–23803/Solo23900–23903; known20800/21300/21700/
22100 по4seeds. Final update4 выбран до eval, не подбирать cap/коэффициент
по оценке. Validation bank только diagnostics, не gradient/training
bank selection. Исторические/новые PPO/eval seeds объявлены forbidden.

31 batches/124 новых episode captures,4 instances x2, индивидуальные seeds,
CUDA-only обучение/gradient tests. Source/native fingerprints одинаковы
между текущими captures; historical bank inputs имеют отдельную старую
provenance. Без копий PAK, без изменения live. Проверить sources,
semantic count-bank identity, composition mass/coverage, independent
analytic bank KL, native bytes/rewards/Adam/RNG/exact CUDA replay и
uninterrupted ownership. Сравнивать per-seed новые смерти, не только
сумму побед и mean KL. GRU/weapon/reward/horizon не менять.

Root `workspace/artifacts/combat-composition-bank-v4-20261006`.


## Результат и решение

Обе ветки завершили по 4 CUDA updates / 40 actor steps. После общей
первой партии (928 rows) каждая собирала собственные on-policy данные:
count 3302 rows, composition 4377 rows. Итоговые counters обеих 57/564.
74 Python tests прошли; native re-export, reward consumption, exact CUDA
replay actor/Adam, counters/RNG и направление шага проверены. Direction
fallbacks: 0. Независимый analytic bank KL также совпал с отчётами.

Банк count: 988 train / 570 validation / 21 train contexts; composition:
1046 / 570 / 25. Composition train counts: empty238, Gunner79,
Parasite291, обе438; масса каждого состава ровно 25%. Raw observation
identity/features и episode split проверены. Старые данные банка — 20
исторических training captures, не новые прогоны и не evaluation data.
Count selected rows/features/IDs и float32 weights точно равны прежним;
в JSON doubles есть округление до 4.34e-19 train / 1.74e-18 validation,
поэтому полное byte equality файлов не заявляется.

| Набор (по 4 seeds) | Parent | Count | Composition |
|---|---:|---:|---:|
| Fresh Mixed 23800–23803 | 3 | 1 | 3 |
| Fresh Solo 23900–23903 | 4 | 4 | 4 |
| Known occlusion 20800–20803 | 3 | 3 | 3 |
| Known Mixed 21300–21303 | 3 | 0 | 2 |
| Known fresh 21700–21703 | 3 | 1 | 3 |
| Obstacle 22100–22103 | 3 | 4 | 3 |

Победа означает оба монстра убиты в первой жизни без передачи управления
rules до последнего убийства. На 20 Mixed: parent 15, count9,
composition14 побед; count23 kills / 9 deaths / incoming1582,
composition30 / 4 / 1510. Одиночные бои обе ветки сохраняют4/4,
incoming217→219. Это один fresh Mixed quartet и четыре известных
regression quartets: результат описательный, не независимая оценка
обобщения. Повторные parent captures не новые независимые seeds.

Общий initial rollout bytes и parent resume точно совпали; update1 weights
различаются из-за bank penalty. Для всех 24 parent evaluation seeds
проверено равенство observed/action uninterrupted learned prefixes.
Полные outcomes после rules handoffs не равны: на21300 parent received
248 против317 и deaths0 против1, при тех же3 uninterrupted wins.
Поэтому разницу общего incoming с parent нельзя целиком приписывать
обновлению политики. Изменение mass both-threat 38.1%→25% также входит
в экспериментальный фактор.

Composition новые неудачи относительно parent:20802 (смерть),21300
(alive unfinished),22103 (смерть);21701 остался неуспешным и теперь
умер. На21302 completion отсутствовал и прежде. Низкий независимый
bank mean KL train0.00448812 / validation0.00461254 не гарантирует
сохранения поведения. Composition полезнее count в этом опыте, но
**не принят новым reference; live не заменён**.

## Технический сбой и воспроизводимость

124 корректных episode captures / 31 batches:4 concurrent instances,
x2, каждому инстансу отдельный подтверждённый seed. Всего предприняты
132 episode attempts / 33 batches. Две неполные parent quartets
`composition/obstacle-holdout/before` (3 usable) и `before-retry1`
(2 usable) сохранены и целиком исключены. Причина: Get-FileHash общего
immutable AAS shard asset-0.pak, файл занят другим процессом. После
проверки SHA исходный source.bin кеша атомарно заменён идентичной
копией с новым NTFS link counter; старый source сохранён как backup.
`cache-recovery.json` фиксирует операцию. Полный parent retry
`before-retry2` и последующий after прошли.

Все 31 корректные партии имеют одинаковые source/native fingerprints:
`abe11d5c0569cd79e9bb7d5d4226d52e3d3708e7e20729a226fa7933002fe3fe` /
`99ca5b165e8c1e2683c0ae8ecc5800385c75cedfd0895cf3dd7ebb928bb9a08c`.
Исходники и веса во время восстановления не менялись. После завершения
всех captures и аудита Install-RuntimePak переведён на существующий
Get-RuntimeImmutableHash с ReadWrite/Delete sharing и bounded retry.
Focused PowerShell test проверил concurrent writable/delete-sharing
handle, saturation/rollover, corrupt pool rejection и atomic snapshot
isolation. Этот post-capture fix не входит в fingerprints эксперимента;
его игровая проверка будет в следующей партии.

Итоговые weights SHA count:
`b2ac71b61d923345bf8588e59c8896f32de5d369932595b920a8c8abb7f78d2c`,
composition:
`9891fe9107374cab57e105781a658665b1910fc1a25e811d246622ac94b13bb4`.
Composition bank SHA:
`2574125b4f7812572a809df174263756f0df18409eef9529bba5b4ed731a6a6b`.
Артефакты: completion-status.json, bank-audit.json, comparison-audit.json,
per-arm training/direction/consumption/evaluation/blockage-audit.json и
retention native-reexport/shaping-audit.json внутри указанного root.

## Следующий отдельный опыт

Training closure diagnostic по 16 stochastic episodes каждой ветки:
count0 raw full successes / 9 deaths / 7 alive unfinished;
composition1 / 3 / 12. Общая первая четвёрка включена в обе суммы,
поэтому эти результаты не независимы; raw success может включать rules.
Consumed kill reward rows count5 / composition4. Успешных полных
stochastic траекторий мало; это наблюдение, не доказательство ошибки
reward/bootstrap. `training-closure-audit.json` сохраняет last consumed
row, terminal/truncated/next_value для каждого episode.

Далее проверить завершение rewards/GAE: смерть, horizon truncation,
bootstrap и control handoff, отдельно сохранение kill transitions.
После подтверждения контракта сравнить прежний rollout budget с более
крупной партией (например,4 instances ×3 episodes) при тех же4 CUDA
updates, reward/horizon/архитектуре. Новые train/eval seeds, frozen
heldouts, без обучения на evaluation traces. Не менять одновременно
reward, память сети и размер партии.
