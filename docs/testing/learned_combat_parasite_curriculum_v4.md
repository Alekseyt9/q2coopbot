# Parasite curriculum v4

06.10.2026. Протокол до evaluation результатов.

Отдельный фактор состава training episodes: control по3 Mixed на worker;
curriculum последовательность Mixed/Solo/Mixed на каждом из4 workers,
то есть8 Mixed+4 Solo в каждой12-episode партии. Только существующие
native fixtures: Solo имеет stock175HP Parasite и исходную позицию;
Mixed дополнительно Gunner. Это начальный curriculum, не воспроизведение
состояний после Gunner kill и не snapshots из evaluation traces.

Обе ветки original obstacle parent53/524, identical Adam/RNG, composition
bank SHA2574125b4f7812572a809df174263756f0df18409eef9529bba5b4ed731a6a6b,
anchor/bank constant coefficients1, fixed4 CUDA updates/40 actor steps,
reward maneuver-v4, horizon300, release100, Blaster, MLP810→64→64→8.
4 concurrent servers/clients x2, каждому episode отдельный seed;48
training episodes на ветку24300–24347. Initial common Mixed8 seeds
должны иметь одинаковые learned prefixes; Solo4 меняет условия этой
части initial data. Далее обе ветки own-policy, overlapping seeds не
означают одинаковые trajectories. Не изменять reward/GRU/noise/бюджет.

Fresh paired Mixed24400–24403/Solo24500–24503; known20800/21300/21700/
22100 по4 seeds, final update4 выбран до eval. Ожидается192 captures/
32 batches:96 training,96 evaluation. Старые bank inputs training-only,
validation no-grad. Baseline manifest episode_pattern и per-episode
fixture_mixed/actual native Gunner entity count должны точно подтверждать
состав; несовместимые параметры/InitialBatch отклоняются до output.
75 Python tests до collection прошли, overlap/source/receipt/seed checks
сохранены. Root workspace/artifacts/combat-parasite-curriculum-v4-20261006.

Измерять kills по классам и consumed reward transitions; full stochastic
success отдельно для Mixed (2 kills) и Solo (1), непрерывный ownership;
Mixed/Solo deterministic wins/deaths/damage и per-seed regressions.
Маневрирование: Фактические sent movement/backward/strafe/crouch/jump commands
и observed airborne по provider-owned alive first-life frames; zero
jump не маскировать падениями/knockback. Наличие движения не доказывает
осознанного уклонения от конкретного projectile. Все4 action heads
участвуют PPO; прыжки не форсировать эвристикой и не добавлять jump bonus
в этот опыт. No live promotion до preservation acceptance.

## Результат

Опыт завершён: 192 valid captures, 32 batches, 96 training и 96 evaluation.
Все партии использовали 4 server/client instances, timescale 2; каждому
эпизоду подтверждён отдельный seed. Обучение и optimizer replay только CUDA.
75 Python tests прошли; отрицательные PowerShell проверки несовместимых
EpisodePattern/EpisodesPerWorker и InitialBatch отклоняют запуск до output.
В capture сверены fixture_mixed и фактическое число Gunner в native entity
lump: Mixed содержит одного Gunner, Solo — ноль.

Обе ветки получили 4 updates/40 accepted actor steps, итоговые счётчики
57/564; direction fallback отсутствует. Initial common Mixed8 learned prefixes,
parent resume и bank bytes равны. Дальнейшие партии собраны своей policy.

| Training задача | Эпизоды | PPO rows | Gunner kills | Parasite kills | Смерти | Полные непрерывные победы |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control Mixed | 48 | 11729 | 5 | 0 | 16 | 0 |
| curriculum Mixed | 32 | — | 4 | 0 | 16 | 0 |
| curriculum Solo | 16 | — | 0 | 16 | 0 | 16 |

Curriculum всего 9398 PPO rows. Суммарные 16/48 побед нельзя сравнивать
с control 0/48: Solo требует одного убийства, Mixed — двух. Все 25 kill
transitions вошли в reward; closure audit проверил все 21127 eligible rows,
samples/rewards, terminal/truncation flags и независимое разложение GAE.
Death/handoff дают zero bootstrap, последний полный horizon transition
сохраняет V(next); неполная последняя команда исключается. Независимый
пересчёт critic здесь не выполнялся; оптимальность handoff objective
этими проверками не доказана.

Детерминированная оценка: полные победы под непрерывным learned ownership.

| Набор (4 seeds каждый) | Parent | control | curriculum |
| --- | ---: | ---: | ---: |
| Fresh Mixed 24400–24403 | 1 | 3 | 2 |
| Solo 24500–24503 | 4 | 4 | 4 |
| Known 20800–20803 | 3 | 1 | 2 |
| Known 21300–21303 | 3 | 3 | 2 |
| Known 21700–21703 | 3 | 2 | 1 |
| Known 22100–22103 | 3 | 3 | 3 |
| Всего 20 Mixed | 13 | 12 | 10 |

На 20 Mixed parent: 28 kills/6 deaths/incoming1606, control:
29/4/1385, curriculum: 24/7/1518. Solo incoming197→164/154.
Все 24 повторных parent prefixes и полные native outcomes в этом опыте
совпали; повторные parent captures не являются независимыми наблюдениями.
Небольшие eval-наборы не доказывают обобщение на произвольные карты.

Control потерял прежние победы 20802/20803/21303/21702; новые смерти
20803/21701. Curriculum потерял 20802/20803/21300/21303/21702/21703/22101;
новые смерти 20803/21300/21303/21701/21702/21703. На 20803/21300/21303/21703
curriculum убивает Gunner, но погибает при живом Parasite. Перечень взят
из before/after именно этого опыта, а не из старых результатов тех же seeds.
Train/validation mean bank KL control0.002870/0.003163 и
curriculum0.003342/0.003875 не гарантируют сохранение навыков.
Обе final модели не проходят preservation; original obstacle reference
и live policy сохранены.

## Движение и условия завершающей фазы

Training control: 16 policy jump commands, 16 sent commands и 16 observed
ground-to-air transitions после них; curriculum: 11/11/11. Итого 27
фактических прыжков при stochastic training. В deterministic evaluation
обеих веток jump commands отсутствуют. Airborne51/50 frames сами по себе
не доказывают прыжки: возможны падение или knockback.

На unique alive first-life provider frames (Mixed20+Solo4; для завершённых
боёв до последнего kill) control: 5007 frames, движение73.0%, strafe55.9%,
назад21.8%, crouch31.1%; curriculum: 4529 frames, движение69.2%,
strafe56.3%, назад18.9%, crouch25.1%. Это подтверждает маневрирование,
но не осознанное уклонение от определённого выстрела или освоенные прыжки.
Jump bonus и принудительные jump rules не добавлялись.

Диагностика только training captures: 9 состояний после Gunner kill,
8 уникальных seeds (общий initial24302 присутствует в обеих ветках).
Следующее UDP observation показывает только Parasite: HP бота28–94,
дистанция330.51–401.46, minimum known standing hull clearance17–64.
Fresh Solo начинает с HP100 и дистанции около168, в другой позиции
и оружейной фазе. Следовательно, Solo success не покрывает наблюдаемые
условия завершения Mixed. Разница условий — гипотеза переноса;
причинный эффект HP/геометрии/дистанции отдельно пока не измерен.
Это не engine snapshots и не воспроизведение post-kill world state.

Следующий отдельный опыт: fresh synthetic remaining-Parasite fixtures
с условиями из training-only диапазона; сначала изолировать дистанцию/
геометрию от HP, затем отдельный HP factor. Сохранить Mixed coverage,
bank/reward/MLP/update budget, CUDA, 4 instances x2 и заранее фиксированный
fresh eval. Начальные состояния из evaluation traces не использовать.
GRU и выбор оружия остаются отдельными последующими факторами.

## Проверяемые артефакты

Root: `workspace/artifacts/combat-parasite-curriculum-v4-20261006`.
`completion-status.json`, `comparison-audit.json`, `bank-audit.json`,
`stochastic-audit.json`, `regression-audit.json`, `movement-audit.json`,
`phase-coverage-audit.json`, `closure-contract-audit.json` и per-arm audits.
Generated bank auditor сначала ссылался на старое имя expanded-bank.json;
после исправления на curriculum-bank.json все audits завершены. Captures,
модели и критерии не менялись; технически неполных game batches не было.

- Parent weights SHA: `3c5cd86df29e73b1709a02b5e969e35cfa262ff4464c6cded022d403653a8d7d`.
- Parent checkpoint SHA: `969b8a9b174e193081a444f1444f726148c08a2687d0f3e3cc9f4da6925ed39c`.
- Control weights SHA: `559111041aa2c2d2285ee191276762edb42da9a2a49574c6aeab699a2d00f80b`.
- Control checkpoint SHA: `545ca9310d7afd5023efb613071334432c7fe6cdecab9604a6a4aa47aa489731`.
- Curriculum weights SHA: `f1ae305f4bd9065dee4eac9dd2826398983a3759660e88e00af0ab147629ee74`.
- Curriculum checkpoint SHA: `b49da8526c5d9ae9f8f6853dff99e38eb99306c127df7bbdb92295821aaeaa39`.
- Reward SHA: `3fb87a9f771a2cca578417fe17530bd3bcd70576540a5fa65dfdb8166b2ea005`.
- Source fingerprint: `2a8f02823840c7e4de95c3e7a35c1612e3bbbe94c520dbd85c58c74bada98aed`.
- Native fingerprint: `99ca5b165e8c1e2683c0ae8ecc5800385c75cedfd0895cf3dd7ebb928bb9a08c`.
