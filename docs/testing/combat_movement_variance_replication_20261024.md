# Повтор уменьшения случайности движения — 24.10.2026

Два независимых набора validation seeds на тех же24рецептах закрыты: **144native/96CUDA**, по48боёв дляsource/noise-only/rules. Первая оценка offset44, повторoffset45. Предыдущие8discovery условий не включены. Source policy, noise-only policy, registry и native environment между оценками совпали; seeds не пересекаются. Это новые seeds на существующих геометриях четырёх карт и двух точек каждой, не подтверждение полной кампании или human coop.

| Набор | Source победы | Noise-only победы | Rules победы |
|---|---:|---:|---:|
| Первый,24условия | 13 | 15 | 12 |
| Повтор,24условия | 10 | 13 | 11 |
| Вместе,48условий | 23 | 28 | 23 |

| Вместе | Смерти | Убийства | Получено HP | Нанесено HP |
|---|---:|---:|---:|---:|
| Source | 22 | 65 | 3078 | 12366 |
| Noise-only | 18 | 79 | 2909 | 13158 |
| Rules | 15 | 64 | 2218 | 11456 |

Парные новые/потерянные победы:noise/source7/2,noise/rules9/4. Описательные exact paired sign tests p0,17969 и0,26685: не общий статистический gate и не достаточное доказательство превосходства. У rules меньше смертей и урона. Разбиение по составу: при двух врагах noise19/24 против rules21/24; при трёх noise9/24 против rules2/24. Страты описательные, не основание выбирать удобное подмножество и объявлять победу модели.

## Действия в сумме двух оценок

Только actual native MG contacts в provider execution windows до конца первой жизни/цели:source916/1312 выстрелов с уроном живому врагу(69,82%),noise965/1352(71,38%). Solid contacts340→323. Стоячие grounded-visible кадры1710/4487(38,11%)→1730/4847(35,69%); большиеyaw356/5548(6,42%)→281/5498(5,11%). Attack без выбранной цели546/5548→291/5498; это команды, не физические промахи. Первая оценка отдельно имела ухудшениеMGдоли, повтор — улучшение. Неравные игровые состояния и длины боёв ограничивают причинное толкование. Лишние действия остаются; устранение вращений/выстрелов в стены не доказано.

## Согласованный чекпоинт и продолжение обучения

`combat-movement-variance-checkpoint-v1-20261024` закрыт: bytes weights совпадают с ранее оценённым noise-only(4b1097038137bff824149836e951f3c6126bde234047a07bc0d1e1e03149d479). CUDA probe143реальных TRAIN строк: rawactor/value неизменны, conditional density difference maxerror9,5367e−7. Сохранены actor/value/geometry, Adam clocks, RNG, история, config/retention; обнулены только моменты двух изменённых movement std координат. Checkpoint/optimizer roundtrip проверен. Persisted optimizer steps0; это миграция, не обучение. Старыеsourceтраектории использованы только для технической проверки.

Для нового чекпоинта собраны **32freshown-policyTRAIN/32CUDA,4298переходов**,offset216..219,8семействsite02 на4картах. Все32seed независимо сверены с полной frozen историей7326seeds. Это пока обучение наsite02, а оценка включаетsite01/site02. Native-credit и GAE аудиты пройдены; оценочные данные в fitting не включены.

`../combat_movement_variance_ppo_protocol_20261024.json` фиксирует2собственных32-caseраунда, полноуправляемые actor/critic/geometry, прежнюю reward13 и прежние KL/stepguards. Создан отдельный строгий adapter `scripts/train_combat_movement_variance_ppo_cuda.py`, который проверяет provenance exploration fork; исходный geometry trainer не меняется. Запущен первый CUDA update, результат обучения/качество ещё не заявлены. Для второго обновления потребуется новый собственный корпус. В8-case screen сравнивается только финальное второе обновление сsource,noise-onlyдообучения иrules. Default/promotionнет.

## Свидетельства

- `combat-movement-variance-replication-v1-20261024/complete.json`:72native/48CUDA.
- `combat-movement-variance-quality-audit-v2-20261024/report.json`:native MG/movement повторной оценки.
- `combat-movement-variance-two-seed-summary-v1-20261024/report.json`:48пар/вариант,all24recipes×2seeds,matchingmodels/registry/environment, inputSHA, descriptive strata.
- `combat-movement-variance-checkpoint-v1-20261024/report.json`:точныеevalweights и согласованныйoptimizer/checkpoint.
- `combat-movement-variance-training-corpus-v1-20261024/complete.json`:32TRAIN/32CUDA.
- `combat-movement-variance-ppo-chain-v1-20261024/freshness.json`:независимая проверкаTRAINfreshness;progress.json/logs — состояние первогоupdate.


## Первый update закрыт; второй поставлен в очередь

`combat-movement-variance-ppo-round1-v1-20261024` завершён наCUDA:4298TRAINпереходов,1336eligible;принято5actor+5geometryшагов и40criticшагов. FullKL0,00249334,baseKL0,00249412;оставшийсяactorbudgetостановленпрежнимguard,а невыданкак30выполненныхшагов. ValueMSE2,92857→0,65010наTRAIN неявляетсябоевымрезультатом. JSONreplaymaxerror0,checkpoint/optimizerroundtripexact,RNGpreserved. Trainingseconds8,62;основноевремяциклазанимаютсами32бояиихобработка. ВесаSHAa685ed224fdbc640fd5641c118e5158cf9dc0df8625548c1241cf38425e7aeec. Qualityпослеобученияещёнет.

Запущен`combat-movement-variance-training-corpus-v2-20261024`:ещё32собственныхTRAINсэтимчекпоинтом,offset220..223,16×2. Независимаяfreshnessпроверкав`combat-movement-variance-ppo-chain-v2-20261024/freshness.json`исключаеткакfullfrozenhistory,такипервыйкорпус. Живаяцепочка`combat-movement-variance-ppo-chain-v2-20261024`послесборапроведётnativecredit/GAEаудиты,второйCUDAupdate,затемscreenфинальныхвесов. Приошибкецепочкаостанавливается;неперезапускаетбоиавтоматически.

Screenзафиксировандоисходовв`../combat_movement_variance_ppo_screen_spec_20261024.json`:8условий,validationoffset46,source/noise-only/trained/rules,32native/24CUDA. Промежуточныйround1неоценивалсяидляqualityselectionнеиспользуется. Продолжениеобученияпоканедоказываетросткачества;defaultнеизменён.


24.10.2026. Noise-only continuation: второй32TRAIN/32CUDA корпус (4658rows) иupdate2 закрыты; всего64боя/8956переходов,13совместныхactor+geometryшагов,80critic. Final8-case screen закрыт32native/24CUDA:source2/8,noiseдо2/8,после2/8,rules4/8;gate=false,24-case расширения иpromotionнет. MG live fraction53,60%→46,43%,attack безtarget72→346. Следующая независимая ветка: head-only PPO на том же собственном первом32TRAINкорпусе, actor/std/baseAdam frozen,headLR.003,guards/reward прежние. CUDA v2 закрыт:9headsteps,baseKL0,categoryKL≈.0025,base actor/optimizer exact; technical RNG checker fix не изменил веса. Запущенfresh32native/24CUDA screen validation47 baseline/head/full-PPO/rules,16×2. См.testing/combat_movement_variance_ppo_20261024.md иtesting/combat_geometry_head_only_ppo_20261024.md.
