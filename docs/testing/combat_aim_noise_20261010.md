# Доводка прицела: шум, переключения и дистанция

## Native variance experiment

`aim-noise-eval-v1-20261010` запущен после полностью закрытого curriculum560
и отрицательного independent-test eligibility. Один frozen parent actor,
continuous yaw/pitch std factors1/0.5/0.25, две sampling ветки20261011/20261012
на одинаковых80 registered validation24..27 условиях плюс rules80:560 боёв,
16 refill slots,×2. Actual evaluation Python PID14228 подтверждён живым.

`scripts/run_combat_aim_noise_evaluation.py` меняет только log_std[2:4]
посредством прибавления log(factor). Movement log_std[0:2], все веса,
categorical target/aim-mode/fire/vertical/weapon, memory и features одинаковы.
Factor1 равен исходной JSON metadata целиком; изменённые std допустимы
contract[-8,1]. Проверены реальные model metadata и syntax; это не NN replay.
Изменение распределения действий может изменить наблюдения и последующие
выходы той же сети. Categorical aim mode/target всё ещё stochastic.

Исходные yaw/pitch log_std:−3.901834/−3.903350. Coarse action scale180°,
fine scale15° (`internal/policy/precision_head.go`). Фиксированный noise factor
не является адаптацией к дальности или native muzzle parallax correction.

До dispatch проверены complete diagnostics,ownership/storage seals,
development-only selection, source bindings, полное совпадение generated
fixtures/engine seeds всех7 планов. Captures сжимаются terminal-member LZX
с сохранением SHA. После окончания выполняются member proofs, actual sampling
config audit,9 diagnostics,ownership,outcome/storage. Парные effects и
дополнительные aim transitions ждут actual OS handle14228. Результатов новой
оценки пока нет; тестовый split и promotion не затрагиваются.

## Подтверждённые наблюдения reference560

`report_combat_aim_transitions.py` добавляет отчёт на sealed480 learned
captures curriculum reference:480 source receipts/steps hashes. Только
first-life provider команды послеframe100, native matched/exclusive.
Смежность требует следующего frame и одинаковых connection/map/spawncount/
actor/life/weapon. Явная выбранная цель с track, clear shot и известным bbox;
невидимая/невыбранная цель не заменяется ближайшим противником.

| Parent arm | Стабильная цель и режим: кадров / ошибка | Смена режима той же цели | Смена цели |
|---|---:|---:|---:|
| A |2360 /5.65°|359 /8.82°|386 /12.43°|
| B |2358 /6.15°|420 /9.40°|338 /12.28°|

Это applied view-ray к текущему bbox. Не учитывает lead/recoil/muzzle
parallax; не точность пуль. Mode transitions имеют больший угол самой
команды:10.77/11.53° против4.25/4.58° на стабильных кадрах. Корреляция
не доказывает вред переключения: сложное наведение может вызывать смену
режима. Strata по actual weapon/range/transition сохранены для проверки.

Near≤128: parentA995 кадров/9.11°, B1290/8.76°. Medium128..512:
A2210/5.79°, B1897/6.27°. Far>512: A0, B25 (только3 firing frames).
Следовательно, текущая оценка не доказывает поведение на дальней дистанции.

Native machinegun selected-target shot error после recoil:parentA567 shots/
6.30°, B480/5.72°; до recoil11.63/10.84°. Это отдельная native shot
диагностика. Она показывает частичную компенсацию отдачи, а view-ray таблица
выше измеряет другой угол и другую выборку.

## Следующее решение

Сначала закрыть variance experiment и сравнить обе RNG ветки с native hits,
победами и transition/range strata. Не переносить удачный single-arm результат
на все архитектуры. Если уменьшение continuous noise не помогает, следующий
отдельный inference эксперимент должен изолировать categorical mode/target
sampling, сохранив randomness движения; live Go источник меняется только после
закрытия текущего cohort. При пользе — повторить подход для остальных моделей
и проверить на свежих development условиях перед reserved independent test.
Дальнюю стрельбу добавлять в registry отдельными проверенными BSP conditions,
не считать25 кадров достаточной оценкой.

## Сводка качества и условий стрельбы

Добавлен `report_combat_aim_comparison.py`: формирует `aim-comparison.json/.md`
из sealed quality,9 diagnostics acceptance,clean ownership/storage и actual
sampling config audit. Перед чтением сравнивает SHA каждого использованного
diagnostic report и protocol binding. Модель не запускает. На reference560
реальный запуск прошёл для всех7 вариантов; таблицы содержат победы/смерти/
урон, native MG shots/live damage fraction, selected shot angle after recoil,
distance strata и moving/stationary shot strata.

ParentA distance<128:111 selected shots/8.09°;128..256:343/5.78°;
distance≥256:113/6.11°. ParentB соответственно100/9.18°,286/4.87°,94/4.63°.
Distance здесь от native muzzle до observed selected bbox, поэтому отличается
от расстояния в provider frame report. Категория≥256 не доказывает дальнюю
стрельбу>512. Legacy rules не объявляют выбранную цель: selected error остаётся
unknown, а их native MG damage fraction известна; подмены цели ближайшим нет.

Current560 aim-noise в очереди имеет compact reporter за actual handle14228.
Уже запущенный waiter загрузил первоначальную версию reporter (без отдельной
motion таблицы); после закрытия оценки актуальный reporter можно повторно
запустить для дополнения этой таблицы без повторных native боёв. Это отличие
формата дополнительной сводки,9 обязательных diagnostics не меняются.

## Native muzzle parallax

Добавлен `report_combat_muzzle_parallax.py`, реальный запуск на sealed
reference560 завершён,560 source receipts/steps/server SHA сохранены в
`muzzle-parallax.json`. Native selected MG shot counts567/480 для parentA/B
и post-recoil angle6.298664/5.718420° совпали с обязательной shot диагностикой.
Каждый измеренный выстрел сравнивает два направления к одной и той же
наблюдаемой bbox-точке: из snapshot eye и из фактического native muzzle.
Только native matched/exclusive windows, first life afterframe100, explicit
target+track. Серверные позиции врагов не используются и не передаются модели.

| Parent arm | Дистанция от muzzle | Motion | Shots | Parallax | Native shot error after recoil |
|---|---|---|---:|---:|---:|
| A |<128|moving|89|10.22°|8.64°|
| A |<128|stationary|22|5.26°|5.85°|
| B |<128|moving|70|11.89°|10.75°|
| B |<128|stationary|30|4.73°|5.51°|
| A |128..256|moving|277|5.70°|5.95°|
| A |128..256|stationary|66|3.02°|5.06°|
| B |128..256|moving|237|5.08°|4.85°|
| B |128..256|stationary|49|2.74°|4.93°|

Moving здесь native speed≥10. Snapshot eye→native muzzle displacement near
moving20.15/22.26 units, stationary9.25/8.53. Смещение объединяет собственное
движение/время между кадрами, stance и weapon muzzle offset. По этим данным
нельзя приписать весь parallax движению или сложить parallax с ошибкой прицела.
Цель остаётся на наблюдаемой позиции без extrapolation; random spread,
moving-target staleness и causal counterfactual не исправляются этим отчётом.

Та же дополнительная диагностика поставлена за actual current evaluation
handle14228. Перед следующей правкой actor оценивать и noise factor, и
conditional moving-near/mode-switch ошибки. Native muzzle остаётся offline
diagnostic/label, не runtime oracle. Текущие веса не менялись; принятого
улучшения по новым шумам пока нет.

## Core quality560: continuous noise

Все560 native captures завершены без ошибок,560 member proofs приняты.
Actual sampling configs480,повторённых stochastic pairs0. Quality SHA256:
`3735ffe3234b9f0de8d925b147e0945573d2e0d5625914f9ec3a2fc04fae7815`.

| Noise factor | Wins armA / armB | Deaths armA / armB | Mean wins /80 |
|---|---:|---:|---:|
|1|70 /63|7 /11|66.5|
|0.5|67 /60|8 /13|63.5|
|0.25|70 /64|7 /11|67.0|
|rules|68|8|68.0|

Factor1 воспроизвёл core победы/смерти предыдущего parent reference.
Half noise хуже на обеих ветках; quarter добавил одну победу только наarmB,
среднее всё ещё ниже rules. Приобретённые/потерянные победы,9 diagnostics,
ownership/storage/outcome и дополнительные parallax/transition summaries
ещё обрабатываются actual evaluation process14228. Не считать+1 доказанным
улучшением и не переносить на все архитектуры.

За тем же actual handle поставлен sealed selection
`aim-noise-test-selection-v1-20261010` с тремя заранее указанными groups.
Решение будет принято после closure, test dispatch/promotion отсутствуют.
Следующая гипотеза при отрицательном gate: отдельно проверить categorical
aim-mode/target randomness. Непрерывный yaw/pitch noise и случайность
categorical выбора являются разными механизмами. Новая live policy правка
допустима только после закрытия текущего source-bound cohort.
