# Геометрия выстрела с прогнозом моторики, 23.10.2026

Продолжение `combat_action_effect_prediction_20261023.md`: отделяется полезность прогноза физического состояния от полезности выбора угла. Все development fits и numerical NN checks выполнены только CUDA. Политика/default не продвигаются по offline результатам.

## Cross-fitting и первый отрицательный результат

`combat-action-effect-shot-probe-v1-20261023` завершён:10505TRAIN shots/328sourcecases. Для каждого из двух внешних development folds forecaster для classifier TRAIN предсказывает строки из трёх внутренних whole-episode/seed folds; ни одна строка не используется для fitting/normalization своего forecaster. Внешний heldout получает прогноз от frozen outer checkpoint из предыдущего опыта. Группы и checkpoint hashes сохранены, внешний muzzle error совпал с предыдущим sealed report. Native targets никогда не подставляются в geometry inputs.

Шесть внутренних motor fits по300steps и восемь Linear214 classifier fits по200steps заняли около11,1с после загрузки. Нормализация classifier только по внешнему TRAIN. NN вычисления не выполнялись ни на CPU, ни в Go offline.

Первый вариант совмещал learned own-motor forecast с постоянным100ms упреждением цели. Он улучшил new-intervened heldout loss на отложенных seeds0,4582→0,4007, но ухудшил наsite02:0,5169→0,5424. Заранее заданный gate не пройден; этот вариант не направлен в runtime. Позднейший другой вариант не переписывает отрицательный результат.

## Разделение движения бота и цели

`combat-action-effect-shot-probe-v2-20261023` — заранее зафиксированная механистическая ablation. Внутренние моторные модели не переобучались: их веса, provenance, состав train/predict групп и normalization заново проверены; predictions получены на CUDA из sealed checkpoints. Обучены12classifiers, те же200steps и seed, около6,6с.

Development ошибки на новых intervened shots (меньше лучше):

| Geometry для Linear214 | Seed heldout,98shots | Site02 heldout,258shots |
|---|---:|---:|
| Centered current |0,4582|0,5169|
| Observed own velocity×100ms |0,4280|0,5270|
| Observed target velocity×100ms |0,4620|0,5547|
| Own+target velocity×100ms |0,4453|0,5733|
| Learned own motor, current observed target |**0,3932**|**0,4988**|
| Learned own motor + target velocity×100ms |0,4007|0,5424|

В данном опыте постоянная экстраполяция цели ухудшает перенос. Из этого не следует, что целям вообще не нужно упреждение или что100ms является измеренной сетевой задержкой. Выбран вариант с прогнозом собственной моторики и текущей наблюдаемой целью. Legacy heldout loss0,4615→0,4200 и0,4230→0,4090; условие regression budget≤0,02 выполнено. Его gates пройдены на обеих development folds. Это всё ещё model selection внутриTRAIN.

## Итоговый refit и контракт

`combat-action-effect-shot-predictor-v1-20261023` refit на всех10505TRAINshots. Три whole-battle seed%3 forecasters строят out-of-fold geometry для обучения Linear214. После фиксации этих признаков final forecaster обучается на всехTRAIN. Веса forecaster2949params, classifier215params. JSON round-trip обоих и combined scores проверен на CUDA. Внешний actor/value/std/Adam/RNG здесь не обучаются.

Новый контракт `combat_predicted_action_geometry_v1` включает centered muzzle, Motor192 normalization, отдельные ветви position76→32→3/direction21→16→2 и прогноз displacement3/pitch+yaw2. Target extrapolation строго0; native hand строго2, scales32units/5degrees. Старый geometryv2 остаётся отдельным контрактом. Metadata validator отвергает иные hand, scales, feature columns или подстановку motor в старую версию.

`internal/policy/action_effect.go` отделяет pure observed packing/projection от NN. Pure CLI никогда не исполняет learned motor; forecast values для offline geometry proof получены только наCUDA. NN forecast вызывается Go только в live `Decide`, при оценке25candidate yaw/pitch offsets. Actor sampling target/mode/weapon/pose/movement/fire и RNG остаются исходными. Новые LP не объявляются обычным on-policy PPO; стандартный PPO replay этого inference режима запрещён. Firemask, hard target lock и rule aiming не добавлены.

Проверки: pure geometry/metadata Go tests прошли; CUDA physical invariants прошли. Матрица10505реальных+3200candidate inputs прошла pure Go/CUDA packing/projection proof. FP32 soft-band deviation соответствует менее0,001world unit, ниже протокольного position quantum0,125. Проверка Go NN scores выполняется отдельно на реальных играх, не offline.

`combat-action-effect-aim-ranker-v1-20261023` — inference bundle, sealed наCUDA: actor/value/std/Adam/RNG/history точно совпали с `combat-conditional-fire-zero-v1-20261022`; добавлены execution metadata и новые predictor weights. Источник и предыдущие rankers не перезаписаны, default не изменён.

## Native screen

Зафиксирован `combat-action-effect-ranker-screen-spec-v1-20261023.json`:32native/8sourceCUDA, по8боёв start/old-ranker/motor-ranker/rules. Registry:base3/ware2,site01/02,ordinary/recovery. Validation offset27 выбран как неиспользованный после сверки1246historical plans/6562seed values. Восемь новых seeds одинаковы между вариантами для парного сравнения. Пул16инстансов ×2,port51000; порт49600отклонён до запуска из-за стороннего MSI процесса, он не останавливался.

Оба rankers после terminal evaluation проверяются отдельными live CUDA audits всех25scores/выбранных углов. Source восемь случаев проходят обычный CUDA replay. Native MG ray outcomes и action-efficiency оцениваются отдельно от выигрышей.

На момент записи pipeline запущен; боевой результат нового motor-ranker неизвестен. Малый положительный screen сам по себе не даёт promotion и не доказывает превосходство над rules на других картах/оружии. Исходная задача остаётся активной.


## Native screen завершён

`combat-action-effect-ranker-screen-v1-20261023/decision.json`:32native/8sourceCUDA завершены,exit0. Live CUDA подтвердил все25scores/выбранные углы: old360ranked/356changed frames; motor372/369. Это реальная проверка Go NN исполнения в игре; offline Go NN replay не выполнялся.

| Вариант | Победы | Убийства | Нанесено / получено | Смерти | Actual MG living damage / shots |
|---|---:|---:|---:|---:|---:|
| Start |3/8|11|1947 /466|3|150/255 (58,82%)|
| Historical ranker |5/8|14|2488 /457|3|191/277 (68,95%)|
| Motor ranker |4/8|16|2703 /568|3|180/280 (64,29%)|
| Rules |4/8|12|2263 /533|4|Этот MG audit не включает rules|

Motor против old:2полученные/3потерянные победы; против start2/1; против rules3/3. Нанесённый урон и kills выше old, но shots с живым уроном меньше и полученный урон выше. Solid contacts103/76/88,corpse2/10/12,targetless shots17/19/22 для start/old/motor. Muzzle-blocked0у всех.

Stationary-clear13,96%/32,62%/33,67%;large-yaw5,70%/4,38%/5,59%. Это малый descriptive screen; сравнение на других seeds ранее не показывало стабильного ухудшения stationary, поэтому конкретная причина здесь не доказана.

**Новый вариант не продвигается; расширенная confirmation не запускается.** Основная политика/default и опубликованные модели не заменены. Положительная точность TRAIN forecaster/shot-classifier не превратилась в преимущество native боя.

Следующий ограниченный шаг — audit фактической ошибки motor forecast и калибровки выбранных scores на уже закрытых native shots. Он должен различить ошибку физического прогноза, сдвиг распределения при argmax и недостаточность immediate-hit цели; данные validation используются только для диагностики, не для нового fitting. Не объявлять изменение мирового движения или конкретный механизм причиной без проверки. Новые игровые корпуса до этого аудита не нужны.
