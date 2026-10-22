# Калибровка вероятности coarse/fine после balanced BC

Статус:train-only CUDA calibration,independent audit иmatched32 native/24 CUDA probe закрыты. Побед не прибавилось;large screen/promotion не назначены.

## Проверяемая причина

В [предыдущем BC](combat_mobile_motion_bc_20261022.md) редкий coarse mode получил такой же вес в CE,как fine. В данных1881 train labels:coarse66 (3.51%),fine1815;development coarse23/1255 (1.83%). Но у aim-weight component средняя coarse probability17.11% наtrain и8.98% наdevelopment. Такое изменение priors — гипотеза регрессии,а не доказательство,что все лишние повороты происходят из-за него.

Ранее source/aim/full component matrix дала4/8,5/8,3/8 соответственно;positive aim signal слабый,yaw/targetless attacks хуже. До дорогого screen нужна малая проверка proper probability calibration.

## Fit без evaluation leakage

`calibrate_registered_aim_modes_cuda.py` берёт замороженный `combat-rules-mobile-aim-weights-v1-20261022` и полные train histories прежних native teacher demonstrations. Не меняются encoder,aim direction weights,movement,target/fire/weapon/vertical,std. На target slots с обоими классами выбирается только один logit-difference intercept,минимизирующий естественный,а не balanced,train NLL. Single-class slots пропускаются,никакой hard mask крупного поворота не добавлен.

Алгоритм:64 итерации CUDA float64 bisection по нулю производной,то есть средняя coarse probability должна совпасть с train label frequency. Symmetric половины shift добавляются к двум base head biases;softmax normalisation сохраняется. Получены только slot1 biases65/66,shift(coarse−fine)=−2.8421054. У других slots нет достаточных labels,они оставлены прежними. Это один fitted scalar,не новые слои или глобальное ограничение углов.

Protocol зафиксирован до fit. Development читается только после завершения train fit;intercept не подбирается по нему,порогам точности или native бою. Новый root `combat-rules-mobile-aim-calibrated-v1-20261022`,со своим CUDA checkpoint/JSON/seal и snapshot calibrator.py. Value/Adam/config/PPO consumption history сохранены;actor Adam уже был пустым после BC и остаётся пустым. Следующий PPO требует свежих own-policy rollouts.

## Offline результат

| Метрика | Train до/после | Development до/после |
|---|---:|---:|
| Natural mode CE |0.2099→0.0629|0.1063→0.0293|
| Coarse probability mean |17.11%→3.51%|8.98%→2.00%|
| Coarse label fraction |3.51%|1.83%|
| Coarse argmax fraction |10.47%→1.86%|4.86%→1.27%|
| Mode agreement |92.82%→97.82%|96.81%→98.80%|

На тех же teacher histories,target фиксирован явным доступным teacher entity;CUDA world-input audit использует собственный mean movement/argmax mode. Development aim RMSE:source3.987°,uncalibrated aim4.324°,calibrated3.464°. При фиксированном teacher mode aim direction RMSE не меняется(4.234°);это подтверждает вмешательство в выбор mode,не перенастройку направления. Mean movement/world-input ошибка остаётся примерно прежней. Это deterministic mean diagnostic,не полный stochastic combat result.

`audit_registered_aim_mode_calibration_cuda.py` независимо проверил,что меняются только bias65/66,остальные actor tensors/std/value/оба Adam/config/history сохранены. Delta проверена сatol2e−6 для float32;неизменные tensors сравниваются точно. На1026 native source rows/1026context rows memory error9.09e−6,остальные raw outputs совпали точно. JSON/checkpoint finite и согласованы.

## Fresh native probe

`combat-aim-mode-calibration-probe-evaluation-v1-20261022`,offset624,порт45600,16 ×2. Четыре варианта на одних8 новых условиях(4ordinary/4recovery):лучший source target/fire BC,uncalibrated aim component,calibrated aim component,исправленные mobile rules. Всего32native/24CUDA,пять парных сравнений. Веса закреплены до старта,automatic promotion=false.

Report bundle допускает4-variant component probe наряду с прежним5-variant иlegacy3-way. Проверяются уникальные identities,budget и всеlearned CUDA replay;четыре focused tests прошли. Никакой новый native pool не выполняется одновременно с этим.

Закрытый результат должен включать wins/dealt/received,stationary/yaw/targetless actions,реальные native shots и ownership. Правильная частота mode/NLL не является доказательством побед или эффективных движений. При отрицательном probe не назначать большой screen автоматически.

## Закрытый probe

Все32native/24CUDA и пять отчётов закрыты;hashes verified. Decision `calibration_not_advanced_to_large_screen`,promotion=false. Оценочные сцены не использованы для нового fitting.

| Метрика | Source target/fire | Uncalibrated aim | Calibrated aim | Mobile rules |
|---|---:|---:|---:|---:|
| Победа |3/8|3/8|3/8|5/8|
| Убито монстров |7|10|9|13|
| Нанесённый урон |1504|1854|1716|2282|
| Полученный урон |423|478|437|314|
| Большой yaw ≥30° |4.20%|7.05%|5.68%|—|
| Clear-enemy standing |11.75%|16.62%|13.49%|—|
| Attack без selected target |39.06%|34.31%|23.96%|—|
| Реальный MG damage contact |69/146(47.26%)|60/156(38.46%)|62/144(43.06%)|—|

Калибровка исправила probabilistic model и часть лишних действий относительно uncalibrated aim,но не превзошла source поyaw/standing/received/MG rate и не улучшила wins. Этот инструмент сохраняется как experimental calibration,а не новый default. Следующий опыт использует [совместные mobile labels и encoder adaptation](combat_joint_mobile_encoder_20261022.md),свежие native условия и исходные source weights.
