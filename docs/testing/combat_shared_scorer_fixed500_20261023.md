# Shared scorers: fixed500 CUDA, отрицательный результат

23.10.2026. Все3прототипа обучены и проверены наCUDA,500шагов каждый, наодном17-state наборе с12train/5development. Ни один не прошёл заранее заданныйgate. Новыеnativequality прогоны иruntimeинтеграция не запускались.

| Голова | Параметры дополнения | Useful mass development | Собственный constant control |
|---|---:|---:|---:|
| Flat MLP500 |88665|0,111%|2,001%|
| Calibrator |2|около0,752%|около1,295%|
| Shared191→16→1 |3090|0,046%|1,920%|

Это средняя вероятность проверенных улучшающих действий на5targeted development состояниях, не точность/победность. Исходная S4 даёт около0,015%. Каждыйconstant получен только изtrain средних25вероятностей своегоfit. Масса непроверенных действий не получает syntheticlabels; conditional known HP не является полной ожидаемой полезностьюpolicy.

Flat500 повторяет слабый перенос предыдущегоFlat200: train полезнаямасса≈14,99%, development≈0,11%. Shared учится наtrain≈13,55%, ноdevelopment≈0,046%. Малыйcalibrator не запоминает rawсцену, однако тоже уступает своемуconstant наизвестных полезных действиях. Это не доказательство непригодностиPPO/attention/shared scoring вообще;17состояний и2исправимыхTRAIN случая мало покрываютдействия/ситуации, аnativeодинspread sample неоценивает ожидаемуюточность.

Source actor/value/std/fire/predictor и source файлы/чекпоинты неизменны. Новыефайлы имеют prototypekind/native_runtime_compatible=false, неготовыеигровыеPPOweights. Исходы всехfixed endpoints сохранены, noearlystop/checkpointselection. Разные заранее заданныеLR означаютотборпрактическихвариантов, анечистоеизолированиеархитектуры.

Следом свежаякогорта16baseline боёв идо2позднихone-frame контекстов/seed, включая движение/отдачу. Новыйsplitlower/upperseed фиксируетсядополученияальтернатив, старыеdevelopment не переходятвtrain. Протокол `../combat_native_aim_late_cohort_20261023.md`. Ненаращиватьто жеобучениебесконечно иневключатьgreedyrankerвdefault.

Артефакт: `workspace/artifacts/combat-native-shared-scorer-probe-v1-20261023` (report, per-context probabilities, head-only checkpoints, source snapshots). Протокол `../combat_native_shared_scorer_protocol_20261023.md`. Коммиты/push/default безизменений.
