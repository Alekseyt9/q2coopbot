# Свежие поздние native пары: новая когорта

23.10.2026. Фиксируется после трёхfixed500 GPU прототипов (flat/calibrator/shared), ни один не прошёл development gate. Цель — расширить число независимых ситуаций, не выбирать checkpoint/split по удобным исходам. Default и sourcebody S4 не меняются.

## Baseline16

Тот же запечатанный constant baseline S4,8registry условий base3/ware2 site01/02 ordinary/recovery. По2новых TRAIN seed наусловие, минимальный свободный общий offset≥112 по доступной истории.16native/16CUDA, пул до16,×2. Старые16baseline captures/17label states не перезаписываются. История удалённых планов не объявляется полностью доступной.

До альтернативных labels зафиксировать split новых seeds: lower slot каждого episode — train, upper slot — development. Старые7train/4development seed сохраняют прежнюю принадлежность. Глобальный registry split этих сцен остаётся TRAIN; это внутреннее разделение, не независимая validation победности. Новыеdevelopment никогда не включаются в gradient.

## Поздние состояния

В каждом новом baseline выбрать до2состояний: один реальный MG solid miss и один live-monster hit. Только matched/exclusive/unrecovered first-life provider windows, один native MG shot, current target eligible и baseline category0. Кадр не раньше первого подтверждённого provider frame+10. Предпочитать движение≥50units/s и |recoil pitch|≥1°, затем меньшийframe. Между двумя выбранными моментами≥5frames. Если позднего подходящего состояния или второй категории нет — показать отсутствие, не возвращаться к ранним кадрам ради квоты.

Отбор использует только baseline факты, не будущие результаты альтернатив. Показать полное покрытие всех16seed, включая recovery безMG. До2состояний/seed сохраняют группировку данных; разные состояния одного боя не пересекают split.

Для каждого выбранного состояния baseline0+2лучших ненулевых frozen hint кандидата (ties поиндексу)+2других поSHA256(seed:frame:category), безповторов. Metadata one-frame logits изменяются ровно в указанном кадре, внеего исходный kernel. Все neural параметры/source checkpoints неизменны; отдельные training checkpoints накаждую копию не нужны — указать общую source provenance и запечатанные веса. Сетка25прежняя, никакого hidden native state вpolicyinputs. Полная CUDA проверка probabilities/value/memory и native prefix обязательна.

Все новые пары запустить одним динамическим пулом16×2 (не отдельнымпуломнакаждыйконтекст). Maximum32contexts/160native/160CUDA, фактическийобъём определяется доступностью кадров. Development acquisition имеет явную purpose и сохраняет split вlabels; это не разрешение его обучения и не production quality evaluation. Старый запрет обычногоPPO дляprobe сохраняется.

После приёмки создать child dataset сparent17, сохранивстарые записи/split. Новый fitting protocol фиксировать отдельно до gradient; не запускать игры качества для уже отклонённых прототипов. Если данных дляcoverageнедостаточно или все±5°вариантыплохие, зафиксироватьрезультат и разбирать physics/labels/диапазон, не ослаблятьgateзаднимчислом.
