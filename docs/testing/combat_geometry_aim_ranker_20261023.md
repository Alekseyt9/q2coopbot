# Выбор углов обученным MG-предиктором

23.10.2026. Prototype после `combat_action_geometry_predictor_20261023.md`. Цель — проверить результат в реальном бою, не подменять его качеством offline classifier.

## Исполнение

Source: `combat-conditional-fire-zero-v1-20261022`. Дополнительный learned predictor — Linear214 с 215 параметрами, обученный на pooled TRAIN. Политика выбирает target, coarse/fine, movement, vertical, weapon и fire прежним sampler. В известном MG-домене с доступной выбранной наблюдаемой bbox predictor ранжирует 25 yaw/pitch candidates: сетка offset0/−5/−2.5/+2.5/+5° относительно sampled actor aim. Исходный candidate идёт первым; почти равные оценки сохраняют прежний выбор. Fine/coarse bounds сохраняются.

Итоговый угол выбирает learned score, а не минимизация геометрической ошибки правилами. World raycasts, native outcomes и будущие наблюдения не используются. Геометрия — вход predictor. Fire veto/mask не добавлен; conditional fire, если есть, вычисляется после окончательного выбора aim. Новых RNG draws нет. Команды движения сохраняются, но мировой вектор движения и дальнейшая траектория могут измениться вместе с взглядом.

Режим `geometry_shot_rank_aim_v1` явно inference-only. `IsStochastic`/Python training validators/evaluator не допускают его как стандартный PPO sampler; записанная стандартная log-probability контрфактическая. Исходные actor/value/std, Adam/RNG/history совпали на CUDA с source. Новые predictor weights добавлены отдельно. Default не меняется.

## Пограничная геометрия

Первая проверка pure Go/CUDA geometry покрыла 9248 исходных inputs и 3200 candidate inputs. В одном из 12448 случаев двоичный ray/bbox hit расходился FP32/FP64 из-за почти касательного луча. FP64 reference совпадал; это реальный дефект устойчивости бинарного признака.

В `combat_observed_action_geometry_v2` заменены только два intersection inputs: `clamp((slab_gap+1)/2,0,1)`, плавный переход в полосе ±1 world unit. Input181/recoil4, labels, cases и folds сохранены. Dataset пересчитан из компактного binary, без перечитывания больших captures. Переобученные fixed200-step линейные метрики сохранились: seed/site02 loss0.4625/0.4287, site02 AUC0.8851. Старый predictor geometryv1 к runtime не допускается.

Принятая проверка: `combat-action-geometry-native-parity-v3-20261023/complete.json`. Pure Go geometry не выполняла NN inference. CUDA FP64 и Go совпали по математике; максимальное FP32 расхождение непрерывных признаков 0.0000562, soft-hit признаков 0.000205. Последнее соответствует примерно 0.00041 world unit в переходной полосе, существенно меньше квантования наблюдаемой позиции 0.125 unit. Допуск для soft inputs связан с этой физической шкалой, а не с побитным равенством строк.

CUDA coordinate/gradient tests и focused Go geometry/sampler metadata tests прошли. Go learned scoring проверяется только на реальных игровых вызовах; offline native neural benchmark не запускался.

## До начала pilot

Predictor: `combat-geometry-shot-predictor-v2-20261023`, полный TRAIN refit 9248 shots и JSON/CUDA проверка. Model fork: `combat-geometry-aim-ranker-v1-20261023`. Имеющиеся categorical greedy режимы сохраняют прежнее поведение; новый ranker сохраняет stochastic target/mode.

Заранее задан `combat-geometry-aim-ranker-screen-spec-v1-20261023.json`: fresh validation offset764, восемь base3/ware2 условий (две точки, ordinary/recovery), start/ranker/rules, 24 native боя, пул16 ×2. Восемь исходных stochastic captures проверяются обычным CUDA replay; ranker получает отдельный live audit, без ложной PPO-проверки.

`audit_combat_geometry_aim_ranker_cuda.py` должен подтвердить активацию и изменение aim, затем на CUDA пересчитать learned scores всех 25 candidates по реальным native наблюдениям, проверить максимальность выбранной оценки и совпадение выбранных углов. Pure Go CLI используется offline только для упаковки наблюдений и geometry, без learned score. Помимо качества боя проверяются реальные MG shots, контакты с геометрией/трупами, повороты и stationary-clear. Автоматическое назначение default запрещено.

## Native pilot завершён

`combat-geometry-aim-ranker-screen-v1-20261023/complete.json`: 24 native/8 CUDA, все основные audits завершены, процесс exit0. Отдельный `live-ranking-cuda/complete.json` подтверждает **393** matched provider кадра с ranking, **384** выбора не исходного угла. CUDA пересчёт всех 25 candidates подтвердил максимальность выбранной оценки и совпадение углов. Математических/активационных ошибок не найдено; это проверка исполнения, а не качества управления.

| Вариант | Победы | Убийства | Нанесённый / полученный урон | Смерти |
|---|---:|---:|---:|---:|
| Start |4/8|15|2492 / 518|3|
| Learned ranker |4/8|14|2420 / 461|2|
| Rules |7/8|18|2865 / 476|1|

Против start одна полученная и одна потерянная победа. Уменьшение полученного урона/смертей не сопровождается улучшением победности или нанесённого урона. Кандидат отклонён для promotion/default; expanded confirmation не запускается.

Реальные MG shots в exclusive first-life alive provider windows:

| Метрика | Start | Ranker |
|---|---:|---:|
| Live monster damage / все shots |158/283 (55,83%)|176/319 (55,17%)|
| Solid contacts |123|140|
| Corpse contacts |2|3|
| Без выбранной видимой bbox |28|37|
| Measured rays: damage любого живого monster |158/255|176/282|
| Damage именно выбранной цели среди measured rays |156/255|169/282|

Средняя post-recoil ошибка относительно прежней observed AimPoint22: 7,96° → 8,71°, signed pitch error +0,35° → +3,27°. Predictor учитывает весь bbox, поэтому эта метрика не обязана оптимизироваться; однако улучшения действительной damage fraction тоже не получилось. Большие yaw commands 5,47% → 4,60%; stationary-clear 20,79% → 19,40%; attack commands без target 18,44% → 29,30%. Последняя величина не равна числу реальных выстрелов.

## Что проверять далее

Улучшение классификации уже состоявшихся shots не обеспечило полезного выбора неисполненных действий. Возможная причина — слабая поддержка контрфактических действий в данных: исходный fine std около 0,30° на ось, а grid проверяет offsets до5°. Это гипотеза о распределении опыта, не доказанная причина неудачи.

Следующий эксперимент должен получать **реальные TRAIN labels при намеренно рандомизированных aim interventions** вокруг actor, а не выдавать offline геометрическую чувствительность за такие labels. Сохранить отдельный predictive-only recipe и sampler identity; запретить обычный PPO replay этих captures. Цель/оружие/fire/movement остаются у базовой политики, perturbation ограничено и помечено как exploration для обучения. Пул16 ×2, fresh TRAIN seeds; сначала проверить метки/активацию на малом наборе, затем собрать данные. Старые validation/test не присоединять к TRAIN. После обучения predictor на новом опыте повторить native pilot; до него не масштабировать ranking по картам и не включать default.
