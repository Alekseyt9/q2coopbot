# Прогноз средней величины преимущества: CUDA discovery и frozen confirmation

26.10.2026. Использован новый continuation registry с 20 состояниями. Для автономного full-return supervision подходят 14 состояний на 13 картах; шесть mixed-control состояний сохранены в исходном реестре и явно исключены из fitting. Ничьи не удалялись. Средняя цель получена из всех 3/7 парных продолжений каждого состояния; один мир имеет единичный суммарный вес независимо от числа streams.

## Зафиксированный discovery протокол

`probe_combat_mean_advantage_cuda.py`: CUDA float64, weighted ridge с непенализируемым intercept, четыре внешних whole-map folds. Три внутренних TRAIN-map folds выбирают ridge из [.1,1,10,100]. Normalization только по текущему TRAIN, std floor=.05. Primary — physical24, secondary — frozen16, baseline — TRAIN constant. Аналитические CUDA regressions проверили известные решения прямой, ridge shrinkage и weighted intercept. Source actor не обучается.

Physical24 — прежняя наблюдаемая геометрия source/anchor и requested-motion contrasts; это не предсказанная будущая физика. Frozen16 — выход первых двух замороженных слоёв source wide head. Диагностический readout имеет соответственно 25 или 17 параметров. Входы не включают future return или stream seed.

Для дополнительной root-surrogate оценки используется небольшое изменение logit: `.5*tanh(predicted_mean/TRAIN_RMS_floor1)`. Эта же формула применяется к постоянному прогнозу. Затем оценивается `(new_probability - source_probability) * observed_mean_delta`. Будущая политика во всех исходных боях остаётся source: это surrogate изменения в одном root, а не качество нового контроллера на протяжении боя.

## Discovery результат

Артефакт: `workspace/artifacts/combat-mean-advantage-probe-v1-20261026/report.json`.

| Прогноз | OOF MSE | Root surrogate gain относительно неизменного source |
|---|---:|---:|
| TRAIN constant | 3.326062 | −0.003744 |
| Physical24, primary | 4.436888 | −0.014107 |
| Frozen16, secondary | 2.389879 | +0.036405 |

Primary gate **не пройден**: physical24 MSE ratio 1.333977. Его excess gain над constant −0.010363, descriptive fixed-OOF map-bootstrap interval [−0.032602,+0.001969]. Это не coverage guarantee: всего 13 карт, predictions фиксированы, reused TRAIN states и streams коррелированы. Предварительная оценка variance/n самих средних targets — 1.889248; она не использовалась как точный noise floor или основание для inverse-variance weighting.

Frozen16 дал примерно на 28% меньшую ошибку, но был **вторичным** вариантом. Он не объявлен primary победителем и не продвинут в игру. Нельзя выбирать его после просмотра таблицы и считать ту же таблицу независимым подтверждением. Новых игровых весов нет.

## Зафиксированное продолжение

`prepare_combat_mean_confirmation_cuda.py` заморозил secondary hypothesis до новых игр: final ridge fit только на 14 TRAIN состояниях, lambda выбрана четырьмя TRAIN-map folds из прежней сетки; выбрана .1. TRAIN constant, normalization, correction scale, coefficients и predictions сохранены в `combat-mean-confirmation-frozen-v1-20261026/`. Checkpoint имеет `actor_compatible=false`; это readout, не игровой PPO checkpoint.

Следующие восемь SHA-ranked различных миров выбираются без outcome/preference/prediction filter, исключая **все 20 известных миров**, а не только совпавшие root ID. Карты: hangar2, ware1, hangar1, lab, command, space, mine3, base3. Для этого regression это новые состояния; они остаются частью исторического TRAIN teacher corpus и не являются глобальным final-test на незнакомых картах.

`run_combat_mean_confirmation_cases.py` связывает заранее сохранённые predictions/inputs/coefficients по SHA и собирает 72 ветви: восемь миров × три продолжения × baseline/choice0/choice1. Source actor и future policy прежние. Сиды suffix выводятся из `(wide-mean-confirm-v1, record_id, future)`, reset на root+1. Применяются прежние строгие trajectory/terminal/historical-root проверки, 16 слотов ×2, CUDA-only numerical replay.

`evaluate_combat_mean_confirmation_cuda.py` не переобучает readout. Он сверяет source root logit с native записью, сохраняет все восемь состояний, а mixed-control исключает из автономного mean только целым состоянием. Gate заранее требует ≥6 автономных состояний, MSE ratio ≤.95 относительно frozen TRAIN constant, положительный root surrogate gain относительно source/constant и положительную нижнюю границу descriptive map bootstrap excess. Даже проход этого gate не означает native качество нового actor и не включает automatic promotion.

На момент записи сбор `combat-mean-confirmation-cases-v1-20261026` ещё выполняется. Завершение и качество подтверждения следует проверять по фактическому процессу, `complete.json` сбора и отдельному confirmation evaluation report.

## Confirmation завершено: преимущество не перенеслось

Сбор завершился по исходному протоколу: **72/72 native и 72/72 CUDA**. Предсказания и readout остались зафиксированными. У семи из восьми миров все пары автономны; lab сохранён отдельно как mixed-control. По зафиксированному критерию минимальных шести автономных состояний данных достаточно для решения gate, но это по-прежнему очень небольшой исследовательский набор.

Первоначальный evaluator v1 отказал: проверка source logit ошибочно применялась также к диагностическим choice0/choice1, где logit намеренно равен −20/+20. Оригинальный отказ сохранён в `combat-mean-confirmation-evaluation-v1-20261026/rejection.json`. Отдельная v2 проверяет source logit только у baseline, а у clamp ветвей требует точные ±20 и заданный choice. Автоматически проверено, что **весь остальной код идентичен**, включая predictions, score, bootstrap seed и gate. Исправление связывает старый/новый SHA в `combat-mean-confirmation-evaluator-correction-v2-20261026.json`; бои не повторялись.

Результат `combat-mean-confirmation-evaluation-v2-20261026/report.json`:

| Прогноз | Confirmation MSE | Root surrogate gain |
|---|---:|---:|
| Frozen TRAIN constant | 0.321648 | −0.000153 |
| Frozen16 | 1.001617 | +0.000045 |

MSE ratio **3.114018**, descriptive excess-gain interval **[−0.009000,+0.008828]**, `confirmation_gate_passed=false`. Замороженный readout ошибся по знаку среднего на hangar1/space/mine3. Почти нулевой surrogate gain не исправляет отсутствие переноса. Readout не экспортирован в игровой actor; дообучения по этой confirmation выборке и подбора другого lambda не было.

Все восемь новых состояний сохранены, включая lab. Версия `scripts/scenarios/combat-teachers/continuations-v2.json` ведёт к корпусу `combat-continuation-evidence-v2-20261026`: 28 разных миров с парными продолжениями. Прежний 20-state индекс остаётся неизменным, чтобы TRAIN и frozen confirmation сохраняли исходные привязки. Неуспех регрессии не превращает эти наблюдения в бинарные train labels.

## Следующая отдельная проверка

После отрицательной confirmation прекращён этот readout-путь к actor. Новый `run_combat_wide_prior_screen.py` проверяет частоту выбора существующей геометрической опции непосредственно на полном бою: постоянные начальные вероятности .2/.5/.9, parent и rules. Три prior-варианта имеют одинаковые legacy actor/critic и два первых слоя gate; отличаются только последним bias. Это инициализация, **не результат обучения** и не применение непринятого mean readout.

Probability .5/.9 созданы существующей CUDA migration с точным сохранением legacy Adam/RNG и проверенным reload checkpoint. Заранее задан небольшой fresh validation screen на восьми MG-start условиях: 40 native /32 CUDA, 16 слотов ×2. История seed и статические geometry preflight проверяются до боёв; outcome filter отсутствует. Primary — goal первой жизни без active rules control; доказанный passive goal-stop не считается takeover. Проход означает только основание для более крупного fresh screen, automatic promotion выключен.
