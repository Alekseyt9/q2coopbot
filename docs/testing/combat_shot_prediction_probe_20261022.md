# Небольшой предиктор результата выстрела

22.10.2026. Реализован и выполнен `scripts/probe_combat_shot_prediction_cuda.py`. Это диагностика непосредственного результата действия; рабочий combat actor не меняется, fire threshold/filter не добавлен.

## Данные и контракт

Источник: только32TRAIN боя первого conditional-fire корпуса. Из них26боёв содержали1000валидных native MG shots:600нанесли урон живому монстру,392попали в solid,8вcorpse. Остальные6боёв не дают MG shot labels; отсутствие выстрела не размечается как промах.

Input181 полностью повторяет causal conditional-fire inputs: наблюдаемая geometry/weapon phase/выбранный противник и тип, sampled target/mode/movement/aim/vertical/weapon. Future/native сведения используются только как label: был ли немедленный health damage живому монстру от реального MG shot. Проверяются accepted TRAIN row, совпадение observation frame, provider ownership, matched/exclusive native window без recovery и sample attack.

Dataset сохранён как `dataset.pt` сCUDA float32 tensors; metadata — `members.jsonl.gz`. Массив не размножается в больших JSONL. Сами исходные captures не изменены.

## Заранее заданная проверка

Артефакты: `workspace/artifacts/combat-shot-prediction-probe-v1-20261022`.

Два разбиения целыми боями, без перемешивания кадров между train и held-out:

- Последний seed каждой family:19боёв сshots вtrain и7вheld-out,280held-outshots.
- Site02 полностью held-out:14боёв сshots вtrain и12вheld-out,480held-outshots.

Это development внутри исходного TRAIN, не новая игровая validation и не доказательство campaign generalization. Normalization вычисляется только на train. Fixed200steps, Adam LR0.001/weight_decay0.0001, один фиксированный seed; early stopping/checkpoint selection нет. Все NN train/evaluation и метрики — CUDA.

Сравниваются train constant, train rates отдельно для target0/selected target, linear181→1 и MLP181→32→16→1(6369params).

| Holdout | Модель | Log loss ↓ | Brier ↓ | AUC ↑ |
|---|---|---:|---:|---:|
| Отдельные seeds | Constant |0.6792|0.2430|0.5000|
| | Target strata |0.6210|0.2197|0.5776|
| | Linear |0.5713|0.1968|0.7396|
| | MLP |1.0248|0.2460|0.7218|
| Site02 | Constant |0.7035|0.2547|0.5000|
| | Target strata |0.5908|0.2062|0.6355|
| | Linear |0.5911|0.2012|0.7564|
| | MLP |1.1852|0.2384|0.7502|

MLP train log loss0.1403/0.1609 против held-out1.0248/1.1852 — сильное переобучение. Большая уверенность неправильных прогнозов ухудшает log loss даже при AUC выше случайного. Linear полезнее constant на отдельных seeds, но на site02 по log loss не лучше простой target-strata baseline. Среди selected-target shots site02 loss0.6724 противbaseline0.6600; это не готовое решение проблемы стен.

Проверка говорит о некоторой предсказуемости результата из текущих input features и слабой переносимости малой обучающей выборки. Она не доказывает, что классификатор знает оптимальное действие, что лишние выстрелы устранены или что pretraining улучшит PPO. Не превращать predicted hit probability в runtime fire mask.

## Следующий набор без новых серверных прогонов

Создан `workspace/artifacts/combat-shot-prediction-source-registry-v1-20261022/index.json`: **9закрытых TRAIN-only корпусов,288боёв,40885eligible policy rows**. Закреплены complete/pool/plan/model/merged digests; у всех v10/precision contracts. Источники: motor feedback, four-round learning curve и текущий conditional-fire corpus. Validation/test captures исключены.

Обновление23.10: [объединённый dataset и larger probe](combat_shot_prediction_registry_20261023.md) закрыты:9248MGshots из247боёв. Linear geometry-holdout loss0.4922/AUC0.8601;MLP всё ещё хуже поgeometry logloss. Подготовлена policy-preserving linear hidden basis;fresh native compatibility запущена. Пункты ниже описывают исходные требования, а текущий статус приведён в новом отчёте.

288 — число исходных боёв, не гарантированное число уникальных траекторий;40885 — policy rows, не количество MG shots. Сбор объединённого shot dataset ещё не выполнен. Перед fit нужно:

1. Извлечь реальные MG labels из каждого TRAIN pool по тем же native attribution правилам; записать counts и исключённые/пустые бои.
2. Найти копии одинаковых battle trajectories; группировать все варианты одного episode/engine seed в одну сторону split. Недостаточно случайно разделить shot rows.
3. На большем наборе сохранить geometry holdout site02 и deterministic seed-fold; fit normalization только на train. Повторить тот же fixed linear/MLP probe, не выбирать checkpoint по held-out результату.
4. Только после устойчивого улучшения над простыми baseline обсуждать predictive representation pretraining новой fire branch. Любая policy initialization требует zero-output compatibility, checkpoint/Adam/RNG migration proof и нового native сравнения. Предиктор остаётся отдельной задачей до такой приёмки.
