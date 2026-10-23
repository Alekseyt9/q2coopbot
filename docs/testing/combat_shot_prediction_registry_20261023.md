# Shot prediction: общий TRAIN dataset и начальное представление

23.10.2026. Из девяти проверенных TRAIN корпусов собран компактный dataset. Predictive basis прошла CUDA/native compatibility, затем полный и fire-only PPO fits. Свежая оценка не показала прироста: start 3/8, full 1/8, fire-only 3/8, rules 4/8. Новые веса не продвигаются.

## Данные

Source recipe: `combat-shot-prediction-source-registry-v1-20261022/index.json`.
Builder: `scripts/build_combat_shot_prediction_registry_cuda.py`.
Dataset: `combat-shot-prediction-dataset-v2-20261023`.

- 288исходных TRAIN боёв,40885eligible policy rows. Ни validation, ни test gameplay captures не использованы.
- 247боёв дали **9248реальных MG shots**;41не дали подходящих shot labels. Отсутствие выстрела не размечается как промах.
- 5567live-monster damage,3536solid geometry,144corpse,1sky.
- Дубликатов полных ordered shot sequences внутри одинаковых episode/engine-seed групп не найдено. Сигнатура включает frame, наблюдаемые features, sampled action и outcome; это проверка одинакового supervised shot material, а не тождественности всех нестреляющих кадров.
- Каждая запись сопоставлена с accepted TRAIN row, provider matched/exclusive native window без recovery; sample.attack=true и observation frame совпадает. Native outcome используется только как label.

Хранятся только causal input181, label/masks в бинарном `dataset.pt` (**6762841bytes, около6.45MiB**) и сжатая metadata `members.jsonl.gz`. Исходные JSONL/asset files не копировались. Пулы разных behavior models объединяются только для predictive supervision; это **не** общий on-policy PPO rollout.

## Разбиения и fixed fit

`fit_combat_shot_prediction_registry_cuda.py`, protocol заранее записан в `combat-shot-prediction-registry-probe-v2-20261023`.

Варианты одного episode/engine seed всегда в одной стороне split. Seed-fold использует `seed%4==3`:183train и64held-out боя сshots,2512held-outshots. Geometry-fold полностью держит site02 отдельно:116train и131held-out боя,4713held-outshots. Это development внутри TRAIN; «не виденная геометрия» означает другую точку тех же двух карт base3/ware2, не неизвестную карту.

Mean/std вычисляются только на train. Одинаковые200steps, LR0.001, weight_decay0.0001, seed20261022; early stopping и выбор промежуточного checkpoint отсутствуют. Вся NN математика и оценка — CUDA. Время fit+metrics около4.76с после загрузки.

| Holdout | Модель | Log loss ↓ | Brier ↓ | AUC ↑ |
|---|---|---:|---:|---:|
| Seed groups | Constant |0.6668|0.2369|0.5000|
| | Target strata |0.6088|0.2136|0.5746|
| | Linear181 |0.4967|0.1648|0.8161|
| | MLP181→32→16→1 |0.4636|0.1526|0.8433|
| Site02 | Constant |0.6879|0.2472|0.5000|
| | Target strata |0.5726|0.2005|0.6382|
| | Linear181 |0.4922|0.1591|0.8601|
| | MLP181→32→16→1 |0.6362|0.1710|0.8557|

У selected-target-only shots linear loss0.5264/0.5549 против target-strata0.6456/0.6464. Значит, улучшение не сводится только к распознаванию target0. MLP стала полезнее на seeds, но её уверенность на другой geometry всё ещё приводит к худшему loss, чем у простого baseline. Для следующей инициализации выбрана linear basis; это выбор алгоритма по development, не утверждение о неприкосновенной итоговой оценке.

## Hidden representation initialization

`initialize_combat_fire_prediction_basis_cuda.py` переобучил linear predictor на всех9248TRAIN shots с теми же200steps. Он преобразует learned affine score в два signed ReLU hidden units существующей fire branch. Остальные случайные units сохраняются; все units далее обучаемы. Размер боевой сети не увеличивается.

Последний fire layer остаётся **полностью нулевым**: branch добавляет0для любого finite input. Рабочая политика продолжает выбирать movement/aim/target/weapon/vertical/fire прежним способом; predicted hit probability не становится fire mask/threshold. Это изменение начального представления, не готовый новый контроллер.

`combat-fire-prediction-basis-zero-v1-20261023`:

- decoded learned score совпадает с исходным linear predictor на9248input rows сmaxerror6.44e-6;
- raw actor outputs и joint log-probability на5058реальных full-history TRAIN rows совпали точно с source;
- non-fire actor/value/std, Adam/RNG/consumed history сохранены; новая fire optimizer group по-прежнему пуста;
- checkpoint сохранён и повторно проверен на CUDA; changed hidden tensors записаны как явная model migration.

Fresh source/zero compatibility: `combat-fire-prediction-basis-zero-compat-v1-20261023`,8native/8CUDA,offset756,base3/ware2,16×2. **Strict full-trace и semantic-episode audits прошли:598provider frames exact**, кроме объявленной model version и допустимой freshness metadata. Код audit допускает source с прежним identically-zero fire block; оба zero outputs проверяются структурно, все non-fire model fields обязаны совпасть.

Начат свежий own-policy TRAIN корпус: `combat-fire-prediction-basis-training-v1-20261023/corpus`,32боя,16×2. Исправлен пропуск `*/corpus/*plan*.json` в historical seed scan; реальный ранее пропускавшийся plan включён. Отдельный `fresh-seed-audit.json` подтвердил32новых сида без пересечения с288TRAIN seeds predictive registry. Этот audit не исправляет старые корпуса и не меняет их receipts.

Заранее зафиксированы full/fire-only fits на одном новом корпусе, каждый до10actor/40critic steps, прежний reward и LR; затем fresh8-case/variant start/full/fire-only/rules screen,offset757. Исходный policy стартует с той же функцией, но с learned hidden features; это не гарантирует лучшего PPO update. Predictive scores сами по себе не дают приёмки качества боя или превосходства над rules; default не меняется.

## Закрытый PPO screen

`combat-fire-prediction-basis-training-v1-20261023/complete.json` и `evaluation/complete.json`: 32 свежих TRAIN боя, 3595 eligible rows, 32 native/32 CUDA для корпуса. Оба fit и их аудиты завершены; full обучает 194333 параметра, fire-only — 6369. Бюджеты одинаковы: 10 actor/40 critic. CUDA fit занял около 3,71 и 2,40 с; время игр сюда не входит.

Свежая validation: 32 native/24 CUDA, по восемь боёв на вариант, offset 757, пул 16 ×2. Источник чисел — `evaluation/quality-*.json`.

| Вариант | Победы | Убийства | Нанесённый урон | Полученный урон | Смерти |
|---|---:|---:|---:|---:|---:|
| Start |3/8|11|2227|585|4|
| Full |1/8|8|1864|654|6|
| Fire-only |3/8|11|2227|588|4|
| Rules |4/8|13|2406|526|4|

Full отклонён; fire-only не дал выигрыша. Это малый development screen, а не доказательство невозможности predictive обучения. Инициализация меняла hidden representation, поэтому сравнение с предыдущим другим TRAIN корпусом не изолирует причинный эффект предобучения. Default остаётся прежним.
