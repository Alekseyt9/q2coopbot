# Следующий опыт: предсказание наблюдаемого будущего

Статус: dataset, первый paired CUDA fit и повтор с другим seed реализованы и закрыты.
Результаты и ограничения: [первый predictive pilot](combat_future_prediction_20261022.md).
Польза прогноза для actor и боёв ещё не проверена.

## Почему этот этап

[Corrective BC](combat_reactive_advice_20261022.md) после малого положительного
smoke проиграл source на 24-case screen: 8/24 против 12/24, rules17/24.
Раздельные переносы movement/aim/target-fire не дали выигрыша на свежих 8 cases.
Поэтому не увеличиваем автоматически число BC steps или размер policy.

В [основном плане](../learned_combat_new_directions_20261021.md) предусмотрена
небольшая модель структурированного будущего. Сначала проверим прямой
supervised прогноз игровых наблюдений. Это baseline для дальнейшей
JEPA-подобной auxiliary задачи, а не уже выполненная JEPA/TD-MPC реализация.
Предсказательная точность сама по себе не принимается за улучшение боя.

## Данные и причинность

Первый источник — закрытый `combat-reactive-advice-corpus-v3-20261022`:
32 TRAIN и 16 development, четыре карты, обычные/recovery группы,
существующие native provenance и CUDA replay receipts.
Ошибочный v1 expert, его советы и данные оценочных smoke/screen/component
боёв не становятся учебными примерами. Рекомендации исправленного учителя
тоже не нужны: используется реальная траектория learner.

Изначально горизонт **один network/server step**, с проверенным интервалом
100 ms игрового времени, независимо от ×2 wall-clock.
Требуются соседние frames, одинаковые map/connection/spawncount/actor/life,
native matched exclusive dispatch без recovery и поддержанная длительность
фактически исполненной команды. Изменение жизни/карты, пропуск кадра,
неоднозначный dispatch исключаются, а не превращаются в нулевое будущее.
Для первой motion-задачи соседнее наблюдение должно быть живым; число
исключённых terminal/death samples указывается отдельно. Риск смерти
этим первым predictor не моделируется.

Вход: канонические наблюдаемые features v10 и **только текущее исполненное
действие**: movement, yaw/pitch delta, up, attack. Не включать будущие
действия behavior policy, награды, native shot-time poses или hidden health.
Текущие признаки уже содержат наблюдаемое оружие/отдачу/геометрию/историю.
Нет повторного добавления имеющихся признаков под видом новой информации.

Метки из следующего клиентского наблюдения:

* смещение самого бота;
* для максимум 8 текущих monster slots — мировое смещение врага,
  выраженное в локальных осях **текущего** yaw бота;
* присутствует ли в следующем наблюдении тот же observed track;
* clear-shot mask при наличии следующего наблюдения этого track.

Сопоставлять врагов по entity **и observed track**, не по будущему слоту,
дистанции или классу. Перестановка distance-sorted slots не является сменой
врага. При потере track нет координатной метки; disappearance не означает
доказанную смерть/физическую невидимость. ID служит join key, не NN feature.
Смещение self переводится в тот же текущий локальный базис.

## Baseline и фиксированный fit

Обязательные baseline: нулевое смещение и продолжение текущей наблюдаемой
скорости на dt. Неизвестная velocity имеет явную маску и отдельную страту;
не выдавать fallback0 за известную остановку. Присутствие/clear-shot — persistence
baseline с отдельными masks. Где лог содержит подходящий pre-dispatch
`pmove_ground_prediction`, дополнительно сравнить self motion с ним на общей
eligible подвыборке, учитывая `support_ambiguous` и ограничения модели.
Известная геометрия/физика не объявляется новым достижением нейросети.

Два одинаковых небольших MLP (2×128) на CUDA:

1. features + исполненное текущее действие;
2. те же входные размеры, но action channels обнулены.

Одинаковые начальные веса, TRAIN minibatches, seed, AdamW budget и loss masks.
Первый бюджет — 2000 updates, batch256, lr0.001, weight_decay0.0001.
Fixed final checkpoint; development не выбирает checkpoint, архитектуру
или коэффициенты. Контроль нужен, чтобы отделить условный эффект действия
от возможности запомнить особенности самого наблюдения.

Предсказывать residual к constant-velocity для motion, вероятности для
track-present/clear-shot. Нормировать displacement на 64 units;
SmoothL1 motion и BCE masks с явными знаменателями.
Неизвестные координаты не заменять обучающими нулями.
Никаких gradients в действующий actor и никаких новых runtime overrides.

## Приёмка и следующий gate

Сначала dataset audit: отсутствие пересечения train/development seeds,
join/reset/mask tests, источник/hash каждого case, число исключённых samples,
равенство feature vectors каноническому экспорту. Хранение — компактный
binary dataset и небольшой JSON manifest; assets и executables не копировать.

После fit: motion RMSE отдельно self/enemy, по осям, картам, known/unknown
velocity и observed monster type. Track/clear-shot оценивать Brier и
precision/recall, а не только accuracy частого класса. Показать case-weighted
среднее и разброс по development cases, наряду с pooled frame average.

Продолжать только если action-conditioned predictor воспроизводимо лучше
constant velocity на enemy motion и action-blind MLP на action-sensitive self
motion без крупного map-specific провала. Простое снижение TRAIN loss не gate.
Если не помогает, сначала установить причину по masks/actions/geometry,
а не переходить сразу к большой видеомодели или MPC.

Горизонты 3/5 steps потребуют отдельной схемы conditioning/rollout.
Нельзя скармливать реальные будущие действия как якобы доступные сейчас
команды. JEPA/EMA latent targets потребуют отдельного anti-collapse audit.
Только затем можно проверять auxiliary representation при том же actor и
бюджете свежих PPO данных. Для принятия модели всё равно нужны новые
native бои против source/rules; прогнозы не заменяют оценку победности.

Все NN вычисления и обучение — CUDA. Для будущих игровых проверок —
существующий harness, pool16 ×2, независимые seeds и совпадающие условия
между вариантами, без автоматической замены рабочего бота.
