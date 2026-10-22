# Обучаемая связка выбранного движения и прицела

Дата:21.10.2026. Статус: два PPO rounds завершены на64 новых боях; PPO screen14/24 против BC12/24 и rules13/24, но интервал включает0 и есть регрессии поворотов/стояния. Начато coverage56 условий/вариант. Общее боевое преимущество не доказано. Основная модель не заменена. Предыдущий BC output-layer пилот отклонён после smoke3/8→1/8 против rules5/8: [отчёт](combat_aim_feedback_experiments_20261021.md).

## Проверяемая гипотеза

Существующий spatial residual получает `tanh(raw[:2])` — средние планируемые forward/side. Реальный Gaussian sample движения выбирается позже. Новая ветка получает выбранный sample и позволяет условному распределению прицела учитывать именно эту команду. Это проверка архитектурной связи действий; добавление скрытых native входов, боевых правил или новой системы наград сюда не входит.

Под выбранным движением понимается объявленная моделью команда до environment guards. Фактическое будущее перемещение после столкновений/guards ещё неизвестно и во входы не передаётся. Текущие наблюдаемые скорость, отдача, геометрия и признаки цели уже есть в исходных spatial inputs; они не объявляются новой реализацией обратной связи. Отдельная память предыдущего выполненного действия новым модулем пока не добавлена.

## Версионированный контракт

Поле весов: `sampled_motion_aim`, version `combat_sampled_motion_aim_v1`. Требуется feature v10, существующий spatial branch и coarse/fine target aim head. Общий temporal encoder, width81 raw logits, target/weapon/vertical/fire/mode heads и feature width1147 сохранены.

Вход125 =119 существующих spatial inputs выбранной цели +2 выбранных `tanh(move_latent)` +2 разности между выбранным и средним движением +2 one-hot режима coarse/fine. Сеть125→64 ReLU→32 ReLU→2 linear:10210 параметров. Выход добавляется к yaw/pitch latent mean после выбора target/mode/движения. Target=none строго нейтрален даже при ненулевых весах.

Последний слой инициализируется нулями. Порядок RNG draws Go сохранён: target, mode, forward, side, yaw, pitch, затем прочие категории в прежнем порядке. Нулевая миграция должна сохранять старую политику; сам факт добавления модуля не является улучшением поведения.

Совместная вероятность раскладывается как:

`P(target) · P(mode|target) · P(move) · P(aim|move,target,mode) · P(fire) · P(vertical) · P(weapon)`.

Go Decide добавляет conditional correction после выборки первых двух latent координат; Review использует сохранённые первые две координаты sample. CUDA `selected_means/probabilities` получает те же pre-tanh координаты. При вычислении PPO importance ratio sample фиксирован; подставлять вместо него среднее движение неверно. Raw81 сохраняет безусловные исходные logits: для полной вероятности необходимо передать conditional branch явно.

Реализация: `internal/policy/sampled_motion_aim.go`, `ppo.go`; `scripts/combat_sampled_motion_aim.py`, `combat_spatial_aim.py`, `combat_precision_head.py`. Primary CUDA replay/finalizer и PPO entrypoints обновлены. Старый head-learning audit, считающий только безусловный Gaussian KL, явно отвергает новый контракт; старый spatial-output BC trainer также не принимает conditional модель. Это не молчаливое игнорирование новой ветки.

## Регуляризация и возобновление

`combat_sequence_retention.policy_kl` сохраняет прежнюю точную формулу для старых моделей. При новом контракте KL движения и категорий считается явно, KL прицела — условно на одной зафиксированной behavior-команде движения на историю. Target/mode интегрируются суммой с весами прежней политики. Это Monte Carlo оценка ожидания по движению, а не точный аналитический KL всей нелинейной совместной политики. Во время повторных проб guarded optimizer этот sample не меняется.

Teacher branch заморожен и отделён от student. Метаданные sequence retention фиксируют `fixed_behavior_move_per_history_v1`, чтобы старый objective не считался тем же контрактом. CUDA проверки требуют нулевого KL одинаковых моделей и положительного KL ненулевой conditional-правки даже при одинаковых raw81 logits.

Миграция: `scripts/migrate_combat_sampled_motion_cuda.py`. Сохранены старые actor/value/log_std и optimizer moments по именам параметров; новые параметры не получают чужих Adam состояний. Старый sequence-retention objective сбрасывается явно. Сохраняются стандартные weights/checkpoint/report complete receipts; возобновление после обучения требует нового собственного rollout.

## Выполненная проверка и сбор

`workspace/artifacts/combat-sampled-motion-zero-v2-20261021`: CUDA, RTX5070,996 реальных историй из закрытого свежего train corpus. Проверены нулевой эффект миграции, temporal state, joint likelihood относительно native записей, ненулевой отклик на выбранное движение, нейтральность target=none, conditional retention и его gradients, перенос optimizer по именам, checkpoint reload. Ненулевая тестовая ветка действительно меняет вероятность при неизменных исходных raw logits; это проверяет подключение нового распределения, а не только его форму.

Go tests только проверяют JSON/размеры/версии/finite metadata, без исполнения NN. Пройдены `go test ./internal/policy -run '^TestSampledMotionAimContract$' -count=1`, сборка CLI `q2coopbot/q2ppo-data/q2episode`, Python compilation и whitespace check. NN вычисления выполняются на CUDA; реальная игра исполняет штатный Go inference.

Начат свежий корпус `combat-sampled-motion-corpus-v2-20261021`:32 train +16 development,4 single-target MG и4 group MG семейства на base3/city1/jail1/ware2. В группах parasite/infantry/gunner, по разным точкам карт;16 ×2. Train offset320, development128 проверяются историей до заморозки планов. Подготовка v1 остановилась до native запуска: общий offset600 не помещался в512-seed split групп. v2 использует допустимые новые ranges; это не перезапуск по таймауту и не потерянные игровые результаты.

`scripts/train_combat_sampled_motion_cuda.py` готовит фиксированный80-step CUDA BC-пилот только нового модуля с native MG body-point/recoil labels. Loss сравнивает желаемое latent действие и добавляет conditional retention к исходной политике на зафиксированных train историях. Development не входит в loss или выбор checkpoint. Слишком большие для замороженного fine режима команды исключаются и считаются отдельно; режим не обучается угадывать недостижимую доводку. Геометрические labels не признаны оптимальным боевым учителем. Результат требует нового native smoke против исходной модели и правил, затем только при положительном сигнале screen.

Критерий механической приёмки — ненулевой native replay: нулевые веса не доказывают правильность ненулевого исполнения. Ниже закрыт этот этап. До независимого качества нет замены основных весов или утверждения превосходства над правилами, кампании либо coop.

## Закрытые результаты первого цикла

Корпус v2 завершён:48 native/48 CUDA,2213 train и1119 development provider rows. Экспорт `combat-sampled-motion-labels-v1-20261021`:1167/601 меток MG выстрела;343/195 случаев с одновременным собственным и наблюдаемым по native pose движением цели. В fit вошли1072/550 reachable labels,95/51 исключены по заранее заданным ограничениям fine режима/среды.

Веса `combat-sampled-motion-bc-v1-20261021` закрыты стандартным complete receipt.80 фиксированных CUDA Adam шагов, lr0.0003, conditional retention weight0.02; trained10210 новых параметров, остальные веса/шум/critic сохранены. Development средняя абсолютная ошибка осей4.185°→4.069°, норма пары6.678°→6.495°. Development не участвует в loss и выборе checkpoint. Это offline метрики, не точность стрельбы в новой траектории.

`combat-sampled-motion-use-v1-20261021.json`: на всех601 development историях ненулевой gradient по выбранному движению; среднее изменение действия при замене sampled движения на его mean —0.034° по оси, максимум0.765°. Значит зависимость подключена и обучена, но её средний эффект мал. Новые параметры и group-inclusive train corpus тоже изменились; приписывать улучшение боя исключительно выбранному движению без ablation нельзя.

Smoke: `combat-sampled-motion-smoke-evaluation-v1-20261021`, validation offset552,24 native/16 CUDA. Проверены все8 ненулевых learned native cases новой модели, включая согласование conditional likelihood с CUDA. Это закрывает механическую проверку ненулевого пути; отличие версий весов/rollout сохранено в receipts.

| Вариант | Победы | Ordinary | Recovery | Полученный урон | Смены цели /100 кадров | Повороты ≥30° | Стояние при clear enemy | Attack без цели |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Исходная learned модель |3/8|2/4|1/4|544|33.51|9.47%|22.55%|3.30%|
| Sampled-motion BC |5/8|2/4|3/4|490|31.69|9.23%|33.80%|15.00%|
| Правила |5/8|1/4|4/4|360|—|—|—|—|

Парно:2 новые/0 потерянных побед против исходной модели;1/1 против правил. Две новые победы — recovery. Сигнал положительный по исходу, но стояние и команды без выбранной цели ухудшились. Последние не равны количеству фактических выстрелов или промахов. Превосходства над правилами и решения исходных проблем лишних действий здесь нет.

Bundle `combat-sampled-motion-smoke-bundle-v1-20261021/complete.json` закрывает пять отчётов. `decision.json` допускает только screen, promotion=false, обе регрессии зафиксированы. Запущен screen `combat-sampled-motion-screen-evaluation-v1-20261021`:72 native/48 CUDA,24 новых условия на вариант,16 ordinary/8 recovery, offset556–557,16 ×2. Итоги screen фиксировать после его полного закрытия; основные веса сохранены.

Перед последующим PPO: нынешний MG корпус собран с `combat_reward_v8`/SHA `ef03cd4a…`, исходный checkpoint закреплён за target-commitment reward/SHA `af9b9e4c…`. BC не использует reward; для PPO такой корпус не является готовым продолжением старого objective. Не обходить `validate_objective` сменой SHA в checkpoint. Нужны новые own-policy captures с согласованным registry reward либо явная проверенная objective fork с новой коллекцией. Primary PPO likelihood/retention подключены и проверены CUDA, но реальный PPO update нового контракта этим циклом ещё не подтверждён.

## Закрытый screen и следующий PPO-пилот

Screen `combat-sampled-motion-screen-evaluation-v1-20261021` завершён:72 native/48 CUDA,3324 control и3004 candidate provider rows. Все24 случая каждого learned варианта прошли CUDA replay. Bundle `combat-sampled-motion-screen-bundle-v1-20261021/complete.json` закрывает пять отчётов.

| Вариант | Победы | Ordinary | Recovery | Полученный урон | Смены цели /100 кадров | Повороты ≥30° | Стояние при clear enemy | Attack без цели |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Исходная learned модель |6/24|4/16|2/8|1676|29.16|9.14%|19.12%|25.29%|
| Sampled-motion BC |9/24|6/16|3/8|1579|31.02|9.77%|18.42%|14.04%|
| Правила |15/24|8/16|7/8|1328|—|—|—|—|

Парно против исходной модели:4 новые/1 потерянная победа. Против правил:2 новые/8 потерянных. Малое улучшение сохраняется на новых условиях, но descriptive95% интервал разницы к исходной модели включает0; общего превосходства над правилами нет. По сравнению со smoke стояние и attack-without-target не показывают устойчивой регрессии, однако смены цели и большие повороты на screen немного ухудшились. Нельзя объединять разные seeds в доказательство, что проблема лишних действий решена.

MG selected-target post-recoil ошибка уменьшилась9.25°→7.04°. Число native damage rays по выбранной цели168/448→187/526: абсолютное число больше, доля37.5%→35.6% не улучшилась. В recovery71/123→51/120. Это разные closed-loop траектории; меньшая угловая ошибка не равна более эффективному расходу или скорейшему убийству.

`decision.json`: development_parent_only, promotion=false, confirmation не запущена. BC-кандидат выбран только как родитель последующего обучения; основные/default веса сохранены. Следующий шаг — результативность полного PPO, без новой архитектуры или изменения закреплённой награды.

Подготовлен `combat-sampled-motion-ppo-registry-v1-20261021/index.json` через `scripts/prepare_combat_objective_registry.py`. Восемь зарегистрированных group семейств на четырёх картах:4 обычных weapons и4 weapons-grenades-recovery. Изменены только recipe.reward_config и revision; генераторы, геометрия, monsters/loadouts и split ranges сохранены. Reward совпадает с checkpoint config (`af9b9e4c…`); receipt расположен рядом с каталогом как `combat-sampled-motion-ppo-registry-v1-20261021-complete.json`, потому что компилятор реестра отвергает незарегистрированные JSON внутри него.

Collector получил `--ppo-ready`: проверяет checkpoint/config/reward SHA до запуска, сохраняет config и pins. `--train-only` исключает ненужную development коллекцию из PPO round. Проверена реальная отрицательная заявка с несовпадающей наградой: отказ до создания output/native запуска, receipt `combat-sampled-motion-objective-guard-v1-20261021.json`. Первичная подготовка round1 v1 остановилась компилятором до native из-за receipt внутри registry; receipt перенесён в соседний файл, round1 v2 запущен после исправления структуры, без повторения боёв.

Зафиксирован protocol `combat-sampled-motion-ppo-pilot-v1-20261021/protocol.json`:2 train rounds по32 новых боя,16 ×2,10 actor/40 critic steps на round, conditional retention weight1, исходный reward сохранён, без выбора checkpoint по промежуточной validation. Родитель первого round — BC-кандидат. Новые кадры каждой следующей итерации собираются её собственными весами; использовать старый BC/native corpus как on-policy после update запрещено.

`scripts/fit_registered_combat_ppo_cuda.py` связывает закрытый PPO-ready corpus с checkpoint objective и anchor/bank, создаёт train-only equal-family/seed retention и вызывает штатный CUDA PPO. Отдельно сообщает число принятых actor steps; полный отказ actor optimizer не объявляется улучшением политики. Сохраняет стандартный update в подкаталоге `update` и receipt связей с исходным corpus. На момент записи round1 corpus v2 прошёл32 native и обрабатывается CUDA; действительный PPO fit ещё требуется завершить.

### Реальный PPO round1

Закрыт `combat-sampled-motion-ppo-round1-corpus-v2-20261021`:32/32 native/CUDA,3579 train rows,8 group families, train offset384–387. Config/reward совпадают с checkpoint; development/validation в corpus отсутствуют. Полностью завершён `combat-sampled-motion-ppo-fit1-v1-20261021`: принят10/10 actor steps,40 critic steps по исходному config, CUDA optimization11.34с. Final sample approximate KL0.00485 при лимите0.005; conditional sequence retention0→0.00449. Это проверка полноценного обновления и восстановления, не доказательство новых побед.

Основная CLI `ppo_recurrent.py` теперь сообщает фактический sampled-motion likelihood/retention contract в отчёте, не повторяет устаревший текст «No learned weapon choice». Во время этого round обучались все actor heads и critic, включая movement/aim/target/weapon/fire/mode/std. Стандартный checkpoint и optimiser states сохранены в `.../combat-sampled-motion-ppo-fit1-v1-20261021/update`; fit receipt отдельно связывает update с train-only corpus.

Второй corpus `combat-sampled-motion-ppo-round2-corpus-v1-20261021` собирается весами update1, train offset388–391,32 новых боя. Это следующий собственный rollout после изменения политики, а не повторное обучение на прежней выборке. Пилот сохраняет зафиксированные2 rounds: промежуточной validation и выбора удачного checkpoint нет. По завершении update2 требуется отдельное сравнение с BC-родителем и правилами.

### Завершение фиксированного PPO-пилота

Оба rounds закрыты:32+32 уникальных native/CUDA train cases,3579+3588=7167 rows,20/20 принятых actor steps и80 critic steps. Второй update final approximate KL0.00416, conditional retention0.00368. На CUDA проверены сохранённые actor tensors (30 изменившихся tensors в каждом round), config/checkpoint и отсутствие пересечения seeds. Родитель второго corpus строго update1. Training complete receipt: `combat-sampled-motion-ppo-pilot-v1-20261021/complete.json`; итоговые веса — `combat-sampled-motion-ppo-fit2-v1-20261021/update`.

Training outcomes:11/32 побед у BC до первого update,10/32 после update1 на других train seeds. Это описательные исходы учебных боёв, не парная оценка и не основание выбора checkpoint.

Независимый PPO smoke, offset560:24 native/16 CUDA, BC4/8, PPO4/8, rules5/8; парно0 новых/0 потерянных к BC. Урон443→394, большие yaw10.43→8.34%, attack-without-target26.0→20.7%, стояние12.48→17.09%. Smoke дал смешанный сигнал эффективности и прошёл только к screen.

PPO screen, offset564–565: `combat-sampled-motion-ppo-screen-bundle-v1-20261021/complete.json`,72 native/48 CUDA,3259 BC/3059 PPO rows; закрыты все пять отчётов.

| Вариант | Победы | Ordinary | Recovery | Полученный урон | Смены цели /100 кадров | Повороты ≥30° | Стояние при clear enemy | Attack без цели |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BC-родитель |12/24|9/16|3/8|1457|30.49|6.01%|17.64%|15.38%|
| После64 PPO боёв |14/24|8/16|6/8|1434|33.20|7.80%|21.33%|10.24%|
| Правила |13/24|5/16|8/8|1241|—|—|—|—|

Парно:4 новые/2 потерянные победы относительно BC;4/3 относительно правил. Exploratory95% интервал разницы к правилам[−16.67;+29.17]п.п., к BC[−8.33;+25.00]п.п. Преимущество в один бой не является доказанным превосходством. Recovery улучшилось3/8→6/8, ordinary9/16→8/16. Регрессия поворотов/стояния относительно BC в более широкой выборке не согласуется с улучшением yaw в малом smoke, поэтому проблема лишних действий остаётся открытой.

`decision.json`: advance_to_coverage_only, promotion=false. Веса frozen. Запущено `combat-sampled-motion-ppo-coverage-evaluation-v1-20261021`,168 native/112 CUDA:28 семейств×2 seeds на вариант,40 ordinary/16 recovery, validation offset568–569,16 ×2. Это ступень coverage, а не финальная confirmation112 условий. Без закрытого coverage нет нового назначения основных весов или широкого вывода о правилах/кампании/coop.

Параллельная read-only [диагностика native geometry](combat_native_ballistic_geometry_20261021.md) обнаружила1° квантование `fire_lead` и немедленное изменение высоты ствола на24 units при crouch/выходе из crouch. Текущая aim-ветка не видит выбранный vertical sample; v2 с таким conditioning пока только следующая изолированная гипотеза. Go/native код не изменяется во время текущего frozen coverage. Также исправлен BC label exporter: начало определяется native network release, а не константой100 из game clock. Это не изменило уже обученные/оценённые веса; PPO включал ранние eligible frames.

### Закрытый coverage: общий выигрыш над правилами не подтверждён

`combat-sampled-motion-ppo-coverage-evaluation-v1-20261021` и bundle того же имени закрыты:168 native/112 CUDA,56 условий на вариант,6224 BC и7022 PPO replay rows. Проверены хеши всех пяти отчётов и закреплённых входов bundle. Seeds568–569,28 семейств,16 инстансов ×2.

| Вариант | Победы | Обычные бои | Recovery | Полученный урон | Смены цели /100 кадров | Повороты ≥30° | Стояние при clear enemy | Attack без цели |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BC-родитель |22/56|16/40|6/16|3853|37.52|10.01%|17.98%|14.12%|
| PPO после64 учебных боёв |25/56|19/40|6/16|3598|35.50|10.37%|16.32%|15.70%|
| Правила |28/56|13/40|15/16|3307|—|—|—|—|

Парно PPO против BC:6 новых/3 потерянных победы, разница+5.36п.п., exploratory95% интервал[−3.57;+14.29]. Против правил:11 новых/14 потерянных,−5.36п.п., интервал[−25;+14.29]. Ordinary лучше по точечной оценке, recovery сильно хуже; общее превосходство отсутствует. Повороты и attack без цели не устранены. Attack обозначает команду, а не подтверждённый выстрел/промах.

В каждом из16 recovery боёв обе learned модели действительно переключились на оружие из поддерживаемого набора. Первая такая экипировка наблюдалась через медианные4 network frames, максимум17; это время от native release, не чистая задержка оружейной анимации. Следовательно, recovery дефицит нельзя полностью объяснить отсутствием переключения. Это не исключает неудачного выбора оружия или последующих переключений.

`decision.json`: development_parent_only, promotion=false, confirmation_started=false. Большую confirmation для этих весов не запускаем. Default не меняется. Следующий ограниченный опыт — [выбранный vertical в conditional aim](combat_vertical_aim_protocol_20261021.md). Существующий PPO остаётся зафиксированным результатом, а не победителем над правилами.
