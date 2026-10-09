# Исправление переноса FireBC, 2026-10-09

Основание: [1440 боя архитектур](combat_architecture_results_20261009.md), новые модели слабы уже до PPO. FireBC остаётся исходной learned baseline; новые модели не продвигаются до живой оценки.

## Изменения и CUDA-проверки

В старой дистилляции оптимизировалось общее MSE сырых actor logits и value с весом 1. Новая версия `combat_architecture_prior_v2` использует отдельно движение после tanh, циклическую ошибку yaw и обычную ошибку pitch, fire BCE, pose KL и weapon KL с маской наблюдаемой доступности. Critic weight 0,01; actor/critic gradients клипуются раздельно. В v2b огонь балансируется по положительному/отрицательному решению учителя, pose — по классам. Недоступные weapon logits не входят в распределение и не влияют на objective.

Первый v2-кандидат с усиленным aim loss отвергнут: aim RMSE около 3°, но fire-choice disagreement около 22% на validation. Это показало, что улучшение одного непрерывного head недостаточно.

Завершены v2b CUDA-обучения Attention128, seeds 20261007/08: по 1000 epochs на одном train corpus, validation только измеряется. Отдельный frozen Attention64 identity control копирует FireBC без переинициализации и без optimizer steps. У identity weights SHA совпал с исходным FireBC, все ошибки действий равны нулю. Это контроль сохранения поведения, а не новая обученная архитектура.

Независимый offline CUDA audit использует 6531 validation row, 82 сегмента; teacher/student получают одинаковые features и нулевую память на identity/frame gaps. Проверены экспорт/повторная загрузка CUDA весов и causal-prefix invariance. Go model parity/tests не запускались. Оптимизация и model inference только CUDA.

| Attention128 | Movement RMSE v1 → v2b | Aim RMSE, градусы v1 → v2b | Fire disagreement v1 → v2b | Pose disagreement v2b |
| --- | ---: | ---: | ---: | ---: |
| seed 20261007 | 0,3470 → 0,1722 | 7,20 → 3,11 | 1,62% → 0,38% | 1,82% |
| seed 20261008 | 0,3613 → 0,1634 | 7,39 → 3,47 | 1,82% → 0,51% | 1,53% |

Aim p95 уменьшился с 14–15° до 6,8–7,0°. Masked weapon disagreement на этом корпусе равен нулю, что не доказывает освоение разнообразного выбора оружия в игре. Улучшилась offline имитация; рост побед ещё не доказан. Этот эксперимент меняет objective, балансировку и бюджет epochs одновременно, поэтому вклад каждого изменения отдельно не установлен.

Receipts: `workspace/artifacts/architecture-priors-v2b-20261009/report.json`, `cuda-audit.json`, per-arm `complete.json`. Отвергнутый v2 сохранён отдельно в `architecture-priors-v2-20261009`. Train/validation inputs и исходный FireBC не изменены.

## Парная живая проверка

Root: `workspace/artifacts/distill-eval-v2b-20261009`. Пять вариантов × 80 боёв = 400: две пары FireBC identity → Attention128 v2b для двух initialization seeds плюс rules. Before здесь означает точную копию FireBC, after — CUDA-дистилляцию, а не PPO. Все 20 семейств, стартовые позиции, ресурсы, состав монстров и validation seeds совпадают с предыдущей матрицей.

Запущен фоновый пул 16 слотов, ×2, порты 34500–34515. Frozen prior seals и template/registry SHA проверены. Источники Go/PowerShell во время capture не меняются. `go build -buildvcs=false` исключает предыдущую ошибку агрегации из-за VCS dirty stamp. Итоговые quality reports создаются автоматически после успешного завершения. Это reused validation; independent final test отложен. До подтверждения побед новые кандидаты не заменяют FireBC.

Первая живая проверка: 39/400 завершённых боёв, 39/39 native captures валидны, ошибок заданий нет, source fingerprint не изменился. Пул продолжает работу; это проверка запуска, не результат эффективности.

## Промежуточная диагностика завершённых пар

`diagnose_combat_evaluation_pairs.py` собрал только полностью завершённые парные условия: первый seed всех 20 семейств, пять вариантов, 100 боёв. Snapshot `trace-snapshot-first-seed.json` сохраняет trace/server/report SHA. Это неполная диагностическая выборка, не полный результат 400 боёв.

Средняя по эпизодам минимальная горизонтальная ошибка команды при стрельбе относительно любого clear observed enemy: FireBC identity около 25°, Attention128 v2b около 22° / 32° для двух initialization seeds; rules около 0,045°. Геометрическая метрика игнорирует pitch, lead и recoil, поэтому это не измеренная точность попаданий. Ground movement stall proxy 0–0,5%: массовое застревание в этой выборке не проявилось. Вывод для следующей диагностики: проверить согласование aim/movement и фактическую цель стрельбы; одного улучшения offline копирования действий недостаточно для заявления улучшения боя.

Mean per-episode selection p95: identity Attention64 около 13 ms, Attention128 около 34–39 ms на активном пуле. Это время выбора под нагрузкой, не изолированный benchmark. Стоимость inference увеличилась; при дальнейшем выборе модели учитывать её вместе с победами и native frame proofs.

## Итог 400 боёв и следующий эксперимент

Пул завершён: 400/400, ошибок заданий 0, source unchanged, агрегация успешна. Итоговые `quality-report.md/json` в `distill-eval-v2b-20261009`:

| Вариант | Победы / 80 | Смерти | Средний нанесённый урон | Средний полученный урон |
| --- | ---: | ---: | ---: | ---: |
| FireBC identity, seed 07 pair | 50 | 29 | 63,0 | 45,5 |
| Attention128 v2b, seed 07 | 43 | 36 | 63,6 | 48,3 |
| FireBC identity, seed 08 pair | 50 | 29 | 63,0 | 45,6 |
| Attention128 v2b, seed 08 | 41 | 37 | 62,9 | 58,5 |
| Rules | 68 | 8 | 72,2 | 15,4 |

Seed 07: 8 новых побед и 15 потерянных; seed 08: 5 новых и 14 потерянных. V2b значительно сильнее старых стартов Attention128 (2/80 и 5/80), но не превосходит FireBC. Кандидаты не продвинуты. Rules получили 68/80 против 70/80 в предыдущей серии; одинаковые условия не гарантируют побитово одинаковое native исполнение.

Следующий root `distill-apool-v1-20261009`: исходный FireBC и два Attention128 v2b, каждому 80 свежих собственных тренировочных боёв, всего 240. Все 20 семейств, одинаковые seed-offset 64 и геометрия между вариантами; split train, validation seeds исключены. Пул 16, ×2, отдельный native процесс каждого боя. Это равный **добавочный** бюджет, исторический опыт моделей различается. После сбора существующий проверенный processor экспортирует собственные rollout и запускает по одному CUDA PPO update с fresh optimizers, retention/bank weight 0. Общие anchor/bank files сохраняются только для checkpoint lineage; их loss отключён. Обучение не использует 400 оценочных боёв. Затем требуется новое парное before/after сравнение; превосходство пока не установлено.

Сбор запущен; первые 27/27 completed captures проверены по native report/manifest, seed/dispatch и exporter SHA, ошибок нет. `continue_combat_distillation_evaluation.py` открыл handle подтверждённого живого training driver и проверил его command line. После terminal состояния он требует complete processing report всех трёх CUDA updates, проверяет completion seals и запускает следующую оценку. Root `distill-ppo-eval-v1-20261009`: три пары before/after плюс rules, 7 × 80 = 560 боёв; отдельный FireBC baseline не дублируется, он уже является FireBC-before. Validation offset 24, 16 слотов, ×2. Это подготовленное автоматическое продолжение, ещё не завершённая оценка. PID receipts и `evaluation-continuation.json` лежат в capture root; restart по истёкшему observation timeout не выполняется.

## Диагностика всех 400 закрытых трасс

`trace-snapshot-all.json` содержит 80 полностью совпадающих условий × 5 вариантов, с SHA каждой trace/server/report. Для каждой команды огня вычисляется минимум горизонтальной ошибки относительно любого clear observed enemy. Доли ниже считаются по кадрам с огнём и хотя бы одним таким противником; средняя ошибка — среднее эпизодных средних.

| Вариант | Средняя ошибка yaw | Доля yaw > 20° | Доля yaw < 5° | Ground stall proxy | Средний selection p95 под нагрузкой |
| --- | ---: | ---: | ---: | ---: | ---: |
| FireBC identity, pair 07 | 24,41° | 35,40% | 12,76% | 0,58% | 11,61 ms |
| Attention128 v2b, seed 07 | 19,21° | 29,63% | 18,07% | 1,09% | 35,00 ms |
| FireBC identity, pair 08 | 24,41° | 35,36% | 12,75% | 0,58% | 11,01 ms |
| Attention128 v2b, seed 08 | 32,73° | 62,62% | 10,65% | 3,33% | 38,13 ms |
| Rules | 0,08° | 0% | 99,94% | 0% | — |

FireBC machinegun на base1 site-03/site-04: 0/4 побед в каждом семействе, mean yaw около 73°/71°, rules 3/4 и 4/4 с yaw около 0°. На base2 site-02 blaster у FireBC 0/4 при yaw около 11°, у seed08 Attention128 0/4 при yaw около 4°: одной горизонтальной ошибки недостаточно для объяснения поражений. На base2 site-03 machinegun FireBC 4/4, Attention128 1/4 и 1/4. Значит перенос поведения имеет зависимость от позиции/геометрии, и усреднённый offline RMSE её скрывает.

Вывод: после текущего matched PPO сравнения проверять отдельно согласование aim/movement, pitch/recoil и target selection на этих семействах. Трассы не доказывают причинность, stall proxy не равен подтверждённому столкновению со стеной, yaw не учитывает lead/pitch и не является hit accuracy. Новую архитектуру пока нельзя выбирать только по размеру или offline loss.

**Уточнение по препятствиям и latency:** низкий ground stall proxy не подтверждает хорошее самостоятельное избегание стен. По сохранённым `guards` у FireBC pair07 3691 событий `static_hull_blocked` на 5879 наблюдаемых alive frames, у Attention128 seeds07/08 — 4098/6621 и 4526/7406. Это события вмешательства, не частота физических столкновений и не доля уникальных кадров. Дополнительно 569/634/695 событий `unsupported_motion_guard`. Защитная коррекция может скрывать плохой выбор движения; следующий quality анализ должен сохранять эти счётчики рядом с победами. Текущий PPO exporter сохраняет guards как исполнение среды, а не автоматически исключает все такие transitions (`cmd/q2ppo-data/main.go`, metadata scope).

Native manifest описывает synchronous lockstep: мир ждёт outstanding command между шагами. Поэтому измеренные 35–38 ms выбора Attention128 влияют на throughput, но не являются установленной причиной игровых поражений в этом эксперименте. Влияние на обычный realtime client требует отдельной проверки.

`report_combat_architecture_evaluation.py` теперь выводит рядом с результатом native capture diagnostics: среднее число guard events, суммарные provider frames, frame gaps и среднее episode selection p95. Проверен повторным построением полного отчёта 400/400; показатели побед не изменились. Guard events — все события capture, без утверждения об уникальности кадров/столкновениях. Frame gaps всех пяти вариантов равны нулю. Дополнение автоматически попадёт в следующую 560-battle оценку.

## Текущий собственный PPO-корпус

`distill-apool-v1-20261009/pool/report.json`: complete, 240/240 jobs без ошибок, source unchanged. Все 240 member receipts отдельно проверены по report/manifest/exporter SHA, seed и dispatch; общий source fingerprint один.

`distill-aproc-v1-20261009/firebc/update`: первый CUDA update завершён, 6050 eligible transitions, 10 accepted actor steps, approximate KL 0,006550. Native rollout → CUDA log-probability max error 0,0000810, value max error 0,00000897. `complete.json` SHA весов, checkpoint и report проверены. Следующие два Attention128 ещё обрабатываются; свежей оценки качества PPO пока нет. Это подтверждение обучения/целостности, не улучшения побед.

**Обработка полностью завершена:** `distill-aproc-v1-20261009/report.json` complete, 3/3 CUDA updates. Все per-model completion seals повторно проверены.

| Модель | Свои training battles | Eligible transitions | Accepted actor steps | Approximate KL | CUDA old log-prob max error | CUDA old value max error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FireBC | 80 | 6050 | 10 | 0,006550 | 0,0000810 | 0,00000897 |
| Attention128 seed07 | 80 | 6635 | 10 | 0,008767 | 0,0000421 | 0,00000559 |
| Attention128 seed08 | 80 | 7388 | 10 | 0,005567 | 0,0000403 | 0,00000930 |

Автоматическое продолжение перешло в `evaluating`: `distill-ppo-eval-v1-20261009`, 560 battles. Проверены SHA всех семи планов и одинаковые условия/fixtures/seeds, только split validation. Первые 9 jobs завершены без ошибок, все 16 слотов активны; полный результат качества ещё впереди. Training driver terminal, evaluation supervisor жив и ведёт новую очередь. Модели не продвинуты по одному факту завершения обучения.

### Промежуточный matched snapshot PPO

`distill-ppo-eval-v1-20261009/trace-snapshot-first-seed.json`: 19 полностью совпадающих условий первого seed, 133 закрытых боя. Семейство `campaign-base2-site-04-machinegun` ещё не вошло; покрытие неполное, источник общий. Только завершённые условия всех семи вариантов, с SHA закрытых traces/server/report.

| Модель | Before wins / 19 | After wins / 19 | Новые победы | Потерянные |
| --- | ---: | ---: | ---: | ---: |
| FireBC | 11 | 10 | 1 | 2 |
| Attention128 seed07 | 10 | 10 | 4 | 4 |
| Attention128 seed08 | 10 | 9 | 0 | 1 |

Rules 17/19. Mean episode minimum horizontal attack-yaw error: FireBC 25,72°→26,75°, Attention128 seed07 22,75°→18,44°, seed08 32,34°→29,44°. Меньшая геометрическая ошибка не дала роста побед в этой неполной выборке. Это диагностическое наблюдение, не окончательный выбор checkpoint/архитектуры и не основание прерывать frozen 560-battle cohort. Финальный отчёт остаётся ожидаемым.

**Полный результат:**560/560 завершены без ошибок; Attention128 seeds07/08 43→46 и 41→45/80, FireBC 50→46/80, rules68/80. Новые модели не продвинуты. Полные метрики, закрытые трассы, повторные before-условия и следующий этап: [итог PPO](combat_distillation_ppo_results_20261009.md). Этот итог заменяет промежуточные снимки при принятии решений.
