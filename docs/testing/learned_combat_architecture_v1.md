# MLP / GRU / temporal attention / entity attention

06.10.2026. Протокол фиксируется до игрового сравнения.

По запросу пользователя сравниваются оба вида multi-head attention с GRU
и существующим MLP. Используется прежний observation/features v4/810;
оружие Blaster, reward maneuver-v4, HP100/stock monsters, horizon300,
native release100. Training Mixed skill1, evaluation Mixed/Solo skill1 и
Solo skill3. Fixed MG и learned weapon choice в этот опыт не добавляются.

Все варианты начинаются с одного obstacle reference. GRU/attention имеют
нулевую residual output projection, поэтому initial actions/value совпадают
с reference; скрытые клетки/projections инициализируются до обучения.
В каждой ветке новый Adam, включая MLP: старые optimizer moments не дают
одной архитектуре преимущество.2 updates по4 fresh episodes, actor budget10,
value40, CUDA-only. Constant anchor и composition bank weight1 одинаковы.

Возраст обучения различается у базовых и добавленных параметров. Общая
MLP база во всех четырёх ветках уже прошла53 PPO updates/524 actor steps
и предшествующие supervised fits. Новые GRU/attention параметры
инициализированы заново и получили лишь2 updates/20 actor steps;
общая MLP база также получила эти2 updates во всех вариантах. Поэтому
это comparison краткой адаптации residual additions к обученной MLP,
не сравнение самостоятельных архитектур после сопоставимого полного
обучения. По результату нельзя объявлять MLP лучше GRU/attention.
Для следующего опыта нужен больший заранее заданный общий бюджет,
кривые качества/checkpoints по ходу обучения и отдельный учёт прогрева
новых блоков. Выбирать лучший eval checkpoint для continued fit нельзя.

GRU: отдельная actor/critic GRU64 поверх существующего hidden64, residual
heads8/1. Go повторяет torch.nn.GRU r/z/n candidate equation. В dataset
сохраняется hidden state перед действием; native exporter проверяет всю
цепочку provider calls, включая исключённые из loss команды. Sequence
context содержит все наблюдаемые provider frames, loss только eligible
native transitions. На каждом PPO proposal prefixes пересчитываются с
текущими весами, градиенты отсекаются каждые32 frames. Memory сбрасывается
при смене life/map/connection/spawncount/actor и разрыве/откате frames;
повторная Decide того же кадра не продвигает memory/RNG.

Temporal attention:4 головы, hidden64, causal sliding window32, fixed
sinusoidal positions от последнего reset. Только прошлые и текущий кадры;
Go хранит до31 encoded frame, Torch пересчитывает их с текущими весами.
Entity attention:4 головы, query из текущего global hidden64, общий
token encoder68→64 и cross-attention.29 masked slots: self+8 monsters+
4 projectiles+8 geometry probes+4 pickups+4 props. Tokens собраны только
из существующих v4 значений; masks, monster types/velocity/direction/aim/
bbox сохранены. Movers/beams/short history остаются в global input, но
отдельных токенов здесь не получают. Никакой server truth в inputs нет.

Bank не имеет sequences: для GRU/temporal attention distillation на bank
измеряется при zero memory. Это ограничение retention, не гарантия
сохранения полного recurrent поведения. Контекстные anchor targets на
fresh rows — исходный MLP; reference не обучается.

Training seeds26000–26007, одинаковые парные seeds разных архитектур,
отдельный seed каждого из4 concurrent native server/client instances, x2.
Evaluation26100–26103/26200–26203/26300–26303, не участвует в fit.
Ожидается92 valid captures/23 batches:32 training+60 evaluation.
Две CUDA updates — ограниченный первый architecture pilot, не оценка
предельной ёмкости сети.4 seeds на условие не доказывают superiority.

Первый stochastic training batch должен иметь identical initial actions,
latent samples, native executed commands/events и physical outcome totals
во всех вариантах. Асинхронная rules/setup telemetry до barrier100 и
порядок visible-enemy display arrays не являются controlled-window proof.
Второй own-policy batch проверяет Go/Torch log-probability/value согласие
после ненулевых residual updates. Native re-export/source/reward/GAE,
optimizer resume/counters, seed exclusion, current-model sequence state и
causality/padding проверяются отдельно. Inference latency измеряется в Go,
training time — на CUDA; parameter count не приравнивается к compute budget.
Selection elapsed включает decision, conversion в command и safety guards;
это не отдельный microbenchmark матричного forward.
No live promotion; для принятия потребуется более длинное обучение и
полный набор известных regressions/cooperative safety.

Runner: scripts/run_combat_architecture_compare.ps1.
Root: workspace/artifacts/combat-architecture-v1r2-20261006.

Первый preflight root combat-architecture-v1-20261006 сохранён отдельно:
8 MLP episodes/2 updates и4 GRU episodes. GRU update остановлен до
оптимизации, hidden parity max0.002045 против tolerance0.00002.
Independent no-grad precision diagnostic: CPU error<0.000001;
CUDA cuDNN TF32 enabled0.002045, disabled0.00000751. Причина в точности
cuDNN GRU, не в ослабленном допуске. Для основного R2 cuDNN/matmul TF32
отключены, обучение остаётся FP32 CUDA; tolerance не менялся.
R2 снова начинает все варианты с parent и fresh Adam; эти12 preflight
captures и MLP updates в основной comparison не включены.

После всех R2 updates обнаружен export/resume bug: Python loop variable
`key` перезаписывала architecture key значением `value`. Вторые checkpoint
тензоры/оптимизаторы правильные, но final JSON critic был object вместо
трёх dense layers;4 GRU Mixed evaluation attempts остановились до UDP
capture. Исходные malformed exports и эти attempts сохранены. Loop variable
исправлена; отдельный двух-update CUDA integration test для GRU/обоих
attention проходит (82 Python tests суммарно). Final JSON восстановлен
в export-repaired из immutable checkpoint tensors, без повторных gradients,
изменений optimizer/RNG/counters или потребления новых rows. Repair manifest
pin SHA исходных файлов; audit сравнивает original/repaired state tensors
и все optimizer/RNG данные точно. Успешные evaluation используют эти
исправленные exports; initial parent/MLP captures не перезапускаются.
Тот же shadowing оставил bptt_steps=null в исходном втором GRU report;
реальный compute использовал32, заданные до resume loop. Это metadata
дефект сохранённого report, не отключение TBPTT; дальнейшие exports с
исправленным trainer записывают32 корректно.

## Завершённый R2

92 valid captures/23 batches,4 native server/client instances x2;32 fresh training episodes+60 evaluation. Initial stochastic provider commands/samples/native outcomes во всех четырёх ветках совпали точно (versions и memory payload различаются и не сравниваются как действия). Source/native fingerprints: `901d0df25472a304d1e80a3f03f2194568f2d1dafaa96b164af6cae4bb87e2b3` / `96bada6beb4d6242a023ce0210bde7fb8debd714729cee7d8a84f3836f103f1e`.

| Вариант | Actor+critic parameters | Consumed rows | Accepted actor steps | CUDA update seconds | Mixed inference p95, µs |
|---|---:|---:|---:|---:|---:|
| MLP | 112717 | 1938 | 20 | 2.60 | 1068 |
| GRU64 | 163222 | 1686 | 20 | 5.09 | 1550 |
| Temporal attention4×32 | 146582 | 1588 | 20 | 3.15 | 3627 |
| Entity attention4 heads | 155414 | 1751 | 20 | 4.61 | 3089 |

Parameter counts не уравнены: этот пилот сравнивает рабочие варианты, а не изолирует пользу памяти от дополнительной ёмкости. CUDA update seconds — report wall time actual optimizer stages, без одинакового standalone benchmark; MLP имеет отдельный warm-up benchmark, GRU/attention его не имеют. Native inference измерен при isolated synchronous capture, не приёмка1× coop/live.

Полные uninterrupted победы, kills, deaths и incoming первой жизни; каждая ячейка wins/4, kills, deaths, incoming:

| Вариант | Mixed skill1 | Solo skill1 | Solo skill3 |
|---|---|---|---|
| Reference | 2/4, 5, 2, 356 | 4/4, 4, 0, 183 | 4/4, 4, 0, 273 |
| MLP | 4/4, 8, 0, 318 | 4/4, 4, 0, 153 | 4/4, 4, 0, 253 |
| GRU64 | 1/4, 4, 3, 395 | 4/4, 4, 0, 194 | 4/4, 4, 0, 273 |
| Temporal attention4×32 | 2/4, 5, 2, 347 | 4/4, 4, 0, 194 | 4/4, 4, 0, 273 |
| Entity attention4 heads | 2/4, 4, 2, 341 | 4/4, 4, 0, 157 | 4/4, 4, 0, 273 |

Training first-life outcomes и consumed kill transitions:

- MLP: 0/8 full wins,1 native kills,5 deaths; consumed kill transitions1.
- GRU64: 0/8 full wins,1 native kills,6 deaths; consumed kill transitions1.
- Temporal attention4×32: 0/8 full wins,0 native kills,5 deaths; consumed kill transitions0.
- Entity attention4 heads: 0/8 full wins,1 native kills,6 deaths; consumed kill transitions1.

Все ветки прошли exact native replay, source hashes, eligible transition coverage и independent GAE, own-policy model lineage, checkpoint/Adam counters и resume. GRU/temporal attention проверили всю Go hidden/cache/reset chain, а Torch recomputed current-model sequence states и log-probability/value на обоих own-policy batches. Entity/MLP проверили log-probability/value; zero-memory bank не даёт guarantee temporal retention. Второй batch использовал обновлённые residual projections, не только zero-head initialization.

Go tests и82 Python tests прошли. Gradient tests/updates только CUDA; precision diagnostic CPU выполнялся только no-grad. GRU cuDNN TF32 отключён, исходный tolerance сохранён. Causality/window/padding/token offsets/reset/duplicate-frame semantics проверены. Исходный reference/live сохранён, автоматического выбора оружия нет: все модели имеют прежние8 action logits, weapon output остаётся пустым.

Это первый ограниченный пилот по2 updates/8 training episodes на ветку. Четыре evaluation seeds на условие не доказывают превосходства архитектуры; known regression suite, другие карты, продолжительное обучение, память напарника и learned weapon choice не проверены. До live требуется больший frozen-budget опыт и полные regressions.
