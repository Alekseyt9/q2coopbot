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
latent samples, native executed commands и outcomes во всех вариантах.
Второй own-policy batch проверяет Go/Torch log-probability/value согласие
после ненулевых residual updates. Native re-export/source/reward/GAE,
optimizer resume/counters, seed exclusion, current-model sequence state и
causality/padding проверяются отдельно. Inference latency измеряется в Go,
training time — на CUDA; parameter count не приравнивается к compute budget.
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
