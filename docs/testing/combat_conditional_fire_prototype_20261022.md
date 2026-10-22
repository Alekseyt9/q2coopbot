# Условный fire residual: проверенное ядро, 22.10.2026

Статус: **Go runtime, CUDA likelihood, recurrent trainer и optimizer/retention подключены; checkpoint migration проверена на 4205 TRAIN rows**. После [исправления supervisor](combat_goal_boundary_repair_20261022.md) fresh v2 comparison8native/8CUDA прошёл semantic и strict full-trace audits:612 provider frames точны. [Первый matched fit](combat_conditional_fire_training_20261022.md) закрыт:32 TRAIN боя/5058rows,full иfire-only по10CUDA actor steps;fresh32-case evaluation закрыта: start/full/fire-only по3/8, rules4/8; promotion=false. Основание и полный план: [conditional fire](combat_conditional_fire_plan_20261022.md). Последняя [четырёхраундовая кривая](combat_ppo_learning_curve_20261022.md) не дала прироста побед: 10/24 в трёх точках, у rules 13/24. По умолчанию игровые веса не заменены.

Созданы `scripts/combat_conditional_fire.py` и `internal/policy/conditional_fire.go`. Optional spec `combat_action_conditioned_fire_v1`,MLP181→32→16→1,**6369параметров**. Выход добавляется к исходномуfire logit. Zero-output initialization сохраняет старую CUDA Bernoulli probability/log-probability и RNG;это пока не checkpoint migration или native parity.

| Input offsets | Содержимое |
|---|---|
| 0:73 | Наблюдаемые common20+geometry53 |
| 73:99 | Текущее equipped weapon/phase26 |
| 99:111 | Выбранный observed enemy12,либо нули дляtarget0 |
| 111:151 | Его typed features40,либо нули |
| 151:160 | Sampled target one-hot9 |
| 160:162 | Sampled coarse/fine mode2 |
| 162:166 | Sampled movement tanh иyaw/pitch delta/180;fine scale15/180 |
| 166:169 | Sampled release/jump/crouch3 |
| 169:181 | Sampled weapon request12,включаяkeep |

Context — фактический **model sample доenvironment guards/quantization**,не будущий результат выстрела. Недоступныеtarget/weapon,невалидныекатегории,NaN иотсутствующийcontext отвергаются. Attack не маскируется правилом «нет цели». Входыfeature/action context detached;обучаемыйresidual иисходныйraw fire logit сохраняют градиент. Черезsampled movement/aim не возникает дополнительногоpathwise градиента.

## Проверки

- `F:/src/strat/.venv-gpu/Scripts/python.exe scripts/test_combat_conditional_fire_cuda.py`: **8 tests, OK**. Первые пять проверяют input layout, zero identity/RNG, чувствительность/градиенты и отказ при неверном контексте/параметрах. Шестой загружает фактические source weights `combat-motor-feedback-v1-20261022/round-2-full-update`: raw outputs, joint log-probability и entropy точно совпадают после zero initialization; export сохраняет JSON; ненулевая fire bias даёт ожидаемую Bernoulli-разницу и градиент. Пропущенная fire branch и неподдерживаемые optimizer/retention вызовы отвергаются. Все численные NN операции на CUDA.
- `go test ./internal/policy -run '^TestConditionalFireContractAndInputs$' -count=1`:passed. Толькоlayout/availability/contract/finite checks,без выполненияNN наCPU.
- Snapshot источников/результатов: `workspace/artifacts/combat-conditional-fire-prototype-v1-20261022/report.json`.

Полная последовательность RNG в игре, Go/CUDA native inference parity, польза в бою **ещё не проверена**; первый training resume новой ветки проверен. CUDA checkpoint migration и causal replay проверены отдельно на 4205 rows. `PPOFile` принимает optional `conditional_fire` и проверяет header/параметры. `Decide` сохраняет fire uniform на прежнем месте RNG, но применяет conditional logit после окончательного aim/vertical/weapon; `Review` использует тот же logit. Без optional блока прежний путь остаётся. Go NN inference проверяется только в самой игре.

`SpatialActor` содержит и экспортирует отдельную branch; raw width 81 не изменяется. CUDA finalizer считает совместную likelihood на полном recorded sample и фиксирует SHA нового helper. Recurrent PPO, optimizer и conditional retention теперь поддерживают новый формат; feedforward PPO и SAC явно отказывают. Это ещё не обученная улучшенная модель. Исторические standalone и runtime-likelihood-stage snapshots сохранены; текущие доказательства находятся в [migration report](combat_conditional_fire_migration_20261022.md).

## Следующее обязательное подключение

1. Реализовано в исходниках: Go optional field/header/spec validation, deferred fire decision с прежним draw order и matching Review. Нужна native parity.
2. Реализовано: Python actor attachment/export и precision/recurrent/PPO/CUDA-replay likelihood с полным recorded context. Неподдерживаемые training consumers явно отказывают.
3. Реализовано и проверено на CUDA: sequence retention с frozen teacher и fixed complete behavior action context. Отдельная metadata MC expectation; остальные KL terms прежние.
4. Четвёртая optimizer group и CUDA zero checkpoint migration проверены с сохранением actor/critic/std/Adam/RNG/consumed history. Fresh v2 comparison после исправления supervisor прошёл semantic и strict full-trace audits; см. отчёт boundary repair.
5. Только послеэтого train-only fit иfresh paired quality test. Сначала фиксированнаянаграда,безодновременной смены reward или movement control.

Reward audit:впрежнемv13 ужеесть `off_target_attack=-.01`,но `actionQualityCosts` применяет его лишь дляdirect weapon приknown bbox,безuncertain target иприотсутствииalignment. Приполномотсутствиивидимогоknown enemy этотcost0;`machinegun_miss=0`. Поэтому conditional head сампосебе не гарантирует исчезновение стрельбы приocclusion. Первыйтест факторизации иотдельный будущийobjective test нельзя смешивать или объявлятьготовымисправлением.
