# План: отдельный обучаемый motor residual

Статус22.10.2026: [core Go/CUDA, migration и native compatibility реализованы](combat_motor_residual_20261022.md), новый корпус48native/48CUDA закрыт. [Первый matched fit](combat_motor_training_20261022.md) завершён, mechanical gate пройден; native comparison запущено, улучшение боя ещё не доказано. Ниже сохранён исходный протокол после закрытых [motor-scope probes](combat_movement_scope_20261022.md). Цель остаётся полностью обучаемым боем, а не исполнением движения правилами.

## Проверяемые изменения

1. Небольшая optional NN branch получает уже наблюдаемые geometry/state/enemy/navigation/threat-memory features напрямую, минуя замороженный combat encoder. Начальный вариант: common/environment/enemy prefix169 +typed enemies320 +navigation27 +remembered threats240 =756 inputs,64→32→2 outputs. Это draft input layout: сверить Go/CUDA offsets и доступность перед замораживанием schema. Выход — additive forward/side Gaussian means, не готовая команда «отойти от края».
2. Последний layer инициализируется нулём. Branch отсутствует — старый путь полностью сохраняется. Branch равна нулю — все outputs/likelihoods/выполненные команды должны совпасть с исходной моделью.
3. Движение добавляется после существующих spatial/target/weapon transformations. Старые контроллеры продолжают получать reference means исходной ветки. Sampled-motion aim получает фактически выбранный movement sample и тот же reference. Модель по-прежнему учитывает физическую связь движения с прицелом.

Предлагаемая причинная схема:

```
delta = motor(observed_features)
movement_mean = base_movement_mean + delta
reference_mean = movement_mean - delta
aim = existing_aim(observation, reference_mean, actual_sampled_move)
```

Motor inputs не должны включать base movement means: тогда reference восстанавливается без второго base inference. Повторный motor forward для reference допустим, но должен быть детерминированным и правильно учитывать gradient cancellation. Предпочтителен явный `reference_branch`/reference parameter; не прятать новую семантику в mutable cached tensors или неявных weakref callbacks.

## Обязательные gates перед обучением

- Отдельный versioned model block и validation; существующие feature/action widths не менять без необходимости. Никаких future/native hidden state в inputs.
- Go и CUDA используют одинаковые feature offsets, order, normalization и reference semantics.
- CUDA zero migration и ненулевые motor perturbations. При фиксированном actual sampled movement изменение motor mean не меняет conditional aim; при изменении actual movement сохранённая aim branch продолжает реагировать. Проверка ненулевого случая обязательна: одной zero-equivalence недостаточно.
- Реальные полные causal histories. JSON/checkpoint, actor/value/std/Adam/RNG и optimizer parameter IDs сохраняются; новая optimizer group имеет только новые branch parameters. Явный migration/fork и source binding.
- Native source/zero compatibility на fresh matched ordinary/recovery cases через16-slot ×2 pool, проверка выбранных и исполненных действий. Если policy RNG привязан к изменившемуся model identity, сначала разобрать sampling lineage, а не объявлять equivalence по одному logits test.

## Обучение и приёмка

После compatibility собрать новые собственные TRAIN случаи на flat и elevation geometry; источник и conditions закрепить в registry. Первый fit обучает только residual branch, остальные actor параметры и их optimizer state сохраняются. Critic/advantage и фиксированный reference control одинаковы; CUDA only. Не вводить fall penalty одновременно с архитектурой.

До большого native screen проверить размер изменения movement и conditional KL при фиксированном sampled action. Затем сравнить source, matched control и candidate на независимых seeds, контролируя победы, здоровье/смерти, фактические MG outcomes, глубокие спуски и восстановление видимости. Веса не продвигать по training loss или отсутствию падений: полезные спуски/прыжки не должны стать запрещёнными.

Это проверка представления и dataflow. Полезность нового residual, достаточность его ёмкости и перенос на неизвестную геометрию ещё не установлены.
