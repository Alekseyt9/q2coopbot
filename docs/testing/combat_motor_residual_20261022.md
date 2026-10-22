# Motor residual: реализация и совместимость, 22.10.2026

Реализована отдельная обучаемая ветка forward/side с прямыми входами наблюдаемой геометрии. Go/CUDA migration и native zero-compatibility прошли. [Первый matched CUDA fit](combat_motor_training_20261022.md) закрыт и прошёл механический gate; сравнение в игре запущено, доказанного улучшения боя пока нет.

## Контракт

Optional block `motor_residual`, version=`combat_observed_motor_residual_v1`. Сеть756→64→32→2 с ReLU между слоями, 50 594 новых параметра. Последний layer первоначально нулевой. Input layout: features[:169] +[466:786] +[854:1121]; это существующие common/geometry/enemy/type/navigation/remembered-threat данные. Ни movement means, ни future/server hidden state во входы не входят.

Go: `internal/policy/motor_residual.go`, интеграция в `PPOFile`/sampling/replay и `sampled_motion_aim.go`. CUDA: `scripts/combat_motor_residual.py`, SpatialActor wrapper и явный reference parameter в precision/correction/conditional retention. Feature width1147 и action width81 сохранены.

Residual прибавляется после исходных spatial/target/weapon transformations. Conditional aim восстанавливает base reference вычитанием того же motor(features). Поэтому при одной фактически выбранной команде движения обучение residual не должно менять aim; при другой actual command связь движения с прицелом сохраняется. Используются явные arguments, без mutable tensor caches/weakrefs.

## Проверки до обучения

`combat-motor-residual-zero-v1-20261022`: CUDA migration на2496 native/CUDA TRAIN rows и полных реальных историях.

- Нулевые raw outputs и log likelihoods точно совпали с parent; прежние actor/value/std/Adam/RNG/consumed-rollout поля сохранены, старые optimizer IDs не менялись.
- Новый motor Adam group добавлен последним, состояния его параметров пустые.
- При ненулевом offset (+0.015,−0.015) fixed-action aim error ниже1e-5, nonmovement conditional KL ниже1e-6; изменение actual sampled movement продолжает менять aim.
- Aim→motor gradient ниже1e-5, gradient движения в новую ветку ненулевой. Native/source LP error ниже прежнего допуска0.003; memory parity ниже2e-5.
- `go test ./internal/policy -run '^TestMotorResidualContractAndInputs$'`: только contract/layout, без исполнения нейросети. CUDA tests:2 новых motor scope/reference checks и4 сохранённых target/decision/movement checks. Новая проверка вынуждает Adam fallback и подтверждает сохранность замороженных momentum/weights.

Migration receipt подтверждает численную совместимость, не качество боя. CPU используется для чтения checkpoint/RNG metadata; вся нейросетевая математика проверок и будущего обучения — CUDA. Go выполняет inference в самой игре.

## Native compatibility

`combat-motor-zero-compat-v1-20261022`: source/zero на4 fresh ordinary/recovery условиях base3/ware2 site02, offset720,8 native /8 CUDA replay, pool16 ×2. Все655 released provider frames сравниваются через `scripts/audit_combat_motor_native_parity.py`.

Canonical features и memory histories совпали точно; предложенные/исполненные/отправленные команды и samples совпали, за исключением ожидаемой model version. Допускается только entity-ID bijection из одинаковых наблюдаемых class/track/pose.

Сначала raw observation comparator обнаружил packet age1ms против0. Полный аудит нашёл maximum jitter2ms: это wall-clock freshness metadata, не закодированный feature. Обе стороны во всех кадрах проходят существующий policy freshness guard0…300ms. В аудит добавлено именно сравнение класса свежести; geometric/state fields, canonical inputs и все actions по-прежнему сравниваются точно. Остальных различий не найдено.

## Обучение и следующий gate

Существует training scope `motor-residual`: только новые параметры, прочие actor/std и их Adam states проецируются неизменными, в том числе после fallback, очищающего momentum. Actor optimizer имеет отдельную motor group/LR; explicit fork/migration history сохранены. Conditional retention использует отдельные копии teacher motion **и motor reference**.

`combat-motor-corpus-v1-20261022` закрыт:32 TRAIN +16development,48native/48CUDA,4903/2586rows,base3/ware2,site01/site02,ordinary trio иgrenade recovery. Все6 наблюдаемых глубоких спусков вTRAIN имеют ground warning. Новые seed ranges проверены по истории, PPO-ready reward совпадает с parent. Резерв диска24GiB закреплён явно. Fit запускался только после закрытия корпуса.

Matched full-PPO/motor-only CUDA fit10actor/40critic иaudit закрыты: frozen actor/std/Adam иобщий critic подтверждены, motor mean shift0.005101 против full0.003262, conditional nonmovement KL ниже1e-6. Запущен новый32native/24CUDA smoke. Default и опубликованные веса не заменялись; детали в отчёте первого fit.

Старые исследовательские утилиты, не передающие reference для нового model block, должны завершаться ошибкой; такой fail-closed check добавлен. Приёмка текущего изменения покрывает Go runtime, CUDA canonical finalization, PPO/sequence retention и migration, а не автоматически все исторические SAC/BC tools.
