# Инвентарь V6: основание для learned weapon choice

07.10.2026. Добавлены UDP-visible признаки инвентаря и availability masks. Обучаемый categorical weapon head ещё не реализован; actor по-прежнему выдаёт 8 параметров движения/прицела/выстрела/vertical. Результаты ниже подтверждают наблюдения, миграцию и replay, а не обучение выбору оружия.

## Контракт

`combat_features_v6`: 845 признаков. Первые 814 в точности сохраняют V5. Добавлены known/fresh, 11 owned flags, 6 нормированных количеств ammo и 12 availability flags. Порядок действий: keep, Blaster, Shotgun, Super Shotgun, Machinegun, Chaingun, Grenade Launcher, Rocket Launcher, HyperBlaster, Railgun, BFG10K, Grenades.

Unknown inventory или age вне 0..20 оставляет доступным только keep. Известные устаревшие количества сохраняются вместе с freshness=0. Для смены нужны наблюдаемое наличие оружия и stock минимум ammo: SSG 2 shells, BFG 50 cells, остальные расходующие ammo 1. Маска не обещает готовность оружия к выстрелу. Negative/duplicate relevant inventory counts отклоняются.

Сохранена загрузка V1–V5 моделей. `migrate_combat_inventory_v6.py` добавляет по 31 нулевому столбцу actor/value first layer; Temporal attention и выходные слои сохраняются. Adam checkpoint пока не мигрируется. Entity attention требует отдельного token contract и здесь явно отклоняется.

## Проверки

- Go contract tests: unknown/empty/stale inventory, duplicate/negative counts, stock SSG/BFG thresholds; независимая Go/Python parity на 9 observation cases.
- Реальная Curriculum24 Temporal policy: 65 кадров, включая пересечение окна32 дважды, unknown/stale inventory. В Go FP64 действия и value после миграции в точности равны старой модели.
- `go test ./cmd/... ./internal/...`: прошёл с включёнными real-model и Python parity tests.
- `go test ./...`: завершился с ошибками duplicate `main` в старых generated `workspace/artifacts/temporal-mixed-demo-20261006` и `combat-mobile-recoil-aim-v1-20261007`. Product packages прошли; generated материалы не удалялись.

## Live UDP и CUDA replay

Артефакты: `workspace/artifacts/combat-inventory-v6-validation-v1-20261007/`. Source weights Curriculum24 SHA256 `7b01b4999df979aaef85f9b7c908246edd74f5fb68c583b2083800de24931c40`; migrated weights `6b0160609b30109de29f1264d95cd0fe3dbd67dfa6e9e5fc47e88af5a307479c`.

4 сервера/клиента x2, distinct seeds44000–44003, Mixed, fixed Machinegun, skill1, release100, max300, StopOnGoal, unchanged recoil-v5 reward. Все4 capture/dispatch/reset/seed проверки приняты, decode errors0. Три goal receipts, один first-life death, всего6 monster kills. Это небольшой stochastic compatibility run, не paired оценка улучшения модели. Baseline report `gameplay_accepted=2` использует свой gate; число goal receipts3 отдельно проверено, эти метрики не смешиваются.

Go `q2ppo-data` перепроверил native command/reward proof и sample replay; экспортировано441 first-life transition и441 context rows. Все имеют width845, inventory known=1 и fresh=1. Availability:364 строки keep+Blaster+Machinegun;77 строк keep+Blaster при исчерпании bullets.

`verify_combat_inventory_v6.py`: CUDA forward/replay, RTX5070, optimizer steps0. FP32 input-width change даёт небольшой GEMM drift при нулевых новых columns; FP64 Go тест выше требует exact equality. Максимальные ошибки:

| Проверка | Ошибка |
|---|---:|
| Actor V5→V6 CUDA logits | 0.000003338 |
| Value V5→V6 CUDA | 0.000012160 |
| Actor context Python/Go | 0.000007987 |
| Value context Python/Go | 0.000007614 |
| Joint legacy logprob Python/Go | 0.000031948 |
| Value Python/Go | 0.000018835 |

Bounds совпадают с текущим recurrent trainer: logprob<0.003, value<0.0001, context<0.00002. Миграционные logits/value сравниваются с bound0.0001; фактические ошибки сохранены в `cuda-replay.json`.

## Следующая работа

Добавить explicit versioned 12-way weapon head поверх actor, masked sampling и joint likelihood в Go/Python. Мигрировать Temporal residual/output и Adam moments, сохраняя старые 8 outputs и training counters. Проверить unavailable/stale choices и native use/weapon transition. Затем открыть в существующем харнесе несколько доступных оружий, обучать только CUDA и сравнить на новых paired seeds,4 instances x2. До этого fixed Machinegun runs не доказывают learned weapon choice; live/default policy не переключалась.
