# Learned combat при потере видимости

06.10.2026. Продолжение [fixed Mixed reset](learned_combat_mixed_fixed_reset_v4.md).

## Изменение ownership

Ранее `len(observation.enemies)==0` немедленно возвращал управление rules,
даже если ранее наблюдавшийся живой монстр просто исчез из UDP snapshot.
Теперь Go хранит только ID/class наблюдавшихся врагов для определения
границы боя. Координаты/скорости невидимых врагов не сохраняются в этом
контроллере и не подставляются в observation.

После последнего видимого врага learned provider сохраняет управление
до30 последовательных игровых frames —3 секунды game time, независимо
от timescale. Во время этого окна сеть получает **пустой** текущий список
enemies и обычную историю client observations. Каждый возврат видимого
врага запускает новое окно. Убийство всех известных врагов по UDP death
animation (`Snapshot.Defeated`) немедленно завершает engagement. Server
reward/hidden health/позиции в этом решении не используются.

Если неизвестный исход сохраняется дольше30 frames, передача записывается
как `combat_visibility_timeout`, а не объявляется завершённым боем или
убийством. Смена жизни/карты/connection/spawncount, откат frame и пропуск
наблюдений сбрасывают ownership history. Death/stale/setup/geometry/provider
failures сохраняют прежние явно записанные границы. Rules/shadow режимы
не получают learned движения. В Selection добавлен `combat_continuation`;
observation/action/810 features и веса исходной MLP не изменены.

## Синхронный inference budget

Первый Mixed repeat20800–20803 выявил ещё одну зависимость от wall time:
один local decision занял5862µs и вернул rules при лимите5000µs. Это
изменило первый life outcome20801. В20800 trajectories расходились после
явного visibility timeout и передачи rules. Initial repeat не прошёл
`--require-equal`; его результат не объявлен воспроизводимым.

В native synchronous test мир удерживается до команды. Завершённое local
решение теперь применяется даже при превышении5ms; превышение отдельно
логируется `inference_budget_exceeded=true` и сохраняется elapsed_us.
Обычный asynchronous/live budget и remote deadline продолжают давать
явный fallback. Это корректировка ownership в test lockstep, не новая
тактическая помощь. Panic/invalid action/stale observation не игнорируются.

## Проверка runtime

Все прогоны4 instances x2,300 frames,Blaster,fixed100,stock HP, reward v4,
immutable runtime assets. Initial Mixed20800–20803 дважды, Solo20900–20903;
после budget fix новая повторная пара20800–20803. В final repeat четыре
непрерывных provider prefix полностью совпали:226/206/208/226 rows.
Сравнение исключает setup и заканчивается на первом фактическом возврате
rules; transition reason не считается командой provider.

Полные first-life trajectories совпали3/4; outcomes совпали4/4.20800
расходится после явной потери видимости/передачи rules; эта часть не входит
в доказательство learned repeatability. В каждом final наборе5 kills,
1 death,2/4 полных побед, incoming243. Solo сохраняет4 kills/0 deaths,
incoming163. Это validation ownership, не парный рост качества.

В final20800 записаны167/168 cumulative held frames. Максимальная
непрерывная серия30, timeout3/2 раза; различие относится к части после
первого возврата rules. Во всех held observations enemies пуст,
candidate identity совпадает, native command matched. Один последний
held row в каждом наборе не имеет следующего observation: это
`game_frame_limit`, reward unavailable, effect window не объявлено
проверенным. Такие rows не входят в PPO. Все observed kill reward windows
в проверенных captures принадлежат provider. Само наличие kill windows
не превращает timeout-assisted episode в uninterrupted acceptance.

`go test ./...` прошёл с полным BSP/AAS root
`workspace/runtime/q2go/baseq2`. Четыре новые проверки включают bounded
grace/reset, observed defeat, provider ownership при пустом observation и
local synchronous/asynchronous inference budget. Первоначальный запуск с
assets-only root не имел base2/base3 AAS; после выбора существующего
полного root весь suite прошёл, assets не скачивались.

## Один fresh GPU PPO update

От исходного barrel parent выполнен один fresh stochastic Mixed batch
21000–21003 и update, anchor retention beta1 (schedule1,2/3,1/3,0 pinned).
Он выполнен **до** synchronous budget fix и сохранён как диагностическая
ветка. Не менять задним числом её provenance и не считать текущей live policy.

644 native on-policy rows;240 rows с пустым enemy observation вошли в
training, один kill reward+5 также включён. PyTorch2.10+cu128/RTX5070,
только CUDA training/benchmark,10 accepted actor steps/40 critic steps,
behavior KL0.00804966, anchor KL0.00847264. Итог54 cumulative PPO updates/
534 actor steps. Independent audit проверил consumed rewards/components
(max error8.88e-16), checkpoint/model tensor parity, Adam10/40, parent/resume
SHA и отсутствие повторных rollout. Native re-export644 rows побайтно
совпадает. Deterministic evaluation batch был корректно отклонён native
PPO exporter: для обучения использован новый stochastic batch.

Weights `045100cb0351a44c40d2497328de4b1496b6a96f32396fe73cc3470b989b73ee`,
checkpoint `78ef6323abbf8925238b9cca54d8beb076ad37e927cef65583f7268cac2bae65`.
Резюмирование этой ветки требует того же anchor/schedule.

| Diagnostic Mixed21100–21103 | Полные победы | Kills | Deaths | Incoming |
|---|---:|---:|---:|---:|
| Before | 4/4 | 8 | 0 | 199 |
| After | 3/4 | 6 | 0 | 186 |

Полного улучшения нет; before/after ещё захвачены с прежним budget
fallback, поэтому это descriptive diagnostic, не причинный выигрыш от PPO.
Не использовать их states для нового обучения. Ветка не принята.

Всего32 captures этого шага:12 initial runtime/Solo,12 PPO train/eval,
8 final repeat. Пользовательский live runtime/config не менялись.

Следующий опыт — fresh GPU PPO на окончательном lockstep runtime, с
новыми seeds и самостоятельным Solo regression. Timeout/иная помощь до
завершения боя отдельно исключается из боевой приёмки. Оставшаяся проблема
— learned поиск/добивание после длительной потери видимости; нужны свежие
успешные transitions и проверка памяти, а не скрытые позиции или rules aim.
GRU/attention/обучаемое оружие пока не реализованы.

Evidence roots:`combat-occlusion-continuity-v4-20261006`
(`continuity-audit.json`),`combat-occlusion-continuity-confirm-v4-20261006`
(`prefix-repeat-audit.json`),`combat-occlusion-ppo-v4-20261006`
(`training-audit.json`,`native-reexport-audit.json`,checkpoint/datasets).

```powershell
& F:/src/strat/.venv-gpu/Scripts/python.exe scripts/audit_combat_repeat.py workspace/artifacts/combat-occlusion-continuity-confirm-v4-20261006 --seed 20800 --learned-prefix --require-equal
```
