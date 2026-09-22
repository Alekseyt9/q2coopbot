# Quake II client harness

Этот harness подключается к `q2ded.exe` по настоящему Quake II UDP-протоколу
и проходит тот же сетевой вход, что и клиент:

```text
getchallenge -> connect -> client_connect -> new -> serverdata -> begin
```

Он не заменяет графический тест Yamagi: рендеринга, клавиатуры и полноценного
human-playability здесь нет. Зато он позволяет проверить серверный контракт и
появление реального non-bot player entity, когда графический клиент не
запускается в текущей среде.

## Предварительные условия

- собранный `q2ded.exe` и `libgladiator_x64.dll` в
  `F:\src\quake2\q2coopbot-runtime-bot`;
- DLL игры после сборки скопирована в runtime (`gamex86_64.dll` и используемая
  сервером `baseq2\game.dll`/`baseq2\gamex86_64.dll`);
- Python 3;
- команда выполняется из
  `F:\src\quake2\q2coopbot-release`.

## Запуск dedicated server

Используйте отдельный UDP-порт и уникальный `episode_id`, чтобы не смешивать
результат с другим запуском:

```powershell
$runtime = 'F:\src\quake2\q2coopbot-runtime-bot'
$stdout = Join-Path $runtime 'protocol-client.stdout.log'
$stderr = Join-Path $runtime 'protocol-client.stderr.log'
$serverArgs = '+set game baseq2 +set dedicated 1 +set coop 1 +set deathmatch 0 +set maxclients 8 +set minimumplayers 1 +set port 27934 +set logfile 2 +set botlib libgladiator_x64.dll +set coopbot_seed 305 +set coopbot_episode_id protocol-client-305 +map base2'
$server = Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $serverArgs -WorkingDirectory $runtime -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$server.Id
```

Порт и episode ID должны совпадать с параметрами harness. Не запускайте второй
server на том же порту.

Для сценариев с монстрами обязательны `+set coop 1 +set deathmatch 0`.
Deathmatch-прогон на `base2` оставляйте только для транспортных/следящих
проверок: в нём карта не активирует монстров.

Для combat-прогона используйте `+set minimumplayers 2`: один слот занимает
Gladiator companion, второй — подключаемый UDP human. При `minimumplayers 1`
сервер может удалить бота сразу после входа human-клиента.

## Кеш AAS

`base2.aas` уже содержит готовый lump reachability, поэтому при co-op старте
сервер сразу пишет `AAS initialized.` без `calculating reachability...`.

Если у карты reachability-lump пустой (так было у исходного `base1.aas`),
один раз прогрейте карту и сохраните результат:

```powershell
$serverArgs = '+set game baseq2 +set dedicated 1 +set coop 1 +set deathmatch 0 +set framereachability 2000 +set forcewrite 1 +set maxclients 8 +set minimumplayers 1 +set port 27950 +set logfile 2 +set botlib libgladiator_x64.dll +set coopbot_seed 510 +set coopbot_episode_id aas-prewarm-base1-510 +map base1'
$server = Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $serverArgs -WorkingDirectory $runtime -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
```

Ждите `AAS initialized.` и только после этого останавливайте этот PID. В
проверенном прогоне `base1.aas` был переписан с reachability length `470492`;
следующий старт `base1` уже загрузил AAS без расчёта. Не используйте
`forcereachability 1` для обычных тестов: это принудительная регенерация, а не
загрузка кеша.

## Подключение и вход в игру

Из корня `q2coopbot-release`:

```powershell
python .\tools\q2_client_handshake.py `
  --host 127.0.0.1 `
  --port 27934 `
  --duration 8 `
  --name ProtocolHuman `
  --event-log 'F:\src\quake2\q2coopbot-runtime-bot\coopbot_debug_events.jsonl' `
  --episode-id protocol-client-305 `
  --require-human-marker
```

Успешный результат содержит:

```json
{
  "challenge_received": true,
  "client_connect_received": true,
  "spawncount": 1232907478,
  "begin_sent": true,
  "human_marker_seen": true
}
```

Значение `spawncount` меняется между запусками. В stdout dedicated server
должны появиться строки вида:

```text
ProtocolHuman connected
ProtocolHuman entered the game
```

Флаг `--require-human-marker` делает тест строгим: harness завершится с кодом
`0` только после события `bot_snapshot` с явным
`player_is_human=1`. Одного факта сетевого `client_connect` для этого гейта
недостаточно.

## Управление через harness

После `begin` harness умеет отправлять Quake II client-команды через
`--server-command` и настоящий проверенный `clc_move` usercmd-поток через
`--move-forward`:

```powershell
python .\tools\q2_client_handshake.py `
  --port 27934 `
  --duration 8 `
  --name ProtocolHuman `
  --event-log 'F:\src\quake2\q2coopbot-runtime-bot\coopbot_debug_events.jsonl' `
  --episode-id protocol-client-305 `
  --require-human-marker `
  --server-command 'say ProtocolHarnessActive' `
  --server-command 'use blaster' `
  --move-forward 3 `
  --attack
```

Команды передаются после входа в игру, поэтому в stdout сервера можно увидеть,
например:

```text
ProtocolHuman: ProtocolHarnessActive
```

Это подтверждает не только подключение, но и обработку сервером команды от
созданного player slot.

`--move-forward 3` отправляет forward usercmds с частотой 20 Hz в течение трёх
секунд. `--attack` добавляет `BUTTON_ATTACK` в эти пакеты. Harness считает
`move_packets` и применяет Quake II command checksum; в проверенном прогоне
получено 60 move-пакетов, а player origin в bot telemetry изменился с
`(-911.2 -72.0 -127.9)` до `(-608.1 -72.0 -135.9)`.

Для сценариев retreat/strafe/поворота используются те же пакеты:

```powershell
python .\tools\q2_client_handshake.py `
  --port 27934 `
  --move-forward 4 `
  --forward-speed -400 `
  --side-speed 200 `
  --yaw-rate 35 `
  --attack
```

`--forward-speed -400` моделирует движение назад, `--side-speed` — strafe,
`--yaw-rate` задаётся в градусах в секунду, `--jump` удерживает up input.
Если Windows возвращает UDP `WSAECONNRESET`, harness завершает транспортный
цикл и сообщает его в JSON-поле `socket_error`, не выдавая traceback.

## Проверенные сценарии через UDP

| Плановый сценарий | Episode | Что реально проверено |
| --- | --- | --- |
| Follow-style | `protocol-client-309` | 100 move-пакетов; 31 human snapshot; бот записал 26 кадров `following` и 5 `regroup` |
| Retreat-style | `harness-retreat-402` | 60 move-пакетов с backward/strafe/yaw/attack/jump; 31 human snapshot; 31 `regroup` |
| Co-op combat probe | `coop-combat-509` | настоящий co-op `base2`; 41 монстр; 360 move-пакетов stationary-spin/attack; human получил урон от монстров; botlib записал выбор целей `entity=45/306/293/374` |
| Co-op combat with bot fire | `coop-combat-531` | `minimumplayers 2`; 200 move-пакетов; 99 human snapshots; 13 bot Blaster launches/shots; 2 target acquisitions; 10 damage живому monster entity `302` |
| Co-op combat baseline | `coop-combat-538..557` | 20/20 UDP+human gate; 20/20 bot-owned fire; 6/20 эпизодов с bot damage по monster; 7 hit / 70 damage |
| Co-op follow baseline | `coop-follow-600..619` | 20/20 UDP+human gate; 0 stuck/regroup; scripted forward path не вышел за leash threshold |
| Co-op retreat baseline | `coop-retreat-660..679` | 20/20 UDP+human gate; 227 bot-owned shots; max distance 759.7; 0 stuck/regroup |
| Co-op phase retreat baseline | `coop-phase-retreat-740..759` | 20/20 UDP+human gate; 12/20 per-episode botlib logs contain `coopbot_regroup`; 295 bot-owned shots; 20 bot damage events / 842 damage; max distance 761.5 |
| Co-op cover baseline | `coop-cover-920..939` | 20/20 UDP+human gate; 20/20 strict `role=COVER`; 756 player-intent events; 104 cover-role events; max distance 678.1 |
| Co-op rescue baseline | `coop-rescue-850..869` | 20/20 strict `rescue_position`; 97 rescue-position events; 48 `role=RESCUER`; 20/20 UDP+human gate; 12/20 map transitions требуют отдельного persistence-теста |
| Co-op end-level transition + persistence | `coop-transition-1114..1133` | 20/20 strict UDP+human; 20/20 штатный runtime `map_transition`; 20/20 exact bot health/max-health/armor/ammo index+count/weapon match после карты |
| Co-op kill-steal default probe | `coop-kill-steal-980..986` | 7/7 UDP+human gate и player-focus telemetry; 0 `kill_steal_yield` при default radius 192 |
| Co-op kill-steal controlled probe | `coop-kill-steal-990..1009` | 20/20 UDP+human gate; 6/20 эпизодов и 20 `kill_steal_yield` events при controlled radius 64; default acceptance не закрыт |
| Co-op lost-LOS probe | `coop-lost-los-1015` + `1020..1039` | 21/21 UDP+human gate; 1/21 `target_lost`; 0 map transitions; acceptance не закрыт |
| Co-op elevator probe | `coop-elevator-1151` | 1/1 strict UDP+human; `coopbot_map_model` показывает 1 elevator и vertical edge 806 -> 751, delta 93.7; live `TRAVEL_ELEVATOR`/reacquire не зафиксирован |
| Co-op elevator runtime fixture | `coop-elevator-fixture-1155` | 1/1 strict UDP+human; opt-in UDP fixture разместила bot в area 806 и human в area 751; botlib записал `coopbot_elevator_route`, `coopbot_elevator_reacquired` и `coopbot_regroup_complete`; это runtime-ветка, не доказательство прохождения карты обычным движением |

Первые две строки — исторические transport/input прогоны; они были выполнены
в deathmatch и не являются доказательством боевой кооперации. Все строки с
`Co-op` выполнены в правильном co-op режиме. `coop-combat-531` дополнительно
доказывает bot-side запуск Blaster и попадание по живому monster entity.
Acceptance baseline `N >= 20` частично подтверждён: cover закрыт отдельной
строгой серией, rescue закрыт в 20/20 seed, phase-retreat подтверждает
regroup в 12/20 seed, а end-level persistence закрыт отдельной серией
`coop-transition-1114..1133`. Default-radius kill-steal, стабильный lost-LOS,
elevator traversal и полные cooperative-level статистические пороги всё ещё
не подтверждены; transport/input проверен без foreground окон.

Для записи projectile/shot/damage hooks в combat-команде нужен
`+set coopbot_log 2`; при обычном `coopbot_log 1` базовые снапшоты остаются,
но подробная боевая телеметрия не пишется.

Отчёт `coop-combat-531` сохранён в
`artifacts/coop-combat-531-report.json`.

## Повторяемые co-op серии

Для baseline можно использовать скрытый batch-раннер. Он не открывает окно
`q2ded`, не выводит его в foreground и сохраняет stdout/stderr каждого эпизода
в `artifacts/udp-*-baseline/`:

```powershell
Start-Process pwsh.exe -WindowStyle Hidden -ArgumentList @(
  '-NoProfile', '-ExecutionPolicy', 'Bypass',
  '-File', '.\tools\run_udp_coop_combat_batch.ps1',
  '-Scenario', 'retreat',
  '-FirstSeed', '660', '-Count', '20',
  '-Port', '28060', '-MoveSeconds', '10'
) -RedirectStandardOutput '.\artifacts\udp-retreat-baseline\retreat-batch.stdout.log' `
  -RedirectStandardError '.\artifacts\udp-retreat-baseline\retreat-batch.stderr.log'
```

`-Scenario combat` отправляет forward+attack и требует bot damage по monster;
`-Scenario follow` отправляет forward без attack; `-Scenario retreat`
отправляет backward+strafe+yaw+jump+attack; `-Scenario phase-retreat`
отправляет фазу advance, затем backward+strafe+yaw+jump+attack и финальный
hold+attack через repeatable `--phase` usercmds; `-Scenario rescue` включает
`coopbot_roles/rescue` и оставляет игрока под monster pressure, требуя
`coopbot_rescue_position`; `-Scenario cover` включает cooperative roles и
требует отдельный `role=COVER` в botlib-log; `-Scenario kill-steal` включает
player intent/shared focus/kill-steal control и требует `kill_steal_yield` в
botlib-log; batch-сценарий kill-steal намеренно задаёт controlled
`coopbot_kill_steal_radius=64`, тогда как default `192` проверяется отдельной
серией. На каждый эпизод создаётся новый
co-op server с `minimumplayers 2`; повторно использовать тот же `episode_id`
в общем JSONL не следует — для чистой статистики нужен новый диапазон seed.

`-Scenario transition` включает только для тестового прогона opt-in cvar
`coopbot_test_mode 1`, отправляет через UDP сначала маркировку состояния бота,
затем штатную команду перехода `base2 -> base1`, и требует
`--require-runtime-map-transition --require-bot-state-persistence`. Проверяются
`health`, `max_health`, `armor`, `ammo_index`, `ammo` и `weapon` до/после
перехода. Production-путь по умолчанию этот hook не включает:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run_udp_coop_combat_batch.ps1 `
  -Scenario transition -FirstSeed 1114 -Count 20 -Port 28604
```

`-Scenario elevator` включает `coopbot_map_model 1`, идёт к вертикальному
участку `base2` через real-time UDP phases и строго проверяет наличие
вертикального `TRAVEL_ELEVATOR` edge в AAS. Это отдельный map/route probe;
для полного runtime acceptance дополнительно нужны `WAIT_ELEVATOR` /
`TRAVEL_ELEVATOR` / `reacquired` события.

`-Scenario elevator-fixture` — отдельный opt-in runtime smoke test. Harness
через UDP отправляет тестовые команды `coopbot_test_bot_position` и
`coopbot_test_player_position`, ставит bot в нижнюю area `806`, human в верхнюю
area `751` и требует `TRAVEL_ELEVATOR` route, reacquire и completed regroup.
Команды доступны только при `coopbot_test_mode 1`; они не имитируют обычное
прохождение карты и нужны для проверки live runtime-ветки без foreground окна:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run_udp_coop_combat_batch.ps1 `
  -Scenario elevator-fixture -FirstSeed 1155 -Count 1 -Port 27995
```

Для ручного timeline harness поддерживает `--server-command-at SECONDS:COMMAND`
(repeatable) и общий `--server-command-delay SECONDS`; это позволяет отправлять
серверные команды после `begin`, не подменяя real-time `clc_move` поток.

Пример многофазного живого клиента напрямую:

```powershell
python .\tools\q2_client_handshake.py `
  --port 28100 --duration 16 --move-forward 0 `
  --phase 4:400:0:0:0:0 `
  --phase 6:-400:200:35:1:1 `
  --phase 4:0:0:0:1:0 `
  --event-log 'F:\src\quake2\q2coopbot-runtime-bot\coopbot_debug_events.jsonl' `
  --episode-id coop-phase-retreat-700 --require-human-marker
```

Reducer различает общий combat telemetry и bot-owned показатели:

```powershell
python .\tools\coopbot_event_report.py `
  'F:\src\quake2\q2coopbot-runtime-bot\coopbot_debug_events.jsonl' `
  --episode-id coop-combat-531 `
  --require-human-player `
  --require-bot-shot `
  --require-bot-monster-damage
```

Для kill-steal используется дополнительный botlib gate:

```powershell
python .\tools\coopbot_event_report.py `
  'F:\src\quake2\q2coopbot-runtime-bot\coopbot_debug_events.jsonl' `
  --episode-id coop-kill-steal-980 `
  --botlib-log '.\artifacts\udp-kill-steal-baseline\coop-kill-steal-980-botlib.log' `
  --require-human-player --require-kill-steal-yield
```

Поворот, strafe, прыжок, attack и повторяемая многофазная временная
последовательность уже вынесены в CLI и batch-режим. Следующий уровень —
map-aware waypoint/conditional timeline для гарантированных LOS и default-radius
kill-steal; для этого не требуется возвращаться к графическому окну.

## Проверка отчётом

После запуска можно проверить строгий human-player gate существующего reducer-а:

```powershell
python .\tools\coopbot_event_report.py `
  'F:\src\quake2\q2coopbot-runtime-bot\coopbot_debug_events.jsonl' `
  --require-human-player `
  --output .\artifacts\protocol-client-305-report.json
```

Reducer читает переданный JSONL целиком. Для сравнения конкретной серии
эпизодов сначала используйте отдельный лог или выделенный файл событий.

## Остановка сервера

Останавливайте только PID, который вернул `Start-Process`:

```powershell
$owned = Get-Process -Id $server.Id -ErrorAction SilentlyContinue
if ($null -ne $owned -and $owned.ProcessName -eq 'q2ded') {
  Stop-Process -Id $server.Id -Force
}
```

## Ограничения доказательства

Успешный harness-тест доказывает:

- UDP challenge/connect контракт сервера;
- установку Quake II netchan;
- переход клиента через `new` и `begin`;
- создание и обработку реального non-bot player slot;
- попадание server command и human marker в runtime telemetry.
- обработку real-time `clc_move` usercmds с forward movement и attack flag.

Он не доказывает прохождение кооперативной кампании, качество прицеливания,
полный keyboard/mouse control, обычное прохождение elevator scenarios,
scripted waypoint coverage
или полные cooperative-level acceptance thresholds.
Runtime elevator branch отдельно smoke-tested через opt-in fixture, но
полное перемещение к лифту и прохождение карты по-прежнему требуют отдельного
map-aware scripted/runtime сценария.
