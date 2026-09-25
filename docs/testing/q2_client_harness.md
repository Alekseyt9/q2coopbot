# Quake II client harness

**Архивный Python-пример.** Его исходники перенесены в `examples/python-harness/`; основной клиент и тестовый харнес теперь написаны на Go. Пути `tools/*.py` и инструкции для Gladiator ниже относятся к прежней структуре.

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
  `F:\src\quake2\q2coopbot-src`.

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

Из корня `q2coopbot-src`:

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

Для map-aware проверки живого маршрута используется closed-loop режим
`--waypoint X:Y:Z[:JUMP]`. Координаты читаются из свежих `bot_snapshot`
событий (`player_is_human=1`), поэтому следующий usercmd вычисляется по
фактической позиции игрока, а не по таймеру. `--use` удерживает `BUTTON_USE`,
а `--post-move-duration` оставляет клиента подключённым после последней точки,
чтобы AI успел обработать regroup:

```powershell
python .\tools\q2_client_handshake.py `
  --port 28338 --duration 120 --name ElevatorHuman `
  --event-log 'F:\src\quake2\q2coopbot-runtime-bot\coopbot_debug_events.jsonl' `
  --episode-id coop-elevator-natural-route-1348 `
  --require-human-marker --use --post-move-duration 45 `
  --waypoint 758.4:2296:-232 `
  --waypoint 715.8:2297:-232 `
  --waypoint=-84:1408:-176 `
  --waypoint=-56:1408:-176 `
  --waypoint=-84:1408:-176
```

Это сокращённый пример; полный 33-point маршрут задаёт batch-сценарий.

Отрицательное значение waypoint передавайте как `--waypoint=-X:Y:Z`, иначе
старый Windows PowerShell может принять его за параметр. В прогоне
`coop-elevator-natural-route-1348` UDP-клиент достиг `30/33` точек, а botlib
записал реальный `TRAVEL_ELEVATOR`, `coopbot_elevator_route`,
`coopbot_elevator_reacquired` и `coopbot_regroup_complete`. Это первый строгий
natural-route успех. В серии `1348..1372` strict bot elevator acceptance
прошёл `2/19` seed; отдельные `1368/1369` дошли до `31/33` точек и дали
вертикальный подъём живого UDP-игрока, но bot-reacquire не завершился.

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
| Co-op kill-steal default-radius fixture | `coop-kill-steal-default-fixture-1176..1195` | 20/20 strict UDP+human/report; 40 `kill_steal_yield` events при radius 192; 2306 player-focus events; live player focus на живом `monster_infantry` entity 306; bot separation 240 > 192; 287 bot shots и 99 monster-damage events / 990 damage |
| Co-op kill-steal natural default probe | `coop-kill-steal-default-1312..1331` | 20/20 UDP+human gate; 134 player-focus events; 0 `kill_steal_yield` при штатном radius 192; natural acceptance не закрыт |
| Co-op lost-LOS probe | `coop-lost-los-1292..1311` | 20/20 UDP+human gate; 16/20 strict `target_los_lost` (46 LOS-loss edges, 70 target acquisitions); `target_lost` отдельно не подменяется; естественный 20/20 acceptance не закрыт |
| Co-op lost-LOS map fixture | `coop-lost-los-map-fixture-1272..1291` | 20/20 strict UDP+human/report; 60 `target_los_lost` edges, 80 target acquisitions; bot held on valid `base2` AAS area 445 and then moved to the initial sector; real UDP player remained connected |
| Co-op elevator probe | `coop-elevator-1151` | 1/1 strict UDP+human; `coopbot_map_model` показывает 1 elevator и vertical edge 806 -> 751, delta 93.7; live `TRAVEL_ELEVATOR`/reacquire не зафиксирован |
| Co-op elevator runtime fixture | `coop-elevator-fixture-1155` | 1/1 strict UDP+human; opt-in UDP fixture разместила bot в area 806 и human в area 751; botlib записал `coopbot_elevator_route`, `coopbot_elevator_reacquired` и `coopbot_regroup_complete`; это runtime-ветка, не доказательство прохождения карты обычным движением |
| Co-op elevator player ride | `coop-elevator-player-1160` | 1/1 UDP+human; реальный UDP-игрок стартовал на нижней `func_plat` и через live server physics поднялся примерно с `z=-38` до `z=112`; отдельный gate проверяет вертикальный delta `>=64`; bot route fixture этим не подменяется |
| Co-op live elevator route probe | `coop-elevator-natural-route-1348..1372` | 19/19 UDP+human; `2/19` strict bot elevator acceptance (`1348`, `1358`); latest route reached `31/33` and produced live player vertical rise in `1368/1369`, but bot-reacquire remained incomplete |
| Co-op natural lost-LOS spot-check | `coop-lost-los-1354..1357` | 4/4 UDP+human; `3/4` strict episodes с `target_los_lost`, один seed не записал LOS-loss; это spot-check, не новый `20/20` baseline |
| Co-op natural kill-steal melee spot-check | `coop-kill-steal-default-melee-1377..1380` | 4/4 UDP+human; `2/4` strict `kill_steal_yield`, `1379/1380` не записали yield до смерти bot; это spot-check, не новый `20/20` baseline |

Первые две строки — исторические transport/input прогоны; они были выполнены
в deathmatch и не являются доказательством боевой кооперации. Все строки с
`Co-op` выполнены в правильном co-op режиме. `coop-combat-531` дополнительно
доказывает bot-side запуск Blaster и попадание по живому monster entity.
Acceptance baseline `N >= 20` частично подтверждён: cover закрыт отдельной
строгой серией, rescue закрыт в 20/20 seed, phase-retreat подтверждает
regroup в 12/20 seed, end-level persistence закрыт отдельной серией
`coop-transition-1114..1133`, а default-radius kill-steal закрыт
детерминированным live UDP fixture в 20/20 seed. Естественный uncontrolled
probe `coop-kill-steal-default-1312..1331` также дал 0 yield при 20/20
живых UDP-подключениях, поэтому map-aware kill-steal, стабильный natural
lost-LOS, elevator traversal и полные cooperative-level статистические
пороги всё ещё не подтверждены; отдельный map fixture для LOS-механики
закрыт в 20/20, transport/input проверен без foreground окон.

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
серией и deterministic fixture. `-Scenario kill-steal-default` использует
штатный радиус `192` и не задаёт позицию игрока или бота; серия
`coop-kill-steal-default-1312..1331` зафиксировала `0/20` yield при `20/20`
живых UDP-клиентах. Fixture через UDP-команды в
`coopbot_test_mode` ставит живого игрока рядом с живым `monster_infantry`, а
bot на 240 units; игрок всё равно создаётся обычным UDP handshake и держит
real-time attack usercmds. На каждый эпизод создаётся новый
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

`-Scenario elevator-player` проверяет именно движение живого UDP-игрока на
нижней `func_plat`: тестовая команда только ставит игрока на нижнюю площадку,
после чего `clc_move` идёт в real time, а reducer требует вертикальное
перемещение позиции игрока не менее чем на 64 units. Bot и игрок при этом
могут оказаться на одной движущейся платформе; это проверка player-side
physics, а не полная проверка bot traversal:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run_udp_coop_combat_batch.ps1 `
  -Scenario elevator-player -FirstSeed 1160 -Count 1 -Port 28000
```

`-Scenario elevator-natural-route` использует тот же live UDP-клиент, но
проходит map-aware closed-loop waypoint-маршрут к elevator-зоне без position
override; в конце harness удерживает `BUTTON_USE` и остаётся подключённым ещё
45 секунд. В probe включён health-guard для живого UDP-игрока и bot, а также
`god/notarget` для игрока; команды не задают координаты. Первый строгий успех
есть в seed `1348`, но стабильность `TRAVEL_ELEVATOR`/reacquire ещё нужно
довести:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run_udp_coop_combat_batch.ps1 `
  -Scenario elevator-natural-route -FirstSeed 1348 -Count 6 -Port 28348
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

Default-radius fixture запускается так:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run_udp_coop_combat_batch.ps1 `
  -Scenario kill-steal-default-fixture -FirstSeed 1176 -Count 20 -Port 28026
```

Это закрывает live runtime-механику yield при штатном радиусе `192`, но не
подменяет естественный map-aware маршрут: позиции fixture задаются только
opt-in UDP-командами при `coopbot_test_mode 1`.

Natural default-radius probe запускается так:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run_udp_coop_combat_batch.ps1 `
  -Scenario kill-steal-default -FirstSeed 1312 -Count 20 -Port 28280
```

Он использует обычный co-op маршрут и real-time usercmd-поток UDP-клиента,
без position/entity override. Отсутствие yield в этой серии — открытый
map-aware acceptance item, а не ошибка подключения harness.

Строгий LOS map fixture запускается так:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\tools\run_udp_coop_combat_batch.ps1 `
  -Scenario lost-los-map-fixture -FirstSeed 1272 -Count 20 -Port 28230
```

В нём UDP-игрок проходит handshake и отправляет real-time usercmds, а только
бот удерживается на валидной AAS-точке `area=445` (`160 1896 -168`) до выбора
живой цели; затем его переносят в другой известный сектор. Это доказывает
LOS-механику, но не заменяет естественную навигацию бота по карте.

Поворот, strafe, прыжок, attack и повторяемая многофазная временная
последовательность уже вынесены в CLI и batch-режим. Для LOS reducer различает
`target_lost` (цель исчезла) и `target_los_lost` (та же живая цель перестала
быть видимой). Strict map fixture закрыт, но естественная серия `16/20`
остаётся ниже acceptance; natural default-radius probe `1312..1331` дал
`0/20` yield. Следующий уровень — conditional map-aware timeline для
естественного lost-LOS и default-radius kill-steal. Для этого не требуется
возвращаться к графическому окну.

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

## Локальное управление через OpenJev

Опциональный режим использует установленную на F: модель
`hf.co/apus-ailab/APUS-OpenJev-v1-4B-GGUF:Q8_0` через локальный API Ollama.
Он управляет **UDP test player**, а встроенный CoopBot продолжает работать своим
обычным BotLib. При `-OpenJev` сервер включает `coopbot_openjev_world=1` и
передаёт живой снимок: здоровье, броню, оружие и активные патроны игрока,
позицию спутника, до четырёх видимых игроку живых врагов (позиция, скорость,
здоровье) и до трёх ближайших активных предметов. Модель примерно раз в
секунду выбирает атаку конкретного врага, движение к аптечке, следование,
отход или остановку. Харнесс пересчитывает угол наведения по свежей позиции
цели и отправляет `clc_move` каждые 50 мс. При потере цели, устаревшем
снимке или ошибке ответа он прекращает атаку.

```powershell
.\tools\run_udp_coop_combat_batch.ps1 -Scenario follow -FirstSeed 2037 -Count 1 -OpenJev
```

Результат и журнал решений сохраняются в `artifacts/udp-follow-openjev/`.
`openjev_attack_packets` считает отправленные команды атаки. Подтверждение
попадания требует отдельного `damage_applied` с ID UDP-игрока как `attacker`.
Граф проходимости AAS и надёжный маршрут к предмету модели пока не передаются;
движение к видимой аптечке здесь прямолинейное и может упереться в препятствие.

Для чистого Yamagi без CoopBot DLL используется `--udp-snapshot-log`: харнесс
разбирает стандартные пакеты протокола 34. В них есть позиции, модели и кадры
анимации сущностей, но нет здоровья монстров и гарантированной прямой видимости.
Кадры анимации смерти солдат и пехоты исключаются из списка целей. Атака по
цели, перед которой попал в стену бластер, временно блокируется.

Долгую игровую сессию можно остановить без перезапуска сервера и окна игры:
запустить харнесс с `--stop-file <путь>`, затем создать этот файл. Харнесс
отправит серверу команду `disconnect` и завершится. Перед следующим запуском
удалите файл остановки и дождитесь выхода предыдущего процесса.

## Ограничения доказательства

Успешный harness-тест доказывает:

- UDP challenge/connect контракт сервера;
- установку Quake II netchan;
- переход клиента через `new` и `begin`;
- создание и обработку реального non-bot player slot;
- попадание server command и human marker в runtime telemetry.
- обработку real-time `clc_move` usercmds с forward movement и attack flag.
- closed-loop waypoint navigation по telemetry-позиции игрока, включая jump/use.

Он не доказывает прохождение кооперативной кампании, качество прицеливания,
полный keyboard/mouse control или обычное прохождение elevator scenarios.
Waypoint coverage теперь доступна как маршрутный инструмент, но сама по себе
не доказывает успешный elevator traversal
или полные cooperative-level acceptance thresholds.
Runtime elevator branch отдельно smoke-tested через opt-in fixture, а live
player ride подтверждён отдельным vertical-position gate, но
полное перемещение к лифту и прохождение карты по-прежнему требуют отдельного
map-aware scripted/runtime сценария.
