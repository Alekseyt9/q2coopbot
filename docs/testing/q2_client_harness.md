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
$serverArgs = '+set game baseq2 +set dedicated 1 +set deathmatch 1 +set maxclients 8 +set minimumplayers 1 +set port 27934 +set logfile 2 +set botlib libgladiator_x64.dll +set coopbot_seed 305 +set coopbot_episode_id protocol-client-305 +map base2'
$server = Start-Process (Join-Path $runtime 'q2ded.exe') -ArgumentList $serverArgs -WorkingDirectory $runtime -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$server.Id
```

Порт и episode ID должны совпадать с параметрами harness. Не запускайте второй
server на том же порту.

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

## Проверенные сценарии через UDP

| Плановый сценарий | Episode | Что реально проверено |
| --- | --- | --- |
| Follow-style | `protocol-client-309` | 100 move-пакетов; 31 human snapshot; бот записал 26 кадров `following` и 5 `regroup` |
| Retreat-style | `harness-retreat-402` | 60 move-пакетов с backward/strafe/yaw/attack/jump; 31 human snapshot; 31 `regroup` |

Это уже проверяет transport/input/telemetry слой игрока. Это ещё не финальная
кооперативная acceptance-проверка: на `base2` в этих deathmatch-прогонах нет
монстров, поэтому cover, rescue, target focus и kill-steal нельзя честно
подтвердить только этим запуском.

Поворот, strafe, прыжок и произвольная временная последовательность клавиш
пока не вынесены в CLI. Их следующий шаг — расширение того же usercmd-потока,
а не возврат к графическому окну.

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
полный keyboard/mouse control, elevator scenarios или acceptance baseline
`N >= 20`.
Эти пункты по-прежнему требуют отдельных scripted/runtime сценариев.
