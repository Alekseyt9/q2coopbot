# Локальный канал внешних действий

Дата: 05.10.2026. Добавлен `combat_bridge_v1`: существующий Go UDP-клиент отправляет observation внешнему процессу и получает низкоуровневое действие. Игровой протокол, guards, capture и native-харнес остаются прежними. Обученных весов и optimizer пока нет.

## Подключение

В `combat.provider_file` можно передать JSON вида:

```json
{
  "kind": "combat_remote_v1",
  "address": "127.0.0.1:33000",
  "timeout_ms": 1500,
  "policy_version": "diagnostic_probe_bridge_v1",
  "episode": "runner",
  "seed": 0
}
```

Сетевой Provider разрешён только в изолированном loopback-пилоте с `test.synchronous`, combat barrier, frame pacing, capture и Blaster. Runner создаёт собственную конфигурацию для каждого эпизода: `episode=worker-N-seed-S`, отдельный seed и SHA256 файла. В каждом процессе клиента генерируется новый случайный session ID. Один внешний процесс может обслуживать четыре независимых клиента; сессии и игровые миры не смешиваются.

Для воспроизводимой диагностики:

```powershell
./scripts/run_learned_combat_bridge.ps1 -Workers 4 -EpisodesPerWorker 2 -Timescale 2 -GameFrames 300 -Seed 11800 -Port 32996 -BridgePort 33000 -Mixed -HealthKit
```

Этот скрипт только запускает диагностический peer и существующий baseline runner. Он проверяет готовность порта, сохраняет бинарник/конфигурации/транскрипт/хеши и останавливает свой peer в `finally`. Занятый порт отклоняется; чужой listener не останавливается. `q2policy-bridge` воспроизводит уже существующий diagnostic probe, загруженный один раз и клонированный для сессий; это не обученная политика и не новый игровой бот.

## Протокол

TCP IPv4 loopback, UTF-8 JSON, один объект на строку. На соединении сохраняется reader. Размер сообщения ограничен 256 KiB, неизвестные поля и второй JSON-объект в строке отклоняются. Входной config может быть обычным многострочным JSON.

| Request | Содержание |
| --- | --- |
| `version` | `combat_bridge_v1` |
| `session`, `request` | Случайная сессия клиента и возрастающий uint64 номер |
| `episode`, `seed` | Независимый эпизод worker и его seed |
| `policy_version` | Явно ожидаемая версия peer |
| `observation` | Полное `combat_observation_v3`, включая историю, геометрию и previous applied command; без серверной ground truth |

Ответ содержит `version`, `session`, `request`, `policy_version` и `action` версии `combat_action_v1`. Идентичность action обязана точно совпасть с identity исходного observation, включая жизнь, карту, connection, spawncount, actor и frame. Затем существующий Go-конвертер проверяет численные диапазоны, оружие и физические ограничения. Никаких дополнительных поворотов прицела или готовых манёвров peer не получает от adapter.

Один deadline 10–2000 мс покрывает connect, запись и чтение. Ошибка/timeout/несовпадение identity закрывает соединение, чтобы поздний ответ не стал ответом следующего шага. Ошибка отмечается обычным fallback rules в capture; последующий запрос может подключиться заново. Локальный diagnostic Provider сохраняет прежний бюджет 5 мс; сетевое ожидание разрешается только при native-паузе мира и ограничивается timeout конфигурации. `policy_version` — проверяемая метка протокола, а не доказательство содержимого удалённых обученных весов.

## Границы текущего интерфейса

Канал вызывается на свежих боевых кадрах, где Provider владеет выбором действия. Setup, смерть, noncombat и явные fallback остаются у правил и сохраняются в полном client capture. Peer получает следующий боевой observation и предыдущую реально применённую команду; серверная reward, terminal/truncated и reset proof сейчас экспортируются offline в отдельных файлах. Поэтому это ещё не полный `Reset/Step` API внешнего learner: отдельная доставка окончания жизни и онлайн reward остаётся следующей работой. PPO/BC не запускались.

## Проверка

Unit-тесты проверяют правильный ответ, чужие frame/session/request/policy/version, timeout с закрытием соединения, переподключение после позднего ответа, запрещённые адреса, несинхронный режим, неизвестные поля, переполнение, неполный JSON и многострочный config. Общий `go test ./...` прошёл.

Первый live-прогон: `workspace/artifacts/learner-bridge-20261005-182001-633/`, четыре seed 11700–11703, x2/300 игровых кадров. Внешний peer вернул 999 решений с нулём несовпадений identity, каждая session принадлежала одному seed. Все четыре captures и native synchronous proof прошли, frame gaps/decode errors — 0. Первые ответы намеренно задержаны на 250 мс: измеренные максимумы 250748–252043 мкс, при этом один tick на действие сохраняется. Отдельно доступны 252 first-life reward-шагa. Gameplay acceptance — 0, поскольку peer диагностический.

Прогон повторяемого runner: `workspace/artifacts/learner-bridge-20261005-182357-178/`. Четыре worker, два холодных эпизода на worker, восемь отдельных seed 11800–11807. Восемь сессий, 2019 внешних решений; число транзакций каждого эпизода точно совпало с provider-controlled frames его capture. Reply mismatches — 0. `bridge-report.json`: transport accepted и source/probe provenance valid; каждый эпизод имеет native synchronous proof. Первые ответы задержаны на 250 мс, максимум 250795–251767 мкс. Доступны 404 first-life reward-шагa. Диагностический peer остановлен после проверки.
