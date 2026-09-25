# Q2 CoopBot — Go UDP-клиент

Кооперативный напарник для Quake II. Бот входит на обычный сервер Yamagi Quake II как сетевой клиент по протоколу 34. Исполнение, наблюдение мира, маршрутизация и подключение моделей находятся в Go-модуле этого репозитория. Серверу не нужны Gladiator Botlib, мод `game_clean` или специальные команды добавления бота.

## Запуск

Нужны Go 1.23+, выделенный сервер Quake II и легально полученные игровые данные `baseq2`. AAS-файлы в `baseq2/maps` используются Go-маршрутизатором, но игровой сервер их не загружает. Для тестов подготовьте изолированный runtime с **оригинальной** `baseq2/game.dll` Yamagi. Локальные игровые PAK/AAS лежат в `F:\src\quake2\assets\baseq2`; `prepare_runtime.ps1` берёт их оттуда по умолчанию.

```powershell
# После конфигурации отдельной сборки Yamagi:
cmake --build F:\src\quake2\yquake2\build\codex-speed-test --target q2ded game
./scripts/prepare_runtime.ps1
```

`prepare_runtime.ps1` берёт из указанного `AssetsRoot` только PAK/AAS-файлы, копирует сервер и `game.dll` из сборки Yamagi в игнорируемый Git каталог `workspace/runtime/q2go`. Пути источников можно задать через `-AssetsRoot`, `-ServerExe`, `-GameDll`; пригодные AAS из отдельного каталога можно добавить через `-AASRoot`. Файлы из этого каталога должны содержать переходы между областями, иначе скрипт завершится ошибкой.

```powershell
go test ./...
New-Item -ItemType Directory -Force workspace/build/go | Out-Null
go build -o workspace/build/go/q2coopbot.exe ./cmd/q2coopbot
Copy-Item config.example.json config.local.json
./workspace/build/go/q2coopbot.exe --config config.local.json
```

Все настройки Go-клиента находятся в JSON-файле; при запуске нужен только `--config`. Пример — [`config.example.json`](config.example.json). Относительные пути внутри JSON считаются от каталога самого файла, `run.duration` задаётся в формате Go (`30s`, `5m`, `0s` для работы без ограничения). Локальный `config.local.json` исключён из Git. Автоматический тест сохраняет отдельные конфиги обоих клиентов в каталоге результатов. Тестовый RCON-пароль передаётся через `Q2COOPBOT_TEST_RCON` и в JSON не записывается.

Для автоматического теста скорости с неподвижным вторым Go-клиентом:

```powershell
./scripts/run_speed_trial.ps1 -GameFrames 100 -SynchronizedStart -UnlimitedLoopbackRate
```

Переключатели `-SynchronizedStart` и `-UnlimitedLoopbackRate` относятся только к отдельной тестовой сборке Yamagi. Без них скрипт может проверить обычный сервер, но не сверяет применённые сервером команды. Подробности: [ускоренные прогоны](docs/testing/q2go_speed_trials.md).

## Структура

- `cmd/q2coopbot/` — запуск с единственным аргументом `--config`;
- `internal/bot/` — UDP-сессия, планирование и подключение моделей;
- `internal/quake/` — протокол 34, команды движения, BSP и AAS;
- `go.mod` — Go-модуль;
- `scripts/` — запуск и оценка тестовых эпизодов;
- `workspace/tools/bspc/` — отслеживаемые исходники отдельной утилиты подготовки AAS;
- `workspace/build/`, `workspace/runtime/`, `workspace/artifacts/` — локальные сборки, игровые данные и отчёты, исключённые из Git;
- `examples/python-harness/` — прежний Python-прототип и его тесты, сохранённые только как пример;
- `docs/` — текущий [план Go-бота](docs/system2_strategy_tactics_plan.md) и исторические материалы прежней реконструкции.

Для сборки отдельной утилиты AAS с предварительным расчётом переходов:

```powershell
cmake -S . -B workspace/build/aas -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build workspace/build/aas --target bspc
./scripts/compile_aas.ps1 -BspPath 'F:\path\to\base2.bsp' -OutputRoot workspace/artifacts/aas-base2
./scripts/prepare_runtime.ps1 -AASRoot workspace/artifacts/aas-base2
```

Цель `bspc` собирает Quake II-совместимый форк BSPC: AAS версии 5 содержит области и рассчитанные заранее переходы, которые Go-клиент читает без BotLib. Исходники форка, версия и лицензия GPL-2.0-or-later указаны в [UPSTREAM.md](workspace/tools/bspc/vendor/q2bspc/UPSTREAM.md). Старая неполная реализация сохранена только как отдельная цель `bspc_reconstruction` для исторических тестов. Проверенный запуск и ограничения описаны в [проверке генерации AAS](docs/testing/bspc_reachability.md).

Игровые PAK/AAS-файлы и исполняемый сервер не входят в репозиторий. Исторические документы могут описывать удалённый C-бот; они сохранены для справки и не задают текущую сборку.
