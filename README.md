# Q2 CoopBot — Go UDP-клиент

Кооперативный напарник для Quake II. Бот входит на обычный сервер Yamagi Quake II как сетевой клиент по протоколу 34. Исполнение, наблюдение мира, маршрутизация и подключение моделей находятся в Go-модуле этого репозитория. Серверу не нужны Gladiator Botlib, мод `game_clean` или специальные команды добавления бота.

## Запуск

Нужны Go 1.23+, выделенный сервер Quake II и легально полученные игровые данные `baseq2`. AAS-файлы в `baseq2/maps` используются Go-маршрутизатором, но игровой сервер их не загружает. Для тестов сначала подготовьте изолированный runtime с **оригинальной** `baseq2/game.dll` Yamagi; каталог `q2coopbot-runtime-vanilla` на этой машине содержит старую Gladiator DLL и не должен использоваться как готовый сервер.

```powershell
# После конфигурации отдельной сборки Yamagi:
cmake --build F:\src\quake2\yquake2\build\codex-speed-test --target q2ded game
./scripts/prepare_runtime.ps1
```

`prepare_runtime.ps1` берёт из указанного `AssetsRoot` только PAK/AAS-файлы, копирует сервер и `game.dll` из сборки Yamagi в игнорируемый Git каталог `workspace/runtime/q2go`. Пути источников можно задать через `-AssetsRoot`, `-ServerExe`, `-GameDll`.

```powershell
go test ./...
New-Item -ItemType Directory -Force workspace/build/go | Out-Null
go build -o workspace/build/go/q2coopbot.exe ./cmd/q2coopbot
./workspace/build/go/q2coopbot.exe --game-dir .\workspace\runtime\q2go\baseq2 --port 27910 --name GoCoopMate
```

Для автоматического теста скорости с неподвижным вторым Go-клиентом:

```powershell
./scripts/run_speed_trial.ps1 -GameFrames 100 -SynchronizedStart -UnlimitedLoopbackRate
```

Переключатели `-SynchronizedStart` и `-UnlimitedLoopbackRate` относятся только к отдельной тестовой сборке Yamagi. Без них скрипт может проверить обычный сервер, но не сверяет применённые сервером команды. Подробности: [ускоренные прогоны](docs/testing/q2go_speed_trials.md).

## Структура

- `cmd/q2coopbot/` — CLI, флаги и запуск;
- `internal/bot/` — UDP-сессия, планирование и подключение моделей;
- `internal/quake/` — протокол 34, команды движения, BSP и AAS;
- `go.mod` — Go-модуль;
- `scripts/` — запуск и оценка тестовых эпизодов;
- `workspace/tools/bspc/` — отслеживаемые исходники отдельной утилиты подготовки AAS;
- `workspace/build/`, `workspace/runtime/`, `workspace/artifacts/` — локальные сборки, игровые данные и отчёты, исключённые из Git;
- `examples/python-harness/` — прежний Python-прототип и его тесты, сохранённые только как пример;
- `docs/` — текущий [план Go-бота](docs/system2_strategy_tactics_plan.md) и исторические материалы прежней реконструкции.

Для сборки утилиты AAS:

```powershell
cmake -S . -B workspace/build/aas -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build workspace/build/aas --target bspc
```

Игровые PAK/AAS-файлы и исполняемый сервер не входят в репозиторий. Исторические документы могут описывать удалённый C-бот; они сохранены для справки и не задают текущую сборку.
