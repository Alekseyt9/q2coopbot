# Q2 CoopBot — Go UDP-клиент

Кооперативный напарник для Quake II. Бот входит на обычный сервер Yamagi Quake II как сетевой клиент по протоколу 34. Исполнение, наблюдение мира, маршрутизация и подключение моделей находятся в Go-модуле в корне репозитория. Серверу не нужны Gladiator Botlib, мод `game_clean` или специальные команды добавления бота.

## Запуск

Нужны Go 1.23+, выделенный сервер Quake II и легально полученные игровые данные `baseq2`. Укажите каталог `baseq2` с PAK-файлами; AAS-файлы в `baseq2/maps` используются Go-маршрутизатором, но игровой сервер их не загружает.

```powershell
go test ./...
go build -o q2coopbot.exe .
./q2coopbot.exe --game-dir F:\src\quake2\q2coopbot-runtime-vanilla\baseq2 --port 27910 --name GoCoopMate
```

Для автоматического теста скорости с неподвижным вторым Go-клиентом:

```powershell
./scripts/run_speed_trial.ps1 -GameFrames 100 -SynchronizedStart -UnlimitedLoopbackRate `
  -ServerExe F:\src\quake2\yquake2\build\codex-speed-test\release\q2ded.exe
```

Переключатели `-SynchronizedStart` и `-UnlimitedLoopbackRate` относятся только к отдельной тестовой сборке Yamagi. Без них скрипт может проверить обычный сервер, но не сверяет применённые сервером команды. Подробности: [ускоренные прогоны](docs/testing/q2go_speed_trials.md).

## Структура

- `*.go`, `go.mod` — основной бот и UDP-харнес;
- `scripts/` — запуск и оценка тестовых эпизодов;
- `tools/bspc/` — отдельная утилита подготовки AAS из карт; собирается через корневой CMake, не входит в Go-бота;
- `examples/python-harness/` — прежний Python-прототип и его тесты, сохранённые только как пример;
- `docs/` — текущий [план Go-бота](docs/system2_strategy_tactics_plan.md) и исторические материалы прежней реконструкции.

Для сборки утилиты AAS:

```powershell
cmake -S . -B build-aas -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-aas --target bspc
```

Игровые PAK/AAS-файлы и исполняемый сервер не входят в репозиторий. Исторические документы могут описывать удалённый C-бот; они сохранены для справки и не задают текущую сборку.
