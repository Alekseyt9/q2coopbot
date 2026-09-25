# Предварительный расчёт AAS-переходов

Основная цель `bspc` — [Quake II-совместимый форк BSPC](../../workspace/tools/bspc/vendor/q2bspc/UPSTREAM.md). Она создаёт области и таблицу переходов при обработке BSP; игровой BotLib не требуется. Это отдельная утилита подготовки данных. Во время игры AAS версии 5 читает Go UDP-клиент, а сервер работает со стандартной `baseq2/game.dll`.

Для сборки нужны CMake, Ninja и компилятор C/C++ (на Windows проверен MinGW GCC 15). Каталог `bin` MinGW должен быть в `PATH`, чтобы компилятор мог запустить свои вспомогательные программы. Игровые BSP и PAK не входят в репозиторий. Пример запуска из корня проекта:

```powershell
cmake -S . -B workspace/build/aas -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build workspace/build/aas --target bspc
./scripts/compile_aas.ps1 -BspPath 'F:\path\to\base1.bsp' -OutputRoot workspace/artifacts/aas-base1
./scripts/compile_aas.ps1 -BspPath 'F:\path\to\base2.bsp' -OutputRoot workspace/artifacts/aas-base2
./scripts/prepare_runtime.ps1 -AASRoot workspace/artifacts/aas-base1
./scripts/prepare_runtime.ps1 -AASRoot workspace/artifacts/aas-base2
```

`compile_aas.ps1` не перезаписывает AAS, сохраняет лог компилятора и JSON с SHA-256, сверяет версию, диапазоны и кратность секций областей, настроек и переходов. Для разовой проверки без копирования в runtime можно передать `-AASDir` в `run_speed_trial.ps1`. Проверка перехода карты:

```powershell
./scripts/run_speed_trial.ps1 -TransitionMap base2 -TransitionAfterFrames 10 -GameFrames 20 -Timescales 1,2 -SynchronizedStart -UnlimitedLoopbackRate -RequireTransitionAAS
```

Проверенный результат для игровых BSP `base1` и `base2` (2026-09-25):

| Карта | Области AAS | Записанные переходы | SHA-256 AAS |
| --- | ---: | ---: | --- |
| `base1` | 3913 | 5482 | `FAD1821579FBD6E2846B182DA2E13910F0949996A574A880DDF595745BA87019` |
| `base2` | 3454 | 4570 | `2A12842195BDA257239D7EB70E022786C710D88D889BA73C76A425392F7118DA` |

При принудительном переходе `base1 → base2` Go загрузил 3454 области и 4568 поддерживаемых им переходов; `navigation=ready` на 1× и 2×, второй клиент виден после перехода, пропусков кадров и ошибок декодирования нет, все 32 команды в каждом прогоне приняты сервером. Скорости — 9,43 и 18,52 кадра/с. Артефакт: `workspace/artifacts/transition-bspc-v5-final-20260925/summary.json`.

Это доказывает пригодность нового файла для загрузки графа и короткого маршрута к напарнику. Компилятор записал для `base2` ноль лифтовых переходов; поведение на лифтах, дверях, длинных маршрутах и при естественном выходе с карты требует отдельных сценариев. Цель `bspc_reconstruction` остаётся только для [исторических тестов CLI](bspc_cli.md) и не создаёт рабочий AAS.
