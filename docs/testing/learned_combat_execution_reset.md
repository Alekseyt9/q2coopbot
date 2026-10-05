# R2: native dispatch usercmd и проверка полей старта

Дата: 05.10.2026. Продолжение [наблюдений v2 и серверных исходов](learned_combat_observation_outcomes.md).
Это проверка исполнения и данных; обученных весов и online reset/step API пока нет.

## Подтверждение команды

В native `SV_ClientThink` уже существует `sv_test_applied_cmd`: запись после
проверки commandMsec и непосредственно перед `ge->ClientThink`. Теперь worker
включает `sv_test_trace_client SoloRetreatBot`, а exporter использует существующие
Go parser/verifier из `internal/harness`. Parser дополнен выбором имени клиента;
прежний default GoCoopMate и его проверки сохранены. Native код не менялся.

Capture и step сохраняют `client_sequence`. Вся trace, включая setup и последнюю
команду, сопоставляется с сервером по connection, spawncount, sequence, виду
`new` и точному равенству **всех** полей usercmd. Connection берётся из отдельного
успешного `<имя> connected`, а не из повторов handshake. Пропущенная, дублированная,
лишняя или изменённая команда отвергает required execution proof. Дополнительные
`old`/`oldest`/`last` учитываются как recovery и не заменяют доказательство `new`.
Отдельные actor/map в applied-log отсутствуют: принадлежность задаёт native
фильтр одного имени; actor/map сохраняются в клиентском step, мир привязан через
spawncount. Два клиента с одним именем не являются поддержанным worker-режимом.

Каждый step получает `server_execution`, версия `server_dispatch_v1`:

- `matched`: эта команда точно передана native ClientThink;
- `dispatch_frame`: реальный серверный кадр передачи;
- `window_exclusive`: между observation F и snapshot F+1 была ровно одна
  переданная команда — именно эта, без других new/recovery;
- количество recovery-команд в окне и причина неподтверждённого окна.

Сетевая отправка observation F предшествует следующему tick; usercmd, полученный
между snapshot F и F+1, исполняется при server frame F. Для обычного перехода
проверяется `[F, F+1)`, тогда как события **эффектов** по-прежнему собираются в
`(F, F+1]`. Эта разница не исправляется сдвигом damage log на глаз.

`matched=true` при поздней доставке не означает `window_exclusive=true`.
Последняя команда может быть matched, но без next observation её окно не
подтверждается. Глобальный accepted proof подтверждает все dispatch; он не
гарантирует пригодность каждого перехода для обучения. Для последовательного
RL-перехода будущий reader должен требовать `window_exclusive=true`, затем
обрабатывать terminal/truncated, handoff и intervention. Неоднозначные окна
сохранены как диагностика и не исправляются чужими командами.

Телеметрия непосредственно перед вызовом подтверждает dispatch, а не возврат
callback, попадание или успешное перемещение. Состояние после обработки видимо
в следующем snapshot. Урон от старого снаряда не приписывается новой атаке:
серверные outcome остаются отдельными временными окнами, scalar reward — null.

## Проверка первого состояния

Runner создаёт `reset-expectation.json` только из безопасных полей worker-конфига:
карта, placement, исходные health/armor, Shotgun и 20 shells, класс/позиция
наблюдаемого врага. Exporter сравнивает именно **первое** пригодное observation
после overrides; не ищет поздний удобный кадр для получения acceptance.

Проверяются life=1, свежесть и identity, health/armor/ammo, ground/ducked,
позиция (не более 1 unit по каждой оси), оружие и реально видимый враг с
ожидаемыми class/position. Начальный viewmodel Shotgun распознаётся как известный
native model path, если inventory/item names ещё не получены. Эта проверка
не добавляет серверных или ожидаемых координат во вход Provider.

`episode_start.json` содержит `observed_reset_proof`. В report
`observed_reset_confirmed=true` означает совпадение этих полей; одновременно
`full_server_reset_confirmed=false` и старый `fixture_reset_confirmed=false`.
Inventory, RNG, hidden monster AI, серверные поколения и полное состояние мира
перечислены как unverified. Cold restart и seed ready marker остаются отдельными
проверками; точное восстановление всего мира или save/load equivalence не заявлены.
Полный manifest/runtime assets сохраняются прежним runner.

## Прогоны: четыре инстанса x2

Все worker runs — loopback, 300 игровых кадров, Blaster-пилот, существующий
solo tactical harness, без LLM. 301 команда на клиент.

| Каталог baseline | Режим / seeds | Native commands matched | Reset fields | Игровая приёмка |
| --- | --- | ---: | --- | --- |
| `20261005-171506-408` | Direct probe, 10600–10603, до добавления reset check | 1204/1204 | не проверялись | 0/4 |
| `20261005-171953-620` | Shadow, 10700–10703 | 1204/1204 | 4/4 | 4/4 |
| `20261005-172137-458` | Direct probe, 10800–10803 | 1204/1204 | 4/4 | 0/4 |

Полный путь каждого: `workspace/artifacts/learned-combat-baseline-<каталог>`.
Capture/provenance прошёл во всех случаях; decoder errors, frame gaps,
command mismatches и recovery commands — ноль. Direct остаётся диагностическим
probe, а не политикой, обученной побеждать.

Shadow: 1168 steps, 1160 exclusive execution windows. Четыре последних команды
не имеют next observation; ещё по одному late dispatch и окну с двумя new
dispatch у seeds 10700 и 10702. Это реальные ограничения alignment, несмотря
на нулевые потери кадров и полное совпадение команд. Около 54.78 commands/s
вместе с cold restart/export.

Финальный direct: 1113 steps, 1109 exclusive windows, ещё четыре последних
команды без next observation. Все проверяемые стартовые поля совпали.
19.79–19.82 игровых кадров/реальную секунду; около 52.39 commands/s с полным
runner. p95 Provider/conversion/guards 0.511–0.518 мс — не latency будущей сети.

`go test ./...` прошёл. Новые тесты проверяют выбор traced client, sequence и
connection/spawncount, command mismatch, duplicates, late dispatch, recovery,
tail без next state; reset identity/resources/pose/weapon/visibility/finite
coordinates и явное отсутствие подтверждения скрытого server state.
В shadow artifact сохранены `negative-checks/report.json`: неверное ожидаемое
здоровье и удалённая native запись команды оба отвергнуты CLI. У неуспешного
экспорта нет успешного report; частичные файлы нельзя принимать как dataset.

## Дальнейшая работа

Нужны подтверждённый полный старт с неподвижным AI до release, синхронный online
reset/step с управлением tick, полноценные geometry/projectile признаки и
управление в воздухе. После этого — delayed reward/версии коэффициентов,
отбор хороших демонстраций и BC/PPO. Текущий экспорт не заменяет online среду.
