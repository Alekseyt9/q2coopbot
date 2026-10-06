# Mixed: фиксированный старт и повтор замороженной политики

06.10.2026. После [PPO retention](learned_combat_retention_ppo_v4.md)
проверяется воспроизводимость до следующего сравнительного обучения.

## Протокол

Runner теперь разрешает Mixed со stock health, synchronous Blaster и
`ReleaseGameFrame=100`. Нативный движок уже поддерживает этот барьер:
монстры не думают до release, первый controlled tick получает подтверждённый
post-frame seed, Blaster idle frame синхронизирован на9. Нативный код и
политика не изменяются. Curriculum HP и health kit ограничения сохранены.
PPO runner допускает fixed Mixed; его before/after уже передают release
frame в baseline runner.

Два последовательных запуска одинаковой deterministic barrel parent
`91697086e5c57d21f85ad393e0674e044ccd08fb54f77e91485e98ee3df7fd6b`,
seeds20400–20403,4 instances x2,300 controlled frames, прежний reward v4.
В каждом запуске собственные процессы/каталоги/эпизоды. Повтор seeds —
проверка воспроизводимости, не новые независимые условия. Обучения нет.
Root:`workspace/artifacts/combat-mixed-fixed-repeat-v4-20261006`.

Проверить native dispatch/provenance, RNG/release/weapon receipts; первые
наблюдения и последовательность состояния/команд, per-seed outcomes,
handoffs до убийств. Равенство видимых state не является доказательством
равенства полного мира/AI: native fixture доказательство ограничено
логируемыми полями. Не включать новую policy в live на основании повторов.

## Локализация и исправление

На первом этапе20400–20403 первые observations совпали4/4, полные
траектории2/4. Первое расхождение20400 — ID Blaster projectile в frame109,
20402 — ID projectile в104; положение/скорость этого первого projectile
ещё одинаковы. После исключения IDs первое различие enemy angles/action
появляется в126/107. IDs действительно принадлежат native edicts.

Во втором diagnostic наборе20500–20503 добавлен один offline
`g_test_entity_start` dump при release. Он подтвердил разные `func_timer`
nextthink, `DelayedUse` slots и свободные edicts до первого решения сети.
Приготовление замораживало только monsters; таймеры/объекты карты продолжали
выполняться и расходовать RNG между sign-on и release. В `G_FindFreeEdict`
выбирается первый доступный slot; его номер задаёт положение projectile в
нативном цикле обработки. Поэтому post-frame seed и weapon reset не
восстанавливают уже созданные timers/entities. Расхождение таймеров и
пула подтверждено dump; влияние каждого отдельного поля не изолировано.

`yquake2/src/game/g_main.c` теперь в **single-client cheats fixed-release
test** также удерживает non-client map entities до release. Положительные
nextthink сдвигаются вместе с preparation clock, чтобы сохранить ожидание
после старта; stock AI/projectile/world processing после release продолжается.
Неиспользуемые edicts получают единую начальную фазу freetime. Режим
release0 и обычный gameplay эту ветку не используют. Заморожены только
подготовительные состояния, без телепортов/дополнительных предметов во время
приёмочного боя. Geometry/BSP и learned weights не изменены.

Release логирует `g_test_world_start frame=100 phase=fixed_map_hold
free_pool_reset=1`; baseline runner требует ровно один такой receipt вместе
с seed и gunframe receipts. Manifest содержит `fixed_world_hold=true`.
Дамп native сущностей остаётся offline диагностикой и не входит в features.
Он перечисляет часть полей, а не сериализует весь мир/AI.

Game DLL успешно собрана CMake/Ninja. Проверены PowerShell parsers обоих
runners. Тестовая сборка использует immutable snapshots; пользовательский
live runtime и его настройки не перезапускались.

## Проверка исправленного протокола

Две новые четвёрки20600–20603 и20700–20703, каждая выполнена дважды:
16 valid captures,4 concurrent instances x2,300 controlled frames.
`scripts/audit_combat_repeat.py` независимо сверяет одинаковые fingerprints,
policy SHA, параметры, confirmed seed/dispatch, native reset receipts,
полную последовательность observed states/applied actions и native first-life
outcomes. Из наблюдения исключается только wall-clock observation age.
Все8/8 пар совпали полностью. Последняя четвёрка дополнительно прошла
receipt check обновлённого baseline runner, manifest `fixed_world_hold=true`.

| Seed | Kills в каждом повторе | Incoming в каждом повторе | Deaths |
|---|---:|---:|---:|
| 20600 | 2 | 30 | 0 |
| 20601 | 1 | 59 | 0 |
| 20602 | 1 | 69 | 0 |
| 20603 | 0 | 100 | 1 |
| 20700 | 2 | 59 | 0 |
| 20701 | 1 | 38 | 0 |
| 20702 | 2 | 59 | 0 |
| 20703 | 0 | 100 | 1 |

Это проверка повторяемости **замороженной** policy на восьми условиях,
не рост её качества:1/4 и2/4 полных побед, один death в каждой четвёрке.
Новый fixed world protocol меняет стартовую фазу карты относительно прежних
release0/старого fixed100; их результаты нельзя смешивать в парное улучшение.

Native start dumps20600 совпали по timers/DelayedUse/free pool; для одного
seed остаются различия player animation и freetime активного Parasite.
Наблюдаемые траектории при этом совпадают. Полное скрытое состояние/AI
не объявляется восстановленным. Все32 captures текущей работы сохранены:
8 initial,8 native diagnostic,8 fixed-world,8 final confirmation.

Диагностика fallback20400 подтверждает `system2_noncombat` при пропаже
видимых enemies. Доубийственный handoff остаётся отдельной нерешённой
границей learned control; наличие такой помощи должно исключать эпизод из
приёмки непрерывного learned combat. Handoff в том же frame, что последний
provider kill window, может быть завершением этого window и сам по себе
не означает выстрел rules.

Следующее обучение: два сравниваемых CUDA-only PPO arm с fixed100 на новых
train seeds, одинаковым objective/architecture и отдельным замороженным
eval. Сначала повторять before policy и проверять этот auditor; применять
такой же world hold в Solo regression. Отдельно решить управление при
кратковременной потере видимости без скрытых monster positions; reward
ground truth остаётся только offline.

Evidence: roots `combat-mixed-fixed-repeat-v4-20261006`,
`combat-mixed-fixed-native-diagnostic-v4-20261006`,
`combat-mixed-fixed-world-v4-20261006`,
`combat-mixed-fixed-confirm-v4-20261006`; `repeat-audit.json`,
`native-start-audit.json`, `fallback-audit.json`, manifests/datasets/server
logs. Команда повторного аудита:

```powershell
& F:/src/strat/.venv-gpu/Scripts/python.exe scripts/audit_combat_repeat.py workspace/artifacts/combat-mixed-fixed-confirm-v4-20261006 --seed 20700 --require-equal
```
