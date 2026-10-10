# Super Shotgun в обучаемом бою

Добавлен loadout `weapons-ssg`: старт Super Shotgun, Blaster/Machinegun/Shotgun
также доступны, Bullets40/Shells20. Оружейная голова уже содержит этот вариант;
архитектура, ширина входов и порядок действий сохранены для существующих моделей.
Новые четыре registry recipes: первые точки base3, ware2, jail1, city1,
с отдельными train/validation/test/confirmation seed ranges от60000000.

Контроллер сохраняет provider ownership при Super Shotgun. Guards проверяют
обе native группы дроби yaw±5°, spread1000/500 и удвоение spread под водой.
Barrel cone30° и полный двухотрезковый underwater range; проверки не добавляют
движение, прицеливание или attack. Старая геометрия MG/Shotgun сохранена.
`TestHitscanFriendNativePathCoverage`: 6480 native path samples protected,
включая SSG. Fresh inventory readiness требует SSG и полный объявленный stock.
Снятие native fixture hold фиксирует stock idle gunframe18, без изменения ammo.
Native RNG/weapon receipt verifier и q2ppo-data требуют эту же фазу.

## Исправление завершения первого выстрела

Первый smoke v1 завершил 16 valid captures, но быстрые убийства не остановили
эпизоды. Native log показал первый provider cmd seq126 в server frame98 после
release game frame100; shotgun kill имеет тот же server frame98. Это реальный
первый command, а не стрельба харнеса. Reward уже учитывала урон и kill.
Ошибка была в goal supervisor/verifier: фильтр `event.Frame > release.Frame`
выбрасывал первый released ClientThink. Исправлен на `>=`; всё ещё требуется
следующая наблюдаемая frame после убийства, native kill, alive first life,
совпадение actor/world/classes. События до release остаются запрещены.
Новые regression case release_frame и старые invalid cases прошли.

Повторный smoke v2: 16/16 usable captures, все16 goal stops, 16 native kills,
0 deaths. Из них13 kills именно MOD_SSHOTGUN3, SSG health damage648.
Суммарные actual game frames337 вместо2379 в v1; это исправление supervision,
не улучшение модели. Wall time31.04s. Реальная защита напарника новым оружием
пока проверена геометрическими тестами; живой paired SSG ещё не выполнен.

`combat-ssg-validation-processing-v1-20261010` отклонён при native verifier:
старый weapon-start parser не принимал Super Shotgun. Первая train processing
v1 затем отклонена отдельным first-policy-idle check (оставался Blaster9).
Оба исправлены; ошибки сохранены, не обойдены. Captures повторно используются
для fresh native/CUDA export v2 без изменения immutable исходных файлов.

Отдельный train pool: `combat-ssg-train-pool-v1-20261010`, 32/32 usable captures,
16 slots ×2, 65.24s, train seeds; validation captures не идут в update.
CUDA processing/update: `combat-ssg-training-processing-v2-20261010`.
CUDA validation-only: `combat-ssg-validation-processing-v2-20261010`.
Оба v2 processing завершены: validation16/16, 274 rows, training=false;
train32/32, 985 rows, 10 actor steps, updates_completed7, KL0.004999744.
Resume от расширенного campaign checkpoint update6; actor/critic/Adam/RNG
сохранены, objective v8 не изменена. Новые weights SHA
`cd84d88a692aa292c712c0a07957073cb4673b12523371528072175e06fe1853`.

## Сравнение после обучения

`combat-ssg-paired-eval-pool-v1-20261010`: 32/32 captures, один общий пул16,
две frozen plans (before/after), одинаковые validation seeds и условия.
Свободный slot берёт очередной бой любого варианта. Comparer проверил равенство
client/exporter/reward SHA и полного generated fixture. Он поддерживает
`--before-plan-index`/`--after-plan-index` для такого смешанного пула.

`combat-ssg-quality-v1-20261010.json`:

| Показатель | До | После |
|---|---:|---:|
| Выполненные цели / kills | 16 / 16 | 16 / 16 |
| Смерти | 0 | 0 |
| Нанесённый health damage | 760 | 760 |
| Полученный health damage | 15 | 13 |
| Actual game frames | 326 | 304 |

Время−6.75%. Base3 61→64, city1 34→28, jail1 185→168, ware2 46→44.
В jail1 infantry HP100, на остальных проверенных точках soldiers HP30.
Все37 наблюдаемых последовательных SSG ammo depletion transitions тратили
ровно2 shells; `ssg-ammo-audit.json`. Это не shot accuracy.
Результат малый, win rate насыщен; веса экспериментальные, общая superiority
и promotion не установлены. Нужны группы и live teammate SSG проверка.

Chaingun пока не разрешена direct guard. Native spin-down может продолжать
стрельбу после отпускания attack: простое снятие кнопки не прекращает эффект.
Нужны учёт phase/остаточных выстрелов и соответствующий teammate guard.
Railgun, HyperBlaster и splash weapons остаются следующими этапами.
