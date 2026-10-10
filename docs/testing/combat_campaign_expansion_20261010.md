# Расширение карт и оружия

В основной реестр добавлены 60 рецептов: base3, bunk1, ware1, ware2, jail1,
jail2, mine1, fact1, hangar1, city1; по две оригинальные точки встречи на карту.
Точки разнесены минимум на 512 единиц; BSP hull/floor/visibility и 16
сэмплов каждого split проверены генератором. Native doors/movers сохраняются.
Это изолированные бои в геометрии кампании, не полное прохождение уровней.

На точку: blaster, machinegun, weapons (MG/Shotgun/Blaster inventory,
только learned controller). Сиды новых рецептов начинаются с 50000000;
train/validation/test/confirmation и варианты оружия не пересекаются.
`go test ./internal/trainingepisodes` прошёл; NN на CPU не проверялась.

Генератор поддерживает `q2episode --campaign-sites N --campaign-maps ...`.
Порядок выбранных карт не меняет seed ranges генератора. Старый API сохраняет
base1/base2 и прежние диапазоны. Реестр принимает дополнительные карты только
с campaign geometry descriptor, обычные неописанные fixtures не расширены.
Убраны ограничения base1/base2 в чтении generated fixture и death-stop receipt;
death-stop по-прежнему требует совпадение map/life/native terminal/damage.

Живая проверка: `campaign-expanded-smoke-pool-v1-20261010`, 16 slots ×2,
4 независимых боя на base3/ware1/mine1/city1, machinegun rules controller.
Первый запуск v1 был отклонён клиентским ограничением combat barrier на карты.
Исправлены synchronous teleport map guard и goal/death supervisor map guards.
Повторный `campaign-expanded-smoke-pool-v2-20261010` завершён:
16/16 usable captures, state complete, source_unchanged true, 27.99 s.
Это подтверждает native capture на четырёх проверенных картах; пригодность
всех 60 рецептов для PPO пока не установлена. GPU training ещё не запускалось.

`campaign-expanded-learned-pool-v1-20261010` также завершён: 16/16 usable
captures, source_unchanged true, 42.54 s. Те же четыре карты, inventory
MG/Shotgun/Blaster, learned v9 из исходной модели (не проигравший quality update).
Один base3 capture прошёл native-only export и CUDA finalization:
`campaign-expanded-learned-cuda-v1-20261010`, 11 eligible/context rows,
device cuda, state passed. Проверены sampled actions, likelihood, critic и
recurrent context; это проверка совместимости данных, не GPU weight update.
Затем все 16 validation cases CUDA finalized:
`campaign-expanded-validation-processing-v1-20261010`, 208 rows, training=false.
Эти записи не использованы для обновления весов.

## GPU обучение и оценка во вторых точках

Новые рецепты revision2 используют `combat-reward-selected-aim-v8.json`,
совместимую с исходным checkpoint objective. Generator seed_revision1 сохраняет
условия ранее подготовленных сцен. Для обучения собраны отдельные train seeds:
`campaign-expanded-train-pool-v1-20261010`, 80/80 usable captures, 16 slots ×2.
Использованы первые точки всех десяти новых карт, MG/Shotgun/Blaster inventory.

`process_registered_combat_pool_cuda.py` проверяет закрытый source-stable pool,
frozen plan/model, native replay, точное сохранение remembered_threats из trace,
CUDA likelihood/value/context и bootstrap, затем объединяет отдельные sequences.
При `--config` допускает только train splits. Validation/test/confirmation
не могут обучать веса. Fresh output обязателен; resume optimizer передаётся явно.

`campaign-expanded-training-processing-v1-20261010`: 80/80 CUDA finalized,
1616 rows, 10 actor steps, updates_completed6, final KL0.004999839.
Actor/critic/Adam/RNG продолжены от исходного v9 checkpoint; objective сохранён.
Weights SHA `43229caa6cb291c0a1bdd700638ce5e60e1002644546155b991014104aee27cd`.
Файлы weights/checkpoint/report/complete записаны в `update/`.

Held-out: по четыре validation seeds во вторых точках base3/ware1/mine1/city1,
которые не использовались в этом update. Before и after: 16/16 usable captures
каждый. `compare_registered_combat_pools.py` подтвердил равенство fixtures,
составов, loadouts, map/seed/geometry и client/exporter/reward SHA.
Native first-life comparison `campaign-expanded-quality-v1-20261010.json`:

| Показатель | До | После |
|---|---:|---:|
| Убийства / выполненные цели | 16 / 16 | 16 / 16 |
| Смерти | 0 | 0 |
| Нанесённый урон | 480 | 480 |
| Полученный урон | 52 | 48 |
| Суммарные game frames до завершения | 402 | 334 |

Время уменьшилось на 16.9%. На base3 frames70→73; city1 210→163,
mine1 45→42, ware1 77→56. City1 received46→48, ware1 6→0.
Это малое сравнение одиночных soldiers с уже насыщенным win rate;
общая superiority и promotion не установлены. Нужны более сложные группы,
другие loadouts и остальные карты. Рецепты расширения включают soldiers и
infantry, но оценочные вторые точки здесь все soldier.

Далее:

1. Принять native reset/terminal для новых карт; проверить learned inventory
   choice и CUDA export на новых геометриях, затем собрать свежий train pool.
2. Добавить Chaingun/Super Shotgun/Railgun с проверкой spread, recoil,
   penetration и teammate guards. Сейчас direct combat guard разрешает
   Blaster/Machinegun/Shotgun, несмотря на более широкий weapon head.
3. Добавить HyperBlaster/Rocket Launcher/Grenade Launcher с projectile travel,
   splash/self damage и guard геометрии; нельзя просто выдать оружие при
   запрещённой стрельбе и считать это обучением.
4. Расширить группы монстров и paired coop registry; проверить подавление
   NPC infighting без подавления атак на игроков.
5. GPU-only updates и held-out сравнение на других точках/сидах/картах;
   исходные веса сохраняются до доказанного улучшения.

Power1 не добавлена: текущий генератор не нашёл двух допустимых точек с
поддерживаемыми ground monsters; неподдерживаемые сцены не подменялись.
