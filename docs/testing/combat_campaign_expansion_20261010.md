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
Остальные 15 записей ещё не CUDA finalized, held-out quality не измерялась.

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
