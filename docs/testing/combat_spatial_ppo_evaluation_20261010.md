# Spatial PPO: завершённые480 native боёв

Все480 captures прошли строгую проверку source/native fingerprints, registry/plan/model bindings и actual client/exporter binary hashes; два binary variants. Progress complete. Root `workspace/artifacts/spatial-ppo-eval-v3-20261010`, quality SHA256 `28810bb95672be9c7b6939dc47ce0e228049553873a1996a9823fa3ee7cba6c6`, protocol SHA256 `f82771788f89aba414acf954a18015b67934f8ed70708c01963b0984a57ff18c`. По80 боёв на20 семействах, повторно использованный validation24; final test отложен.

| Вариант | Победы /80 | Смерти | Средний полученный урон | Blaster live hits / shots |
| --- | ---: | ---: | ---: | ---: |
| Instant BC | 47 | 26 | 37.30 | 264/360 (73.33%) |
| Instant PPO | 51 | 22 | 34.34 | 273/439 (62.19%) |
| Postmove BC | 53 | 20 | 32.76 | 272/434 (62.67%) |
| Postmove PPO | 51 | 23 | 35.24 | 257/362 (70.99%) |
| FireBC | 49 | 30 | 46.95 | 231/536 (43.10%) |
| Rules | 70 | 8 | 14.90 | 184/195 (94.36%) |

У native blaster join нет unknown endings. Это actual MOD1 shots в eligible first-life command windows, включая возможную смену оружия; не machinegun accuracy и не selected-target hit credit.

| Вариант | Blaster loadout /40 | Machinegun loadout /40 |
| --- | ---: | ---: |
| Instant BC | 28 | 19 |
| Instant PPO | 33 | 18 |
| Postmove BC | 31 | 22 |
| Postmove PPO | 32 | 19 |
| FireBC | 25 | 24 |
| Rules | 35 | 35 |

Парный Instant update:10 gained wins,6 lost, net+4; Postmove:4 gained,6 lost, net−2. Instant выигрыш приходится на Blaster (+5), MG−1. Postmove Blaster+1, MG−3. У Instant доля actual blaster hits снизилась, хотя выигрышей стало больше; у Postmove наоборот. Нельзя оптимизировать или принимать модель только по одной из этих метрик.

В разборе applied ray на явно выбранную цель Instant firing angular error вырос near19.53→23.34°, medium7.13→9.41°. Postmove near19.57→20.66°, medium10.28→8.30°. Это геометрическая диагностика без lead/recoil/muzzle correction, не hit rate; модели создают разные распределения кадров/дистанций. Far firing frames: только12 у Postmove BC, у остальных spatial variants0. Старый набор не проверяет дальнюю стрельбу.

## Решение и ограничения воспроизводимости

Общее превосходство над rules не получено. Postmove PPO не принимается как улучшение своего BC parent. Postmove BC остаётся сильнейшим learned вариантом по общим победам этой оценки53/80; Instant PPO — отдельный Blaster candidate33/40, не универсальная замена. Live/default и canonical registry не переключены. Следующий сбор должен покрыть дальнюю геометрию и Machinegun; простое увеличение сети этими данными не обосновано.

Повторный Rules результат отличается от прежнего400-battle прогона67→70/80, хотя source/native fingerprints и все recipe/seed/generated instance условия совпадают. Все три изменения исхода находятся в `campaign-base2-site-04-machinegun`, seeds1735025..1735027. Прежде это были живые timeout без kill; теперь kills/goal подтверждены. Для1735025 уже на observed frame101 различаются Soldier animation55 vs177, previous command buttons1 vs0 и kick angles, хотя стартовые position/HP/weapon/view angles и относительная позиция primary совпадают. Sent/applied attack на этом кадре также различается. Это подтверждённая разница начальной истории исполнения, не доказанная причина и не установленная ошибка генератора или CUDA модели. Одинаковый seed/fixture не доказывает побитовое повторение полного боя. Поэтому net+4 Instant пока не является доказательством устойчивого выигрыша: нужны повторные cohorts и разбор release/command timing. Source hashes не менялись между оценками.

## Следующий native прогон

Дальняя оценка96 боёв `workspace/artifacts/combat-far-eval-v1-20261010` автоматически начала прежний16-slot pool ×2 после complete/hash/proof gate предыдущего процесса. Четыре draft рецепта, те же шесть frozen вариантов, отдельные validation seeds. На момент этой записи прогон активен; итогов качества ещё нет. Это первая native пригодность новой геометрии, не обучение и не final-test acceptance.
