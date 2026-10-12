# Checked wall-slide guard: первые live проверки

Новый opt-in combat.checked_wall_slide сохраняет полный command модели при static_hull_blocked, если collider step/slide paths и все support alternatives проходят дополнительные проверки. Guard не добавляет движение, цель или огонь. Начальная версия разрешает checked static contact; прыжки, неизвестные overlap cases и неописанные mover behaviors остаются незавершённой применимостью. Это эксперимент, не общий новый default.

Проверяются непрерывный hull sweep каждого сегмента, one-unit samples static flat support, dry surface, lethal contents, lasers, неизвестные двери; близкие packet bodies с1/8 position uncertainty не считаются надёжно предсказанными. Это ограниченная sampled проверка, не формальная гарантия всех невидимых/подвижных объектов или narrow-gap geometry.

Первая версия blanket-rejected любой observed mover:24 native/16 CUDA,176.30с; control иcandidate3/8, правила5/8. На всех902 provider frames были movers; флаг дошёл до runtime, но не изменил движения. Этот smoke выявил ошибку механики до большого прогона.

Исправление: учитывается полный stock travel envelope door/button/plat (lip/height parsed), постоянный корпус wall/explosive, rotational sphere, reachable path_corner train bounds. Даже открытая дверь может закрыться через путь; текущая pose не заменяет travel range. Неподдерживаемые классы остаются conservative. Packet nonsolid render brush models не считаются collision movers. Unknown/missing geometry rejected.

Вторая версия:24/24 native,16/16 CUDA,175.74с,1979 проверенных переходов. Одни и те же weights/checkpoint, fixtures иseeds; разница только checked flag. Все24 manifest/runtime flags сверены с frozen plans. Обучения/назначения основных весов нет.

| Метрика | Legacy | Checked |
|---|---:|---:|
| Победы |2/8|2/8|
| Ordinary/recovery победы |1/4 +1/4|1/4 +1/4|
| Полученный урон |654|656|
| Fully suppressed / proposed movement |347/941 (36.88%)|269/1033 (26.04%)|
| static_hull_blocked interventions |137|54|
| static_hull_component_clipped |131|16|
| Стояние при clear enemy |29.56%|28.41%|
| Повороты>=30 |10.70%|13.33%|

Правила6/8 (2/4 ordinary +4/4 recovery),урон541. Из-за разных траекторий число кадров различается; raw counts без denominator не являются сравнением качества. После ослабления лишнего clipping у модели появились другие состояния. Блокировок меньше, но выигрышей больше не стало и крупных поворотов больше. Нужны screen24 условий pervariant и обучение на новом исполнении, а не вывод о превосходстве по этому smoke.

Targeted Go tests: сохранение всего записанного wall command, legacy fallback, unsafe/missing alternative rejection, nearby body/mover uncertainty, floor edge/slick/laser, closing-door travel и finite train loop. Это физика/metadata без NN. Numerical replay только CUDA. Существующие сценарии и old plans сохраняют legacy default; checked flag включается через frozen plan/registry compiler и проходит все harness layers.

Artifacts: combat-wall-slide-smoke-evaluation-v1/v2-20261012, соответствующие sealed reports; evaluator scopes без training/promotion. Все входные raw данные активных проверок сохранены. Следующий72-battle screen — validation492–493, ещё не результат.

## Screen: 72 боя завершены

Свежие validation492–493: 12 семейств, 24 условия на вариант (16 ordinary, 8 recovery), одинаковые веса и условия. Пул 16 ×2 завершил native часть за 190.29 с. Seal подтверждает 72/72 native и 48/48 CUDA replay cases, 5894 перехода. Все проверки завершены; обучения и promotion нет.

| Метрика | Legacy | Checked | Правила |
|---|---:|---:|---:|
| Победы |8/24|5/24|10/24|
| Ordinary |4/16|2/16|2/16|
| Recovery |4/8|3/8|8/8|
| Полученный урон |1788|1971|1386|
| Полностью подавленные / предложенные движения |827/2735 (30.24%)|524/3156 (16.60%)|—|
| Крупные повороты |8.32%|9.83%|—|
| Стояние при clear enemy |24.30%|22.07%|—|
| Attack command без цели |9.70%|11.47%|—|

Парно против legacy: ноль новых и три потерянные победы. Это screening, не независимое подтверждение превосходства или универсальной регрессии. Guard меняет исполнение, но текущая модель не использует эту свободу эффективно. Default остаётся legacy; расширенный confirmation без нового кандидата не запускаем. Следующий эксперимент должен сравнить одинаковый CUDA-бюджет дообучения на legacy и checked исполнении, затем пройти новые smoke/screen условия. Полноценные прыжки и human recorder остаются отдельной незавершённой частью плана.

Проверенные отчёты: `workspace/artifacts/combat-wall-slide-screen-report-v2-20261012/complete.json`, `results.md`, `combat-wall-slide-screen-execution-v2-20261012.json`; исходный seal: `combat-wall-slide-screen-evaluation-v2-20261012/complete.json`. Raw captures сохранены. Доли подавления используют предложенные ненулевые команды; остальные поведенческие показатели описывают разные траектории и не доказывают причинность. Attack command не равен физическому выстрелу.
