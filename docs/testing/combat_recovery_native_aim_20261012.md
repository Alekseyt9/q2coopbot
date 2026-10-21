# Восстановление оружия: native урон и доводка

Источник — завершённый matched screen `combat-wall-slide-trained-screen-evaluation-v2-20261012`:72 native/48 CUDA, свежие validation524–525. Анализ ниже не выполняет NN и не обучается на оценочных данных. Все input seal/spec/pool/capture/log/steps SHA проверены повторно.

## Что происходит после смены

Обе модели сменили оружие во всех8 recovery случаях; средняя задержка первого observed supported equipment7.5 network frames, первого applied attack10.375. В recovery обнаружены21/28 кадров с HUD ammo ниже stock minimum; во всех таких кадрах видна альтернативная поддерживаемая категория. Это наблюдение не доказывает пустые физические выстрелы: часть кадров может совпадать с автоматической сменой/анимациями и задержкой HUD. Статистика phases отделяет activation/firing/lowering animation, не подменяя их скрытым WeaponState.

| Recovery метрика | Legacy | Checked | Правила |
|---|---:|---:|---:|
| Победы |4/8|3/8|8/8|
| Native Blaster health damage |400|260|0|
| Native Shotgun health damage |593|557|188|
| Native Machinegun health damage |583|795|2012|
| MG fire events |128|160|258|
| MG ray contacts с живым monster |71|99|249|
| Blaster spawn events |68|63|0|

Урон разделён по native MOD и ограничен оставшимся HP, трупы исключены. Contacts луча и health damage имеют разные определения; 99/160 и249/258 не называем полной точностью. Для Shotgun/Super Shotgun число физических выстрелов здесь не считается: damage contacts могут быть отдельными дробинами. Это разные траектории, не доказательство, что принудительная выдача MG изменит исход.

## Явно выбранная цель

`report_registered_machinegun_aim.py` связал MG события с exclusive native command windows, declared target entity/track и наблюдаемым bbox. Ближайший враг не подставляется вместо выбранной цели. Missing/no-target случаи вынесены отдельно.

- Checked ordinary:427 measured MG rays, средняя ошибка после native recoil8.38°; recovery158 rays,13.79°. Из160 eligible recovery shots2 не объявили цель при наблюдаемом bbox.
- Checked recovery: до recoil17.06°, после13.79°; компенсация в среднем помогает, но не устраняет ошибку. Native view совпадает с приложенной командой в пределах округления; изменения физики/сетевое исполнение не обнаружены как источник этих больших отклонений.
- Recovery measured rays с native damage именно объявленной цели:69/158. Попадание в другую живую цель не считается ущербом автоматически. Observed target pose отличается от полной native истины и не доказывает точность/ошибку на каждом движущемся враге.
- В нескольких recovery случаях MG расходовал40 патронов; правила в соответствующей выборке завершали бои за27–37 MG fire events. Это основание проверить доводку и расход, а не только вероятность выбора MG.

Отчёты: `combat-wall-slide-trained-screen-native-weapon-damage-v2-20261012/{complete.json,results.md}`, `combat-wall-slide-trained-screen-machinegun-aim-v2-20261012.json`, `combat-wall-slide-trained-screen-weapon-effectiveness-v2-20261012.json`, два `phases-*` отчёта. Два metadata/native tests проверили исключение чужого actor/life/pre-release, cap overkill, запрет duplicate shot IDs и разные game/network clocks; нейросеть в тестах не исполнялась.

## Следующая проверка: шум доводки

В новом `combat-aim-noise-quarter-fork-v1-20261012` уменьшены только Gaussian std latent yaw/pitch в4 раза. Сохранены actor/value tensors, все mean logits, target/weapon/attack/vertical distributions и forward/side std для фиксированных историй. На3320 реальных строках CUDA verified outputs/state exact; actor/value совпали с checkpoint. Adam clocks/RNG/history сохранены, обнулены только моменты yaw/pitch log_std; прежний sequence-retention архивирован, новый fit требует свежего own-policy corpus. Held-out данные использовались лишь для проверки, не обучения.

Новая модель остаётся stochastic, средние команды и пространство действий не ограничены. Closed-loop histories могут измениться. Это проверка вклада exploration noise, не исправление всех причин промахов и не новая архитектура.

Запущен smoke24: `combat-aim-noise-smoke-evaluation-v1-20261012`, validation532,4 ordinary+4 recovery условий на вариант, пул16 ×2. Обе learned ветки используют checked wall execution; разница только aim noise. Третий вариант — rules. Основные веса не заменены, результат пока отсутствует.

## Результат smoke, проверен 21.10.2026

Сохранённый evaluator имеет stage complete:24 native/16 CUDA; `combat-aim-noise-smoke-report-v1-20261012/complete.json` имеет state complete. Контроль4/8, уменьшенный aim-noise6/8, rules4/8. Ordinary2/4 у обеих learned веток; recovery2/4→4/4. Полученный урон540→414, смены цели37.95→30.31/100 кадров, yaw≥30°14.23→12.10%. Attack-команды без выбранной цели6.47→23.10%; они не равны физическим выстрелам/промахам. Это положительный малый сигнал, но не независимое подтверждение превосходства и не основание назначить основные веса. Нового screen в рамках этого разбора не запускалось.

Общий разбор и приоритеты: [новые направления от21.10](../learned_combat_new_directions_20261021.md).
