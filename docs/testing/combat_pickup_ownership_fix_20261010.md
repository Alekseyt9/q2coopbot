# Подобранное оружие остаётся под управлением сети

В combatCommand поддерживаемые Machinegun/Shotgun теперь разрешены
в synchronous parasite_blaster/parasite_machinegun и multiweapon fixtures.
Прежнее условие допускало Shotgun только для multiweapon или campaign
evaluation, хотя policy могла выбрать его из фактического инвентаря.
Именно это возвращало управление правилам с pilot_equip_not_ready.

Изменение применено после terminal560 evaluation и всех девяти seals;
перед применением проверены исходные SHA. Четыре targeted проверки
прошли без skips: supported ownership и Machinegun/Shotgun barrel guards.
Провайдер в этих проверках фиксированный probeProvider, нейросеть не
выполняется. Сама inference проверена настоящими native captures.

Свежий56-case validation завершён и прошёл provenance, sampling-config
audit, девять диагностик и ownership проверку. Все48 effective learned
configs используют объявленные разные offsets. У всех шести learned
вариантов нет equip fallback и rules-with-clear-target frames.

| Вариант | Победы /8 | Смерти |
| --- | ---: | ---: |
| m1 deterministic | 0 | 8 |
| m1 stochastic-a | 4 | 4 |
| m1 stochastic-b | 4 | 3 |
| parent3 deterministic | 0 | 5 |
| parent3 stochastic-a | 5 | 2 |
| parent3 stochastic-b | 4 | 3 |
| Правила | 8 | 0 |

m1 stochastic ранее имела5/8 в обоих arms; после прекращения передачи
Shotgun правилам стало4/8. Это изменение runtime при одинаковых весах,
не новый этап обучения. Нельзя приписывать каждую потерю победы отдельному
rules выстрелу без причинного эксперимента. Ограничение результата —
только site02; полная20-family проверка нового runtime остаётся следующей.

Root: `workspace/artifacts/pickup-ownership-fix-v1-20261010`.
Quality SHA256: `9263dbb3fbdc95c3b5523f998178585c4892bffaf1816dda5ca2213c70ff12dc`.
Ownership SHA256: `f3a7bf72da107e7ea3f5ce7c27b952f4adda4b0bc86fb40d6f4251c3bf4a5be0`.
Contract checks SHA256: `8193d7aa8df27d8530d72b91d88cadd27df94b17069eeeb77f6864a21d527e5c`.

## Продолжение CUDA PPO

Подготовлен и запущен общий160-case own-policy сбор m1 и parent3, по80
train164..167 условий на20 семьях. Это новые исполнения train условий;
не заявляется отсутствие этих seeds во всей исторической базе. Пул16 ×2,
автоматическое заполнение слотов и LZX завершённых streams с SHA проверкой.

Каждая модель использует exact bytes прежних weights/checkpoint/report.
Actor/value/std и Adam сохраняются из предыдущего подтверждённого CUDA
этапа, counter1→2. Config/anchor/bank hashes совпадают с parent reports;
learning rate, KL и reward objective не меняются. Никакого нового Adam
после BC в этом продолжении. После сбора ownership проверяется на всех160
train записях; только затем CUDA-only export, batched finalization и
обновление. Результат обучения потребует новых exact CUDA checkpoint audits.

Own-policy outcomes не являются validation quality. Веса не продвигаются,
независимый test остаётся зарезервированным. После GPU обновления нужно
парное сравнение исходных/новых весов и правил в исправленном runtime.

Capture: `workspace/artifacts/pickup-ppo-capture-v1-20261010`.
Processing: `workspace/artifacts/pickup-ppo-processing-v1-20261010`.
Driver: `scripts/run_combat_pickup_ppo_continuation.py`.

## GPU continuation и закрытие общего сравнения

Обе модели завершили CUDA update2. Оба checkpoint reload audit подтверждают
actor/value/std exact и optimizer state exact на RTX5070; config/objective
сохранены. Это проверка обучения, не победы над rules.

`pickup-ppo-eval-v1-20261010`: пул завершил1040/1040 задач без ошибок,
source_unchanged=true. Индивидуальная native/source/member проверка приняла
все1040 записи. SHA `recovery/verified-members.json`:
`dc030c2922be7aab03ca2b00e63e4ccd44ecab1f8c14489c31820037f30c3e2f`.
Фактические sampling configs и quality/diagnostics ещё обрабатываются;
итогового качества и promotion на этом этапе нет. Очередь wall curriculum
ждёт полного закрытия всех gates, а не только завершения native боёв.
