# Награда за подтверждённое убийство монстра

05.10.2026. По запросу пользователя добавлен `combat_reward_v2`. V1 и его прежние datasets сохраняют прежний смысл. Новая функция задаёт цель обучения, а не доказывает, что боец её освоил.

| Компонента | Коэффициент |
| --- | ---: |
| Урон здоровью живого монстра | +0.01 / HP |
| Подтверждённое убийство монстра этим ботом | +5 / monster |
| Полученный урон здоровью | −0.02 / HP |
| Дополнительный урон себе | −0.02 / HP |
| Урон напарнику | −0.10 / HP |
| Смерть бота | −5 |
| Игровой tick | −0.001 |

Для текущего Parasite fixture со 175 HP полный урон даёт +1.75, финальное убийство добавляет +5. До time/damage costs итог +6.75. Гибель бота со 100 HP стоит −7 с учётом received damage и death; обмен убийства на свою смерть остаётся нежелательным. Это фиксированные экспериментальные коэффициенты, не подобранные по новым evaluation seeds.

## Доказательство kill и граница управления

Bonus берётся из отдельного native outcome: attacker=this actor, target class начинается с `monster_`, health_before>0 и health_after≤0. Перепроверяются world identity, согласованность health/take и число уникальных lethal targets с MonsterKills. Удар по трупу, исчезновение entity, чужое убийство или убийство игрока не создают бонус. Отложенный projectile может получить бонус в окне своего фактического эффекта; текущий action не обязан нажимать fire. Эти native данные остаются отдельно от observation policy.

В v1 все truncations маскировались, в том числе control_handoff. В v2 последний **полный и точно исполненный** шаг до control_handoff сохраняет reward. Иначе добивание могло выпадать, когда следующая наблюдаемая цель исчезала и управление переходило к System 2. Проверки свежих соседних кадров одной жизни, native tick и exclusive dispatch остаются. Хвост без next, frame gap, harness override, stale/world/life mismatch и неподтверждённый dispatch всё ещё маскируются.

`q2ppo-data` сохраняет такой segment-end row с truncated=true и next_value=0: это конец learned сегмента, не утверждение победы на всей карте или смерть игрока. GAE разрывает продолжение на этой границе. Последующие эффекты после handoff не приписываются несуществующему provider action.

## Смена цели обучения

Reward config SHA теперь фиксируется в PPO rollout report. Training config `combat-ppo-kill-v2.json` содержит `objective_reward_sha256`; trainer отказывается принимать v2 без pin или с другой функцией награды. Для смены objective требуется отдельная явная ветка checkpoint; просто продолжать старый optimizer с новым config нельзя.

`fork_combat_objective.py` сохраняет actor/log_std, обнуляет выход critic и сбрасывает actor/value Adam. Critic hidden layers, RNG, consumed history и counters сохраняются. Родитель не перезаписывается; проверяется точное соответствие checkpoint весам. Новый checkpoint оценивает только свежие on-policy данные новой награды. Counters включают родительскую историю, fork сам не является update.

Go tests покрывают kill, delayed projectile, corpse, чужого attacker, player, nonlethal, duplicate, отрицательный count, неверный world/health, handoff и отказ иных truncations; проверяют v1 совместимость. Python tests проверяют objective pin и reset critic/optimizers с сохранением actor/parent. Общий `go test ./...` и восемь Python GAE/resume/objective tests прошли, PowerShell syntax проверен.

## Проверка в игре и новый цикл

Proof: `workspace/artifacts/combat-kill-reward-proof-v2-20261005`, четыре инстанса x2, seeds 14200–14203, rules с фиксированным Blaster. Во всех четырёх capture/provenance-valid прогонах ровно одно окно с bonus +5. Это проверка расчёта native награды, не learned-policy достижение.

Objective fork: `workspace/artifacts/combat-ppo-kill-objective-fork-v2-20261005`; родитель — exploration branch, SHA `e7c0462922cbbc3be6c044e36a85319b1580fff764200de80e3050d39db3c0ab`. Начальные новые веса SHA `7595745f914ee08065f4800fa552fd51e8c48362321bcb435cf244462996a1c7`.

Cycle: `workspace/artifacts/combat-ppo-kill-cycle-v2-20261005`, 4 fresh iterations × 4 server/client instances, x2, 300 кадров. Training seeds 14300–14315, separate paired deterministic evaluation 14400–14403. Reward config frozen в artifact и сравнивается по SHA перед каждой стадией; fourth update выбран заранее. Оценочные seed не входят в training и не служат настройкой reward.

Бонус sparse: если learned policy не убивает монстра ни в одном training window, градиент не получает примеров +5. Наличие бонуса необходимо отличать от его фактического использования обучением. При отсутствии таких примеров следующий этап — curriculum с доступным добиванием/удержанием прицела и отдельными свежими контрольными seed; teacher kill не подмешивается в PPO как якобы on-policy опыт.

## Результат первого цикла v2

| Batch | Переходы | Actor steps | Final KL | Value MSE своего batch |
| --- | ---: | ---: | ---: | ---: |
| 1 | 747 | 10 | 0.007222 | 0.405685 |
| 2 | 663 | 10 | 0.009071 | 0.201556 |
| 3 | 693 | 10 | 0.009837 | 0.111764 |
| 4 | 736 | 10 | 0.005932 | 0.113784 |

2839 fresh transitions; 40 actor / 160 value шагов. Финальный checkpoint содержит 17 updates / 170 actor steps с учётом прежней истории. Веса `iteration-4/update/weights.json`, SHA `3516e5f7050996310bc645ab0216f63989a0d5e9f3d60018b31e1c2cb751b3e1`. Вся новая training ветка pinned к SHA reward v2. Max Go/PyTorch logprob error 0.0000421, value error 0.000000716; все четыре native replay прошли. CPU выбран во всех обновлениях по benchmark.

**В 16 training первых жизнях native kills=0; ни одной provider-owned строки с bonus +5.** Утилита начисляет награду правильно, но этот короткий опыт ещё не содержит её положительного применения политикой.

| Eval seed | Урон врагу до / после | Полученный урон до / после | Первая смерть до / после |
| --- | ---: | ---: | --- |
| 14400 | 30 / 30 | 0 / 45 | нет / нет |
| 14401 | 30 / 30 | 5 / 100 | нет / да |
| 14402 | 30 / 30 | 0 / 48 | нет / нет |
| 14403 | 30 / 30 | 0 / 50 | нет / нет |

Kills 0→0, monster damage 120→120, received 5→243, deaths 0→1. Fixture acceptance false. Движение изменилось: stationary requested motion 960/1092→461/1029 (87.9%→44.8%), path length 1068→4743 units; horizontal attack error >15° вырос 18/1096→885/975. Политика стала больше перемещаться, но хуже удерживает цель и получает больше урона. Улучшение боя не подтверждено; четыре пары недостаточны для статистических выводов. Это не сравнение только reward v1/v2: одновременно были critic/optimizer reset и новый опыт.

В процессе проверки обнаружена ошибка только счётчика в новом exporter report: zero-bootstrap handoff считался в `terminals` вместе со смертью. После полного окончания сбора счётчик исправлен. Replay первого batch в `count-fix-replay-iteration-1` даёт 3 terminal rows вместо старого report count=4, при **побайтово одинаковом rollout SHA**. Флаги terminal/truncated и обучение не изменились, прежние artifact reports не переписаны. Новые exports считают только actual terminal rows.

Всего 28 capture/provenance-valid эпизодов, включая четыре rules proof. Guards/native/source условия before/after совпали. Собственные порты освобождены, пользовательский live runtime прежний. Общий Go test, восемь Python tests, native replay и syntax/whitespace checks прошли. Full-world reset equivalence остаётся неподтверждённой. Следующий приоритет — отдельный curriculum для первых on-policy убийств и удержания прицела; дальнейшее увеличение kill bonus без таких примеров не подтверждено как решение.
