# Признаки углового отклонения цели v2

06.10.2026. Предыдущий [tracking HP60 v2](learned_combat_tracking_hp60_v2.md) обнаружил устойчивый уход взгляда к ±89° и отрицательный результат обычного продолжения PPO. Новый ограниченный опыт проверяет доступность явного углового представления, без автоматического наведения и изменения награды.

`combat_features_v1` и его 386 значений сохраняются побитово. `combat_features_v2` добавляет 40 значений: восемь enemy slots в прежнем stable distance order, каждый содержит valid mask, sin/cos yaw error и sin/cos pitch error. Геометрическая точка — **наблюдаемый origin монстра**, отсчитанный от глаза (+22 стоя, −2 ducked); это не upper-body AimPoint. Clear-shot false/unknown и вырожденные направления маскируются. Используется только UDP observation, без seed, ID, reward/server health, future state. Признаки не выбирают действие, weapon или target и не вмешиваются в команды сети. Пока observation не содержит enemy bbox, точность попадания из этих углов не выводится.

Go loaders маршрутизируют обе версии по model header и проверяют соответствующую input width. PPO exporter использует версию самой behavior policy, pins feature_version в rollout report; Python trainer проверяет совместимость версии. Старые BC/rollout exports остаются v1. Архитектура нового actor/value 426→64→64→8/1, без GRU/attention.

Fork: `workspace/artifacts/combat-features-aim-v2-fork-20261006`. Родитель — фиксированный последний tracking-v2 checkpoint, weights SHA256 `97651633277075a74f3ac041d2ce244a03321397fb800a0fb022a8eff5eb5c6d`. Новые колонки первого слоя actor/value нулевые; функция до обучения сохраняется. На 954 frozen probe rows с произвольными ненулевыми добавленными признаками max error actor/value=0. Fork weights SHA256 `0f947be1abce19ee8882e53a4d4b0b540d7d762ed5f47f73b2d83274f729a6eb`. Обе Adam state сброшены из-за нового размера параметров; objective/std/RNG/consumed history/counters сохранены. Это дополнительное отличие от родителя: без matched v1 optimizer-reset control причинное преимущество именно новых features не устанавливается.

Протокол до запуска: четыре fresh batch × четыре инстанса, x2, разные training seeds 15500–15515; 300 post-barrier game frames, release frame 100, Blaster idle 9, post-frame seed reset; fixture HP60, прежние reward v2/PPO config. Fixed final fourth update; eval before/after на 175 HP, held-out seeds 15600–15603. Eval не обучает и не выбирает checkpoint. Full world reset не доказан. Новая политика не устанавливается в пользовательский live.

## Результаты

Root: `workspace/artifacts/combat-ppo-aim-features-v2-20261006`. Все 24 capture прошли native provenance/dispatch/seed проверки, сохранён `capture-audit.json` с SHA reports/steps. Четыре PPO batch: 1099/935/942/923 = **3899** переходов; actor steps 10/8/10/10, KL 0.007381/0.009992/0.008385/0.006021. CPU быстрее CUDA на каждом benchmark; накоплено 33 updates/326 actor steps. Новые колонки первого слоя получили ненулевые actor/value веса: признаки действительно входят в обучение.

Одно training убийство HP60: seed **15506**, step **113**, observation frame **210**, owner=provider. Reward component monster_kill=+5, score=4.999; строка включена в PPO iteration-2 rollout, SHA rewards/rollout сохранены в `kill-audit.json`. Это единичный положительный пример, а не освоение сопровождения цели.

| Eval seed | Исходящий урон до → после | Полученный урон до → после | Kills до → после | Смерть после |
| --- | ---: | ---: | ---: | --- |
| 15600 | 20 → 20 | 75 → 100 | 0 → 0 | да |
| 15601 | 20 → 20 | 73 → 100 | 0 → 0 | да |
| 15602 | 20 → 20 | 79 → 100 | 0 → 0 | да |
| 15603 | 20 → 20 | 75 → 100 | 0 → 0 | да |

Суммарно исходящий урон 80→80, полученный 302→400, deaths 0→4. До обучения два capture содержат handoff/возврат rules; native first-life агрегаты before включают эти интервалы. After во всех четырёх capture handoff нет — политика напрямую дошла до наблюдаемой смерти. Отличие длины жизни и управления ограничивает сравнение счётчиков.

Provider observations 1071→571; visible attack origin samples 1024→519. Pitch error >15°: 973/1024 (**95.0%**)→491/519 (**94.6%**); 3D angle >15°: 1000/1024 (**97.7%**)→499/519 (**96.1%**). Pitch возле ±89°: 597/1024→321/519. Удержание прицела не освоено; незначительное изменение угловой доли на коротких жизнях нельзя объявлять улучшением боя.

Fixed final weights `iteration-4/update/weights.json`, SHA256 `4eaa3b10ca67473b79c084d0567d676cc258a5a72f24008cccd1d50a4ba0aa30`. Результат отрицательный, без live promotion или выбора более выгодного промежуточного checkpoint. Явных угловых features в этом бюджете оказалось недостаточно; следующий опыт должен проверить обучающий сигнал длительного удержания и точку прицеливания с UDP bbox, прежде чем объяснять проблему только отсутствием GRU/attention. Архитектура остаётся MLP.

`go test ./...` прошёл; 10 Python tests (feature fork, наследуемые exploration проверки и diagnostics) прошли. Initial function parity=0 проверена offline; Go runtime v2 и exporter/PyTorch likelihood parity подтверждены свежими игровыми capture. Harness порты 33100–33103 освобождены; пользовательский live marker сохранён. Исходники/native config пользовательского live не менялись.
