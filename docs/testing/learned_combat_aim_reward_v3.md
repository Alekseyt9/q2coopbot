# UDP bbox и potential shaping прицела v3

06.10.2026. После отрицательного [aim-features v2](learned_combat_aim_features_v2.md) добавлены optional `observed_solid` для видимых врагов и `combat_features_v3`. Старые observation v3 без поля по-прежнему читаются; отсутствующие bounds явно неизвестны. Поле берётся из UDP entity solid, копируется в history и не содержит server health/reward. Feature v1/v2 сохраняются; v3 добавляет 40 значений (466 inputs): восемь masked sin/cos yaw/pitch offsets до bbox-clamped Go `Object.AimPoint`, без угадывания высоты при неизвестных bounds. Прямое управление сетью сохранено.

Reward `combat_reward_v3` сохраняет damage/death/tick и +5 за native verified kill. Новый component: `aim_potential = gamma*Phi(next)-Phi(current)`, gamma=0.99 совпадает с PPO и проверяется до capture/при training. `Phi = -0.5*(1-dot(view_forward, observed_aim_direction))/2`, диапазон [-0.5,0], для ближайшей видимой цели с известным bbox; если её нет, Phi=0. Направление использует текущие наблюдаемые eye/angles/bounds, без applied-action shortcuts и серверных данных. На terminal death и verified control handoff next Phi=0, PPO segment bootstrap=0; unproven gaps/tails не получают награды. Normal complete transitions retain standard next-value bootstrap.

Основа — [Ng, Harada, Russell, 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf). Дисконтированная сумма shaping телескопируется в `-Phi(start)+gamma^T*Phi(end)`: круг прицеливания не создаёт дополнительной дисконтированной награды. Отдельный static step при negative Phi может быть положительным из-за gamma; это компенсируется всей последовательностью, поэтому положительный component сам по себе не означает полезный бой. Никакого постоянного бонуса за стояние/attack нет. Здесь partial observation, guards, truncations, finite PPO и непроверенный полный reset: теорема для MDP не объявляется доказательством эффективности или неизменности оптимальной политики этого runtime.

Forks сохраняют latest fixed parent v2 weights SHA `4eaa3b10ca67473b79c084d0567d676cc258a5a72f24008cccd1d50a4ba0aa30`:

- Bbox feature fork: 426→466, новые input weights=0; actor/value function parity 0 на 923 frozen probe rows. SHA `a5392356f09541d98e216ba81b5591588da8042b6e03e1b30c9a3d62c0a2dca0`.
- Objective fork: actor/std сохраняются, critic output обнулён, Adam reset, прежние RNG/consumed history/counters сохраняются. SHA `b8d7a08b656aef56adc854fc4bfae6607dcc2132505541e28eec5dcafa79b7c5`. Reward SHA `b39507018fe9a8075b6cfeda85239a5d3c0497cf0761094abd98f4bf9265a90e`.

Протокол до запуска: четыре fresh PPO batch, четыре инстанса, x2, отдельные seeds 15700–15715; HP60, 300 post-barrier frames, release100, Blaster idle9, post-frame RNG reset. Eval fixed fourth update до/после на 175 HP, held-out seeds 15800–15803; no selection по eval. Root `workspace/artifacts/combat-ppo-aim-reward-v3-20261006`. Изменены сразу representation и objective/critic/Adam; это bounded combined experiment, причинный вклад каждого изменения без отдельных control arms не доказан. GRU/attention не добавлены; full reset не доказан; live promotion отсутствует.

До запуска прошли `go test ./...`, 19 Python checks и PowerShell parser. Go tests покрывают packed bbox/unknown masks/v1-v2 prefix, shaping loop telescope, death/handoff boundary, finite/range config; Python — two-stage zero-extension, objective fork, gamma mismatch, bbox diagnostic parity с Go geometry, старые PPO/diagnostics проверки.

## Завершённый цикл

24 capture прошли native provenance/dispatch/seed проверки. Получены 825/873/1117/889 = **3704** fresh PPO transitions; actor steps 10/10/10/10, KL 0.006051/0.009826/0.009121/0.009446. CPU быстрее CUDA во всех четырёх benchmark; checkpoint содержит 37 cumulative updates и 366 actor steps. Training kills: **0 из 16**.

`observed_solid=7266` реально получен для Parasite. Independent offline `audit_shaping.py`/`shaping-audit.json` проверяет packed bbox geometry и каждый доступный shaping component всех шести batch, а также точное равенство reward score каждой из 3704 consumed PPO rows; max component discrepancy **2.220446049250313e-16**. Reports/steps/rewards/rollout SHA сохранены. Для восстановления score используется последовательное сложение в порядке Go, а не Python 3.12 compensated `sum`. Корректность reward pipeline подтверждена; gameplay success из неё не следует.

| Eval seed | Урон монстру до → после | Полученный урон до → после | Kills до → после | Deaths до → после |
| --- | ---: | ---: | ---: | ---: |
| 15800 | 20 → 20 | 100 → 100 | 0 → 0 | 1 → 1 |
| 15801 | 20 → 20 | 100 → 100 | 0 → 0 | 1 → 1 |
| 15802 | 20 → 20 | 100 → 100 | 0 → 0 | 1 → 1 |
| 15803 | 20 → 20 | 100 → 100 | 0 → 0 | 1 → 1 |

Все восемь eval являются provider first-life до смерти, **без control handoff**. Суммарно outgoing 80→80, incoming 400→400, kills 0→0, deaths 4→4. Не следует сравнивать before этого опыта с before более раннего опыта на других seeds как одинаковые траектории; full reset не доказан.

Provider frames 621→641. Bbox aim samples visible attack: 524→548; pitch error >15° — 512/524 (**97.7%**)→520/548 (**94.9%**); 3D angle >15° — 512/524 (**97.7%**)→528/548 (**96.4%**). Pitch limit interventions 331→253; requested stationary frames 349/621→373/641. Слабое изменение угловых долей не улучшило убийства/урон/смерти и не доказывает надёжность удержания прицела.

Fixed final weights `iteration-4/update/weights.json`, SHA256 `d36ad955b8351698db4e3d2384db384c97d8c83f3ca8d72f35688807aed9d19b`. Отрицательный ограниченный опыт сохранён, промежуточные checkpoints не выбирались по eval. Следующий шаг — изолированная проверка обучаемости yaw/pitch и масштаба действий, перед повторным увеличением общего PPO бюджета. GRU/attention по-прежнему не реализованы. Пользовательский live marker сохранён, порты 33100–33103 освобождены; live policy не заменена.
