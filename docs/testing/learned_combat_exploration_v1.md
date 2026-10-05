# Отдельная ветка PPO с расширенным исследованием

05.10.2026. После диагностики второго PPO-цикла выполнен аудит корпуса и отдельный эксперимент. Награда `combat_reward_v1`, архитектура, guards и правила Go-исполнения не меняются.

## Аудит демонстраций

В `combat-bc-data-blaster-v2-20261005` есть размеченные движения у стен: train 188/260 movement rows, validation 40/58, test 50/68. Определение near-wall: хотя бы один standing-hull probe <16 units. SHA исходных split и счётчики сохранены в `workspace/artifacts/combat-ppo-exploration-cycle-v1-20261005/teacher-coverage.json`.

Это отвергает простую гипотезу полного отсутствия движения у препятствий. Не доказывает покрытие состояний остановки собственной политики: корпус мал, teacher двигается по ограниченным траекториям, selectors исключают команды с corrections/guards и полученным уроном.

## Замороженная гипотеза

У предыдущей модели latent movement std ≈0.1003, yaw/pitch std ≈0.00499. До tanh это примерно 0.9° разброса изменения угла при масштабе 180°. Исследование может недостаточно часто менять направление относительно небольшой BC-команды. Это гипотеза, не установленная причина остановок.

`scripts/fork_combat_exploration.py` создаёт новую ветку: log_std = [-1.2,-1.2,-3.9,-3.9], latent movement std ≈0.301, yaw/pitch std ≈0.02024 (~3.64° возле нулевого latent). Нелинейность tanh меняет фактический command std; величины не являются измеренными распределениями игровых команд.

В исходной модели/critic веса не меняются. Actor Adam moments сбрасываются явно, value Adam, RNG, consumed rollout history и накопленные counters сохраняются. Новый weights SHA связывается с v2 checkpoint. Fork не считается PPO update. Родительские веса/checkpoint не перезаписываются; проверяются SHA и точное соответствие actor/value/log_std checkpoint исходному JSON. Только fresh on-policy rollout новой версии входит в PPO, прежние rollout повторно запрещены.

Два теста проверяют сохранение весов/critic optimizer/RNG/history и неизменность родителя, сброс actor Adam, отказ нечислового/out-of-range std и несовпадающего actor. Оба прошли. Восстановление fork проверяется также реальным первым PPO update.

## Протокол

Frozen experiment: `scripts/scenarios/combat-exploration-v1.json`, его копия `experiment.json` в root artifact. Родитель: второй цикл, веса SHA `7de6c1c7620ca0f2ca2542871128da4836c669977f4d86757b892b541e5b0308`. Fork: `workspace/artifacts/combat-ppo-exploration-fork-v1-20261005`, веса SHA `edc881e90da1dc1d338370baf4f8db21e77bfb9b239117846473105ec3b0fd39`.

Cycle root: `workspace/artifacts/combat-ppo-exploration-cycle-v1-20261005`. Четыре batch, 4 server/client инстанса, x2, 300 игровых кадров. Training seeds 14000–14015, separate deterministic evaluation 14100–14103. Фиксируется четвёртый update; selection по оценке и настройка параметров по этим seeds не выполняются. Deterministic before имеет ту же actor/value сеть, что родитель: замена stochastic std не меняет deterministic action.

Это сравнение ветки до/после обучения, не A/B-доказательство пользы более широкого std против прежнего: одновременно есть свежий опыт и reset actor Adam. Для причинного сравнения нужны отдельная контрольная ветка и больше независимых условий. Узкая fixture-приёмка остаётся kills ≥1 и отсутствие первой смерти во всех четырёх валидных after-эпизодах; автоматического live-переключения нет.

## Результат

| Batch | Fresh transitions | Actor steps | Final KL | Value MSE своего batch |
| --- | ---: | ---: | ---: | ---: |
| 1 | 733 | 10 | 0.005487 | 0.068945 |
| 2 | 814 | 10 | 0.004921 | 0.262311 |
| 3 | 834 | 10 | 0.006748 | 0.065325 |
| 4 | 789 | 10 | 0.008175 | 0.048140 |

3170 новых переходов, ещё 40 actor / 160 value шагов. Ветка содержит 13 updates, 130 actor steps и 13 consumed rollout SHA с учётом родительской истории. Веса `iteration-4/update/weights.json`, SHA `e7c0462922cbbc3be6c044e36a85319b1580fff764200de80e3050d39db3c0ab`; checkpoint рядом. Все четыре native replay прошли; max Go/PyTorch logprob error ≤0.000057, value error ≤0.000001. CPU снова быстрее: 3.50–3.96 мс против CUDA 5.35–6.28 мс на actor step. Optimizer total ≈1.13 с без game/replay/startup/evaluation.

| Seed | Урон врагу до / после | Полученный урон до / после | Первая смерть до / после |
| --- | ---: | ---: | --- |
| 14100 | 30 / 30 | 100 / 0 | да / нет |
| 14101 | 30 / 30 | 19 / 5 | нет / нет |
| 14102 | 30 / 30 | 13 / 0 | нет / нет |
| 14103 | 30 / 30 | 19 / 0 | нет / нет |

Суммарно: monster damage 120→120, received damage 151→5, deaths 1→0, kills 0→0. `fixture_promotion_eligible=false`. Это выживание в коротком fixture, не успешное завершение боя.

Диагностика `diagnostics.json` автоматически записана runner: requested motion stationary 783/997→960/1092 motion pairs (78.5%→87.9%); movement zeroed 795/1000→972/1096 provider frames. После обучения во всех четырёх эпизодах непрерывные stationary серии 227–229 кадров, около 23 игровых секунд. Horizontal error >15° при атаке ближайшей clear цели снизился 361/999→18/1096, но эта метрика не проверяет pitch или попадания.

Гипотеза выхода из блокировки через эту короткую ветку **не подтверждена**: остановок больше, враг не уничтожен. Не утверждаем, что широкий std вреден сам по себе: контрольной ветки прежнего std с тем же reset Adam нет, sample мал. Выживание может давать высокий score при низком боевом прогрессе; действующая награда остаётся экспериментальной. Следующий шаг должен проверить направление pitch/реальные выстрелы и credit от остановок, а не просто повторять короткие update как доказанное обучение.

Всего 24 capture/provenance-valid эпизода; seed/dispatch подтверждены, before/after source/native/conditions совпали. Пакеты не скачивались, пользовательский live runtime не менялся. Собственные порты 33100–33103 освобождены. Общий Go код в этом продолжении не менялся; два новых fork tests, native replay всех batch, PowerShell syntax и whitespace checks прошли. Full-world reset equivalence остаётся неподтверждённой.
