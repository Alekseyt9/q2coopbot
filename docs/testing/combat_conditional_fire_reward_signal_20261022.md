# Fire reward signal и проверка бюджета обучения

22.10.2026. Диагностический relabel32TRAIN боёв закрыт. Он уточнил прежнюю интерпретацию: группа «нет выбранной цели» неоднородна; реальные MG misses без цели уже имели отрицательный локальный fire signal. После этого выполнен отдельный300-step fire-only fit **с прежней наградой**, не с новым штрафом. Fresh native screen закрыт:300-step вариант3/8 против start4/8 иrules5/8; длинный fit отклонён.

## Диагностический relabel

Артефакты: `workspace/artifacts/combat-conditional-fire-reward-signal-v1-20261022`.

`scripts/diagnose_combat_conditional_fire_reward.py` использовал только закрытые32TRAIN captures из первого conditional-fire корпуса. Новые бои не запускались. Через сохранённый, проверенный SHA Go exporter выполнены64экспорта: control и cost для каждого боя.

- Контрольные steps/rewards/server-outcomes побайтно совпали с исходными.
- В cost steps побайтно совпали с control; наблюдения, actions, labels и native evidence не изменились. В server outcomes меняется только reward score.
- Все reward components, кроме `machinegun_miss`, сохранены точно. Для400eligible command windows новый cost равен числу проверенных geometry/corpse misses ×−0.005. Native dispatch matched/exclusive, recovery0; cost привязан к sample с attack. Goal boundary не пересечена.
- Среди1000MG shots:600live-monster damage,392solid geometry,8corpse. Отрицательный cost затронул400из5058TRAIN rows; суммарная награда269.4399→267.4399, разность ровно−2.
- Это **diagnostic only, PPO-ready=false**: checkpoint/reward metadata и consumed history не подменены. Исходные TRAIN данные сохранены.

GAE и signal сравниваются при одних recorded actions, value estimates и causal histories. NN inference/градиенты выполнены только на CUDA.

| TRAIN группа | Rows | Начальный mean fire score, control | С cost−0.005 |
|---|---:|---:|---:|
| Нет selected target, все команды | 688 | +0.004218 | +0.004081 |
| Native MG misses | 400 | −0.006132 | −0.007288 |
| Native MG misses без selected target | 68 | −0.073252 | −0.075102 |

Score=`normalized_advantage*(attack-probability)` — локальный сигнал для отдельного fire logit, не причинная оценка каждого промаха. Нельзя выводить из положительного среднего по688командам, что68реальных слепых MG misses поощрялись: их сигнал **уже отрицательный**.

Полные initial gradients6369fire parameters проверены отдельно в budget audit: norm0.004060→0.004145, cosine0.999988. Малый штраф усиливает почти прежнее направление, а не открывает принципиально отсутствующий gradient. В нулевой инициализации ненулевой gradient имеет16параметров выходного слоя; дальнейшие слои начинают получать signal после изменения нулевого выхода. Это не означает, что вся branch остаётся замороженной.

## Бюджет:10 против300шагов

`scripts/fork_combat_actor_budget_cuda.py` создал явный альтернативный fork исходного **не потребившего этот rollout** zero checkpoint: меняется только actor-step cap10→300. Weights JSON побайтно прежний; reward/LR/std, critic budget40, Adam, RNG, counters и consumed history сохранены. Fork записан в checkpoint. Это не продолжение обученного10-step checkpoint на повторно использованном rollout.

Артефакты: `workspace/artifacts/combat-conditional-fire-budget-v1-20261022`.

В прежнем `ppo_recurrent.py` выполнен fire-only fit:300принятыхactor steps,6369trainable parameters, CUDA48.18с. Награда прежняяv13 с `machinegun_miss=0`; LR0.0003, sequence retention1, target KL0.005. Итоговый approx KL0.003496, retention KL0.003849.

`audit-v2.json` проверяет initial resume, frozen old actor/std/Adam, consumed-once history, новые clocks300, runtime/checkpoint equality. Critic и его optimizer точно совпали с прежним10-step fit: различие не объясняется дополнительным critic training.

Средняя вероятность выстрела на фиксированных TRAIN behavior contexts:

| Группа | Start |10steps |300steps |
|---|---:|---:|---:|
| Все5058rows |83.36%|83.25%|83.16%|
| Нет selected target,688rows |75.15%|75.01%|75.35%|
| Реальные MG misses,400rows |90.77%|90.71%|90.08%|
| Реальные MG misses без цели,68rows |88.12%|88.04%|84.93%|

Длинный fit проявил различие между группами; он не просто уменьшает огонь везде без цели. Это TRAIN observation, не доказательство сокращения physical misses на новых боях. Не менять reward или default на основании этих вероятностей.

## Следующий native gate

После закрытого audit-v2 отдельно зафиксированы `native-gate-decision.json` и `evaluation-spec.json`:32native/24CUDA, start/fire-10/fire-300/rules по8freshусловий, seed offset755, base3/ware2 с разнымиsite/loadout/composition, pool16×2. Промежуточные checkpoints не выбирались. Сравнение закрыто.

Критерии: verified wins/deaths/received damage вместе с настоящими MG hits/misses и лишними действиями. Без выигрыша native metrics длинное обучение не объявлять решением проблемы, а удачный малый screen отдельно подтверждать. Default и публикуемые weights остаются прежними.

| Вариант | Победы | Убийства | Received damage | Смерти |
|---|---:|---:|---:|---:|
| start |4/8|12|513|3|
| fire-10 |4/8|12|513|3|
| fire-300 |3/8|12|531|4|
| rules |5/8|14|481|3|

Long fit потерял1победу без новых; short не изменил набор побед. Native MG live damage167/280→167/280→153/280(59.64→59.64→54.64%); targetless solid shots13→13→14. Large yaw3.56→3.56→8.27%; stationary clear15.44→15.44→22.01%. Снижение вероятности на TRAIN subset не перенеслось на полезное управление новыми боями.

На base3 site02 исходная модель и оба кандидата продолжают спускаться на168–192units и терять бой; в одном recovery92/80frames находятся ниже старта без видимого противника и с remembered threat наверху. Потерянная long-fit победа относится к site01 без спуска: геометрия не объясняет все проигрыши.

`combat-conditional-fire-budget-v1-20261022/decision.json`: reject long fit, no promotion, прежний zero source остаётся исходной точкой. Не увеличивать PPO steps этого же опыта вслепую. Следующий [action-outcome prediction probe](combat_shot_prediction_probe_20261022.md) проверяет обучаемость физического результата на сохранённых TRAIN captures до какой-либо интеграции предиктора в policy.
