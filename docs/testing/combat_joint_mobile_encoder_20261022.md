# Joint mobile BC: projected heads против обучаемого encoder

Статус:labels,два200-step CUDA fit,audit иfresh32 native/24 CUDA probe закрыты. Heads6/8 противsource5/8 иrules5/8;encoder4/8 не продвигается. Heads допущен только кfresh72 native/48 CUDA screen.

## Пробел в данных

Прежний decision BC обучал fire только при явной видимой цели. В mobile corpus есть1223 raw кадра `teacher_aim_source=none`,proposed/applied attack=false,но они оставались контекстом. Не все годятся для обучения:нужны актуальное observation,native exact/exclusive dispatch и отсутствие доступных наблюдаемых target slots. Нельзя автоматически трактовать grenade_release_pose,prop/navigation aim или guarded proposed attack как targetless negative.

`prepare_registered_joint_mobile_labels.py` соединяет canonical Go features,ground labels иdecision masks по полному seed/identity. Добавлены647train/414development явных no-target fire releases. Три focused tests проверили required proof,доступную цель,иной aim source,guarded attack иstaleness. Никакой runtime fire clamp не добавлен;ученик получает supervised labels,которые позже RL может изменить.

Корпус `combat-rules-joint-mobile-labels-v1-20261022`:

| Метки | Train32 | Development16 |
|---|---:|---:|
| Полный context |4277|2365|
| Target/fire labels |4086|2305|
| Fire positive |2254|925|
| Fire negative |1832|1380|
| Новые targetless releases |647|414|
| Coordinated ground labels |1881|1255|

Features не меняются и остаются причинными;future displacement участвует только в quality gate ground labels. Target0 при пустом availability всё равно принудительно доступен текущим action contract,поэтому высокая target agreement не доказывает новую сложную target tactic. Weapon/vertical supervision не добавлена.

## Равный бюджет

`train_registered_joint_mobile_cuda.py` читает новыйversioned corpus. Source у обоих — прежний `combat-rules-target-fire-bc-v1-20261022`,не отрицательный full-motion candidate и не просмотренные evaluation trajectories. Два checkpoint:

- `combat-rules-joint-mobile-heads-v1-20261022`:train projected output rows0:5,20:81 иspatial/conditional aim MLP;encoder/attention frozen.
- `combat-rules-joint-mobile-encoder-v1-20261022`:те же права плюс encoder иtemporal attention. Новых слоёв/ёмкости сети нет,меняется возможность переучить существующие representations.

Одинаковые source tensors,labels,200 steps,lr0.0003,seed20261022,fixed final endpoint иtrain retention Gaussian draws/keys. Loss:balanced moving/explicit-hold MSE + latent aim Huber(beta0.1) +0.2natural mode CE +0.5natural target CE +natural fire BCE +0.1policy KL. Mode/fire class priors не выравниваются искусственно. Development не входит вloss/selection. Context пересчитывается с текущими encoder weights во всех causal window32 sequences.

Parameter rows5:20(weapon/vertical) иweapon-state residual,std,value заморожены. **Их runtime outputs могут меняться при новом encoder/target представлении**,поэтому сохранение weights не выдаётся за неизменное поведение. Policy KL ограничивает drift,но игровой контроль обязателен. Actor Adam после BC явно сброшен;critic Adam/objective/config/PPO consumption history сохранены. Следующий PPO — только fresh own-policy rollouts.

## Offline development

| Метрика | Source | Heads | Encoder |
|---|---:|---:|---:|
| Balanced movement RMSE |0.464|0.430|0.296|
| Moving RMSE |0.523|0.475|0.365|
| Hold RMSE |0.396|0.381|0.206|
| Aim RMSE,°/ось |4.169|3.750|3.688|
| Natural fire BCE |0.500|0.469|0.370|
| Fire probability на no-target labels |22.13%|17.39%|10.29%|
| Natural mode CE |0.0300|0.0245|0.0244|
| Fixed-draw policy KL |0|1.495|4.152|
| Trainable parameters |—|28748|123020|

Optimizer loop21.85с/12.45с — point-in-time timings,не нормализованный benchmark скорости. Оба fit используют CUDA,CPU fallback нет. Эти числа измеряют teacher imitation,не живые wins/эффективность манёвров.

## Независимый аудит

`audit_registered_joint_mobile_cuda.py` подтвердил одинаковый начальный результат,бюджет/данные/source/seed/loss иточные train retention draws. JSON/checkpoint finite и согласованы. В heads изменились16 actor tensors,вencoder24;encoder/attention действительно изменились только во второй ветке. Protected parameter rows иvalue/std/value Adam/config/history сохранены;пустой actor Adam загружается стандартным PPO optimizer.

На2890source native rows/2906context rows baseline encoded memory проверена поsource;у новых моделей prefixes **пересчитаны из raw features**,а не сопоставлены со старым source cache. Full-sequence output идва независимыхprefix outputs совпали сmax3.81e−6(heads)/4.77e−6(encoder). Нельзя требовать совпадения новых encoder tokens сcache старого encoder.

На этих counterfactual source histories mean weapon KL:2.49e−10/3.40e−5,vertical KL:0/0.0678. Это ограниченная диагностика drift,не гарантия любых новых live states. Полная проверка текущих собственных native memory/log probabilities выполняется в CUDA replay probe.

## Игровой probe

`combat-joint-mobile-encoder-probe-evaluation-v1-20261022`:source/heads/encoder/mobile rules,одинаковые8 fresh условий(4ordinary/4recovery),32native/24CUDA,offset628,порт45700,16 ×2. Frozen weights иpair audit закреплены до запуска,5 comparisons,automatic promotion=false.

Проверяются native outcomes,CUDA replay,ownership/weapon/behavior и пять отчётов. Улучшение боёв и достаточность representations не устанавливаются по одному снижению BC error. При отсутствии выигрыша не масштабировать checkpoint автоматически;исследовать собственные train состояния/DAgger/PPO,не тренировать на текущем evaluation.

## Закрытый probe

Все32native/24CUDA и пять отчётов закрыты,hashes verified. Heads/source1gained/0lost,heads/rules1/0. Decision `projected_heads_screen_only`,encoder_screen=false,promotion=false.

| Метрика | Source | Heads | Encoder | Mobile rules |
|---|---:|---:|---:|---:|
| Победа |5/8|6/8|4/8|5/8|
| Убито монстров |11|14|10|11|
| Нанесённый урон |2046|2256|2018|1838|
| Полученный урон |521|421|519|352|
| Смерти |3|1|3|2|
| Clear-enemy standing |22.31%|17.82%|19.39%|—|
| Большой yaw ≥30° |6.12%|3.36%|5.44%|—|
| Смена цели/100frames |11.10|4.64|5.84|—|
| Attack без selected target |4.85%|18.15%|13.44%|—|
| Native MG damage contacts |125/169(73.96%)|124/194(63.92%)|107/166(64.46%)|—|

Encoder хорошо имитирует demonstrations,но хуже играет:его не продвигать. Heads имеет малый положительный сигналwins/death/received,меньшеstanding/yaw/churn,нохужеMG rate/targetless attack. Нельзя утверждать,что добавление no-target labels решило пустые выстрелы:в собственных новых состояниях они выросли несмотря на lower probability вteacher states. Это аргумент проверять covariate shift/собственные train trajectories,а не маскировать его offline метрикой.

Frozen screen `combat-joint-mobile-heads-screen-evaluation-v1-20261022`:source/heads/mobile rules,24fresh условия/вариант(16ordinary/8recovery),72native/48CUDA,offset632–633,порт45800,16 ×2. Source/code/weights закреплены до запуска,нового fitting нет. Default сохранён,automatic promotion=false.

## Закрытый screen

Все72 native/48 CUDA и пять bundle reports закрыты; SHA отчётов/входов проверены. Bundle: `workspace/artifacts/combat-joint-mobile-heads-screen-bundle-v1-20261022`.

| Вариант | Победы | Ordinary / recovery | Убийства | Полученный урон | Смерти | Yaw ≥30° | Стояние | Attack без цели |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Source target/fire BC |11/24|7/16;4/8|27|1441|12|5.08%|17.84%|9.15%|
| Joint heads |11/24|6/16;5/8|35|1305|7|4.69%|16.61%|19.01%|
| Mobile rules |14/24|7/16;7/8|37|929|6|—|—|—|

К source:2 новые/2 потерянные победы;описательный95% интервал разницы[−16.67;+16.67]п.п. К rules:1 новая/4 потерянные. У heads больше убийств и меньше смертей,но6 незавершённых живыми боёв против1 у source. Actual MG live-monster damage302/515(58.64%) у source и378/605(62.48%) у heads;solid204→205,corpse9→22. Увеличение attack без цели относится к командам,не к физическим выстрелам.

Решение `candidate_not_advanced`,promotion=false:большой coverage/PPO от heads не назначен,default сохранён. Лучший offline encoder ранее тоже не прошёл smoke.

Дополнительный CUDA audit на собственных историях закрытого8-case probe:source50/883 и heads249/1100 кадров не имеют доступного bbox target. На собственных состояниях heads выдаёт mean fire probability55.66%,source на тех же состояниях62.41%;на состояниях source71.57%→65.66%. Улучшение fire head на фиксированном входе не предотвращает попадание в другие состояния. Это вероятности sampled command,не факт выстрела/стратегической бесполезности.

Следующая отдельная гипотеза — [native MG miss reward вместе с sequence objective](combat_sequence_hitscan_reward_20261022.md),от прежнего source,без продвижения heads.
