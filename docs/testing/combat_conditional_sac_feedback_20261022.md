# SAC/PPO: два раунда свежего опыта

Итог:initial4/8,PPO4/8,SAC0/8,rules5/8 наfresh paired smoke. SAC branch отклонена,default сохранён;цикл завершён.

Продолжение [отрицательного offline warm опыта](combat_conditional_sac_20261022.md). Цель — проверить обучение по новым действиям самого обновлённого агента;не объявлять один повторный fit на прежних32 боях полноценным online SAC.

Root: `workspace/artifacts/combat-conditional-sac-feedback-v1-20261022`. До запуска закреплены protocol/source weights/initial replay/CUDA audits и итоговый evaluation spec.

## Бюджет и условия

У каждой ветки общий initial32-case корпус,затем два новых собственных32-case раунда. Итого96 боёв на алгоритм в этом эксперименте,из них128 новых unique native training battles на обе ветки. Предобучение общего ancestor не выдаётся за эти96 боёв. Восемь одинаковых семейств четырёх карт,MG/mixed weapons и recovery;train seeds уникальны между запусками,распределение условий одинаковое.

Пул16 ×2,один native pool за раз,слот автоматически получает следующую задачу. Все training/numerical NN checks CUDA. Рецепт награды сохранён:v13/control с machinegun_miss=0. Actor scope одинаковый:10212 conditional aim/std параметров,остальные heads и encoder frozen. PPO даётся10 actor/40 value steps на свежий корпус,SAC256 actor/256 twin-Q steps с продолжением уже обученных critics/targets/Adam. Intermediate quality не выбирает checkpoints/бюджет.

SAC использует cumulative TRAIN replay32→64→96 случаев;PPO использует только свежие own-policy rows. Разное число optimizer steps и время измеряются отдельно. Здесь одинаково число боёв и их preset frame caps;фактическое число переходов различается из-за ранних побед/смертей/таймаутов и измеряется отдельно. Это case-budget pilot,а не строго transitions-matched benchmark sample efficiency. Для общего вывода нужны согласованный бюджет переходов,полное время и несколько training seeds.

## Проверяемое продолжение и replay

`train_combat_conditional_sac_cuda.py --resume` принимает только закрытый source с проверенными JSON/inference/SAC checkpoint SHA. Actor/Q/targets/оба Adam/RNG восстанавливаются;hyperparameters должны совпадать,изменение требует отдельного fork. RNG сохраняется на границе обучения до export/read-back constructors. Disposable CUDA test подтвердил точное равенство uninterrupted2 и closed1+1,включая actor,critics,targets,optimizers,weights SHA и RNG. Это не восстановление произвольно прерванного optimizer step.

`combat_replay_catalog.py` объединяет binary tensors/indexes. Все shards TRAIN-only,objective одинаковый,source weights закреплены. Разные behavior веса допускаются только при одинаковом fingerprint всей политики кроме conditional aim/std и value:encoder/attention,discrete/motion/spatial/weapon factors и movement std должны совпадать. Duplicate(seed,index),cross-reset/неподтверждённые successor и validation/test отвергаются. Старые assets/traces не копируются.

После каждого SAC fit independent audit проверяет точные optimizer step increments,формы/finite моменты,неизменность protected family и value,сохранение PPO counters. Дополнительно сравнивает Q с фактической native reward на zero-bootstrap новых TRAIN действиях до/после fitting этих же данных. Это training diagnostic,не held-out calibration. После каждого парного раунда CUDA audit повторно проверяет обе модели на3450 общих исходных историях.

## Итоговая оценка

Только после всех128 новых training captures/четырёх fits/audits запускается32 native/24 CUDA на8 свежих matched условиях:initial source,PPO-final,SAC-final,rules. Offset648,порт46220;default не заменяется автоматически. Победы первой жизнью — главный исход;также damage/death/timeout,actual MG ray outcomes,повороты/стояние/атака без цели. Старый warm smoke сохраняется отдельно и не смешивается с новым.

Неполные corpus/update сохраняются и не запускаются заново автоматически. OS lock предотвращает второй executor;`--resume` переиспользует только закрытые фазы после проверки закреплённых исходников. Во время pipeline нельзя менять pinned Go/Python/PS/native code. Документация не влияет на executable snapshot.

Состояние перед запуском:подготовка и проверки продолжения завершены,новый feedback цикл ещё не закончен. Ниже зафиксирован его закрытый результат.

## Закрытое обучение и итоговая оценка

Цикл завершён. Восстановление PPO round1 сохранило31 capture и повторило только один исходный seed. Существующий recovery finalizer встретил уже созданные aggregate binaries;derived group outputs архивированы отдельно,повторный бой уже не запускался,выполнен неизменённый original finalizer-only snippet и обычный32-case CUDA replay. `recovery-v1/finish-recovery.py` и receipts описывают эту derived closure;она не выдаётся за исходное успешное завершение collector. SHA retained members проверены до и после. Pool wall_seconds восстановленного корпуса относится только к финализации и не сравнивается как native performance;старые/повторные member timings сохранены.

| Свежий раунд | Алгоритм | Бои | Новых accepted rows | Context rows | Native game frames | Принятые actor steps |
|---|---|---:|---:|---:|---:|---:|
|1|SAC|32|3979|3982|4135|256|
|1|PPO|32|2963|2963|3078|10|
|2|SAC|32|4066|4067|4201|256|
|2|PPO|32|4074|4079|4221|10|

С учётом общего initial3450 rows:SAC11495 accepted transitions,PPO10487;случаи одинаково ограничены preset caps,ноэто не точный transitions-matched benchmark. На текущем pilot дополнительные переходы SAC не привели к выигрышу. Суммарно SAC768 actor/1024 critic steps от начала warm ветки;PPO30 actor/120 value steps в этой ветке. Более старые shared ancestor counters не включаются в эти числа. CUDA pair audits обоих раундов прошли;protected raw logits остаются exact на3450 исходных историях. До/после fitting новых TRAIN zero-bootstrap действий Q RMSE4.52→3.93 и4.34→3.81;это training fit,не качество управления.

Native TRAIN outcomes до соответствующего update:round1 SAC11 goals/21 deaths,PPO11/21;round2 SAC5 goals/26 deaths/1 other,PPO17/14/1 other. Это разные train seeds и собственные closed-loop trajectories,а не matched quality comparison или основание выбирать промежуточные веса. Таблица final held-out ниже — независимая оценка.

Все32 native/24 CUDA итоговой оценки и пять bundle reports закрыты;SHA отчётов/входов и отдельного actual MG report проверены. Только final checkpoints оценивались,intermediate validation selection не было.

| Вариант | Победы | Ordinary / recovery | Убийства | Полученный урон | Смерти | Yaw ≥30° | Стояние | Attack без цели |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial |4/8|2/4;2/4|11|452|3|6.35%|23.90%|12.79%|
| PPO final |4/8|2/4;2/4|11|378|3|4.62%|25.07%|13.84%|
| SAC final |0/8|0/4;0/4|0|764|8|12.32%|23.50%|1.73%|
| Rules |5/8|1/4;4/4|11|451|3|—|—|—|

У SAC нет новых/четыре потерянные победы к PPO и initial. Native MG с уроном живому монстру:initial104/200(52%),PPO96/178(53.93%),SAC20/200(10%). У SAC180 solid contacts;меньше attack без цели не значит лучше стреляет. На этой постановке продолжение SAC резко ухудшило бой несмотря на снижение учебного TD/terminal fit error.

## Диагностика оценок действий и решение

Read-only `critic-action-support-audit.json` сравнил текущий Q на записанном действии и текущем среднем действии в тех же TRAIN states/категориях/movement. На initial replay текущий прицел в среднем43.42 behavior std от source mean;93.45% дальше3 std. На двух свежих shards соответствующие значения49.68/32.54 std и96.86%/96.63%. Gaussian имеет ненулевую математическую поддержку везде:здесь речь о выходе из плотно представленной данными области,а не о строго нулевой вероятности.

Текущий critic предпочитает новое mean action записанному в76.09%/82.58%/85.46% этих состояний,ожидаемая им прибавка Q≈0.029/0.039/0.028. При этом native final0/8 и10% MG live damage. Это согласуется с эксплуатацией неточных Q оценок вне хорошо покрытой области,ноне является native counterfactual доказательством причины каждого промаха. Большие mean-action shifts11.71–22.21°/ось тоже не являются непосредственно измеренной ошибкой прицела.

Решение `conditional_sac_feedback_rejected`,screen=false,promotion=false. Ещё256/512 updates или увеличение replay этой же ветки автоматически не назначаются. Default/initial сохранены. Не делается общий вывод о непригодности SAC:один training seed,малый smoke иaim-only scope с замороженными остальными factors.

Следующая проверка должна ограничивать доверие к неподтверждённым Q предпочтениям и проверять согласованность всех действий. Нужен отдельный фиксированный протокол:обновления внутри области,поддержанной replay/поведением,либо обновление actor через наблюдённые полезные действия;сравнение с таким же control и независимые новые бои. Простое добавление слоёв/attention или другой penalty не исправляет выявленное расхождение Q и фактического боя.

Во время первого PPO corpus pool остановился:сводно28 usable captures из32,ноindependent per-member проверка нашла31 закрытый пригодный capture и один missing report после reset-release mismatch. Failed pool report перенесён внутри корпуса вrecovery-inputs-v1;исходники/планы/веса не меняются. Используется существующий recover_registered_combat_corpus.py сOS lock:31 retained/1 rerun исходного seed,проверка fingerprints/хешей и обычныйCUDA replay. Весь32-case корпус заново не собирается.

Readonly TRAIN boundary join проверил гипотезу обошибочном раннемzero bootstrap. У initial32 есть15 control_handoff endings,все15 — native goal_complete;у первого fresh SAC32 есть10 таких endings,все10 такжеnative goal_complete. В SAC corpus21 death/11 goal,не побег в timeout. Эти наблюдения не подтверждают,что данный pilot выигрывает reward за уход из боя. Семантика terminal/reward во время эксперимента не меняется.
