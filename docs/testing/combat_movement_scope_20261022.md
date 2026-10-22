# Изолированное обновление движения, 22.10.2026

## Причина проверки

[AWR diagnostic](combat_advantage_regression_20261022.md) показал:79–89% изменения policy KL после полного update приходится на conditional aim; средняя абсолютная разница tanh(move) от source всего0.003–0.005. Проверяется ограничение обновления actor параметрами движения. Само это измерение ещё не доказывает блокировку обучения общим KL guard.

## Реализация и инварианты

Новый training-only `actor-head-scope=movement`: в head/residual обучаются rows0:2 (forward/side means),5:8 (release/jump/crouch logits) и log_std[:2]. Encoder/attention, aim/target/fire/weapon branches и прочие параметры заморожены; после каждого Adam proposal восстанавливаются защищённые rows/moments. Scalar clocks общих tensors всё равно продвигаются; это ограничение проекции, а не per-row Adam clock.

Все runtime действия остаются выходами обученных голов модели. Новый scope не задаёт манёвр правилами и не отключает прицел/стрельбу. У spatial/sampled-motion контроллеров есть вход movement mean: замороженные параметры прицела **не гарантируют неизменность его выхода** при смене движения.

`scripts/test_combat_target_head_scope.py`:4 CUDA tests, включая новый scope на45/81 outputs с уже заполненными Adam moments; прежние decision/target/rejection tests сохранены. `scripts/audit_combat_movement_scope_cuda.py` проверяет реальные source boundaries, runtime weights control, защищённые параметры/moments, общие critic/Adam и TRAIN поведение.

## Два закрытых механических опыта

Одинаковые source,16 TRAIN боёв/2496 rows, reward/sequence retention,10 actor /40 critic steps. Полный PPO control точно совпадает с ранее закрытыми elevation-PPO weights. Ни в одном опыте не использованы development/evaluation rows.

До fit закреплён gate: изменение средней команды движения хотя бы1.5× относительно full при доле nonmovement KL не выше50%. Это механический gate для целесообразности native проверки, не критерий качества боя.

| Вариант | LR | Mean absolute move shift | Total KL | Nonmovement fraction |
|---|---:|---:|---:|---:|
| Полный PPO | 0.00003 | 0.005348 | 0.004246 | 80.54% |
| Только движение, тот же LR | 0.00003 | 0.002733 | 0.000586 | 59.87% |
| Только движение, LR×10 | 0.0003 | 0.004052 | 0.001517 | 52.05% |

Roots: `combat-movement-scope-pair-v1-20261022`, `combat-movement-scope-rate-v2-20261022`. Во втором опыте единственный заранее выбранный больший head LR проверялся только на TRAIN; gate не менялся. Он доступен явно как `--movement-head-lr`, default0.00003 сохраняет первый опыт.

Оба gate не пройдены. `decision.json` каждого root закрывает ветку без native запуска (0 дополнительных боёв), promotion=false. Это экономит прогоны, которые пока не проверяли бы нужный механизм. Архитектура и исходная live модель не заменялись.

Финальный KL обоих scoped fits ниже guard0.005. Следовательно, насыщение guard само по себе не установлено как причина малого изменения. Возможны ограничение представления замороженного encoder, небольшой step budget и характер TRAIN gradient. Данный опыт их не разделяет.

## Выявленная зависимость и следующий прототип

Даже когда менялись только motor parameters,52–60% итогового KL приходилось на остальные компоненты. Это согласуется с проверенным кодом: `combat_spatial_aim.inputs` читает tanh(raw movement means), а `combat_sampled_motion_aim.inputs` дополнительно actual sampled move и его разность с means. В Go те же зависимости заданы `spatialAimInputs` и `sampledMotionAimMeans`.

Следующий [моторный residual prototype](combat_motor_residual_plan_20261022.md) должен проверить прямой доступ к геометрии и раздельное управление распределением движения при сохранении прицела, обусловленного **фактически выбранным** движением. Это новая архитектурная гипотеза; она пока не реализована и не имеет боевого результата.
