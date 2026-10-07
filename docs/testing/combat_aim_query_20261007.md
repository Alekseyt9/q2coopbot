# Aim queries на learned states, 2026-10-07

Исходный BC aim corpus содержал187 train labels со средним yaw_delta0.114°, ни одного поворота >5°. В learned train states тысячи наблюдений требуют коррекции >20°. Старый aim-only BC снижал победы36/80→24/80; fire-only улучшал результат. [Диагностика и контроли](combat_sequence_bc_20261007.md), [новая оценка BC/PPO/rules](combat_fire_bc_rl_20261007.md).

Реализован Go adapter `internal/aimquery`: желаемый номинальный прицел к ближайшему подтверждённому clear observed monster в пределах650 units. Используются observed relative position, protocol bbox и текущий standing/crouched eye; неизвестный bbox или закрытая линия не становятся меткой. Обрабатываются angular wrap и ограничение action contract. Поддержаны текущие Blaster/Machinegun names/model paths. Query не выбирает оружие и не задаёт огонь или движение.

Query не является исполненной командой или точной копией полного rules controller: здесь нет projectile lead, recoil compensation, тактического выбора угрозы или доказательства будущего попадания. Этот ориентир предназначен для обучения восстановления aim в посещаемых состояниях, а не для приписывания себе teacher wins.

## Проверка и экспорт

Новая selection version `counterfactual_observed_bbox_aim_v1` требует explicitly deferred test. `demodata.Build` повторно проверяет original reset/native events/exact dispatch/steps/outcomes/rewards и goal receipts. Frozen per-episode behavior-model SHA и sampling seed проверены; stochastic samples перепроверяются через PPO Review/VerifyMemory, deterministic captures воспроизводятся через Decide. Desired action хранится в отдельном `counterfactual_aim_query`; `target_applied_action`, executed Step и reward остаются исходными.

Legacy `q2bc-data` отвергает query corpus. Sequence exporter проверяет query version/identity/contract и разрешает только aim mask; fire/movement/vertical masks выключены. Каждый row отмечен `counterfactual_nominal_aim` либо `context`. Native verification доказывает исходное состояние и actual command, а не исполнение query.

Взяты все20 семейств реестра. Train80 captures: собственные stochastic battles Fire BC, train offset40. Validation80 captures: исходный Update29 deterministic reference, validation offset24. Policy visitation distributions различаются; validation используется только как validation, final-test не запускался. Spec assignments frozen, engine seeds разделены. Исторические captures revalidated; они не превращены в fresh PPO опыт новой policy.

| Набор | Контекст | Aim queries | Контекст без метки | Fire labels |
|---|---:|---:|---:|---:|
| Train |6987|6364|623|0|
| Validation |6531|6234|297|0|

Root: `workspace/artifacts/aim-query-corpus-v1-20261007`. Все20 query datasets и финальный `sequences-v1` повторно прошли сборку из source receipts. Train SHA `e195f19a5386b0f131cd62f07b4c50f56b33ed00e1541353463bcc2f7dd15200`; validation SHA `1738dabbbbba003ac683365d37b0b05de4342cf84fa3d88dc10000ad05fb6ba0`. Exporter SHA и behavior-model hashes сохранены в metadata.

Первый dataset attempt сохранён отдельно: deterministic validation captures не имеют stochastic Sample; initial adapter ошибочно требовал его. Исправлено раздельной проверкой stochastic samples и deterministic Decide; все завершённые datasets имеют префикс `dataset-v2-`. Незавершённый первый attempt не используется для обучения.

## CUDA обучение

Parent — Fire BC Update29, а не неулучшившая PPO30. Config `scripts/scenarios/combat-sequence-query-aim-v1.json`:150 fixed epochs, lr0.0001, aim heads only. Yaw error сравнивается с учётом периодичности; pitch — обычный error. Нет fire labels, attack loss не используется и report показывает null/0rows вместо ложного BCE успеха. Encoder/MHA и все остальные строки head/residual точно сохранены; critic/log_std тоже сохранены. Actor optimizer для будущего PPO сброшен; PPO counters по BC не увеличены.

| Метрика | Train до | Train после | Validation до | Validation после |
|---|---:|---:|---:|---:|
| Nominal-query aim RMSE, градусы |27.885|23.865|25.609|21.285|

Это согласие с номинальными query labels, не точность стрельбы. Артефакты `aim-head-update-v1`, CUDA training завершён, completion seal сохранён.

## Игровая проверка

Запущена paired оценка Parent Fire BC против Query Aim+Fire BC на новых validation seeds offset28:20 семейств×4 seeds×2 models=160 captures, pool16/x2. Root `workspace/artifacts/aim-query-eval-v1-20261007`. Оценка завершена: query branch регрессировала и не назначена основной. Final-test и полные кампании не запускались.

Проверки: Go tests всего product (`./cmd/... ./internal/...`) прошли, включая большой промах, wrap179→-179, crouched eye, unknown bbox, скрытую ближайшую цель, stale observation, отсутствующий query, запрет fire mask и сохранение actual action. Runtime learned provider не обращается к query adapter: он используется только offline export/training. Веса/checkpoints остаются вне Git.

## Итог native оценки

| Метрика | Parent Fire BC | Query Aim+Fire BC |
|---|---:|---:|
| Победы |47/80|30/80|
| Смерти в первой жизни |30|41|
| Убийства в первой жизни |51|31|
| Полученный health damage |3514|4240|

Все160 captures завершены и проверены, generated fixture pairs совпали. Native outgoing monster damage5539→3497. Доля firing commands с horizontal yaw error >20° к ближайшему clear observed enemy1770/4932 (35.9%)→1987/4778 (41.6%). Этот proxy без pitch/lead не является hit accuracy. Ошибка на offline labels снизилась, но поведение в собственной игровой петле стало хуже. Эта ветка не принимается.

| Семейство | Parent Fire BC | Query Aim+Fire BC |
|---|---:|---:|
|parasite-blaster-generated|4/4|1/4|
|parasite-gunner-blaster-generated|0/4|0/4|
|parasite-machinegun-recoil|3/4|3/4|
|parasite-gunner-machinegun-recoil|0/4|0/4|
|campaign-base1-site-01-blaster|4/4|0/4|
|campaign-base1-site-01-machinegun|3/4|0/4|
|campaign-base1-site-02-blaster|4/4|4/4|
|campaign-base1-site-02-machinegun|3/4|1/4|
|campaign-base1-site-03-blaster|0/4|1/4|
|campaign-base1-site-03-machinegun|0/4|0/4|
|campaign-base1-site-04-blaster|1/4|0/4|
|campaign-base1-site-04-machinegun|1/4|1/4|
|campaign-base2-site-01-blaster|4/4|4/4|
|campaign-base2-site-01-machinegun|4/4|4/4|
|campaign-base2-site-02-blaster|0/4|0/4|
|campaign-base2-site-02-machinegun|0/4|0/4|
|campaign-base2-site-03-blaster|4/4|3/4|
|campaign-base2-site-03-machinegun|4/4|4/4|
|campaign-base2-site-04-blaster|4/4|2/4|
|campaign-base2-site-04-machinegun|4/4|2/4|

## Проверка Go/PyTorch replay

Добавлен cmd/q2bc-sequence-review: deterministic Go replay всех exported provider contexts с теми же identity/frame gaps, сравнение только aim-labelled rows с их отдельно сохранённым query. Before: train27.885373854°, validation25.609161396°; after: train23.864984527°, validation21.285124158°. Это совпадает с CUDA metric до0.00001°. Численное расхождение Go/PyTorch не объясняет этот регресс; повторная source/model/data SHA проверка находится в go-review-v2-*.json.

Next experiment: согласовать movement labels с изменяемым final view frame и сравнить совместное обучение aim/movement с сохранением удачного fire prior. Forward/side в Quake зависят от final yaw и pitch/3; сохранность movement parameters не означает сохранность world-direction при смене aim. Это проверяемая причина, а не установленное единственное объяснение. Нужны tests world-input transformation, проверка качества queries/closed-loop damping, затем собственный свежий PPO опыт; один рост модели не считать исправлением данных/контроля.

Seal query weights SHA86f844f38e1108e128e21fec0ffd8cbad35640cf05c1001bb06dd562427b98f3; checkpoint SHA949f23f75dd38dff2d87c1cdf4bc0822a4cd4883d30fa2130435c2a5b112bb41. Best experimental branch остаётся Fire BC без неулучшивших PPO30 и Query Aim. Превосходство обычного controller и full-campaign acceptance не достигнуты.

