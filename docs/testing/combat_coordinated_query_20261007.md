# Coordinated nominal aim / planar-input BC, 2026-10-07

## Проверяемая причина

Отдельная aim-query ветка снизила offline RMSE, но проиграла игровую paired оценку47/80→30/80. В native `src/common/pmove.c` PM_AirMove planar wishvel зависит от final view frame: horizontal forward с cos(pitch/3), right=(sin(yaw),-cos(yaw)) при roll0. Сохранение параметров movement head не сохраняет world input при изменении yaw. Это диагностическая гипотеза, не установленная единственная причина регресса.

## Новый query contract

Go `internal/aimquery.QueryCoordinated`, version `observed_bbox_aim_world_input_query_v1`; selector `counterfactual_observed_bbox_aim_world_input_v1`.

Исходная командная база — native-проверенное actual applied action, а не hypothetical proposed. Новый aim — прежний nearest clear observed bbox query. Forward/side вращаются в новый quantized command yaw/pitch frame. При выходе за bounds обе команды масштабируются одним factor≤1, сохранённым в `planar_input_scale`. До command rounding направление совпадает с исходным input; амплитуда может снизиться. Это не сохранение actual velocity/trajectory, не оптимальный манёвр, не компенсация recoil/lead.

Labels только grounded standing roll0 vertical-release без guards/interventions; applied action/command должны совпасть с capture. Jump/crouch/air/guarded states остаются masked temporal context. Actual actions, outcomes, rewards не переписываются. Runtime provider не вызывает offline query.

Тесты: yaw−179/−90/0/45/179, pitch−80/0/80, zero/forward/strafe/diagonal/bound saturation; independent native planar vector within1.01 command units после int rounding. Guard rejection и отдельность query targets; запрет fake fire/vertical/weapon masks. `go test ./cmd/... ./internal/...` прошли.

## Corpus

Root `workspace/artifacts/coordinated-query-v1-20261007`. Повторно проверены160 historical captures20families: train FireBC stochastic offset40, validation Update29 deterministic offset24. Source native/runtime/model replay и hashes проверены заново, final test deferred. Это разные visitation policies; validation не используется для обновления gradients.

Первый dataset attempt отклонён: в base1/site01/Blaster validation отсутствуют подходящие movement labels, большая часть grounded release commands имеет static_hull_blocked guard. Сохранён диагностический attempt. Новый coordinated contract разрешает verified context-only семейства: Ready означает проверенный input для combined sequence corpus, а не наличие supervision в каждой семье. Combined trainer требует positive paired aim/movement labels и запрещает fire masks. Обычные teacher/aim-only Ready условия не ослаблены.

| Split | Context | Joint labels | Masked context |
|---|---:|---:|---:|
|train|6987|818|6169|
|validation|6531|365|6166|

| Семейство | Train labels | Validation labels |
|---|---:|---:|
|campaign-base1-site-01-blaster|92|0|
|campaign-base1-site-01-machinegun|46|45|
|campaign-base1-site-02-blaster|34|13|
|campaign-base1-site-02-machinegun|53|3|
|campaign-base1-site-03-blaster|12|0|
|campaign-base1-site-03-machinegun|18|29|
|campaign-base1-site-04-blaster|26|14|
|campaign-base1-site-04-machinegun|18|22|
|campaign-base2-site-01-blaster|52|42|
|campaign-base2-site-01-machinegun|43|6|
|campaign-base2-site-02-blaster|7|0|
|campaign-base2-site-02-machinegun|11|11|
|campaign-base2-site-03-blaster|20|18|
|campaign-base2-site-03-machinegun|22|9|
|campaign-base2-site-04-blaster|16|4|
|campaign-base2-site-04-machinegun|22|1|
|parasite-blaster-generated|98|53|
|parasite-gunner-blaster-generated|51|32|
|parasite-gunner-machinegun-recoil|74|19|
|parasite-machinegun-recoil|103|44|

Train SHA 8849b20106d8be381651253c9f181c84d8a361c890141588b0dd53086a6114c1; validation SHA 4d33ec5827f2656b85dfaed3a7d8d9c5a8839acc51fb7f2a5742aa7ae07ca0ca.

## CUDA update

Parent FireBC29, не PPO30 и не регрессирующая aim-only ветка.150 fixed epochs, lr0.0001, wrapped yaw, aim weight4, movement weight1; output rows0/1/2/3 только. Encoder/temporal MHA frozen; fire/vertical/weapon parameters и outputs на corpus contexts точно сохранены, critic/std неизменны. Fresh actor optimizer для будущего PPO. Counters29/279 не увеличиваются от BC. Config `scripts/scenarios/combat-sequence-query-coordinated-v1.json`.

train: aim RMSE 29.921→26.103°, movement normalized RMSE 0.358262→0.337594.

validation: aim RMSE 33.089→27.366°, movement normalized RMSE 0.273392→0.263375.

Weights SHA ad85edb05b7ad68b59b59bb01da6d3b2e65b776483fd7caf3a9cbd9d4a74b94f; checkpoint SHA 01870f58eec3dfa5d0be89ccfb75dde16ddb29e96b5fde1dea541bf7e86da6e8.

## Live validation

Root `workspace/artifacts/coordinated-eval-v1-20261007`. Запущены160 paired captures20families×4seeds×2models, pool16/x2. Before FireBC29, after coordinated BC. Validation offset28 повторно используется для диагностики совместного обучения; это не новый unseen cohort и не final test. На момент записи оценка ещё выполняется; offline loss не является доказательством игрового улучшения. Main experimental branch пока FireBC29.

## Audit фактических меток

`input-audit.json`: все818 train и365 validation paired labels прошли independent planar-vector check после command quantization. Максимальный component error0.679541/0.662652 native command units; saturation scale применён8/4раза; zero-input labels отсутствуют. Большие yaw correction >20°347/125. Diagnostics SHA проверены до/после, sequence data SHA совпадают с metadata. Это подтверждает корректность геометрического преобразования, не улучшение боёв.

Go replay обеих models завершён на6987/6531contexts с source hashes: aim RMSE совпадает с CUDA до0.00001°. Численный перенос inference проверен; игровые результаты ещё ожидаются.

## Итог paired native оценки joint labels без coupling loss

Оценка завершена160/160 usable captures, source unchanged и native provenance подтверждены, generated fixture pairs совпали. Validation28 повторный diagnostic cohort. Parent FireBC47/80 побед,30смертей,52убийства,3486incoming health damage. Joint-label BC30/80,45смертей,30убийств,4900incoming. Branch не принимается; улучшение offline labels не перенеслось в closed loop. Main остаётся FireBC. Root `workspace/artifacts/coordinated-eval-v1-20261007`.

## Явная связь aim/movement в CUDA loss

После указания пользователя Go replay/model checks больше не запускаются. Python CUDA module `scripts/combat_control_coupling.py` вычисляет continuous planar world-input из predicted forward/side и final yaw/pitch. Новый differentiable MSE сравнивает его с query target world-input; gradients проходят одновременно в aim и movement. Native clamp pitch±89/pitch/3, исходные sin/cos view angles из features6..9; command rounding/collision/water не моделируются. Это явная связь в обучении при неизменном action contract, не новая runtime correction. CUDA basis/gradient checks4cases прошли, включая ненулевые gradients всех4coordinates.

Config `scripts/scenarios/combat-sequence-query-coupled-v1.json`, world_input_weight1:150CUDAepochs от того же parent. Validation aim33.089→27.917°, movement0.273392→0.263204; world-input RMSE0.003973→0.103847, то есть direction objective ухудшился. Отдельный exploratory weight16 из `coupled-strong-config.json` проверен также150epochs с тем же init/data/seed. Это tuning по validation, не final test.

Weight16 validation: aim33.089→31.093°, movement0.273392→0.267093, world-input0.003973→0.065475. Оба coupled checkpoints сохраняют exact fire/vertical/weapon/encoder parameters; PPO counters29/279. Игровая superiority coupled models не проверена.
