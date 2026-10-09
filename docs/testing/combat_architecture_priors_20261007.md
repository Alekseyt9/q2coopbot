# CUDA architecture priors, 2026-10-07

Общий teacher FireBC29, четыре fresh architectures MLP64 / temporal attention64 / temporal attention128 / GRU128, два initialization seeds. Одинаковые6987 train contexts,6531 validation contexts из20registered families. Fixed300 distillation epochs lr0.0003, actor raw MSE плюс critic MSE. Validation только измеряется; final test deferred. Это сопоставимый бюджет BC/distillation, не одинаковое совпадение начального поведения и не PPO/gameplay superiority. Teacher имеет накопленный PPO опыт; fresh arms получают его prior через одинаковый train corpus.

Все8 runs завершены на CUDA. Source hashes unchanged. CUDA export/reload и causal-prefix checks actor/critic прошли, Go replay/tests не запускались. Actor counts ниже без4log_std parameters. `initialization.pt` имеет отдельный contract combat_distilled_prior_checkpoint_v1 и не является PPO resume checkpoint. Собственные PPO updates/transitions ещё0.

| Arm | Seed | Actor parameters | Critic parameters | Validation raw RMSE | Aim agreement ° | Fire probability RMSE |
|---|---:|---:|---:|---:|---:|---:|
|mlp64|20261007|59604|58369|0.4975|6.731|0.1147|
|attention64|20261007|77544|75074|0.5014|8.547|0.1098|
|attention128|20261007|196008|191106|0.3137|7.200|0.0659|
|gru128|20261007|229032|224130|0.3884|7.380|0.0799|
|mlp64|20261008|59604|58369|0.4347|6.388|0.0963|
|attention64|20261008|77544|75074|0.4408|6.200|0.0975|
|attention128|20261008|196008|191106|0.3501|7.390|0.0686|
|gru128|20261008|229032|224130|0.4371|6.404|0.0909|

Root `workspace/artifacts/architecture-priors-v1-20261007`. Python source snapshot сохранён. Config `scripts/scenarios/combat-architecture-priors-v1.json`, trainer `scripts/prepare_combat_architecture_priors.py`. Offline imitation metrics не являются точностью попаданий или победами.

Next: own-policy fresh registered captures, CUDA PPO, pool16/x2, paired validation и rules reference. Разница качества distillation priors должна учитываться в before/after игровых оценках; выбираем architecture по outcomes и принятому опыту, не только raw RMSE.
