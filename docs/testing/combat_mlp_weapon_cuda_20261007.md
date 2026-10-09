# MLP V6 weapon PPO CUDA adapter, 2026-10-07

## Реализовано

MLP PPO поддерживает8legacy outputs и20V6 outputs с combat_masked_weapon_v1. Vertical logits строго5:8; weapon logits8:20 с inventory availability mask. Shared likelihood соответствует action contract; unavailable categories отклоняются. Actor/value/likelihood checks/optimizers теперь CUDA. Initial anchor и bank inference тоже CUDA; weapon mask учитывается в KL/retention. Legacy8-action interface сохранён.

`check_mlp_weapon_cuda.py`: independent analytic uniform distribution checks для8/20 heads, Gaussian/Bernoulli/vertical/weapon log-likelihood и entropy совпали с analytic values без ошибки в FP32. Masked gradients zero, изменение unavailable logit на10000 не влияет на likelihood, недоступная категория отклоняется, self KL0. Go replay/tests не запускались. Native capture/export остаётся существующим Go harness; training и likelihood checks выполняются CUDA.

## Собственный native smoke

4own-policy train captures MLP64 seed20261007 из independent-pool smoke, Parasite/Blaster offset44. Native dispatch/reset/goal/reward/model proof повторно экспортирован перед CUDA update. Root `workspace/artifacts/architecture-priors-v1-20261007`.

Первый export attempt остановился: новый case aggregator сохранил manifests/results всехseed, но не разместил q2combat-export.exe в case root. Incomplete export сохранён. Проверены exporter SHA всех4member batches против manifest, неизменный binary скопирован в case root. Native rollout-v2 успешно экспортирован. Текущая640party не прерывалась и PS sources во время capture не менялись. После её terminal состояния требуется встроить binary publication в case aggregator и проверить аналогично все case roots перед export.

CUDA smoke: 363eligible transitions, 10accepted actor steps, updates1, old-log-prob max error0.000026941, old-value max error0.000000477, final KL0.009914408≤0.01. Weights/checkpoint/report seal сохранён.

Smoke update не используется как initial actor640party; архитектурный pilot продолжает собирать80own-policy episodes для каждого из8неизменных distilled priors. Нет нового PPO rollout sharing или promotion. Weapon learning gameplay acceptance на этом smoke не доказана.

Next: завершить full queue; repeat invalid individual captures on same frozen plan/model; publish case exporter binary with verified SHA; aggregate own-policy rollout20families per model; CUDA update всех8models, then paired validation/rules. Registered training launcher guard for MLP weapon head должен быть снят после завершения active capture; сейчас file не меняется ради source fingerprint.
