# Controlled PPO retention experiment

06.10.2026. После [исправленного joint-threat PPO](learned_combat_barrel_ppo_v4.md),
который ухудшил Mixed, проверяется временное сохранение исходного actor.

## Протокол до результата

Оба arm начинают с barrel parent weights
`91697086e5c57d21f85ad393e0674e044ccd08fb54f77e91485e98ee3df7fd6b`,
checkpoint `6a46f68ed40db4fbd2f9b5a5e1b6b34db9c814c4188eb00ec987c7b9c1f32151`.
Пустой Adam/zero critic output,53 updates/524 actor steps. Config/reward v4,
архитектура810→64→64→8, наблюдения, std и runtime guards неизменны.

Control — исправленный обычный PPO. Retention — тот же PPO +
`beta * KL(anchor || current policy)` на текущих fresh training states.
Analytic KL включает4 pre-tanh Normal movement/aim dimensions, Bernoulli
attack и Categorical vertical. Anchor — замороженный исходный learned actor,
не hidden server state, не rules и не runtime correction. Teacher output
не имеет gradients. Используются только states данного текущего rollout.

Beta заранее1,2/3,1/3,0 в четырёх updates; после четвёртого0 сохраняется.
Schedule/anchor SHA/start update записаны в checkpoint; resume требует тот
же anchor и schedule, silent disable/replacement запрещены. Backtracking
acceptance оценивает полную PPO+retention loss и прежний behavior KL≤0.01.
Это loss regularization, не гарантия сохранения поведения или приёмка.

В каждом arm4 fresh stochastic Mixed batch20100–20115,300 frames,
Blaster/stock health/release0, synchronous,4 independent instances x2.
У arms одинаковые seeds для сравнения, но каждый имеет свои captures/rollouts
с своей behavior policy, и ни один arm не использует transitions другого.
Код CUDA-only, native behavior proofs и exact Adam rollback уже исправлены.

Новые deterministic Mixed20200–20203 для anchor/control/retention,
Solo20300–20303 (175HP/release100);4 instances x2. Anchor Mixed повторён
существующим runner в каждом arm: это повтор одной четвёрки, не8 независимых
baseline episodes. Primary — полные победы без смерти; damage by class/MOD,
guards, solo kills/handoff и actual consumed kill reward rows отдельно.
Сравнение control/retention позволяет оценить эффект regularization на
одинаковом бюджете; четыре seeds не дают общей приёмки.

Fixed final fourth update каждого arm, без best checkpoint selection, eval
training или live promotion.58 focused Python tests прошли, включая CUDA
KL gradients всех heads, annealing/resume pinning и immutable Adam rollback.
PowerShell runner parser check прошёл. Root
`workspace/artifacts/combat-ppo-retention-controlled-v4-20261006`.

## Результат и независимые проверки

Оба цикла и все оценки завершены:32 training captures,16 Mixed eval и12
Solo eval, всего60 captures с подтверждёнными seeds, native dispatch и
provenance. Обучение/benchmark только CUDA. Live policy не заменена.

| Ветка | Fresh rows | Принятые actor steps по updates | Итог updates / actor steps |
|---|---:|---|---|
| Control | 2421 | 8 / 10 / 8 / 5 | 57 / 555 |
| Retention | 2653 | 10 / 10 / 10 / 0 | 57 / 554 |

В четвёртом retention update beta=0; backtracking не принял ни одного
actor шага,40 critic steps выполнены.40 actor steps — максимальный бюджет,
не фактическое число. Adam counters совпали с принятыми шагами; critic
counters40/80/120/160. В checkpoint сохранены только свои свежие rollouts,
исходные53 и четыре новых, без повторов или transitions другой ветки.

Behavior KL control:0.009994/0.008220/0.009993/0.009980;
retention:0.005994/0.008637/0.009776/2.05e-9. Независимый no-grad пересчёт
anchor KL на текущих training states совпал с trainer report. Final KL
control0.040234 и retention0.017486 измерены на разных states своих веток;
их отношение не является чистым измерением причинного эффекта удержания.

Native re-export всех8 rollout побайтно совпал с исходными
`rollout.jsonl`/`reference.jsonl`. Все2421+2653 consumed reward scores
пересчитаны независимо: max error6.66e-16/8.88e-16. Все4 control и7 retention
training kill reward rows вошли в PPO, bonus+5. Проверены export/checkpoint
tensor parity, lineage, anchor/schedule SHA, CUDA-only benchmarks и
неизменность actor/std/value в deterministic evaluation copies.

Final control weights
`a818a5acb172c167a47f08d77199a87c0752cad64e67787e15cf19940ce9c838`,
checkpoint `552a233d334c04f94e19f1784cbcb7e90c8bafa97b170978abe66bec3ea8e575`.
Final retention weights
`3654479a5fce50f853ed9b46a6b5663e69b9af7bfe148b3f6822e56ff9c511c5`,
checkpoint `d342f8112281d9795ac2a3918c765c4ae5b4ab266dda1ba6b60701b037a069f1`.

| Mixed20200–20203 | Полные победы без смерти | Kills | Deaths | Incoming | Grenade incoming |
|---|---:|---:|---:|---:|---:|
| Anchor, первый запуск | 4/4 | 8 | 0 | 288 | 33 |
| Control | 3/4 | 8 | 1 | 371 | 41 |
| Anchor, повтор | 3/4 | 7 | 0 | 281 | 113 |
| Retention | 3/4 | 7 | 1 | 239 | 73 |

Barrel damage0 во всех вариантах. Control20202 убивает второго монстра
в frame277 и умирает в278 без предшествующего handoff; это не смерть после
передачи сопровождению. Retention20200 убивает одного и умирает, также без
handoff. Retention не улучшает число полных побед против своего повторного
anchor и добавляет смерть. На этой оценке причинный выигрыш не установлен.

| Solo20300–20303,175 HP/release100 | Kills | Deaths | Incoming | Kill frame во всех4 |
|---|---:|---:|---:|---:|
| Anchor | 4 | 0 | 116 | 183 |
| Control | 4 | 0 | 173 | 184 |
| Retention | 4 | 0 | 92 | 178 |

Solo retention сохраняет убийство и снижает полученный урон на этой
четвёрке; все kills в provider execution windows, handoff после убийства.
Это локальный результат, не общая приёмка mixed combat.

## Ограничения сравнения и следующий шаг

Даже первый training batch до обучения при одинаковых seeds и исходной
политике дал661 против764 rows. Независимая проверка первых observations
подтвердила **одинаковые feature vectors и первые sampled latents** на всех
четырёх seeds. Frames первых команд32–34 отличаются. Следовательно,
равенство входа первого решения подтверждено, равенство полной динамики
движка/AI не подтверждено; предположение об AI timers остаётся гипотезой.
Mixed пока запрещает fixed release100 в обоих runners. Равенство source
fingerprints и seeds не заменяет проверку полного начального состояния.

Повтор deterministic anchor изменил итог4/4→3/4 на тех же seeds. В
повторном anchor20201 первая передача управления происходит в frame134,
затем многократные возвраты provider/rules; единственный kill в192 выполнен
provider, но весь бой не является непрерывным learned combat. Старый strict
`audit_kill_ownership.py` закономерно отклонил условие «все kills раньше
первого handoff» для этого эпизода. Проверка не объявлена успешной.
Отдельный `handoff-audit.json` подтверждает native-exclusive provider kill
windows во всех28 eval episodes и явно сохраняет `all_clean_handoff=false`.
Rules frames этого эпизода63. Остальные kills precede first handoff; paired
conditions/guards audits и Solo strict ownership проходят. Этот смешанный
по управлению baseline нельзя использовать как самостоятельную боевую
приёмку сети. Существующий общий harness gameplay acceptance также не
пройден: loaded Shotgun switch не проверяется Blaster-only опытом.

Следующий приоритет: фиксированный Mixed barrier/reset с диагностикой
наблюдаемого и полного offline начального состояния; воспроизвести одну
замороженную policy дважды на новых seeds и разделить непрерывный learned
бой от visibility handoff. Лишь после этого повторять сравнительное PPO
обучение на GPU, не меняя одновременно reward, horizon и architecture.
GRU/attention/выбор оружия этим опытом не реализованы.

Evidence в root: `training-audit.json`, `evaluation-audit.json`,
`initial-feature-repeat-audit.json`, `handoff-audit.json`,
`control/shaping-audit.json`, `retention/shaping-audit.json`,
`control/native-reexport-audit.json`, `retention/native-reexport-audit.json`,
`comparisons/*/paired-audit.json` и `guard-audit.json`.
