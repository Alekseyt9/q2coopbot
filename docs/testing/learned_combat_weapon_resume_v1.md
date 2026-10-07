# Weapon head: перенос Adam и первое CUDA обучение

07.10.2026. Продолжено обучение Curriculum24 после расширения V5→V6/actor20. Это проверенный переход состояния и первый weapon-head update, не приёмка полезного выбора оружия.

## Перенос

`scripts/migrate_combat_weapon_checkpoint.py` сверяет SHA source/model/checkpoint и inactive retention pins, сравнивает source module tensors с checkpoint, разрешает только ожидаемые расширения Temporal64: encoder input814→845 у actor/critic и actor head/residual outputs8→20. Старые prefixes weights и Adam moments сохраняются; новые moment cells равны0. Step counters, param_groups, config, consumed rollouts и CPU/CUDA RNG сохраняются. Активный retention отклоняется до отдельной V6 anchor/bank migration.

Два real-checkpoint tests подтвердили exact старые moment prefixes, нулевые новые моменты, счётчики/RNG и отказ при изменении старых весов, активном retention или truncation. Это CPU tensor copies без optimizer steps; обучение ниже выполнялось только CUDA. История model migrations сохраняется в следующих trainer checkpoint/report.

Artifacts: `workspace/artifacts/combat-weapon-resume-v1-20261007/`.

| Артефакт | SHA256 |
|---|---|
| Source Curriculum24 weights | `7b01b4999df979aaef85f9b7c908246edd74f5fb68c583b2083800de24931c40` |
| Source checkpoint | `536c1f04d8f50760dce29165d2e7ba1eff888f69b0a3be1b3321480b6410da16` |
| Initialized actor20 weights | `e97ae237e211975c653566e3db29a1158335996b989398a961bd0ead73919f35` |
| Migrated checkpoint | `f0e89e8caa43254f402f61c6bda6c944e55fcfce11bd25fa9d06cfa5e74ed78a` |
| Update25 weights | `07725dbbf3f462063f716948bfea3d3499667d7403723d46552621409c32fe38` |
| Update25 checkpoint | `d8f87b95f4ef28eb37ff980c3243dd7dc8c156c32f17b379304c4ff7dc6f11ec` |

Retention у Curriculum24 уже имел weights0/0. Старые V5 anchor/bank здесь не участвуют в loss и сохраняют прежние SHA; они не объявляются мигрированным V6 observation bank. Включить их позже без отдельного переноса нельзя.

## Update25

Использован свежий ранее не потреблённый stochastic on-policy batch initialized actor20: seeds44100–44103,4 native server/client instances x2, fixed MG+Blaster, stock HP, release100/max300.377 native eligible transitions/context rows. Diagnostic no-fire probe44200–44203 исключён из обучения. Reward/GAE сохранены: recoil-v5, gamma0.99/lambda0.95.

RTX5070, Torch2.10.0+cu128, CUDA-only gradients/optimization.10 accepted actor steps,40 critic steps, final joint approximate KL0.009204595<0.01. Old Go/Python logprob max error0.000046253; value0.000018224; context errors<0.000005. Existing recurrent3 tests также прошли.

| Проверка | До | После |
|---|---:|---:|
| Updates completed |24|25|
| Actor Adam steps / total_actor_steps |230|240|
| Critic Adam steps |960|1000|
| Consumed rollouts |24|25|

Weapon head weight max change0.000494832;123 new head first-moment entries стали ненулевыми. Таким образом, head действительно включён в loss и optimization. Actor77548 parameters, critic75074. Фактический update time1.433s относится только к optimization377 rows, не к сбору игровых эпизодов/оценке.

## Fresh paired smoke evaluation

8 независимых seeds44400–44407 на модель, deterministic,4 instances x2, одинаковые fixture/reward/native/source fingerprints. Parent — initialized actor20 до update, тело Curriculum24; candidate — update25. Не сравнение архитектур и не независимая окончательная приёмка.

Все16 captures, seed/reset/native command/dispatch proofs приняты, decode errors0. Обе стороны завершены, coordinator terminal. По native damage events monster→monster повреждений0; число explicit first-life weapon requests0 у обеих моделей.

| Метрика | До update | Update25 |
|---|---:|---:|
| Wins |4/8|5/8|
| First-life deaths |4|3|
| Monster kills |11|11|
| Monster health damage |2511|2237|
| Received health damage |567|554|

Новые победы44403/44407, потеря44405. Все deterministic weapon decisions сохраняли текущее оружие: улучшение общего боя нельзя приписывать полезному weapon switching. Выборка мала, live/default не переключены. Update25 — рабочий checkpoint для следующего учебного этапа, Curriculum24 остаётся подтверждённым reference.

## Дальше

Расширить существующий synchronous fixture до MG/Shotgun/Blaster с разными ammo и native reset/equip proof; разрешить соответствующий проверенный firing guard. Обучать совместное release/fire/weapon decision, избегая бесконечных перевыборов при equip delay. Собирать новые on-policy episodes на4 instances x2 и продолжать CUDA Adam от update25. Приёмка требует полезных actual equips и расхода ammo, larger held-out paired cohorts, Solo/Mixed regressions и последующего расширения оружий/монстров/геометрии/коопа. R5–R9 не закрыты.
