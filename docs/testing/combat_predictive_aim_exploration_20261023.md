# Реальные TRAIN-метки при изменении углов

23.10.2026. После learned aim-ranker pilot: classifier стал точнее, но выбор другого угла не улучшил бой. Проверяется гипотеза нехватки опыта с такими действиями; результат заранее неизвестен.

## Режим сбора

`geometry_aim_explore_v1` — отдельный predictive-only sampler. При наблюдаемом MG и выбранной доступной bbox половина подходящих кадров сохраняет sampled actor aim; половина получает равномерный выбор одного из 24 ненулевых offsets сетки0/−5/−2.5/+2.5/+5° по yaw/pitch. Границы coarse/fine сохраняются. Другие решения по-прежнему принимает source policy.

Собственный воспроизводимый SplitMix-style frame hash зависит от sampling seed/frame/life; actor RNG не расходуется. Проверены повторяемость и распределение всех 25 назначений на 10000 кадров, eligibility и запрет стандартного PPO. Не используется младший бит линейного счётчика: gate не должен совпадать с чётностью оружейной фазы.

Source: `combat-conditional-fire-zero-v1-20261022`. Source actor/value/std/Adam/RNG сохранены и сверены на CUDA. Последний conditional-fire layer равен нулю; поэтому выбор смещения не меняет вероятность fire в том же кадре. Это режим получения данных, не кандидат на production/default. Мировая траектория и дальнейшие наблюдения могут измениться из-за поворота взгляда.

## Сбор и формат

`collect_combat_predictive_aim.py` создаёт свежий TRAIN plan из реестра сцен, проверяет уникальность seed и отсутствие пересечения с историческими планами. Используется существующий episode pool, MaxInstances16, скорость ×2. Начальный smoke — восемь боёв base3/ware2, ordinary/recovery на двух точках. Validation/test не собираются.

Записываются условия генерации, source weights hash, sampler identity, конкретные seeds и полный frozen plan в `recipe.json`. После терминального native pool проверяются capture provenance, ownership и execution. Далее `build_combat_predictive_aim_dataset_cuda.py` читает native capture datasets, не создавая обычный PPO rollout и не делая neural replay на Go.

Проверяются first-life alive provider frames после canonical network release, native generation, matched exclusive windows без recovery. Каждое назначение сверяется с независимым Python вычислением frame hash и фактическими углами. В metadata различаются: отсутствие eligibility, baseline assignment, назначенное вмешательство и реально изменившийся после clipping угол.

Метки — настоящие MG events и damage из server telemetry, связанные с командным окном. Входы — только causal observation и фактически выбранный sample. Pure Go CLI пакует185 признаков; CUDA строит geometry29. Native truth не становится входом predictor. Большие исходные JSONL/PAK/AAS не копируются; вспомогательные exports сжимаются без изменения bytes, dataset сохраняется binary, members — JSONL.GZ.

LP/value/advantage targets в predictive dataset не экспортируются. `ppo_ready=false`, `purpose=shot_predictive_train_only`. Такие действия нельзя представлять как обычные Gaussian PPO samples. Source registry поколения боёв остаётся доступен для свежих запусков других моделей; сами off-policy записи пригодны только для явно поддерживающего их supervised/predictive обучения.

## Приёмка smoke

До расширения сбора необходимо получить валидные native captures, фактическую активацию sampler, реальные baseline и intervened shots, корректные actual-action/native labels и compact dataset с SHA provenance. Smoke с одним seed на сцену может не покрывать обе стороны seed-mod4 fold — predictor на таком наборе не обучается.

После успешной проверки — новый TRAIN пакет с несколькими свежими seeds на сцену, достаточный для whole-battle seed/site development folds. Только затем сравнивать predictors, обученные на исходном и расширенном опыте, и повторять native control pilot. Новый режим сам по себе не должен превосходить rules: его качество здесь — достоверность и покрытие учебных данных, не победность.

## Smoke завершён

`combat-predictive-aim-smoke-v1-20261023/complete.json`: восемь native TRAIN боёв, процесс exit0. Все captures, ownership и execution проверки прошли. Получено **242 actual MG shots**: 100 с назначенным вмешательством, все 100 реально изменили угол; 97 eligible baseline shots и 45 shots вне eligibility. Outcomes:120 live monster damage,116 solid,6 corpse. Это показатели сбора опыта, не оценка улучшенного бота.

Сверены 276 eligible frame assignments, все 25 candidate categories встретились:136 baseline frames и140 nonbaseline. Шесть frames с изменённым applied aim сверх квантования и один unmatched/recovered provider frame не использованы для shot labels; исключения отражены в report. Ни fake LP, ни PPO targets не экспортировались.

Binary dataset занимает **214089 bytes** (около209 KiB), metadata сжата. `combat-predictive-aim-source-registry-v1-20261023/index.json` хранит sealed source recipe, модель/сэмплер, условия/seed и dataset hashes. `register_combat_predictive_source.py` отказывается регистрировать повторные engine seeds и источники, не помеченные predictive-only TRAIN. Рецепты можно переиспользовать для генерации нового опыта другими подходящими моделями; назначение записи не превращается от этого в on-policy PPO.

Следующий пакет заранее ограничен **32 TRAIN боями**: четыре новых seed на каждое из восьми условий, один пул16 ×2. Predictor пока не дообучается. После закрытия: проверить sampler/labels и разбиения, зарегистрировать dataset; затем сравнить исходное обучение predictor, новые interventions и их смесь на whole-battle seed/site folds. Smoke и новый пакет входят только в TRAIN, новые validation/test для качества боя остаются отдельными.

## Полный первый пакет и сравнение обучения

`combat-predictive-aim-corpus-v1-20261023/complete.json`: 32 native TRAIN cases, exit0. Получено1015shots:479effective perturbations,460eligible baseline,76outsideeligibility. 1291 sampler frames проверены. Вместе со smoke в реестре **40 случаев /1257shots /579interventions /557eligible baseline**. Episode/seed groups не пересекаются со старыми9248shots. Source policy/default/reward неизменны.

Заранее фиксирован `combat-interventional-predictor-probe-v1-20261023/protocol.json`: один Linear214 (215 параметров),200steps,LR0.001,weight_decay0.0001,seed20261022. Четыре режима: old-only, new-only, pooled без перевзвешивания, balanced с половиной loss от каждого домена. Train-only normalization соответствует весам loss. Eight CUDA fits+metrics завершены примерно за4,15с после загрузки; новые бои в этот замер не входят.

Whole-battle seed/site folds сохранены. New seed-heldout содержит190shots,98intervened; site02-heldout609shots,258intervened. Это development внутри TRAIN, не итоговый campaign/test.

| Training data | Seed: old heldout loss | Seed: new intervened loss | Site02: old loss | Site02: new intervened loss |
|---|---:|---:|---:|---:|
| Old-only |0.4625|0.4690|0.4287|0.5370|
| New-only |0.4753|0.4327|0.6967|1.0395|
| Pooled |**0.4611**|0.4583|**0.4214**|**0.5202**|
| Balanced |0.4632|0.4444|0.4552|0.5591|

Предварительное правило допуска: new-intervened loss должен снизиться на обеих folds; old loss не должен вырасти более чем на0.02absolute. Принят только pooled. New-only/balanced не проходят geometry fold; увеличение веса небольшого нового корпуса не стало универсальным улучшением.

`combat-interventional-geometry-predictor-v1-20261023` — pooled refit на всех **10505 TRAIN shots /328source cases** (старые288 и новые40). Веса экспортированы, logits проверены на CUDA. `training-datasets.json` связывает существующие binary sources вместо их копирования; поле predictor.dataset_sha256 связывает этот composite manifest. Новый inference bundle: `combat-interventional-aim-ranker-v1-20261023`; source actor/value/std/Adam/RNG сохранены точно. Допуск predictor к этому этапу не означает улучшения боя.

Следующий native screen заранее ограничен32battles: start / old-ranker / new-ranker / rules, по8fresh validation условий, offset766, один пул16 ×2. Восемь source cases дополнительно проходят обычный CUDA replay; оба rankers — отдельные live CUDA score/choice audits. Пока default/promotion не меняются.

## Pilot после нового опыта закрыт

`combat-interventional-ranker-screen-v1-20261023/complete.json`:32native/8sourceCUDA,exit0. Отдельные CUDA audits подтвердили все25scores/выбранныеуглы: old-ranker268eligibleframes/265changes, new-ranker283/278. Новые PPO updates не выполнялись.

| Вариант | Победы | Убийства | Нанесённый / полученный урон | Смерти |
|---|---:|---:|---:|---:|
| Start |3/8|10|1978 / 497|4|
| Old ranker |5/8|13|2310 / 469|3|
| New pooled ranker |5/8|13|2310 / 501|3|
| Rules |4/8|10|2003 / 457|4|

Оба rankers против start получили2победы без потерь; new против old не изменил исходы. Против rules у new3полученные и2потерянные победы. Малый положительный pilot не отменяет предыдущий отрицательный опыт на других seeds и не доказывает превосходство над rules.

Actual MG damage fractions: source145/194 (74,74%),old140/203 (68,97%),new152/202 (75,25%). Solid contacts47/57/35,corpse0/6/15,targetless7/16/2. Новый опыт улучшил этот показатель относительно old, но выросла стрельба по corpses и полученный урон. Stationary-clear13,74%/22,11%/24,65%;large-yaw8,57%/8,16%/7,94%. Компонент не устранил неподвижность и лишние действия.

Следующая заранее ограниченная проверка — **96 native cases /24source CUDA**, по24fresh validation боя для start/old-ranker/new-ranker/rules (три новых seed на каждое из восьми условий), пул16 ×2. Оба rankers дополнительно проверяются отдельным live CUDA audit. Это проверка устойчивости и эффекта нового опыта, а не новый TRAIN или проверка неизвестных карт; automatic promotion отсутствует.


## Confirmation закрыта: 96 native / 24 source CUDA

`combat-interventional-ranker-confirmation-v2-20261023/decision.json`: все игровые прогоны и оба отдельных live CUDA audits завершены. Первоначальная подготовка offset768 была отвергнута до игр: validation range заканчивается на768. Повторная spec заранее использовала свободные offsets24..26, по три seed на восемь условий,16инстансов ×2. Это validation, без обучения по этим исходам.

| Вариант | Победы | Убийства | Нанесено / получено | Смерти | Actual MG: живой урон / выстрелы |
|---|---:|---:|---:|---:|---:|
| Start |9/24|30|5798 /1551|11|406/735 (55,24%)|
| Old ranker |11/24|37|6617 /1439|8|481/762 (63,12%)|
| New pooled ranker |11/24|35|6251 /1518|8|485/797 (60,85%)|
| Rules |16/24|40|7094 /1279|7|MG не сравнивался этим отчётом|

New против old:2полученные/2потерянные победы; против start5/3, против rules3/8. Solid contacts:308/264/296; corpses21/17/14 для start/old/new. Первая положительная восьмёрка не дала подтверждения преимуществу нового опыта. Оба rankers уступают rules; default и опубликованные веса не меняются.

Live CUDA подтвердил1057ranked/1041changed frames у old и1033/1019 у new, все25кандидатов и выбранные углы. Это проверка реализации, не доказательство правильности контрфактической оценки. Standard PPO replay для rankers не заявляется.

Stationary-clear:16,97%/12,42%/14,93%; large-yaw3,81%/4,57%/4,87%. Утверждать по предыдущему pilot, что ranker систематически усиливает неподвижность, нельзя. Post-recoil угловая ошибка к выбранной bbox:6,734°/8,502°/7,633°; положительный pitch error у rankers3,85°/3,28° против −0,24° у source. Угловая ошибка не равна фактической баллистической точности; native fire_lead применяет дискретизацию и spread.

Следующий ограниченный диагностический шаг: проверить на уже собранном randomized TRAIN, как часто наблюдаемый луч пересекает bbox выбранного противника, но перекрывается статической геометрией BSP. Это проверка одного возможного пробела входов. Нынешние29geometryfeatures не трассируют candidate ray по карте. Native события служат только метками; движения дверей/врагов, задержка, новый recoil и spread в статический расчёт не входят. Без подтверждения полезности не добавлять runtime firemask, не менять reward и не назначать predictor новым default.
