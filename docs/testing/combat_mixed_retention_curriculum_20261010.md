# Больше сложных групповых боёв, сохранение остальных навыков

Предыдущий warm v8/v9 A/B закрыт. Штраф v9 ухудшил66→63/80 побед,
9→12 смертей и49.0→43.6% MG hit fraction. Он не принят.
Разбор закрытых development traces в `combat_native_waste_family_review_20261010.md`
указывает на mixed Parasite/Gunner и две base1 Blaster сцены.

Новый опыт — один свежий on-policy PPO update из сильного v8 update5.
Actor/value/std, critic и Adam продолжаются; reward v8, конфигурация PPO,
архитектура attention64 и штатная физика не меняются. Обучение и
численное восстановление данных/checkpoint только CUDA. Runtime Go.

## Данные

Нативный компилятор требует минимум4 эпизода на семейство, кратно4.
Первоначальная идея сохранить80 боёв с2 retention сценами на семейство
не прошла эту проверку и не была запущена. Итоговый набор160:

| Семейства | Боев |
|---|---:|
| Parasite/Gunner Blaster |32|
| Parasite/Gunner Machinegun |32|
| base1 site01 Blaster |16|
| base1 site04 Blaster |16|
| Остальные16 семейств, по4 |64|

Все20 семейств сохранены. Генерируются новые train conditions с offset348:
mixed348..379, две проблемные сцены348..363, остальные348..351 внутри
собственных зарегистрированных train splits. Validation/test traces не
используются для PPO. Global unseen-seed claim отсутствует.
Распределение закреплено в
`scripts/scenarios/combat-training/mixed-retention-allocation-v1.json`.
План smoke v2 проверен `q2episode --verify-plan`:20 семейств/160 эпизодов.
Неудачный smoke v1 сcount2 сохранен как диагностика; игровые captures
на нем не создавались. В сравнении с прежними80-бойными опытами меняются
и распределение, и объем данных; отдельное causal преимущество распределения
этот опыт не доказывает.

`compile_weighted_combat_plan.py` собирает native cohorts одинакового размера
и вновь проверяет весь merged plan нативным компилятором. Gameplay/geometry
генерация остается в Go. `run_combat_ppo_series.py --family-counts` сохраняет
существующий default80; для явной allocation принимает ее размер, stride
между последовательными раундами равен максимальному count, чтобы seed
блоки семейств не пересекались внутри серии. Здесь planned rounds=1.

## Запущенные этапы

Training root: `workspace/artifacts/mixed-retention-ppo-series-v1-20261010`.
Parent weights SHA
`211c2d6d8062620a983800b81b18320452aaba4e0a58ebd80c10270ecf2bd27b`.
Для текущих Go/native source receipts использована последняя закрытая
400-бойная `native-waste-parent-rules-eval-v1-20261010`; старый guard
reference не соответствует последующим parser/reward исходникам и был
отклонен до native dispatch. Проверки SHA/ownership/diagnostics сохранены.

Evaluation root: `workspace/artifacts/mixed-retention-ppo-eval-v1-20261010`.
Очередь держит фактический OS training process handle и ждет его завершения.
После закрытия CUDA update и checkpoint seals выполнит400 боёв:
parent/final × RNG20261011/20261012 ×80 common conditions плюс rules80.
Это80 уникальных условий, повторяемых разными RNG arms, не400 независимых
сцен. Explicit validation offset28 исправляет историческое planned32,
которое выходит за зарегистрированный32-эпизодный validation split.
Это повторная development оценка, test остается отдельным.

Один общий16-slot refill pool, timescale2. Второй native pool одновременно
не запускается; evaluation пока ждет training. Terminal streams LZX с
сохранением SHA, immutable assets общие. Человеческую сессию не перезапускать.
Автоматического promotion нет. Перед приемкой нужны результаты всех
семейств:победы/смерти, native попадания и расход, движение/задержки,
отсутствие деградации ранее освоенных сцен. На момент записи идет сбор
training; новых весов и результатов качества еще нет.

Дополнительная read-only очередь `finalize_combat_series_quality.py`
держит OS handle фактического evaluation process. После закрытия core
диагностики проверит SHA quality/ownership/shot/storage отчетов и выполнит
adaptation comparison, native Machinegun waste audit и Blaster credit
audit сmin-frame100. Результат запечатывается отдельно в
`analysis-progress.json`, затем дополнительные SHA добавляются к
законченному evaluation progress. Это не второй native pool.
Live проверка этого нового closing этапа еще предстоит.

Training pool закрыт:160/160 jobs, ошибок0, `source_unchanged=true`.
Receipt SHA
`e872f44ef5936056713353b60e482bdb5b840d797dfa1e5c6dd01ff87e000024`.
Отдельно проверено точное совпадение всех terminal tuples
`(plan_index,task_index,seed,mode)` с weighted plan, без дублей и пропусков.
Capture execution перешел к `cuda_processing`, processing stage
`native-export` на20 корпусов. Root series progress пока отражает старый
`collecting`; актуальная промежуточная фаза — capture execution/processing
progress. CUDA update и400-бойная оценка еще не завершены.

## Обучение закрыто, оценка началась

160 own-policy эпизодов обработаны в20 CUDA corpora; PPO update6 закрыт:
10953 eligible transitions,10 actor steps,55 total actor steps.
Initial resume checkpoint SHA совпадает с сильным v8 parent.
Final weights SHA
`59070295eb47422b873560e9aeb4b689b04009eb46a9e95845f606d7b3b1fe75`;
checkpoint SHA
`15990f0398e627464e884fb125c58b85136c7aa23aff2627d721c30061b59bda`.
CUDA checkpoint audit подтвердил точное восстановление actor/value/std
и optimizer state. Отдельно перечитаны completion seals и SHA.
Series progress теперь `complete`, episodes160, parent optimizer preserved.
Это завершение обучения, не доказательство качества поведения.

Оценка автоматически перешла к400 native боям:147 закрытых jobs без ошибок
на момент записи. Дополнительная диагностика держит handle evaluation
и ждет завершения. Новые веса не публикуются и не заменяют live модель.

Дополнительно разобран закрытый merged rollout: mixed Blaster3362 rows,
mixed Machinegun2412, base1 site01 Blaster1428, site04 Blaster1359.
Mixed занимает5774/10953 rows (52.7%), четыре усиленных семейства
8561/10953 (78.2%), остальные2392 (21.8%). По эпизодам это40%,60%,40%
соответственно: длительность боя меняет долю переходов относительно
числа эпизодов. Это фактический состав данных, не доказательство ошибки
PPO или причины деградации. При выборе следующего curriculum учитывать
обе доли и результаты остальных семейств, а не только число запусков.

## Core400-бойный результат (диагностика пока не закрыта)

Все400 native jobs завершились без ошибок. Core quality SHA
`4382fcd4428fe709c4175642d54ac1050d87e517ec77397a7b89ab746a95822e`.
Parent выигрыши70/80 и67/80, смерти9/7; after68/80 и68/80, смерти8/6.
Итого137→136 побед,16→14 смертей. Received damage17.1375/14.9625
против15.1625/12.5625; outgoing79.275/75.7375 против79.775/77.35.
Rules65/80,9 смертей. Это80 уникальных условий, два RNG повторения для
каждой learned модели; итог160 не следует считать160 независимыми сценами.

По двум RNG arms mixed Blaster0/8→0/8; mixed MG3/8→2/8.
base1 site01 Blaster1/8→5/8, site04 Blaster7/8→5/8;
site01 MG7/8→6/8, site04 MG8/8→7/8.
Больше mixed training данных пока не улучшило эти групповые development
бои. Снижение смертей и урона сопровождается потерями побед в других
сценах. Преимущество над parent и promotion не установлены; final native
diagnostics еще выполняются.

## Итоговая приемка

Evaluation и post-analysis завершились exit0. Проверены9 SHA отчетов:
quality, diagnostics acceptance, ownership, first shot, behavior, storage,
adaptation comparison, Machinegun waste и Blaster credit. Ownership clean.
Post-analysis прошел живую интеграционную проверку после завершения
реального evaluation OS process; не запускал дополнительных native боёв.
Comparison SHA
`e4e4c17064d67bf090b29d0104849b1d3bcd1497abb2096e611305abe2640c2e`.

Победы137→136/160, смерти16→14, средний received damage16.05→13.86.
MG pooled accuracy721/1416 (50.9%)→725/1389 (52.2%);
Blaster464/885 (52.4%)→460/920 (50.0%). RNG arm a gained3/lost5 wins,
arm b gained2/lost1. Преимущество по победам над parent не установлено,
mixed семьи не улучшились, promotion отсутствует. Более низкие смерти
и received damage сохраняются как полезный побочный результат ветки.

Final approximate KL0.00485186 при target0.005: простой рост actor_steps
не означает существенно больше допустимого изменения политики.
Trainer report имеет retention_weight=0 и bank_weight=0. Следующий опыт
должен отдельно проверить мягкое сохранение прежней политики на собственных
training состояниях освоенных семейств, сохранив награду/архитектуру и
явную привязку к parent. Не использовать validation states как retention
bank. Это гипотеза против забывания навыков, не установленная причина
регрессий и не уже выполненное обучение.
