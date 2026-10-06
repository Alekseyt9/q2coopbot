# Remaining-Parasite geometry v4

06.10.2026, протокол до результатов.

Отдельный фактор: заменить fresh Solo средний эпизод Mixed/Solo/Mixed
на synthetic remaining-far. Self [-48,16,24], Parasite [200,-224,24],
дистанция около345; HP бота100, stock Parasite175, Blaster. Позиция выбрана
из диапазона предыдущих training post-Gunner observations. Это новый
старт без Gunner, не engine snapshot и не replay evaluation состояния;
оружейная фаза/скорость начинаются заново. Дистанция и локальная геометрия
меняются совместно; их отдельные причинные эффекты здесь не различаются.
HP/reward/архитектура не меняются.

Control: Mixed/standard Solo/Mixed; candidate: Mixed/remaining-far/Mixed.
По48 training episodes, seeds24700–24747,4 workers×3 episodes×4 updates,
timescale2; обучение и optimizer replay только CUDA. Same original
obstacle parent53/524, composition bank SHA
2574125b4f7812572a809df174263756f0df18409eef9529bba5b4ed731a6a6b,
constant anchor/bank1, maneuver-v4 reward, horizon300/release100,
MLP810→64→64→8. Каждая ветка начинает с одинакового parent Adam/RNG.

Сначала native pilot24600–24603 подтверждает placement/health/enemy reset;
pilot не входит в PPO или retention bank. Frozen fresh Mixed24800–24803,
Solo24900–24903, remaining-far25000–25003, known20800/21300/21700/22100.
Final update4 выбран заранее, eval не входит в fit. Считать классы kills,
uninterrupted full wins отдельно по задаче и per-seed regressions.
Новый fixture selector подтверждается manifest, config и observed reset.
No live promotion без preservation, без jump bonus или forced rules.

Root: workspace/artifacts/combat-remaining-far-v4-20261006.

## Результаты

208 valid captures/36 batches:96 training и112 evaluation; ещё4 valid native pilot captures отдельно. Все4 instances x2, каждому episode отдельный подтверждённый seed. 75 Python tests и4 negative fixture/InitialBatch guards прошли. Source/native/seed/fixture/reset/ownership/reward/exact CUDA actor+Adam+RNG replay/native re-export/bank KL/closure и independent GAE audits прошли. Обе ветки4 CUDA updates/40 accepted actor steps, cumulative57/564, direction fallback0.

| Задача | Parent | Control standard Solo | Candidate remaining-far |
| --- | ---: | ---: | ---: |
| mixed (4 seeds) | 2 | 3 | 1 |
| solo (4 seeds) | 4 | 4 | 4 |
| remaining-far (4 seeds) | 4 | 4 | 4 |
| known-occlusion (4 seeds) | 3 | 3 | 4 |
| previous-mixed (4 seeds) | 3 | 1 | 3 |
| previous-fresh (4 seeds) | 3 | 2 | 1 |
| obstacle-holdout (4 seeds) | 3 | 1 | 2 |
| Всего20 Mixed |14|10|11|

Parent на20 Mixed:29 kills/3 deaths/incoming1472, control26/8/1489, candidate25/4/1457. Повторные parent prefixes и полные outcomes совпали, но их копии не являются независимыми наблюдениями. Обе модели не проходят preservation: original obstacle reference и live policy сохранены. Это один fresh Mixed quartet и четыре known regression quartets; обобщение не доказано.

| Training задача | Эпизоды | Gunner kills | Parasite kills | Смерти | Непрерывные полные победы |
| --- | ---: | ---: | ---: | ---: | ---: |
| control mixed | 32 | 4 | 2 | 9 | 2 |
| control solo | 16 | 0 | 11 | 0 | 11 |
| curriculum mixed | 32 | 4 | 2 | 12 | 2 |
| curriculum solo | 16 | 0 | 13 | 1 | 13 |

control: 11484 PPO rows; lost prior wins 24802/20802/21300/21301/21303/21703/22100/22101; new deaths 24802/20802/21300/21301/21303/21701/21703.
Training jump proposed/sent/ground-to-air: 12/11/11; deterministic evaluation: 0/0/0. Movement не доказывает targeted evasion; jump bonus/forced rules отсутствуют.
Final weights SHA `e88d3d3ef0bddad490d89d3f9ca64e982868d0e98c154b9e6d99f441553b7e6f`, checkpoint SHA `28836897838bde44a125a8005bdb8b67a0d2426792599d48236efd7f1bfb8e58`.

curriculum: 10173 PPO rows; lost prior wins 24802/21700/21703/22101; new deaths 24800/24802/21302.
Training jump proposed/sent/ground-to-air: 18/18/18; deterministic evaluation: 0/0/0. Movement не доказывает targeted evasion; jump bonus/forced rules отсутствуют.
Final weights SHA `0c8477540659a14f64cf37280799ce124b4c20609d554c736d3bfe5e740ee462`, checkpoint SHA `60552bfdef18bbf649e5a7a956e146d0ba1f473f1933689e56dd4c21385d24fe`.

Independent critic check: все 21657 current values и 21445 interior next values сверены Torch CPU no-grad против Go при frozen features; max current abs error 1.43e-06, max next 1.43e-06. 190 nonterminal tail next values и независимое encoding features не пересчитаны. Это диагностика без градиентов; обучение/replay на CUDA. Generated critic auditor сначала получил dtype Long/Float mismatch для all-zero next values; исправлен явный float32, данные/weights/tolerance не менялись.

Обе training ветки дали только2/32 Mixed full successes и2 Parasite kills; дальний Solo13/16 против standard11/16 не доказывает достаточный перенос. Изменение HP остаётся отдельным непроверенным фактором. Следующий приоритет после замечания пользователя — фиксированный Machinegun с конечным запасом патронов, отдельная оценка skill3; выбор оружия сетью и GRU не добавлять одновременно.

Артефакты: completion-status.json, comparison-audit.json, bank-audit.json, stochastic-audit.json, movement-audit.json, regression-audit.json, critic-audit.json, closure-contract-audit.json, result-summary.json и per-arm audits.
Source fingerprint `09a15563606d6228ccc3a2bb510195caa3c82df5d6b2c729c7dfe550b831c77f`, native `99ca5b165e8c1e2683c0ae8ecc5800385c75cedfd0895cf3dd7ebb928bb9a08c`.
