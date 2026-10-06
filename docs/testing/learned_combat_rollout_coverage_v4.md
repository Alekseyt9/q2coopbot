# РџСЂРѕРІРµСЂРєР° closure Рё РѕР±СЉС‘РјР° stochastic PPO rollout

06.10.2026. РџСЂРѕС‚РѕРєРѕР» Р·Р°РєСЂРµРїР»С‘РЅ РґРѕ РЅРѕРІС‹С… evaluation captures.

РџСЂРµРґС‹РґСѓС‰РёР№ count/composition closure audit РїРѕРґС‚РІРµСЂРґРёР» С‚РѕС‡РЅРѕРµ СЃРѕС…СЂР°РЅРµРЅРёРµ РІСЃРµС…
eligible provider transitions/rewards/samples: 3302/4377 rows, 5/4 kill,
9/3 death Рё12/26 handoff transitions. РЎРјРµСЂС‚СЊ Рё handoff РёРјРµСЋС‚ zero bootstrap;
GAE РїСЂРѕРІРµСЂРµРЅ РЅРµР·Р°РІРёСЃРёРјС‹Рј forward expansion, РІРєР»СЋС‡Р°СЏ gaps Рё РіСЂР°РЅРёС†С‹ seeds.
РџРѕСЃР»РµРґРЅРёР№ incomplete game_frame_limit step Р±РµР· next observation РёСЃРєР»СЋС‡С‘РЅ
(7/12); РїРѕСЃР»РµРґРЅРёР№ РїРѕР»РЅС‹Р№ С€Р°Рі СЃРѕС…СЂР°РЅСЏРµС‚ V(next), Р° recurrence РѕСЃС‚Р°РЅР°РІР»РёРІР°РµС‚СЃСЏ
РЅР° РєРѕРЅС†Рµ РґР°РЅРЅС‹С…. РќРµР·Р°РІРёСЃРёРјС‹Р№ critic recomputation РЅРµ РІС‹РїРѕР»РЅРµРЅ; zero handoff
bootstrap вЂ” С‚РµРєСѓС‰РёР№ segment objective, РЅРµ РґРѕРєР°Р·Р°С‚РµР»СЊСЃС‚РІРѕ РµРіРѕ РѕРїС‚РёРјР°Р»СЊРЅРѕСЃС‚Рё.
Reward/GAE РєРѕРЅС‚СЂР°РєС‚ РЅРµ РјРµРЅСЏРµС‚СЃСЏ.

РќРѕРІС‹Р№ РѕС‚РґРµР»СЊРЅС‹Р№ С„Р°РєС‚РѕСЂ:4Г—1 РїСЂРѕС‚РёРІ4Г—3 episodes per update (4 РёР»Рё12 episodes),
РІСЃРµРіРґР°4 simultaneous instances x2 СЃ РѕС‚РґРµР»СЊРЅС‹Рј РїРѕРґС‚РІРµСЂР¶РґС‘РЅРЅС‹Рј seed РєР°Р¶РґРѕРіРѕ
episode. РћР±Рµ РІРµС‚РєРё РЅР°С‡РёРЅР°СЋС‚СЃСЏ РѕС‚ original obstacle parent53/524, РѕРґРёРЅР°РєРѕРІС‹С…
Adam/RNG, reward maneuver-v4, horizon300, fixed release100, MLP810в†’64в†’64в†’8.
Composition bank SHA2574125b4f7812572a809df174263756f0df18409eef9529bba5b4ed731a6a6b
Рё constant anchor/bank coefficients1. РўРѕР»СЊРєРѕ4 CUDA updates,40 actor steps
РЅР° РІРµС‚РєСѓ. Final update4 РІС‹Р±СЂР°РЅ РґРѕ eval. Control train24000вЂ“24015,
expanded24000вЂ“24047; РѕР±С‰РёР№ initial seed subset24000вЂ“24003 РёР· РѕРґРёРЅР°РєРѕРІРѕРіРѕ
parent, РґР°Р»РµРµ own-policy rollouts. РџРѕР·РґРЅРёРµ overlapping seeds РЅРµ РѕР±С‰РёРµ
С‚СЂР°РµРєС‚РѕСЂРёРё: РѕС‚Р»РёС‡Р°СЋС‚СЃСЏ iteration/model. Bank РёСЃС‚РѕСЂРёС‡РµСЃРєРёР№, Р±РµР· eval states.
Validation no-grad. РђСЂС…РёС‚РµРєС‚СѓСЂР°, reward/horizon Рё РєРѕСЌС„С„РёС†РёРµРЅС‚С‹ РЅРµ РјРµРЅСЏС‚СЊ.

Fresh paired Mixed24100вЂ“24103, Solo24200вЂ“24203; known20800/21300/21700/22100.
РЎСЂР°РІРЅРёС‚СЊ native kills/deaths, uninterrupted full wins, class damage,
consumed kill transitions, stochastic completion Рё per-seed regressions.
РћРґРёРЅ fresh quartet РЅРµ РґРѕРєР°Р·С‹РІР°РµС‚ generalization. РќРёРєР°РєРѕРіРѕ live promotion.
РћР¶РёРґР°РµС‚СЃСЏ160 episode captures /32 batches (64 training,96 evaluation).
Seed range/overlap validation С‚РµРїРµСЂСЊ СѓС‡РёС‚С‹РІР°РµС‚ EpisodesPerWorker; eval
РѕСЃС‚Р°С‘С‚СЃСЏ РїРѕ1 episode РЅР° worker.75 Python tests Рё focused PAK-sharing test
РїСЂРѕС€Р»Рё РґРѕ collection. Expanded overlapping eval rejected РґРѕ СЃРѕР·РґР°РЅРёСЏ output.

Root: workspace/artifacts/combat-rollout-coverage-v4-20261006.


## Результаты

Все160 captures /32 batches завершены, 4 simultaneous instances x2;
каждый episode имеет свой подтверждённый seed. 64 training /96 evaluation
captures. Runtime cache lock не повторился. Source fingerprint:
`803a947dd5537e14fc5bb819fa2ffddf37887c2e22d9210bbdc5622cb1202a35`;
native: `99ca5b165e8c1e2683c0ae8ecc5800385c75cedfd0895cf3dd7ebb928bb9a08c`.
Неполных игровых партий нет. В служебном сравнительном auditor были
исправлены оставшиеся старые ключи count/composition на control/expanded;
после этого comparison/completion/closure audits завершились без изменений
captures, моделей или критериев.

| Метрика обучения | Control4 episodes/update | Expanded12 episodes/update |
|---|---:|---:|
| CUDA updates / actor steps | 4 /40 | 4 /40 |
| Episode captures | 16 | 48 |
| Consumed PPO rows | 3252 | 10376 |
| Empty-enemy rows | 1462 | 4456 |
| Consumed kill reward transitions | 0 | 4 |
| Death transitions | 7 | 22 |
| Handoff transitions | 17 | 53 |
| Полные stochastic успехи без handoff | 0/16 | 1/48 |
| Alive unfinished stochastic episodes | 9/16 | 25/48 |

Stochastic смерти составляют7/16 и22/48, поэтому число22 само по себе
не означает ухудшения: объём различается. Expanded native kill transitions:
Gunner24007/24020/24029, Parasite24029; все4 потреблены PPO. Только одна
успешная полная stochastic trajectory; масштаб партии увеличил покрытие,
но полноценных примеров завершения боя пока мало. Это описательный pilot,
не статистическое доказательство преимуществ конкретного budget.

| Полные deterministic победы (по4 seeds) | Parent | Control | Expanded |
|---|---:|---:|---:|
| Fresh Mixed24100–24103 | 2 | 2 | 3 |
| Fresh Solo24200–24203 | 4 | 4 | 4 |
| Known20800–20803 | 3 | 1 | 4 |
| Known21300–21303 | 3 | 3 | 3 |
| Known21700–21703 | 3 | 1 | 2 |
| Obstacle22100–22103 | 3 | 2 | 2 |
| Все20 Mixed | 14 | 9 | 14 |

Победа требует обоих monster kills в первой жизни и непрерывного
provider ownership до последнего убийства. Mixed totals control:
27 kills /11 deaths /incoming1678; expanded32 /5 /1454.
Parent full wins14 в обеих ветках; полный received/death после rules
handoffs различается (1568/4 против1637/5). Все24 parent uninterrupted
observed/action prefixes равны. Initial common seeds24000–24003
имеют одинаковые learned prefixes, parent resume и bank bytes равны;
whole initial datasets/weights не равны:506 против2560 rows. Поздние
overlapping seeds относятся к разным own-policy models, не paired data.
Поэтому total damage нельзя целиком приписывать одному фактору, а
повторные parent baselines не независимые наблюдения.

Expanded новые потери прежних successful seeds:24103/21301/21703
(смерти),22101 (alive unfinished с visibility_timeout/rules handoffs).
21701 уже был неуспешным и теперь погиб после убийства Gunner.22103
также неуспешен, смерть без убийства; control на этом сиде убивал Gunner,
но тоже погибал. Четыре expanded deaths24103/21301/21701/21703 имеют
одинаковую наблюдаемую фазу: Gunner убит, Parasite жив. На22101/22103
не завершено даже первое убийство. Результат **не принят новым reference**;
original obstacle parent и live сохранены.

Final bank mean KL train/validation control0.00608741/0.00563692,
expanded0.00357895/0.00381798; низкий KL не исключает перечисленных
regressions. Solo обе ветки4/4, incoming127→116.

## Проверки и артефакты

75 Python tests прошли (gradient/optimizer tests CUDA-only); focused PAK
pool sharing/saturation/corruption/snapshot tests прошли. Expanded seed
range overlap и несовместимый InitialBatch отклонены до создания output.
Exact CUDA actor/Adam replay, tensor/counter/RNG lineage57/564,
monotonic optimizer clocks, analytic anchor/bank KL, reward/shaping,
kill ownership, independent native byte re-export проверены. Actor
fallbacks0 в обеих ветках. Независимый closure/GAE audit подтвердил
сохранение eligible rewards/sample/terminal flags на всех13628 rows;
last incomplete horizon commands excluded8/22, remaining complete tail
bootstrap сохранён. Zero bootstrap death/handoff — существующий segment
objective; independent critic recomputation не заявляется.

Completion-status, comparison-audit, bank-audit, stochastic-audit,
closure-contract-audit.json и per-arm training/direction/consumption/
evaluation/blockage/native-reexport/shaping audits находятся в root.
Code `scripts/audit_combat_closure.py` позволяет повторять проверку
closure на последующих cycles без изменения training data.

Final weights SHA control:
`4e05f01b274b2db28734c7c9d1ca67faa427016959555009875599aac9454e67`;
expanded:
`3860f56e78d6256cc58fced000950ea8684cb94bfa9d55e04793c3f195b6203c`.
Checkpoint SHA control:
`784e63f26ee277ec31f8593f87667cc68d7d2dc5cb7fd1f58e37c5a767d970fd`;
expanded:
`b020d12333d3d4f260cf20533431d203668472edcedbf3aa995a8ff1c34d8ad7`.

## Следующий фактор

Объём4×3 episodes оставляет редкими verified Parasite-kill transitions.
Следующий bounded training-only curriculum должен отдельно увеличить
fresh stochastic опыт завершения боя с Parasite, сохраняя mixed coverage
и pinned composition bank. Сравнить ordinary Mixed с заранее объявленной
смесью Mixed/Solo/remaining-threat fixtures,4 instances x2, CUDA-only,
тот же reward/horizon/MLP/update budget. Никаких starting states из eval
трасс; новый состав training episodes закрепить до collection, freeze
fresh eval и известные per-seed regressions. Это гипотеза следующего
опыта, а не доказанный рецепт или реализованная смена curriculum.
GRU остаётся отдельным последующим сравнением памяти: не смешивать
новую архитектуру и изменение состава training data в одном опыте.
