# Оценка кандидатов, 12.10.2026

Три файла weights.json в этом каталоге соответствуют последним control/cost/bonus reward variants одной Temporal Attention архитектуры. Исходный manifest и ZIP сохраняют состояние экспорта; веса не изменены.

Полная оценка:1456 native боёв,672 CUDA проверки,84371 rows. На каждом варианте112 одинаковых held-out fixtures с разными seeds и геометрией base3/ware2/jail1/city1; ordinary80/recovery32.

| Кандидат | Stochastic победы | Deterministic победы |
|---|---:|---:|
| control |32/112|33/112|
| cost |31/112|29/112|
| bonus |38/112|34/112|
| rules |60/112|60/112|

Общего превосходства и устойчивого прироста после дообучения не подтверждено. Deterministic recovery0/32 у всех моделей; stochastic bonus7/32 против29/32 rules. Стояние и attack commands без цели остаются; команды не равны физическим выстрелам. Ни один кандидат не принят в обычную игру. Полные карты и реальный coop ещё требуют проверки.

Подробности: [результаты](../../../docs/testing/combat_target_commitment_results_20261011.md).

Источник: `workspace/artifacts/combat-target-commitment-cumulative-quality-v2-20261011/complete.json`, SHA `e709c6587e9ad0d28423d910fa3f5997e5d01cabd4c33c30185ef29a39997ff1`; отчёт `combat-target-commitment-cumulative-quality-v2-20261011-report/results.md`.
