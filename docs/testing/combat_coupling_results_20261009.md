# Парное сравнение архитектур, 2026-10-09

Same FireBC parent and 150 CUDA BC epochs on identical data; joint loss vs world-input coupling weights0/1/16. Paired native validation,20 families, reused conditions; no final-test superiority.

| Вариант | Победы / 80 | Смерти | Урон монстрам, средний | Полученный урон, средний |
| --- | ---: | ---: | ---: | ---: |
| firebc-baseline | 49/80 | 30 | 61.0 | 46.9 |
| joint-candidate | 30/80 | 44 | 50.9 | 65.9 |
| coupling1-candidate | 31/80 | 46 | 50.0 | 64.0 |
| coupling16-candidate | 42/80 | 36 | 61.0 | 55.3 |
| rules-baseline | 69/80 | 8 | 72.5 | 15.0 |

| Модель | Изменение побед | Новые победы | Потерянные победы |
| --- | ---: | ---: | ---: |

Парные изменения относительно firebc-baseline:

| Вариант | Изменение побед | Новые победы | Потерянные победы |
| --- | ---: | ---: | ---: |
| joint-candidate | -19 | 5 | 24 |
| coupling1-candidate | -18 | 6 | 24 |
| coupling16-candidate | -7 | 8 | 15 |
| rules-baseline | +20 | 23 | 3 |

Guard events/provider frames/frame gaps are full native capture totals, not unique first-life collision counts. Selection latency is measured under pool load; synchronous lockstep throughput is separate from realtime behavior.

| Вариант | Guard events, среднее / capture | Provider frames, всего | Frame gaps, всего | Mean selection p95, ms |
| --- | ---: | ---: | ---: | ---: |
| firebc-baseline | 71.2 | 6938 | 0.0 | 10.22 |
| joint-candidate | 80.7 | 10743 | 0.0 | 11.32 |
| coupling1-candidate | 83.8 | 10524 | 0.0 | 11.40 |
| coupling16-candidate | 78.5 | 8094 | 0.0 | 11.28 |
| rules-baseline | 0.0 | 0 | 0.0 | — |

Все три кандидата отвергнуты: baseline FireBC49/80, joint30/80, coupling1 31/80, coupling16 42/80. Новый target-head эксперимент не использует эти неудачные веса. Перед native evaluation на CUDA проверены lineage/config/frozen parameters/optimizer reset.
