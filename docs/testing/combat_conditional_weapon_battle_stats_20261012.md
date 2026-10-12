# Статистика боёв

Принято 1456 native боёв; 672 CUDA cases проверены. Ниже исходы первой жизни на matched fixtures; до/после сравниваются одинаковые условия и seeds. Оценочные данные не использованы для обучения.

| Вариант | Победы / бои | Смерти | Убийства монстров | Урон нанесён | Урон получен | Ordinary победы | Recovery победы |
|---|---:|---:|---:|---:|---:|---:|---:|
| control1-before-stochastic | 35/112 | 75 | 113 | 25107 | 8316 | 31/80 | 4/32 |
| control1-after-stochastic | 33/112 | 67 | 108 | 24395 | 7846 | 27/80 | 6/32 |
| conditional2-before-stochastic | 43/112 | 65 | 132 | 27942 | 7820 | 31/80 | 12/32 |
| conditional2-after-stochastic | 38/112 | 59 | 123 | 27170 | 7387 | 25/80 | 13/32 |
| conditional4-before-stochastic | 45/112 | 65 | 135 | 28737 | 7991 | 31/80 | 14/32 |
| conditional4-after-stochastic | 38/112 | 66 | 126 | 27170 | 7869 | 28/80 | 10/32 |
| rules | 61/112 | 38 | 183 | 32161 | 6075 | 33/80 | 28/32 |
| control1-before-deterministic | 33/112 | 62 | 114 | 22552 | 6825 | 33/80 | 0/32 |
| control1-after-deterministic | 31/112 | 66 | 106 | 21538 | 7388 | 31/80 | 0/32 |
| conditional2-before-deterministic | 33/112 | 62 | 114 | 22552 | 6825 | 33/80 | 0/32 |
| conditional2-after-deterministic | 35/112 | 62 | 116 | 21440 | 6888 | 35/80 | 0/32 |
| conditional4-before-deterministic | 33/112 | 62 | 114 | 22552 | 6825 | 33/80 | 0/32 |
| conditional4-after-deterministic | 32/112 | 64 | 111 | 22479 | 6748 | 32/80 | 0/32 |

Урон — native health damage, не точность выстрелов. Смерть, убийства и выполнение всей цели — разные метрики. Ordinary/recovery имеют разные условия; суммарные результаты не заменяют раздельное сравнение.

| Вариант | Смены цели /100 provider кадров | Повороты ≥30° | Стояние при clear enemy | Attack-команды без выбранной цели |
|---|---:|---:|---:|---:|
| control1-before-stochastic | 40.68 | 11.85% | 21.08% | 9.07% |
| control1-after-stochastic | 36.94 | 9.31% | 15.76% | 16.88% |
| conditional2-before-stochastic | 38.38 | 11.63% | 20.46% | 11.89% |
| conditional2-after-stochastic | 34.88 | 10.10% | 16.78% | 21.63% |
| conditional4-before-stochastic | 39.00 | 10.76% | 21.59% | 8.80% |
| conditional4-after-stochastic | 38.63 | 11.82% | 17.74% | 11.23% |
| rules | — | — | — | — |
| control1-before-deterministic | 8.26 | 3.88% | 34.11% | 13.84% |
| control1-after-deterministic | 8.01 | 3.62% | 37.35% | 11.98% |
| conditional2-before-deterministic | 8.26 | 3.87% | 34.09% | 13.98% |
| conditional2-after-deterministic | 7.70 | 3.52% | 38.67% | 10.83% |
| conditional4-before-deterministic | 8.26 | 3.87% | 34.10% | 13.93% |
| conditional4-after-deterministic | 7.72 | 3.95% | 40.73% | 15.22% |

Поворот или стояние сами по себе не доказаны как лишнее действие. Attack-команды не считаются физическими выстрелами или промахами; для rules provider-only behavior metrics отсутствуют и обозначены «—». Смены оружия и вмешательства исполнителя приводятся в отдельных sealed supplementary reports.

Descriptive matched held-out comparisons. Family bootstrap is exploratory: families share maps and each reward branch has one training seed. No multiple-comparison correction, promotion, campaign or coop acceptance. Attack commands are not physical shots; turns and standing are not automatically waste.

## Выбор оружия и исполнение движения

| Вариант | Наблюдаемые смены оружия, все бои | Recovery: supported equipment /32 | Recovery: первая смена на supported, среднее кадров | Recovery: без запроса оружия /32 | Полностью подавленные команды движения | Из них on_ground=false |
|---|---:|---:|---:|---:|---:|---:|
| control1-before-stochastic | 128 | 21/32 | 54.71 | 7/32 | 3762 | 628 |
| control1-before-deterministic | 29 | 0/32 | — | 32/32 | 4459 | 491 |
| control1-after-stochastic | 143 | 21/32 | 57.00 | 9/32 | 3645 | 762 |
| control1-after-deterministic | 26 | 0/32 | — | 32/32 | 5182 | 444 |
| conditional2-before-stochastic | 164 | 32/32 | 9.19 | 0/32 | 3770 | 591 |
| conditional2-before-deterministic | 29 | 0/32 | — | 32/32 | 4461 | 491 |
| conditional2-after-stochastic | 157 | 32/32 | 9.19 | 0/32 | 3763 | 752 |
| conditional2-after-deterministic | 30 | 0/32 | — | 32/32 | 5097 | 449 |
| conditional4-before-stochastic | 166 | 32/32 | 2.78 | 0/32 | 3849 | 639 |
| conditional4-before-deterministic | 29 | 0/32 | — | 32/32 | 4460 | 491 |
| conditional4-after-stochastic | 157 | 32/32 | 2.78 | 0/32 | 3504 | 678 |
| conditional4-after-deterministic | 27 | 0/32 | — | 32/32 | 5406 | 499 |

Supported equipment здесь означает Blaster/Shotgun/Super Shotgun/Machinegun в текущем пилоте. Задержки в сетевых кадрах, случаи без события исключены из среднего и учитываются отдельно. Наблюдаемая смена оружия не доказывает физическую готовность стрелять. Подавленная команда движения не означает неподвижное тело: прежняя скорость сохраняется.
