# Weapon exploration: закрытая нативная оценка

Confirmation560 закрыта:ошибок0,source_unchanged=true,1519,218с. Все пять контроллеров проходили28 одинаковых семейств × четыре одинаковых confirmation-сида, offset192. Train offset160 не пересекается с оценкой. Подготовка v1 остановилась до боёв; здесь используются проверенные v2-планы и отдельный modes-only confirmation registry.

| Контроллер | Победы /112 | Убийства | Нанесённый урон | Полученный урон | Смерти |
|---|---:|---:|---:|---:|---:|
| control-before | 29 | 111 | 23988 | 8006 | 67 |
| control-after | 34 | 118 | 25193 | 8169 | 68 |
| exploration-before | 10 | 46 | 16364 | 9787 | 95 |
| exploration-after | 6 | 39 | 15940 | 9825 | 99 |
| rules | 58 | 178 | 32934 | 6769 | 47 |

Собственное парное обновление control:13 новых побед,8 потерянных,21 сохранённая победа,70 общих поражений. Собственное обновление exploration:1 новая победа,5 потерянных;10→6. Общая таблица paired_goal_changes имеет baseline control-before; её comparison exploration-after не подменяет собственное before/after temperature2. Отдельные `quality-exploration-own.json` и `quality-trained-cross.json` сохранены.

| Набор | control до → после | exploration до → после | rules |
|---|---:|---:|---:|
| Обычные,80 | 23 →29 | 8 →6 | 27 |
| Recovery-пары,32 | 6 →5 | 2 →0 | 31 |
| trio/weapons,32 | 3 →7 | 1 →1 | 1 |
| trio/weapons-ssg,32 | 13 →14 | 3 →2 | 14 |
| pair/weapons-ssg,16 | 7 →8 | 4 →3 | 12 |

Control получил ограниченный прирост обычных боёв, но recovery-switching не улучшился; deaths/received damage также не улучшились суммарно. Temperature2 заметно хуже и исходного control, и control-after; PPO этого раунда не исправил её поведение. У rules31/32 recovery-побед показывает, что эти сцены в целом исполнимы; это не означает оптимальности rules на всех сценах (на trio/weapons у rules1/32 против7/32 обученного control).

Exploration-веса не продвигаются. Control-after пока не заменяет основной режим: исходная цель выбора оружия на recovery не достигнута, полного campaign/coop acceptance нет. Одно небольшое положительное изменение на80 обычных боях не доказывает общего превосходства.

Источники: `workspace/artifacts/combat-weapon-exploration-confirmation-pool-v1-20261011/report.json`, `combat-weapon-exploration-confirmation-quality-v1-20261011/quality-strata.json`, `quality-exploration-own.json`, `quality-trained-cross-strata.json`. Итоговый `complete.json` закрывает 448 CUDA-воспроизведений и аудиты: 57409 кадров, обучения на оценке нет. `native-quality.json` сохраняет исторический промежуточный статус. Initial-policy probe закрыта на 128 состояниях с полной историей: средняя вероятность keep у control остаётся около 0,984 до/после, у temperature2 около 0,797/0,799. Повышение температуры не научило полезному переключению.

Следующий эксперимент — наблюдаемое состояние в weapon residual с явной миграцией и собственными данными. Нативная интеграция и короткое собственное GPU-обучение реализованы; независимая оценка качества ещё требуется. См. `combat_weapon_state_residual_20261011.md`.
