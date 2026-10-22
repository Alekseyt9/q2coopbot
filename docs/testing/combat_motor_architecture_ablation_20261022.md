# Motor branch: контроль вклада архитектуры, 22.10.2026

Root `workspace/artifacts/combat-motor-base-ablation-v1-20261022`. Контроль обучает весь исходный actor/std, сохраняя новый motor branch нулевым. Это прямое сравнение с совместным full PPO, где motor branch тоже обучается, а не просто сравнение нового fit с исходной моделью.

Добавлен training scope `base-only`, явный `--fork-head-scope`. Исходные состояния,32 TRAIN боя/4903rows,10actor/40critic caps и reward совпадают с [первой парой](combat_motor_training_20261022.md). Её full/motor weights переиспользованы без нового fit. Отдельно обучен только base-only control. НаCUDA проверены сохранность motor params/moments, обновление старого actor, source boundaries/lineage и точное совпадение final critic/Adam с full. Новый scope имеет тест принудительного Adam fallback с сохранением замороженной ветки.

## Fresh architecture probe

Закрыты40 native /32 CUDA,offset732,pool16 ×2; source/base-only/full/motor-only/rules по8 одинаковых условий и seeds.

| Метрика | Source | Base-only | Full+motor | Motor-only | Rules |
|---|---:|---:|---:|---:|---:|
| Победы | 3/8 | 3/8 | 5/8 | 4/8 | 5/8 |
| Убийства | 8 | 9 | 14 | 10 | 12 |
| Полученный урон | 622 | 467 | 465 | 446 | 457 |
| Смерти | 5 | 3 | 0 | 2 | 2 |

Full vsbase:2 gained/0 lost; full vssource2/0; full vsrules1/1. Две новые победы относительно base-only относятся к recovery jail1 иware2. Это не доказательство исправления base3 или общей геометрической навигации. Смертей у full нет, но полученный урон примерно как у base-only/rules.

Full/source веса совпадают с первым smoke: по двум малым development probes full10/16 против source7/16. Это описательный итог, а не independent screen или доказанное превосходство над rules. `smoke-decision.json`: только переход fixed full weights к независимому screen, promotion=false.

## Независимый screen

Запущен96 native /72 CUDA: source /base-only /full+motor /rules,24 свежих случая на вариант,offset736–738,12ordinary+12recovery,16 ×2. Motor-only исключён из большого screen после отсутствия выигрыша над source/full. Никакого refit или промежуточного выбора checkpoint не было.

Primary — full vsbase-only; также full vssource/rules. Награда и оценочные условия прежние.

Screen закрыт: **96 native /72 CUDA**, все случаи валидны, веса не менялись между smoke и screen.

| Метрика | Source | Base-only | Full+motor | Rules |
|---|---:|---:|---:|---:|
| Победы | 9/24 | 7/24 | 9/24 | 15/24 |
| Убийства | 22 | 23 | 25 | 35 |
| Полученный урон | 1739 | 1642 | 1611 | 1063 |
| Смерти | 13 | 14 | 11 | 6 |

Full vsbase:2 gained/0 lost; full vssource1/1; full vsrules1/7. Малый положительный сигнал к base-only повторился, но прироста побед к исходной модели нет, отставание от правил остаётся. **Promotion=false**, текущие игровые веса не заменены.

Native MG: source200/367 выстрелов с уроном живому монстру (54.50%), base207/349 (59.31%), full211/358 (58.94%). Full имеет50 выстрелов без выбранной цели и видимого bbox, все попали в твёрдую геометрию. Это описательная статистика своих разных траекторий, не изолированная точность прицела. Спуски base3 не устранены: full на двух случаях опускается на264/192units и затем теряет видимый контакт.

Новый read-only CUDA diagnostic `scripts/diagnose_combat_motor_inputs_cuda.py` использует4903TRAIN/2586development rows без обучения. Full motor forward offset в среднем−0.00130, среднее абсолютное отклонение между состояниями≈0.00008; motor-only−0.00834/≈0.00061. После10steps ветка преимущественно даёт общую небольшую поправку. Перестановка geometry-band меняет offset мало; navigation-band в этой выборке не даёт изменения при перестановке. Это не доказательство отсутствия информации: вмешательства могут быть вне распределения и измеряют только branch output, не full-policy KL или качество боя.

Доказательства: `screen/complete.json`, `screen/quality-*.json`, `screen/machinegun-aim.json`, `screen/machinegun-outcomes.json`, `screen/elevation-report.json`, `motor-inputs.json`, `screen-decision.json`.

Общий bundle reporter отклонил метку `joint-motor-screen`, потому что принимает только `component-ablation`; sealed spec не переписывался. Отдельные native aim/outcome/elevation reports завершены штатно, основной evaluator завершён. Это ограничение упаковщика отчётов, а не невалидная оценка.

Следующий ограниченный опыт — [два цикла свежего PPO feedback](combat_motor_feedback_20261022.md). Он проверяет накопление обучения на новых собственных боях; победность пока неизвестна.
