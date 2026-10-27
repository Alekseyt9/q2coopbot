# Геометрия и связь движения с прицелом: отрицательный probe

Дата: 2026-10-26. Источник — неизменные веса `804ffd9662ebb7f20a67c5a11326342683993af4cf6af36032b712d31b089172`. Это проверка обучаемости отдельных counterfactual preferences, не оценка нового игрового actor.

## Задача и входы

Предыдущий frozen-representation readout с 17 параметрами не превзошёл bias-only. Новый опыт проверяет явные причинные признаки, доступные до wide-решения, на тех же 47 метках из реестра 65 состояний. Используются 25 карт; все roots/сиды одной карты находятся в одном fold.

`combat_teacher_physical_features.py` принимает только существующий `pre_gate_context187`, не native damage, future state или outcomes. Повторно использованы существующие `combat_action_geometry.py` с CENTER_HANDED v3 и requested-motion transform из `combat_action_effect.py`. Обученный motor predictor не вызывается.

Состав:

- geometry16: расстояние, относительная поперечная скорость и текущие bbox/ray признаки исходного прицела, а также их изменение при wide anchor;
- physical24: geometry16 плюс восемь признаков requested movement в направлении/поперёк цели, изменения вектора движения и текущих наблюдаемых скоростей;
- в обоих кандидатах применяется абсолютный pitch clamp ±89°, как в `policy.Command`.

Это геометрия текущего наблюдения и requested-motion proxy. Предсказания будущего eye/muzzle, фактического перемещения, recoil или quantization здесь нет. В частности, yaw-only requested-motion transform не является полной моделью `PMove`: reference `pmove.c` уменьшает pitch для наземного/воздушного перемещения втрое. Результаты не следует пересказывать как доказанную точность будущей физики.

## Протокол

Настройки зафиксированы до fitting в Hindsight и `combat-teacher-physical-learnability-v1-20261026/protocol.json`; входы и код запечатаны SHA, копии helper scripts находятся в `frozen-scripts`.

Четыре внешних whole-map folds те же, что в предыдущем диагностическом probe. Для каждого из них три внутренних TRAIN-map folds выбирают ridge из [.01,.1,1,10]. Нормализация mean/std и world-balanced weights вычисляются только по соответствующей TRAIN-части, std floor=.05. Предыдущие outer folds повторно используются после отрицательного результата другой архитектуры: это исследовательский screen, не независимое подтверждение.

Модели — logistic delta поверх замороженного source logit: bias-only (1 параметр), geometry16 (17), physical24 (25). Convex logistic Newton, gradient stopping и line search полностью выполняются CUDA float64. Нет CPU neural forward или обучения. Веса игрового actor не экспортируются, checkpoints имеют `actor_compatible=false`.

Заданный primary gate: physical24 OOF world-weighted log-loss ≤95% bias-only, balanced accuracy ≥55%. Hyperparameters выбираются только внутри TRAIN, по внешнему результату их не подбирали.

## Результаты

`workspace/artifacts/combat-teacher-physical-learnability-v1-20261026/report.json` завершён. Все оптимизации сошлись. CUDA checks подтвердили causal-input isolation, pitch clamp, centered bbox geometry, знак Quake side, вращение movement frame и маскирование неизвестной скорости. Оптимизатор проверен на независимом аналитическом решении weighted intercept log(3) и известном упорядоченном сигнале.

| Модель | OOF log-loss | Balanced accuracy |
|---|---:|---:|
| Bias-only | 0,694740 | 48,81% |
| Geometry16 | 0,717668 | 49,85% |
| Physical24 | 0,698081 | 45,68% |

Primary loss ratio **1,004809**, gate не пройден: `advance_to_native=false`, `policy_training_performed=false`, `promotion=false`. На этой выборке явная физическая геометрия с малым logistic readout тоже не дала контекстного преимущества. Это не устанавливает причину: выборка мала, представление ограничено, одна пара продолжений не оценивает среднее по будущей случайности. В игре новый вариант не проверялся и не заменял source.

## Аудит фактического выстрела в root

`audit_combat_teacher_root_shots.py` использует существующие строгие MG readers, pinned server.log и native export steps. Привязка по `(spawncount, begin_frame, actor, sequence)` и порядку native step begin/end; GAME и SERVER frame не смешиваются. Число выстрелов не выводится из attack-команд.

Отсутствие выстрела в root объявляется подтверждённым только при наличии строгой MG telemetry в другом месте того же лога. Все 65 случаев сохранены, unknown отсутствуют. Никакие labels, reward или допуск не изменялись.

`combat-teacher-root-fire-audit-v1-20261026/report.json`:

| Срез | Всего | MG выстрел в обеих ветвях | Ни в одной | Разный мгновенный урон | Противоположные immediate/full-return preferences |
|---|---:|---:|---:|---:|---:|
| Все состояния | 65 | 49 | 16 | 25 | 9 среди имеющих full-return метку |
| Размеченные | 47 | 35 | 12 | 20 | 9/20 |

В размеченной части 42 root имеют отправленный attack хотя бы в одной ветви. Семь пар отправили attack в обеих ветвях, но фактического MG выстрела в этом окне не было. Встречаются observed gun_frame 48/49; reference `Weapon_Machinegun` использует frames46..49 для deactivation, поэтому это может быть смена оружия, а не только cooldown. Само отсутствие MG telemetry в окне не классифицировалось как cooldown. Другие пушки не входят в MG firing denominator.

Из 20 размеченных пар с разным мгновенным уроном anchor дал больше HP в17 и меньше в3. У девяти sign мгновенного HP delta противоположен sign полного discounted observed-return. Это не опровержение причинности full-return записи: одно изменение прицела меняет также движение, наблюдения и последующие sampled действия. Но такую метку нельзя считать изолированным учителем точности выстрела.

По already-frozen physical features изменение requested movement больше .25 в11/47 размеченных roots; средняя норма изменения .189638. Это proxy, не измеренное перемещение. Противоречие с итоговой наградой нельзя приписывать именно этому фактору без отдельного контроля. World-balanced mean полного observed utility delta составляет −0,065696 — описательная оценка одиночных пар, не точное expected Q и не основание автоматически выбирать либо запрещать anchor.

## Связь с общей оценкой боя

Последняя расширенная фиксированная оценка: before14/38 побед/19смертей, update1012/38/19, rules14/38/17. Малый fresh MG screen: parent1/8, wide_fit0/8, rules2/8. Это новые seeds/расстановки, не подтверждение globally unseen maps. Ни улучшения логистической ошибки, ни отдельных удачных counterfactual действий нельзя выдавать за превосходство игровой модели над source или правилами.

Активная очередь гипотез и решения находятся в Hindsight. Здесь сохранены результаты и границы доказательств; более крупная модель, другое распределение продолжений и разделение aim/movement ещё не объявляются проверенными решениями.
