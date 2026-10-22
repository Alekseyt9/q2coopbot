# Следующая проверка: сигнал награды для conditional fire

Статус: [TRAIN relabel/gradient diagnosis](combat_conditional_fire_reward_signal_20261022.md) закрыт:64Goexports/32captures,400miss rows,прочиеданные неизменны. Native blind MG misses уже имели отрицательный signal; cost−0.005 усилил почти тот же gradient. Reward-fork training не запускался. Вместо него отдельный unchanged-reward300-step fire-only fit проверен на CUDA; fresh32-case budget comparison выполняется. Цель — обучить эффективному решению об огне без запрета стрелять при отсутствии выбранной цели.

## Основание

За10fire-only steps probability сдвинулась лишь на−0.11п.п.; native wins3/8 остались3/8, targetless MG solid shots19→19. В688 TRAIN states без selected target средний начальный advantage-weighted fire score положительный. Более ёмкая сеть сама по себе не исправляет неоднозначный objective.

У reward v13 уже есть kill+5 и off_target_attack−0.01. Последний применяется для известных bbox/невыравненного direct aim; при отсутствии known enemy bbox этот cost0. `machinegun_miss=0`. Ранее−0.02 native MG miss penalty без этой изолированной branch не дал достаточного выигрыша; не повторять его как заранее успешное решение.

## Порядок

1. Взять **только** сохранённые32 TRAIN captures `combat-conditional-fire-training-v1-20261022/corpus`, дообучение и validation captures не использовать. Через существующий Go exporter заново сопоставить native hitscan outcomes/command windows и пересчитать reward с контрольным0 и малым отрицательным `machinegun_miss`. Начальная проверяемая величина−0.005; это гипотеза, не выбранное по validation optimum. Другие компоненты, kill reward и terminal boundary оставить одинаковыми.
2. Для каждого варианта проверить полноту causal histories и native execution windows, unchanged observations/actions/values, корректную привязку cost к действию, невмешательство post-goal shutdown tail и отсутствие cost на неатрибутированных rays. Сравнить TRAIN advantages, fire-score direction и долю затронутых записей. NN вычисления — только CUDA; native fields используются только в reward/аудите, не во входах policy.
3. До native fit определить, даёт ли штраф сигнал уменьшать физически бесполезные выстрелы, сохраняя обучение полезному огню. «Нет selected target» не равно «каждый выстрел бесполезен»: предусмотреть unknown/uncertain цели и допустимый упреждающий огонь, не вводить hard fire mask. Анализировать native solid/sky/no-contact/corpse outcomes отдельно от ошибочного выбора цели.
4. Если сигнал подтвердится, заранее зафиксировать matched control/cost experiment с одинаковыми source actor, данными/бюджетами, critic treatment, LR и conditional-fire scope. Обновить явный objective checkpoint fork и проверить всю новую branch/optimizer/likelihood; существующий fork helper требует адаптации и по своему текущему контракту требует fresh on-policy collection. Диагностический relabel сам по себе **не разрешает** объявлять набор PPO-ready или обходить consumed-rollout/fork guards. Возможность повторного использования тех же captures для fit требует отдельного доказанного on-policy/metadata fork, а не замены SHA в report.
5. Только после этих gates провести CUDA fit и отдельный свежий малый native screen. Оценивать победы/смерти/урон вместе с реальным расходом выстрелов. Положительный сигнал подтвердить на более широком наборе карт/составов; кампанию и coop проверять отдельно. Default не заменять автоматически.

Дальнейший бюджет новой branch выбирать по TRAIN loss/KL/градиентам с заранее заданными пределами; сравнивать варианты на новых evaluation seeds. Не увеличивать модель или число боёв вслепую и не считать уменьшение wall shots достаточным, если бот перестал добивать противника.
