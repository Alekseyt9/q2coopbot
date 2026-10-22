# Штраф за зарегистрированный промах автомата: парный PPO опыт

## Гипотеза и границы

В actual pinned objective source используется `combat-reward-target-commitment-bonus-v12.json` (SHA `af9b9e4c…`),а не похожий cost-only файл. Он содержит ограниченную награду за продуктивное удержание цели и штраф за attack в неверном направлении известного bbox,но не штрафует сам native MG miss. Во время отсутствия известной цели off_target_attack=0. Это подтверждено ранее на собственных train trajectories;выстрел без цели может задеть полезный damageable prop,поэтому target0 не является достаточным признаком промаха.

Новая версия `combat_reward_v13` сохраняет все v12 semantics:урон/убийства/смерть/время,selected aim/spacing,action quality,target churn и ограниченный commitment bonus. Добавлена только существующая native MG miss evidence,ранее использованная в v9. Projectile miss objective не включён. Никакого нового runtime запрета fire или изменения observation/action/архитектуры.

Штраф начисляется за реальные shots с native outcome geometry/sky/no_contact/corpse в точном matched exclusive unrecovered окне первого life. Повтор shot,неверное окно и отсутствующие evidence отклоняются. Команда attack во время cooldown сама по себе не штрафуется. Damageable contact без подтверждённого урона не объявляется промахом;damageable props сохраняют эту защиту. Мгновенный промах не гарантирует стратегическую бесполезность:это проверяемый слабый cost,а не доказательство оптимальности.

## Заранее закреплённый опыт

Parent: `combat-rules-target-fire-bc-v1-20261022`;joint-heads после screen11/24 против source11/24/rules14/24 не продвигается.

Root: `workspace/artifacts/combat-sequence-hitscan-pair-v2-20261022`.

| Условие | Контроль | Кандидат |
|---|---|---|
| Reward version |v13|v13|
| Machinegun miss |0|−0.02 за реальный промах|
| Условия/семейства |8 одинаковых|8 одинаковых|
| Новые train бои |32|32|
| Максимум actor steps / value steps |10 /40|10 /40|
| Retention / остальные hyperparameters |одинаковые|одинаковые|

Оба старта имеют точно одинаковые actor/value/std/optimizer/RNG checkpoint состояния на CUDA. Critic output обнулён и оба Adam сброшены одинаково;никаких старых v12 rollout под новым objective. В обеих ветках обязательна одинаковая native evidence даже при нулевом cost. Единственное различие стартового checkpoint — config objective SHA.

Train seeds уникальны между ветками;семейства/распределение условий одинаковы. Пул16 ×2,по одному native pool,автоматическая очередь существующего harness. Всё обучение и numerical NN проверки — CUDA. Данные/assets не копируются;полные старые корпуса не разворачиваются.

После каждой учебной ветки предусмотрены independent CUDA checkpoint/Adam audit и native audit соответствия stored reward реальным MG rays в принятых PPO rows. Затем заранее закреплённый fresh smoke:32 native/24 CUDA,8 paired условий на parent/control/candidate/rules;offset640,порт45980. Общий legacy reward в evaluation нужен для сопоставимой отчётности,эти данные не идут в обучение. Сравниваются победы/смерти/урон/timeouts,физические MG outcomes и лишние команды. Автоматического promotion нет;большое расширение только при положительном боевом сигнале.

## Проверки до запуска

Focused Go tests проходят:нулевой v13 control сохраняет v12 score/components/aim/commitment/target-cycle,проверены неизвестная цель,0/1/2 реальных shot,duplicate/window/recovery masks,goal/death terminal,ограничения cost и изоляция legacy v12/v9/projectiles. `cmd/q2combat-export` tests проходят;Python source компилируется. GPU fork read-back подтвердил точное равенство обоих стартов. Это подготовка эксперимента,боевого улучшения ещё не измерено.

Подготовительная v1 попытка остановилась до создания native pool:registry запрещал artifact reward path. Добавлены два точных canonical scenario paths в allowlist,без ослабления остальных проверок. Registry tests проходят;v2 сохраняет те же условия/бюджет и starts,а failed v1 остаётся для диагностики.

## Завершённое обучение и оценка

Обе ветки закрыты:контроль32 боя/3450 train rows/10 actor steps,штраф32/3179/10;по40 value steps,CUDA. Independent checkpoint/Adam audits подтверждают JSON/actor/value/std,finite moments,сохранённый config/objective и инициализацию пустых Adam states с правильными step counters. Число accepted steps одинаково;изменение исходов на разных train seeds не используется как доказательство эффекта обучения.

Native audit контроля связал822 shots с accepted rows и подтвердил нулевой miss cost на3450 окнах. Для кандидата993 shots:650 live-monster damage,310 solid,33 corpse. Все343 подтверждённых промаха дали суммарно−6.86;3179 reward windows проверены против фактических rays,stored score совпадает с PPO row score. Среди этих промахов47 solid shots без known bbox:прежний off_target_attack=0,новый cost начислен. Нет повторного суммирования одного выстрела/окна.

Fresh smoke закрыт32 native/24 CUDA,пять bundle reports и их SHA проверены. Дополнительный actual MG outcome report и его входные SHA проверены отдельно. `pipeline-complete.json` закрывает весь последовательный запуск;native процессы завершились.

| Вариант | Победы | Ordinary / recovery | Убийства | Полученный урон | Смерти | Yaw ≥30° | Стояние | Attack без цели |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Исходная модель |5/8|2/4;3/4|12|456|2|2.21%|27.94%|17.70%|
| PPO контроль |4/8|1/4;3/4|10|420|3|4.92%|20.70%|12.07%|
| PPO + miss cost |4/8|2/4;2/4|10|494|3|1.77%|16.37%|14.44%|
| Правила |6/8|2/4;4/4|13|399|1|—|—|—|

Native MG с уроном живому монстру:parent134/238(56.30%),контроль127/221(57.47%),cost148/240(61.67%). Solid104/94/91,cost также1 corpse. Это разные closed-loop траектории,не оценка попаданий при одинаковом состоянии и не вся точность оружия. Стояние/повороты/команды attack не автоматически бесполезны.

Кандидат к контролю:1 новая/1 потерянная победа,описательный95% интервал[−37.5;+37.5]п.п.;к parent0/1,к rules0/2. Один timeout у каждой ветки,включая parent/rules. Recovery по полезному weapon damage не объясняется полным отсутствием автомата:cost755 MG damage против control623,parent636;при этом побед меньше. Нельзя заключать,что дополнительная стрельба из автомата сама вызвала поражение.

Решение `miss_cost_not_advanced`,screen=false,promotion=false:штраф технически работает и меняет отдельные действия,ноне улучшил победность в этом малом опыте. Полученный урон вырос к контролю420→494. Default/source сохранён;большой screen/дальнейшие PPO rounds этой пары не назначены. Это результат одного training seed и малого smoke,не доказательство бесполезности любого штрафа или PPO в целом. Следующий цикл требует отдельной гипотезы,а не автоматического увеличения этого cost или бюджета.
