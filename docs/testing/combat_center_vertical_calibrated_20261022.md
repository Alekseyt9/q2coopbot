# Vertical conditioning при одинаковом калиброванном center обучении

Статус:fit, CUDA audit, smoke и screen закрыты. Candidate не продвигается; default сохранён.

## Проверяемая гипотеза

Первый [vertical опыт](combat_vertical_aim_protocol_20261021.md) имел80 steps,retention0.02 и upper teacher. Он не улучшил screen; при этом вход vertical сдвигал предсказание в среднем лишь на0.165° по оси. Последующий [bodypoint pilot](combat_bodypoint_teacher_20261021.md) показал,что200 steps иretention0.002 позволяют заметнее перенять новую метку. Новый опыт проверяет эффект выбранного vertical в этом режиме и с center teacher. Он не устанавливает отдельную причину отличия от первого опыта:между опытами изменились training budget/penalty/teacher.

Текущее парное сравнение изолирует только vertical:обе ветки имеют128→64→32→2 и10402 параметра,одни начальные tensor values,одни1095 train/557 development примеров,200 steps,lr0.0003,retention0.002,seed20261022. В control три дополнительных канала нулевые;candidate получает выбранные release/jump/crouch. Остальные actor/value/log_std заморожены. Нет ручной компенсации угла,запрета прыжка/приседания,новой физики или reward fork.

## Протокол и обучение

`combat-center-vertical-calibrated-ab-v1-20261022/protocol.json` зафиксирован до fit. Проверено,что `combat-vertical-aim-control-zero-v1-20261021` и `combat-vertical-aim-zero-v1-20261021` отличаются только версией conditioning,а все исходные tensor values одинаковы. Control переиспользуется из закрытого `combat-bodypoint-center-bc-calibrated-v1-20261021`;его бюджет/data/hyperparameters проверены. Он не переобучается и не подбирается по новым оценочным боям.

Candidate `combat-center-vertical-bc-calibrated-v1-20261022` обучен на GPU с теми же common-support center labels. Промежуточный checkpoint не выбирался. Development уже использовался при пересмотре протокола,это не независимый финальный holdout. Center bbox не проверен на видимость;невидимая центральная точка остаётся ограничением геометрического teacher.

| Development557 | Control без vertical | Candidate с vertical |
|---|---:|---:|
| Ошибка среднего действия,°/ось |3.781|3.614|
| Диапазон изменения при смене только vertical,среднее по оси |0|2.901°|
| Максимальный такой диапазон |0|23.286°|

CUDA pair audit подтвердил одинаковые начальные метрики,retention histories,сохранение остальных actor/value/std и использование входа. Counterfactual span не доказывает оптимальность компенсации,особенно для редких/физически неисполненных vertical команд. Это причина проверить новый контроллер в игре,не доказательство боевого улучшения.

## Игровая оценка

Первичная подготовка `combat-center-vertical-smoke-evaluation-v1-20261022` остановилась до создания native pool:UDP44301 занят посторонним PnkBstrA. Процесс не останавливался;боёв в v1 не было. Подготовка/ошибка сохранены.

Запущен `combat-center-vertical-smoke-evaluation-v2-20261022`:8 новых условий/вариант,4 ordinary/4 recovery,24 native/16 CUDA. В spec v2 изменён только порт44500 вместо44300 и добавлена причина переноса;frozen условия/seed/weights сохранены,потому что первый запуск ничего не исполнил. Control/candidate/rules имеют одинаковые условия и seed,разные сцены — разные seed. Validation offset596,существующий пул16 ×2. Диапазон44500–44515 проверен свободным;иных native pools нет.

Ownership audit теперь использует actual native network release и включает первые два реальных кадра боя. Веса frozen,promotion=false. Нужны закрытый CUDA replay и пять отчётов;выигрыши,урон,реальные MG outcomes,стояние/повороты и команды без цели сохраняются отдельно. При отрицательном smoke расширять оценку этого checkpoint не следует. При положительном нужен fresh screen;один training seed не устанавливает устойчивое превосходство.

### Закрытый smoke

Пять отчётов и их hashes проверены в `combat-center-vertical-smoke-bundle-v2-20261022`. Решение `advance_to_screen_only`, promotion=false.

| Метрика | Center без vertical | Center с vertical | Rules |
|---|---:|---:|---:|
| Выполненная цель |2/8|3/8|6/8|
| Ordinary |1/4|2/4|2/4|
| Recovery |1/4|1/4|4/4|
| Полученный урон |465|398|394|
| Смена цели/100 frames |25.03|35.34|—|
| Большой yaw ≥30° |8.18%|14.59%|—|
| Стояние при clear enemy |9.01%|14.37%|—|
| Attack без выбранной цели |36.04%|19.49%|—|

Парно к control одна новая/ноль потерянных побед; к rules одна новая/четыре потерянных. Реальные MG выстрелы: control162, из них80 повредили живого монстра (49.38%),81 solid,1 corpse; candidate172,90 (52.33%),82 solid. Это разные траектории, не сравнение прицела на одинаковых кадрах. Повороты, смены цели и стояние ухудшились; проблема лишних действий не решена.

Frozen screen `combat-center-vertical-screen-evaluation-v1-20261022`:24 условия/ветку,16 ordinary/8 recovery,72 native/48 CUDA,новые offsets600–601,порт44600. Native72 закрыты; выполняются ownership/action audits и CUDA replay. До их завершения итог screen не заявляется.

### Закрытый screen

Все72 native/48 CUDA и пять отчётов закрыты, inputs/report hashes перепроверены. `combat-center-vertical-screen-bundle-v1-20261022/decision.json`:candidate_not_advanced,promotion=false,coverage=false.

| Метрика | Control | Candidate | Rules |
|---|---:|---:|---:|
| Цель |7/24|8/24|13/24|
| Ordinary |5/16|6/16|5/16|
| Recovery |2/8|2/8|8/8|
| Полученный урон |1718|1732|1214|
| Убито монстров |23|26|32|
| Смена цели/100 frames |38.50|36.83|—|
| Большой yaw ≥30° |9.91%|8.92%|—|
| Стояние при clear enemy |19.71%|20.92%|—|
| Attack без выбранной цели |15.64%|12.22%|—|
| Реальные MG:урон живому монстру |264/440 (60.00%)|246/440 (55.91%)|—|

Парно к control2 новые/1 потерянная победа; к rules2 новые/7 потерянных. Native MG solid contacts170→193,corpse6→1. Уменьшение больших поворотов/attack без цели не дало общего выигрыша; стояние и полученный урон выше. Возможность использовать vertical доказана, полезность этой конкретной настройки против rules — нет. Не запускать coverage/PPO этого checkpoint автоматически. Следующий отдельный опыт — [совместный teacher](combat_joint_teacher_20261022.md).
