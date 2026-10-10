# Память о ранее наблюдаемых противниках

## Результат предыдущего кооперативного обновления

`coop-learning-eval-v4-20261010/comparison.json`: обе серии завершены,
32/32 accepted и CUDA verified в каждой. До update: 21 learner monster kill,
11 эпизодов с participant death. После: 20 kills, 11 death episodes. По парным
сидам улучшений kills нет, один случай хуже; death улучшился в одном случае
и ухудшился в другом. Новые веса не продвинуты в основной набор.

`behavior-comparison.json` содержит отдельные командные метрики. Медианная
задержка первой attack-команды после release равна 0s в обеих сериях, no-attack
episodes 0. Stationary frames: 185/1193 measured motion frames до, 196/1392
после. Attack commands без наблюдаемой clear-target: 68 до, 93 после.
Это команды, а не число выстрелов/промахов. Сравнение относится только к одному
участку base1 Soldier/Blaster со scripted moving peer, не к human coop.

## Контракт памяти

`internal/policy/threat_memory.go` сохраняет до восьми последних видимых
монстров. Вход — собственные client observations с ClearShot=true и observed
Defeated; hidden current world positions/server reward telemetry не используются.
Сохраняются тип/model, ID, последний наблюдаемый world position и наблюдаемая
скорость. Возраст ограничен 200 native frames (20 game seconds независимо от ×2).

Выход RememberedThreats отделён от Enemies: в нём явно last_observed_relative,
last_observed_velocity и age_frames. Relative корректируется только по текущей
собственной позиции; скрытое движение не экстраполируется. Память очищается
при life/map/connection/spawncount смене, frame gap, stale observation или death.
Наблюдаемая смерть удаляет запись. Память не создаёт selectable firing target.

Feature v9 имеет ширину 1121: неизменённый v8 prefix 881 + восемь slots по 30
значений (presence, age/200, 21 types, yaw-local last XYZ/512, velocity mask,
last velocity XYZ/400). Отсутствующие slots нулевые; нечисловые координаты,
повторные/невалидные IDs и возраст вне диапазона отклоняются.
Target availability продолжает использовать только visible prefix masks.

Focused TestThreatMemory*, TestThreatFeatures*, TestEngagement* прошли;
это проверки наблюдений/feature serialization/ownership, без CPU NN inference.
Go/Python target contracts принимают v9; остальные feature versions сохраняют
прежнее поведение и ширину.

## CUDA checkpoint migration и сбор

`migrate_combat_threat_checkpoint_cuda.py` создал
`threat-v9-checkpoint-v1-20261010` из проверенного v8 baseline checkpoint,
а не из неулучшившегося coop update. Actor/critic input expanded 881→1121,
новые columns и Adam moments нулевые. Прежние параметры, moments/steps,
RNG, consumed rollouts и updates_completed=5 сохранены. CUDA verification:
max output error actor 4.77e-7, critic 3.28e-7. Disposable optimizer step
проверен на GPU, persisted optimizer steps=0, training_performed=false.
Read-back подтверждён; complete seal имеет тип model migration, не update.

Weights SHA: `b6da6ad0f160115f802caeb86501261a53902f5f9c4cce152871a03d9af668ba`.
Checkpoint SHA: `7c996ae6b082e820865a7a8c712594c211c6a37a0c54ca6c87e3fc7472ac1a8d`.

Для v9 PPO ownership grace теперь 200 frames; для старых models остаётся 30.
Observed defeat и reset по-прежнему завершают продолжение. Модель выбирает
движение и поворот сама; remembered records не добавляют выстрелы или маршрут.
Новые нулевые input columns ещё не обучены, поэтому качество поиска не доказано.

Запущен `coop-threat-pool-v1-20261010`, seeds186500–186531, 16 slots, ×2.
Это свежий сбор после изменения observations/ownership, не продолжение старых
on-policy episodes. Следом: CUDA finalize, explicit reward objective fork v11,
GPU update и held-out before/after. Расширение geometries/multi-monster scenes,
проверка остальных architectures и full campaign/human coop остаются впереди.
