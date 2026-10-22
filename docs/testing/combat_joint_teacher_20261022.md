# Совместная имитация правил: движение, цель, прицел, огонь и оружие

Статус:48/48 новых демонстрационных боёв закрыты, readiness audit прошёл. Полный joint BC не готов: учитель почти не двигался. Ученик ещё не обучен. Это пункт4 [плана новых направлений](../learned_combat_new_directions_20261021.md).

Предыдущий BC преимущественно обучал геометрическую доводку прицела. Broad PPO160 не улучшил smoke. Следующая гипотеза — дать модели согласованные примеры всего решения: кого атаковать, когда двигаться и стрелять, какое оружие запросить. Правила служат доступным учителем, но не считаются оптимальным или проверенным человеком поведением.

## Свежие данные

`scripts/collect_registered_rules_teacher.py` использует существующие реестр и динамический пул16 ×2. Корпус `combat-rules-joint-teacher-v1-20261022`:32 train/16 development,8 семейств на base3/jail1/city1/ware2,разные геометрия и loadout (weapons,SSG,grenades-recovery). Реальные seeds/offsets закреплены в protocol до сбора и проверены на пересечение с историческими планами. Финальный test отложен.

Rules владеют всеми командами. Reference PPO weights — только происхождение будущего ученика, нейросеть при сборе не исполняется. Сохраняются исходные pre-command observation, `teacher_aim_source`, `teacher_aim_entity`, applied controls и native logs. Seal проверяет release generation/network frame, capture/dispatch/seed, source fingerprints и hashes. Игровые assets не копируются. Native pools исполняются последовательно.

## История и метки

Обнаружено различие self-history: `combatCommand` в rules сбрасывает previousTarget; provider сохраняет выбранный entity/track для следующего кадра. Поэтому нельзя слепо копировать features правил. На read-only feasibility примере из старой оценки:52 released alive кадров,0 raw previous_target,51 восстановимая предыдущая цель. Эти оценочные записи не включаются в обучение.

Новый `scripts/audit_registered_joint_teacher.py` проверяет закрытый свежий корпус без NN. Он сопоставляет только явный enemy entity с8 отсортированными по расстоянию наблюдаемыми слотами, проверяет clear shot/track и считает историю, восстанавливаемую строго из предыдущего кадра того же life/connection/map/generation/actor. Разрыв кадров и смена life сбрасывают её. Навигация, предметы и отсутствующий enemy остаются неоднозначными, а не превращаются автоматически в метку «нет цели». Три unit tests проверяют причинность/сброс, стабильные slots и отсутствие выдуманного nearest target.

Это readiness audit, а не готовый обучающий dataset. Окончательные availability masks, feature construction и протокольная представимость остаются за Go. Native matched dispatch proof обязателен; отсутствие/разрыв доказательства не заполняется догадкой. Не использовать полученный урон как универсальный повод отбросить пример: иначе теряются важные recovery состояния. Отдельно измерить долю реального движения, смен оружия, разных vertical и положительных/отрицательных fire labels — копирование стоящего учителя не научит маневрировать лучше него.

## Следующие закрываемые этапы

1. Закрыть48 native captures и readiness audit, проверить покрытие действий по train/development.
2. Экспортировать1147 features каноническим Go кодом; добавить past-only target intent и явные masks target/mode/weapon/fire/vertical/movement/aim. Raw logs сохранить неизменными.
3. Проверить причинные temporal окна32: без контекста из будущего и без перехода через life/scene/gaps. При изменении encoder пересчитывать весь контекст на CUDA.
4. Зафиксировать небольшой бюджет joint BC, обучать/проверять tensor operations только на CUDA. Не запускать старый trainer с CPU checkpoint loading как готовую замену; он не покрывает все нынешние81 outputs/conditional heads. Первичный pilot с одним training seed не устанавливает устойчивости.
5. Fresh smoke ученика против source PPO и rules, затем screen только при полезном сигнале. Далее DAgger в состояниях ученика и свежий own-policy PPO, а не PPO на демонстрациях правил. Сравнивать бюджет среды и итоговые бои; ошибки BC сами по себе не качество игры.

Для retention teacher movement не считать выборкой старой NN. Если нужен Monte Carlo KL conditional aim, movement нужно отдельно один раз сэмплировать из замороженной исходной политики на CUDA и затем фиксировать. После BC явно определить сброс Adam state изменённых параметров; продолжение PPO требует новых rollout уже ученика.

## Закрытый сбор и решение по покрытию

Пул завершился штатно,48/48 captures валидны,source unchanged. `joint-readiness.json` проверил hashes и release,явные targets и причинную историю. `joint-data-decision.json` фиксирует сужение первого BC до target/fire/weapon после дополнительного native matched proof/feature export. Это не провал GPU обучения: оно ещё не запускалось.

| Raw evidence | Train32 | Development16 |
|---|---:|---:|
| Released alive frames |2922|1235|
| Явная видимая enemy цель |2782|1163|
| Восстановима цель предыдущего кадра |2772|1159|
| Raw previous_target_intent |0|0|
| Attack commands |1484|797|
| Weapon requests |57|29|
| Movement commands |8|0|
| Jump/crouch |0|0|
| Неоднозначный non-enemy aim |140|72|

Нельзя обучать движение/vertical на этой выборке как полноценную координацию:4157 кадров почти целиком стоят и все отпускают vertical. Маски этих компонентов должны быть отключены. Сохранить движение означает контролировать также shared encoder/общие параметры, иначе их изменение косвенно меняет movement даже при замороженных строках output head. Для полной координации нужен мобильный teacher/human, затем DAgger с проверенными исправлениями;8 команд не создают достаточного покрытия манёвров. Уже закрытые native captures не переписывать и не выдавать за подтверждение обученного ученика.
