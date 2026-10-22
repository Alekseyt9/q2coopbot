# Совместная имитация правил: движение, цель, прицел, огонь и оружие

Статус:48/48 native демонстраций проверены, canonical feature export и200-step CUDA decision BC завершены. Полный decision candidate отклонён с0/8 побед. Exact target/fire ablation дала3/8 против source2/8 иrules6/8;запущен screen72 native/48 CUDA. Полный joint movement BC не готов: учитель почти не двигался. Это пункт4 [плана новых направлений](../learned_combat_new_directions_20261021.md).

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

## Canonical export и первый CUDA ученик

`cmd/q2joint-teacher-data` не загружает NN. Он заново читает native release/steps/damage и dispatch, сопоставляет каждую raw applied command/sequence с реальным исполнением. Label masks требуют exclusive следующего network кадра; неполные границы остаются context-only. Previous intent восстанавливается из прошлого явного доступного target с точной same-life/frame проверкой, features1147 строит `policy.FeaturesForVersion`. Ни server reward, ни future observation не входят в признаки. Полученный урон не служит фильтром удачных примеров.

`prepare_registered_joint_teacher.py` проверяет provenance/hashes и сохраняет новые derived данные gzip с проверкой распаковки перед удалением только своего несжатого output. Raw captures сохранены. Два незавершённых export attempts v1/v2 сохранены: слишком строгая исходная fire mask отбрасывала отрицательные решения правил. Выяснено, что это897 burst pauses,385 barrel risk и16 grenade releases в train; planner command имеет transport Msec50/Light0,а отправленная Msec100/Light115. Финальная v3 маска допускает только перечисленную чистую отмену attack;yaw/pitch/movement/buttons вне attack должны совпадать. Transport duration/light не считаются обучаемым действием,но полная sent command всё равно проверяется native dispatch. Два focused Go tests проверили history reset и запрет копирования остальных guard corrections.

Закрыт `combat-rules-joint-features-v3-20261022`:2922 train/1235 development rows;target2780/1163,fire positive1484/797 и negative1296/366,weapon requests57/29. Сжатые features вместе906 КБ. **Все target labels — slot1 (ближайшая доступная цель)**: корпус не учит сложному выбору дальнего/приоритетного противника. Weapon labels train:keep2729,Machinegun24,Shotgun11,Blaster22;development:1136/12/8/9. Jump/crouch/movement labels не экспортируются.

`train_registered_joint_decisions_cuda.py` создал `combat-rules-joint-decisions-bc-v1-20261022` от прежнего PPO fit2:фиксированные200 steps,lr0.001,seed20261022,retention0.1. Target CE + сбалансированный fire BCE + macro weapon CE;retention — точный KL этих трёх дискретных решений на train histories. Это не conditional continuous MC KL и не PPO на teacher actions. Development не используется в loss/выборе checkpoint. Causal attention пересчитывает полные последовательности с окном32 и сбросом на разрывах/сценах/life.

Encoder,continuous motion/aim/mode,vertical,spatial/weapon-state/conditional-aim branches,std,critic заморожены. Изменились только четыре tensors:base.head/residual weight/bias,только rows4 и8:29. BC Adam свежий;для будущего PPO весь actor Adam state явно сброшен,critic optimizer/config/consumed PPO rollout list сохранены. BC steps не прибавляются к PPO actor steps. Дальнейшее PPO требует свежих own-policy trajectories ученика.

| Development metric | До BC | После BC |
|---|---:|---:|
| Target agreement |55.37%|100%|
| Weapon macro accuracy |25.00%|86.80%|
| Attack balanced accuracy |46.40%|53.22%|
| Attack balanced BCE |1.268|0.748|
| Decision KL к source |0|1.632|

100% target agreement здесь означает копирование ближайшего slot1,не обобщение target tactics. Fire agreement улучшилось слабо. Independent `audit_registered_joint_decisions_cuda.py` проверил CUDA checkpoint/JSON,finite tensors,точную сохранность запрещённых rows/std/value/value Adam/config и явный пустой actor Adam. На1099 native PPO rows и1100 context rows memory проверена с max error9.06e-6,остальные raw heads совпали точно. Это не обещание одинакового движения в игре:новые target/weapon решения меняют траектории и будущие наблюдения.

Запущен `combat-joint-decisions-bc-smoke-evaluation-v1-20261022`:8 fresh условий/вариант,4ordinary/4recovery,24native/16CUDA,offset604,порт44800,16 ×2. Control — исходный PPO fit2,candidate — BC,rules — тот же fixture/seed. Веса frozen,promotion=false. Качество устанавливается только закрытыми боями/пятью отчётами,не процентом imitation agreement.

## Закрытый первый smoke: weapon imitation не перенеслась

Пять отчётов и CUDA replay закрыты;hashes verified,decision `candidate_rejected`,screen=false,promotion=false.

| Метрика | Source PPO | Joint decisions BC | Rules |
|---|---:|---:|---:|
| Победа |4/8|0/8|5/8|
| Убито монстров |9|1|11|
| Нанесённый урон |1803|585|2158|
| Полученный урон |414|604|448|
| Запросы оружия по legacy weapon audit |11|707|—|
| Из них уже взятое оружие |3|291|—|
| Большой yaw ≥30° |11.05%|3.61%|—|
| Смена цели/100 frames |35.33|9.12|—|
| Стояние при clear enemy |15.52%|16.53%|—|
| Реальный MG damage contact |52/161 (32.30%)|49/70 (70.00%)|—|

MG rate вырос на меньшем числе выстрелов и иной траектории;это не компенсирует0 побед/падение общего урона. Более спокойная цель/меньше поворотов тоже не доказывают эффективность боя. Weapon imitation с macro-balancing дала слишком много запросов:вероятная причина регрессии,нужна компонентная проверка. Legacy weapon audit отсекает первые два network frames,а main ownership/action metrics используют actual release;эти11/707 не объявляются полным census.

`fork_registered_decision_rows_cuda.py` создаёт `combat-rules-target-fire-bc-v1-20261022`:из того же фиксированного BC берутся только rows4 и20:29,weapon rows8:20 восстановлены из source. Другие weights проверены точным сравнением на CUDA,нового обучения/выбора промежуточного checkpoint нет. Actor Adam снова явно сброшен. Это проверяет совместный эффект target/fire без weapon-компонента;не изолирует target от fire.

Fresh smoke `combat-target-fire-bc-smoke-evaluation-v1-20261022`:24native/16CUDA,новые offsets608,порт44900,тот же8-family состав/control/rules,16 ×2. Результаты этого smoke нельзя вычитать из предыдущего как парный effect оружия:seeds различаются. Для строгого парного сравнения joint/no-weapon нужна отдельная общая матрица. Если ablation не даёт полезного сигнала,не расширять её автоматически.

## Target/fire smoke закрыт; screen в работе

Все24native/16CUDA,пять отчётов и их hashes проверены. Парно к source1 новая/0 потерянных побед,к rules0/3. Решение `advance_to_screen_only`,promotion=false. Дополнительный independent CUDA audit ablation подтвердил1099 source native rows/1100context,сохранение остальных heads и checkpoint/JSON.

| Метрика | Source PPO | Target/fire BC | Rules |
|---|---:|---:|---:|
| Победа |2/8|3/8|6/8|
| Убито монстров |4|9|13|
| Нанесённый урон |1208|1735|2225|
| Полученный урон |564|497|380|
| Большой yaw ≥30° |17.31%|5.88%|—|
| Смена цели/100 frames |36.20|11.27|—|
| Стояние при clear enemy |8.18%|17.34%|—|
| Средняя задержка первого attack command,frames |10.50|3.00|—|
| Реальный MG damage contact |40/97 (41.24%)|58/97 (59.79%)|—|

Без новых weapon rows сохранилось улучшение churn/yaw и выросли damage/wins,но стояние хуже;это не решение проблемы лишних действий. Малый smoke на одном training seed недостаточен. Frozen `combat-target-fire-bc-screen-evaluation-v1-20261022`:24 fresh условий/ветку,16ordinary/8recovery,72native/48CUDA,offset612–613,порт45000,16 ×2. Веса/control/rules закреплены до запусков;результатов screen пока нет.

После закрытия обоих smoke исправлен `audit_registered_weapon_choices.py` v2:actual native network release вместо game100,проверка generation иserver-log hash. Пять focused tests прошли. Старые seals/отчёты не переписаны;новый отдельный `weapon-releasefix-candidate.json` уточнил полный census первого BC:723 requests/1107 provider frames,295 already equipped (legacy707/1091 и291). Диагноз частой смены оружия сохранился. Новый screen использует исправленный audit.

## Диагностика стояния и неподвижного teacher

`audit_registered_stationary_transitions.py` сопоставляет исполненную команду со скоростью **следующего** соседнего наблюдения:текущая observation предшествует команде,поэтому её неподвижность нельзя причинно приписывать ещё не исполненному guard. Первые/неполные/gap/handoff кадры не получают такую связь. Два focused tests проверили временную сторону и same-life сброс.

На закрытом target/fire smoke:source70 next-stationary-clear transitions,из них41 после suppressed movement,27 после applied movement,2 без motion intent. Candidate172:104 suppressed,67 applied,1 без intent. Среди candidate104 suppressed81 связаны с `no_ground_support`,12 с unsupported airborne motion,10 static hull,1 component clip;у source no-ground только6. Это связь с предыдущей командой,не полный counterfactual causal proof. Новые target/fire решения меняют траекторию и чаще приводят к границе допустимого движения. Обучение простому «всегда двигайся» не устранит такие тупики.

В неподвижном rules teacher найден другой дефект: `Planner.update` при `testCombatOnly` выставляет `combat_fixture` и возвращается без полной навигации. `spacingGoal` в command допускает retreat только для teammate/`reach_level_exit`;combat_fixture не входит. Config собранного teacher действительно campaign+combat_only=true. Отсутствие level-exit route нужно сохранить,но оно не должно отключать короткое боевое отступление. Изменение этого условия следует делать только после закрытия текущего frozen screen;старые сравнения относятся к прежним неподвижным rules fixtures,не к полноценным правилам кампании. После исправления нужен новый live corpus и новая baseline,а не пересчёт старых sealed captures задним числом.

Screen закрыт:13/24 против source10/24 ипрежних rules12/24;MG damage46.21→69.65%,yaw9.67→4.34%,ноstanding16.27→19.34%. Candidate development-only,coverage отложен до baseline repair. После терминального завершения процесса исправлено условие isolated spacing и закрыты48 новых боёв с2708 movement commands/2476 verified displacement transitions. Подготовлены1881 train/1255 development coordinated ground labels;следующее GPU обучение и детали — в [mobile teacher протоколе](combat_mobile_teacher_20261022.md).
