# Два клиента в синхронном харнесе

Для обучения навигации во время боя нужен реальный напарник в наблюдениях.
Прежний native lockstep разрешал только имя sv_test_lockstep_client и
отключал остальных отправителей usercmd. Добавлен test-only opt-in
sv_test_lockstep_peer. Он действует при существующих ограничениях
sv_harness_instance, loopback и cheats; требуется g_test_combat_clients=2
и разные имена участников. Обычная человеческая сессия не менялась.

Сервер принимает по одной новой команде от каждого участника. Неполная пара
не изменяет мир и не вызывает ClientThink. После получения обеих команд
ClientThink выполняется в порядке primary→peer, независимо от порядка
прибытия пакетов. После одного stock world tick записываются две closure
с общей следующей frame. Повторы, посторонние акторы и recovery commands
отклоняются; смена spawncount сбрасывает неполную пару. При исчезновении
участника неполная пара не разрешает дальнейшее движение мира.

Новые записи sv_test_pair_step и sv_test_pair_cmd имеют отдельный формат.
Прежний single-client sv_test_step не выдаётся за paired receipt. Старый
PPO exporter пока не читает paired receipts, поэтому данные пары не trainable.

Native implementation: yquake2/src/server/sv_user.c и
yquake2/src/server/header/test_lockstep_pair.h.
scripts/test_native_lockstep_pair.ps1 проверил оба порядка принятия,
duplicate command/actor, неверную роль и сброс. Native q2ded пересобран.

scripts/run_coop_lockstep_smoke.ps1 использует два настоящих Go UDP клиента,
stock base1 BSP/PMove, но удаляет монстров и бочки для проверки протокола.
Первый диагностический прогон v1 отклонён: телепорт напарника задел бочку,
взрыв изменил health и оба клиента не стали ready. Это не успешный training
reset и не скрытое улучшение поведения. После исключения боевых hazards
v2 прошёл:57 полных пар,114 применённых команд, ready=2,×2.
scripts/audit_coop_lockstep_receipts.py сопоставил все десять полей каждой
native команды с sent_command соответствующего клиента по
spawncount/frame/sequence/actor. Кадры последовательны, каждая пара
закрывается ровно следующим кадром. SHA логов сохранены в audit artifact.
Повтор v3 также дал57пар/114команд, source_unchanged=true и
native_source_unchanged=true; report сохраняет source fingerprints и SHA
обоих executable. Аудит всех команд повторно прошёл.

Прежний одиночный synchronous harness проверен отдельно:
coop-lockstep-solo-regression-v1-20261010, Blaster rules,150frames,release100,
seed186010. capture_complete=true,provenance_valid=true,capture_valid=true.
Это проверка протокола/сериализации, без численных CPU NN проверок.

До trainable кооп-curriculum остаются fixed-frame release для двух акторов,
независимое seed/RNG подтверждение вместо старого двухклиентного seed0,
оружие/позиции/геометрии и reset receipts, seeded движение ведущего,
native damage attribution обоим клиентам, terminal/goal/truncation export,
наблюдаемый navigation-progress reward и CUDA on-policy finalization.
Одного протокольного smoke для начала обучения недостаточно.

## Fixed paired release и Go native proof

Пара с opt-in sv_test_lockstep_peer теперь поддерживает fixed release100.
Режим ограничен cheats, g_test_combat_clients=2, harness instance и ip127.0.0.1.
Map timers/items/movers удерживаются как в одиночном fixed fixture. После
обоих ClientEndServerFrames native RNG повторно сбрасывается на собственный
g_test_seed; оба Blaster приводятся к gunframe9. Старый двухклиентный режим
без fixed release сохраняет прежнее поведение.

Первые поля smoke-v1 подтвердили native release100, но verifier ошибочно
ожидал строку без engine prefix `sv_test_combat ...`; прогон отклонён,
raw report сохранён. После исправления smoke-v2/v3 прошли параллельно с
seeds186100/186101. Каждый дал239пар/478команд; source fingerprints неизменны.
Повтор smoke-v4 с seed186102 дополнительно проверяет native позиции обоих
игроков, health100, fixed world hold/free pool receipt и сохраняет SHA game.dll,
server и Go client. Все проверки прошли. Все эти сцены без боевых hazards;
они не доказывают качество policy или полную эквивалентность reset.

Добавлен learningenv.ReadPairedNativeSteps и cmd/q2coop-proof. Go parser
проверяет обе queued-команды, порядок применения primary→peer, identity,
sequence, actor, exact next frame и release внутри tick. Неполные,
дублированные, recovery/rejected и несогласованные пары отклоняются.
Damage indexes описывают общий tick, включая немедленные ClientThink эффекты;
их нельзя превращать в двойную награду двум игрокам без actor attribution.
q2coop-proof прочитал реальные логи v2/v3 и сохранил SHA и239 native pairs
с явным ppo_trainable=false.

Go проверки охватывают malformed/missing receipts, оба актора, границы
кадров и shared effect window; нейросетевых CPU вычислений нет. Одиночный
fixed harness повторно прошёл на seed186110 после изменения game DLL:
capture_complete/provenance_valid/capture_valid=true.

Далее — adapter парных команд и эффектов в q2combat-export с проверкой обоих
client traces, first-life terminals и reset inventories, seeded маршрут
ведущего и reward за наблюдаемый прогресс. До этих проверок кооп-сценарии
реестра остаются planned, а обучение на их данных запрещено контрактом.

## Парные переходы и проверка реального боя

q2combat-export получил experimental `--paired-peer-trace` и
`--paired-role 0|1`. Перед экспортом читаются ВСЕ команды обоих клиентов.
learningenv.ReadPairedTrace проверяет envelope capture/observation/action,
одну fresh connection и exact identity. NativePairs.BindTraces сверяет
actor/frame/sequence/command каждого участника с полной native парой.
Только после этой проверки выбранный actor передаётся существующим
ExecutionIndex, Assembler и DamageJoiner.JoinNative. Наблюдения напарника
не подмешиваются в observation или features выбранного policy.

Read/Bind tests и отрицательная проверка на настоящем логе подтвердили:
отсутствующий peer, подменённая команда, duplicate actor/reconnection,
неправильная observation/applied identity отклоняются. Tampered copy
с согласованно изменёнными sent_command и capture.applied_command была
отклонена при native binding ещё ДО создания выходного каталога.
Оригинальные trace/log/weights не менялись.

Реальный bounded combat smoke:
`coop-paired-combat-export-smoke-v1-20261010`, seed186200,×2,release100,
один Soldier в geometry base1; learner — stochastic v8 attention64 parent
после update5, peer — обычные правила. Navigation planner активен:
testCombatOnly не используется. Бочки исключены из данного fixture.
Native capture accepted,239 пар/478команд, все source fingerprints стабильны.

Оба actor exports имеют151 переход, observed reset и synchronous proof.
У learner18 provider-owned steps, во всех18 записан navigation_context.
Убийство принадлежит peer:30 monster health damage и1kill. Learner имеет
0 такого урона и0kills; peer effects не стали наградой learner.
Полученный эффект не доказывает superiority модели; качество этого боя
соответствует фактическому отсутствию попаданий learner.

Каждый report сохраняет SHA выбранного trace, peer trace, native log и reset
expectation с проверкой неизменности к концу export.
`scripts/audit_coop_paired_export.py` сверил exported commands/effect actor
attribution и сохранил SHA всех dataset files в
`paired-export-verification.json`. Тесты и audit не выполняют NN inference.

Обычный paired reward export отклоняется, `paired_training_ready=false` явно
записан в report; q2ppo-data отдельно отклоняет paired datasets до validated
двухклиентного training adapter. Состояние registry scenarios остаётся planned.
Следующий этап: закреплённый seeded маршрут ведущего, совместный reset
receipt, first-life/goal supervision, наблюдаемый navigation-progress reward
и CUDA-only on-policy finalization. Веса не дообучались и не promoted.

## Экспериментальная navigation reward v10

Добавлены navigation_reward.go и combat-reward-navigation-v10.json.
Для follow/cover используется только наблюдаемый сейчас напарник;
для search/probe требуется известный last-seen возраст не более 200 кадров.
Недоступные маршруты, неподдерживаемые цели и displacement более 128 units
маскируются. Reference — текущий наблюдаемый goal или waypoint, закреплённый
в мировых координатах на один переход. Перемещение напарника в следующем
кадре не может создать прогресс для неподвижного learner.

Положительный bounded potential: phi(d)=0.1*(1-clamp((d-comfort)/1024,0,1));
добавка gamma*phi(after)-phi(before), gamma=0.99. Terminal/handoff обнуляет
последующий potential. Стояние даёт неположительную добавку; discounted
замкнутый маршрут при фиксированном anchor также неположителен. Меняющиеся
route anchors остаются локальной экспериментальной эвристикой: глобальная
оптимальность или policy invariance не доказаны.

Проверены approach, stationary при движении partner, closed loop, stale/unseen
references, unreachable route, teleport, waypoint, terminal, nonfinite inputs,
коэффициенты и неизменность v8 без navigation component. Выполнено:
`go test ./internal/learningenv -run '^(TestNavigationReward|TestPaired|TestActionQuality|TestSelectedAim)'`.
Это geometry/reward/protocol проверки без CPU NN вычислений.

q2combat-export допускает только явно запрошенный аудит v10 через
`--paired-experimental-reward`. SHA reward config включён в source proof;
paired_training_ready остаётся false и q2ppo-data по-прежнему отвергает
paired training datasets.

Закрытый capture coop-paired-combat-export-smoke-v1-20261010 повторно
экспортирован в dataset-role-0-reward-v10 и dataset-role-1-reward-v10.
Каждый:151 transitions,150 available rewards,1 incomplete final transition.
Learner:18 provider steps, navigation sum -0.1246250520, полный score
-0.4035713820; monster damage reward=0, kill reward=0.
Rules peer: damage reward=0.3, kill reward=5, полный score=5.0171295449.
Scripts/audit_coop_navigation_reward.py проверяет неизменность source hashes,
суммы компонентов, mask counts и раздельный credit; результат сохранён в
navigation-reward-verification.json рядом с исходным capture. Это повторная
обработка записанного боя, не новый live прогон и не обучение.

## Движущийся peer: live проверка и граница совместного эпизода

run_coop_lockstep_smoke.ps1 получил MovingPeer. Только role1 переводится
в idle и выполняет обычные usercmd по AAS маршруту: начало через25 кадров
после server release, bounded80 кадров. Learner не получает scripted movement
и не читает peer trace; navigation_context строится из его наблюдений.
Seed выбирает target: even128,-96,24.125; odd128,-160,24.125.

Первый v1 не открыл readiness barrier: idle actor не обновлял inventory.
client.go теперь запрашивает inventory для synchronous combat-barrier actor
даже при idle; его rules weapon/combat decisions по-прежнему отключены.
Для routed test walk LimitReason теперь сохраняет реальную причину остановки.

Live v2(seed186210) и v5(seed186211):×2,239 native pairs/478commands каждый;
closed source fingerprints, fixed release/RNG/player placements подтверждены.
audit_coop_lockstep_receipts.py сверил все10 usercmd fields у обоих клиентов.
audit_coop_moving_peer.py проверил startserverframe98, firstwalkframe123,
задержку и конец walking window, отсутствие выстрелов peer и соответствие
teammate_relative обычным learner observations. NN проверки на CPU не запускались.

v2: peer displacement131.75, target error10.44,10provider steps. Soldier
убит learner: native actor1 наносит20+10healthdamage. Это один успешный бой,
не оценка качества модели. Веса не изменены.

v3/v4 выглядели как застревание peer; v5 с LimitReason уточнил причину:
actor_dead. Native damage log подтверждает смерть actor2. Первый dead peer
observation на serverframe178; displacement59.31, target не достигнут,
137provider steps. Learner убил Soldier позже, когда peer уже погиб.
Auditor сохраняет route_reached=false/peer_died=true, не объявляет маршрут
успешным и не допускает такой capture к PPO.

Эти captures продолжаются до исходного bounded horizon, поэтому joint
terminal supervision ещё отсутствует. Следующий обязательный шаг — закрывать
оба actor segments на native-confirmed смерти любого участника, учитывать
потерю peer отдельным outcome и сохранять согласованный joint stop receipt.
PPOTrainable=false сохраняется; human session не перезапускалась.

## Общий terminal: native событие плюс оба наблюдения

learningenv.NativePairs.FirstDeathBoundary сначала связывает ВСЕ команды
обоих traces с native receipts. Затем ищет первую post-release смерть
участника в shared DamageIndexes. У обоих игроков требуется положительное
наблюдаемое здоровье до tick; после него обязательны оба next observations
точно на EndFrame. Native смерть должна согласоваться с observed health.
Boundary содержит map/spawncount/seed, оба actor IDs, здоровье before/after,
shared begin/end frames и native death event indexes. Это offline supervisor
evidence, не часть policy inputs. Tests проверяют смерть каждой роли,
одновременную смерть, отсутствие next observation, изменённое здоровье,
повреждённые команды/индексы/кадры/map и selected actor terminal marking.

q2coop-proof --primary-trace/--peer-trace сохраняет joint_death_boundary и
SHA обоих traces. q2combat-export --paired-stop-on-death завершает выбранную
роль на этом общем tick и исключает tail из steps/server effects; все tail
commands остаются проверены в полном dispatch proof. Этот режим пока
несовместим с reward export. Paired PPO guard остаётся включён.

Scripted peer после native release сохраняется как diagnostic transitions
даже при test_idle; teleport/setup overrides остаются исключёнными. Это
включается только для role1 paired-stop export без reward. Реальный peer
может не видеть монстра при reset: explicit allow_unobserved_enemy разрешён
только paired participant reset proof. Pose/resources/inventory проверяются
строго; enemy_presence_and_pose_not_observed сохраняется в Unverified,
FullServerResetConfirmed остаётся false. Одиночный VerifyReset отвергает
такой opt-in. Старый default reset contract сохраняется.

coop-moving-peer-v5-20261010: native boundary177→178, actor2 health4→0,
actor1 health100→100. dataset-joint-role-0 и dataset-joint-role-1-v3 имеют
по80transitions,1terminal,0truncations. Late Soldier kill после peer смерти
отсутствует в обоих exports. audit_coop_joint_terminal.py проверил оба
набора, source SHAs, полный239-command proof каждой роли и boundary;
результат joint-terminal-verification.json. Неудачные diagnostic exports
role1 и их причины сохранены, исходные captures не изменены.

Остаются live joint stop (capture пока всё ещё идёт до bounded horizon),
joint failure reward, validated training reset adapter и CUDA finalization.
Это проверка boundary/export, не запуск обучения и не quality evaluation.

## Живой stop на общей смерти

Native sv_user.c получил opt-in sv_test_pair_stop_on_death. Включается только
в существующем localhost/cheats/harness paired lockstep с combat barrier и
fixed release. После завершения обоих receipts текущего tick проверяется
STAT_HEALTH обоих участников. При смерти фиксируется sv_test_pair_stop,
мир удерживается на этом кадре и последующие ClientThink не выполняются.
Обоим клиентам надёжно передаётся test_pair_stop FRAME. Generation reset
снимает остановку; без opt-in обычный single/paired путь сохраняется.

Go client ждёт именно этот snapshot, не отправляя новые usercmd, и сохраняет
terminal_observation_only перед штатным disconnect. Это только наблюдение:
нет client_sequence/sent_command, provider=terminal_observer, нет policy sample.
ReadPairedTrace и BindTraces принимают такую запись только последней, с
правильным actor/generation/endframe и нулевой командой. Ассемблер получает
next observation завершающего реального действия, а dispatch proof проверяет
только настоящие отправленные команды. PPO guard по-прежнему включён.

Live coop-live-joint-stop-v1 и повтор v2, seed186211,×2: native stopframe178,
168pairs/336commands вместо прежних239pairs/478commands. Source/native
fingerprints стабильны. У обеих ролей169trace rows:168commands+1final
observation. World не продвинулся после stop. Оба экспорта имеют80steps,
1terminal,0truncations. q2coop-proof, command receipt audit и joint terminal
audit подтвердили границу и отдельный учёт actor effects; live_stop_verified=true.

Контроль coop-live-joint-survival-v1, seed186210: stop opt-in включён,
оба участника выжили,239pairs/478commands, ложной остановки нет. Learner
убил Soldier; это повтор bounded smoke, не новая оценка качества/обучение.
Проверки final observation serialization, last-native-tick closure, malformed
terminal/envelope, обеих ролей и reset прошли без NN вычислений на CPU.
Native pair primitive tests обоих arrival orders/duplicates/reset прошли.

Следующий этап: отдельная reward objective за совместный провал, с явным
учётом peer death без подмешивания peer trace в модель; затем согласованный
training reset/manifest и CUDA-only finalization. Веса не изменены.

## Coop reward v11: потеря напарника

Добавлена отдельная objective combat_reward_v11 в
scripts/scenarios/combat-reward-coop-navigation-v11.json. Сохраняет v10
aim/navigation/spacing/action costs, вводит peer_death=-5. Выживший actor
получает этот cost только на native-confirmed joint terminal. Собственная
смерть остаётся death=-5; при одновременной смерти peer cost не складывается
с own death. Чужой monster damage/kill не переносится в actor reward.

JointDeathEvents сохраняются отдельно от actor-specific Events/received damage.
Reward verification сверяет native damage indexes, actor IDs, map/spawncount,
shared tick frame, health arithmetic, оба observed health before/after и
подтверждённый MarkTerminal. Без этой evidence, при неправильном actor/frame,
missing index/health, нетерминальном переходе либо legacy objective reward
отклоняется. Peer trace остаётся offline proof, не входом модели.

q2combat-export допускает v11 только при explicit paired-experimental-reward
и paired-stop-on-death. v10 paired audit и v11 joint export разделены;
PairedTrainingReady=false сохраняется. q2ppo-data eligibility guard не снят.
Исходный v10 config не изменён: повторный export learner старого
coop-paired-combat-export-smoke-v1-20261010 дал побайтно совпадающие steps,
server_outcomes и rewards относительно dataset-role-0-reward-v10.

На coop-live-joint-stop-v2-20261010 экспортированы
dataset-coop-reward-role-0/1: по80steps/80available rewards, terminal178.
Learner peer death cost=-5, own death=0, полный score=-5.2693985510.
Peer own death=-5, peer cost=0, полный score=-7.3388271283, включая собственный
полученный урон. У обоих monster damage/kill reward=0: kill после старого
tail не попал в эпизод. audit_coop_joint_reward.py проверил sums, source SHA,
terminal/shared death evidence и отдельный actor credit; результат
joint-reward-verification.json. Tests death каждого actor, simultaneous death,
invalid evidence/config и nonterminal zero peer cost прошли без NN на CPU.

Это проверка reward arithmetic на закрытой live записи, не обучение модели.
Следующий этап — единый paired training manifest/reset adapter и CUDA-only
finalization с проверкой настоящих stochastic policy samples, затем GPU update.

## Primary paired adapter и CUDA pilot

Добавлен cmd/q2paired-data. Это отдельный experimental adapter; запрет
paired datasets в обычном q2ppo-data не снят. Пока adapter принимает закрытый
fixed-release v11 live-death capture и только role0 learner. Проверяет SHA
исходных/seeded weights, server/client/game binaries, seed и policy version,
оба traces/native dispatch через повторный q2combat-export, observed reset,
joint terminal и rewards. Steps/rewards/server_outcomes повторного export
должны побайтно совпасть с исходным verified dataset. Все inputs хешируются
до/после подготовки. Peer/rules команды не входят в learner rollout.

На CPU выполняются только загрузка весов, sampled action contract и feature
extraction; Review/VerifyMemory/ValueAfter не вызываются. Полные learner
context rows сохраняются отдельно, bootstrap остаётся CUDA-pending. Каждый
sample должен иметь behavior version и sampling_seed данного case.
Wrong behavior model и role1 dataset отклонены до создания output.

coop-paired-ppo-native-v1-20261010:80learner rows/80context rows,1joint terminal,
featurev8 width881, weights source cc941bb38ccb8e46e7530cf457580b58981c604732fe58d8954ad812495567d4.
finalize_combat_cuda_rollout.py проверил это на RTX5070/CUDA:
max log_probability error6.63e-5,value error4.56e-6,actor memory2.26e-6,
critic memory2.15e-6; terminal next_value=0, next resets0, zero bootstraps1.
Результат coop-paired-ppo-cuda-v1-20261010. Pilot paired_training_ready=true
выставляется только после CUDA validation; registry-wide eligibility не меняется.
audit_coop_paired_cuda.py проверил source/model/rollout/context SHAs, sampler
identities, bootstrap и CUDA receipt; paired-ppo-verification.json сохранён.

Это один failure episode для проверки pipeline. Для осмысленного GPU update
нужна смесь успешных/неудачных парных эпизодов с общими weights и objective,
а не дообучение только на этом проигрыше. Следующий шаг: поддержать successful
horizon/goal episodes в adapter, собрать balanced pilot через пул, затем
перенести reward objective checkpoint и выполнить CUDA PPO update. Веса
на данном этапе ещё не обновлялись; superiority не оценивалась.

## Первый кооперативный CUDA update

Следующий этап выполнен: `coop-learning-pool-v1-20261010` содержит 32/32
принятых записи (seeds 186240–186271), 16 slots, timescale 2. Все 32 прошли
native replay и CUDA finalization. Получено 1422 eligible learner/context rows,
20 убийств монстра learner и 12 эпизодов с гибелью участника. Это один участок
base1 с Soldier и бластером, scripted moving peer; результат не является
оценкой полного прохождения или игры с человеком.

Adapter поддерживает surviving horizon и control handoff. В seed 186271
learner передал управление правилам до joint death: retained learner segment
имеет zero bootstrap на handoff, без фиктивной terminal-команды learner.
Поздняя гибель не приписывается действию модели. Это ограничение credit
assignment требует отдельного исправления владения боем при потере видимости.

CUDA verifier sources заморожены внутри каждого output; дальнейшие правки
Python не меняют проверенные receipts. `coop-learning-merged-v1-20261010`
проверяет все source SHAs, сохраняет отдельные seed/context и member outcomes;
не выдаёт исход первого эпизода за исход всего набора. Незавершённый paired
member и смешивание paired/unpaired metadata отклоняются. Focused merge tests
прошли, также выполнено реальное объединение всех 32 CUDA outputs.

`coop-learning-objective-fork-v1-20261010` использует verified migration
receipt v8 и canonical reward v11 SHA
`1aa6167b29e1d2390f90210b29baef73d319f7133c9cd410fc4fa676a94df1c8`.
CUDA read-back подтвердил сохранение actor/critic/std и обоих Adam states.
Objective SHA изменён явно; прежние consumed rollouts сохранены.

`coop-learning-update-v1-20261010` — реальное PPO update на RTX5070,
Torch 2.10.0+cu128: 1422 rows, 10 принятых actor steps, 40 critic steps,
updates_completed 6 (ранее 5), total_actor_steps 55. Final approximate KL
0.004999278 при лимите 0.005. Веса SHA
`2c51b074df52ad589914d26f7b4819df9e9d1a95634c2e13f356011bff38e4d3`.
Сохранены checkpoint и complete seal. Retention weights остаются 0/0 как у
родителя; проверки сохранения старых боевых навыков предстоят.

Запущен `evaluate_coop_learning_pair.py`: before/after по 32 боя на одинаковых
seeds 186400–186431, не пересекающихся с update collection. Всего не более
16 slots одновременно, timescale 2; scripted peer и геометрия одинаковые.
Итог должен находиться в `coop-learning-eval-v1-20261010/comparison.json`.
Пока comparison не получен, улучшение качества и promotion не установлены.

### Перезапуск оценки после signon failures

Первая серия `coop-learning-eval-v1-20261010` остановлена: 19/32 accepted,
13 invalid, ветка after не запускалась. В rejected server logs есть
`SZ_GetSpace: overflow` / `PairLearner overflowed` во время configstrings,
до release боя. Эти случаи не считаются поражениями модели и не входят в
сравнение. Общее завершение серии подтверждено exit code 1 orchestrator.

Harness теперь запускает peer после authoritative `PairLearner entered the
game`, вместо фиксированной паузы 600ms; world по-прежнему held at frame10.
Серия v2 выявила ошибку чтения открытого server log через ReadAllText;
заменено на stream с FileShare.ReadWrite. v2 завершена и не является quality
evidence. Серия `coop-learning-eval-v3-20261010` повторяет тот же protocol,
seeds и модели с исправленным signon readiness. Результат нужно брать из
её comparison.json после завершения обеих веток, а не из частичных v1/v2.

v3 завершилась с 31/32 accepted, без ветки after. В оставшемся seed186416
peer несколько раз отправлял `new` с интервалом 2s до ответа serverdata;
после задержки около 10s получены повторные serverdata/configstrings,
`SV_Configstrings_f: skipping index 60: too big to send` и overflow.
Для synchronous client handshake retry увеличен до 20s. Обычный live client
сохраняет прежние 2s. Focused TestHandshakeAcknowledgesRepeatedRequest прошёл.
Оценка повторно запущена в `coop-learning-eval-v4-20261010`; текущий результат
качества должен браться только из завершённого comparison.json этой серии.
