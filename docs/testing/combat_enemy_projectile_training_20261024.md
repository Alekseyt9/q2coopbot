# Обучение на реальных вражеских снарядах — 24.10.2026

## Итог на 21:03 МСК

Workflow закрыт: 192 TRAIN native / 192 CUDA, 20258 переходов; 40 evaluation native / 32 CUDA. Итоговый report SHA `f9ca1c3cfa672aa21f84703fa74b86d6137eeb374bab64898534f8279611d648`. Объявленный pre-fit source revision сохранён: source_unchanged=true, original_source_unchanged=false, recovery SHA `d97e263307b192acdd2db9fd3e2c4005fe14128d12c7061b983a9182a9d70260`.

| Вариант | Цели / 8 | Смерти | Убийства | Получено HP |
|---|---:|---:|---:|---:|
| Neutral initial | 4 | 3 | 13 | 434 |
| Neutral trained | 5 | 2 | 14 | 328 |
| Typed initial | 4 | 3 | 13 | 434 |
| Typed trained | 5 | 2 | 15 | 386 |
| Rules | 2 | 5 | 9 | 468 |

Типизация снарядов не прошла gate: нет +2 целей относительно обученного neutral, а 386 получено HP больше допустимых 110% от 328. Наличие реальных enemy bolts доказано, но польза их типизации в этой схеме обучения не установлена. Promotion=false, default прежний.

Лишние действия не устранены. Initial → neutral-trained / typed-trained: native MG live damage fraction 55.02% → 56.54% / 56.46%; solid contacts 124/289 → 115/283 / 109/271; standing среди grounded-visible кадров 44.87% → 39.20% / 39.46%; yaw >=30 градусов 86/968 → 79/864 / 107/976. Траектории и число кадров различаются, это описательная диагностика, не причинная оценка манёвров.

Следующая отдельная гипотеза — независимо подтвердить neutral final против своего initial и rules на всех 24 balanced рецептах, без fitting. Выбор neutral выполнен после просмотра этого малого screen; это не успех исходной typed гипотезы. Кандидат не меняется по промежуточным результатам, пороги фиксируются до новых игр.

### Завершение обучения, 20:55 МСК

Все четыре TRAIN раунда завершены: 192 native / 192 CUDA, 20258 принятых переходов, по четыре настоящих обновления каждой ветки. Round04 typed: 2564 rows, 9 actor / 40 critic steps; neutral: 2380 rows, 5 actor / 40 critic steps. Все проверки checkpoint roundtrip и disposable actor/critic Adam continuation прошли.

Final typed weights SHA `194b9a376b5cc7d49ba70851b3fba4cd4f8a2cbaa787b91940959f2ad1c18e89`, neutral `1b96f67cbe5f4d1d89ac3d79f3c425ee8178b942b7e9b8a8c11aaf2d903cf6a7`. Последние exposure gates прошли: typed 38 enemy bolt slots / 25 known velocities в 5 случаях, max L2 1.5720007; neutral 27 / 14 в 7 случаях, max L2 1.6347269. Live infighting=0.

Supervisor PID7476 перешёл к фиксированной оценке, child21932. Подготовлены пять планов для 40 native / 32 CUDA evaluation случаев; исходные и final endpoints заданы заранее. Качество ещё не установлено, default не заменён. Сравнение не использует промежуточные checkpoint для выбора победителя.

### Третий раунд, 20:43 МСК

Третий раунд также завершён, 48 native / 48 CUDA, 4719 переходов. Typed: 2369 rows, 7 actor / 40 critic steps; neutral: 2350 rows, 18 actor / 40 critic steps. Обе ветки scope_updates=3, checkpoint roundtrip и disposable actor/critic Adam continuation=true. SHA typed `952bc73c00ecb71c5e710638d6fc80e96cc7ce4511b8d8a7337b45ae2cbcf248`, neutral `db9d19892f10b7444aa3720cca8ecd5066461a82a4a67c28f7b73715548a7032`.

Pre-fit exposure третьего раунда прошёл у обеих веток: typed 34 enemy bolt slots / 20 known velocities в 6 случаях, L2 max 1.4910168; neutral 22 / 14 в 4 случаях, L2 max 1.1950925. Live infighting=0. Всего за три раунда 144 native / 144 CUDA и 15314 переходов. Supervisor PID7476 жив, началась подготовка четвёртого раунда; stderr пуст. Новых оценочных результатов ещё нет, default не менялся.

### Второй раунд, 20:32 МСК

Второй раунд завершён: ещё 48 native / 48 CUDA случаев, 5625 принятых переходов. Typed: 2717 переходов, 10 actor / 40 critic steps; neutral: 2908 переходов, 30 actor / 40 critic steps. У каждой ветки scope_updates=2; checkpoint roundtrip и реальные disposable продолжения actor/critic Adam прошли. SHA весов typed `8dfe2927d029034d596731c1456adbb1b0ae492acc9ef15a767b181ca55fe45b`, neutral `5dcf0d640d28394817bc105a58b379e4e6723eee747867d16a0cfbd87535ea0c`.

Оба pre-fit exposure gate прошли. Typed: 39 enemy bolt slots, 23 с известной скоростью в 7 случаях, максимальная ошибка 1.3142504 ед/с; neutral: 38 / 28 в 5 случаях, максимум 2.2020073 ед/с. Live monster infighting=0 у обеих веток. Число принятых actor steps различается из-за неизменного KL guard при одинаковом предельном бюджете; это не дополнительный опыт одной ветки.

Итого два раунда: 96 native / 96 CUDA, 10595 переходов. Supervisor PID7476 жив, началась подготовка typed корпуса третьего раунда. Источники и протоколы не менялись. Оценочные бои ещё не начались; эти числа подтверждают выполнение обучения, а не улучшение качества.

Предшествующий typed/neutral опыт получил6/8целей против5/8контроля и4/8rules,нонепрошёлзаранеезаданныйgate. В его TRAIN все198bolt slots были собственными. Последующий exact soldier_light пилот и отдельный native lifetime journal подтвердили реальные вражеские болты:28/28observed slots,19known velocities,L2max1.254ед/с,4same-world пары/356кадров безизменениядействий. Эти проверки технические; качество новых весов ещё не установлено.

## Условия и сравнение

Реестр `workspace/artifacts/combat-enemy-projectile-curriculum-v1-20261024/registry/index.json`:24заранеевыбранныхусловия,base3/city1/jail1/ware2,двеBSPточкинакаждой. Составы light soldier/gunner и parasite/light soldier/gunner:12пар/12троек. Loadout blaster/machinegun/weapons:по8условий. Это сбалансированная дробная схема, а не полный факторный перебор; взаимодействия map/site/loadout отдельно не оцениваются. Builder `scripts/prepare_combat_enemy_projectile_curriculum.py`, все placements проверены обычным Go generator по реальному BSP.

Typed и neutral стартуют из прежней одинаковой пары initializer, а не из выбранных по оценке новых весов. У typed есть виды снарядов, у neutral только их presence; сеть/Adam/RNG/scope/LR одинаковы. Reward v14 cost без изменения, восемь optimizer групп и общий обучаемый bias. Никакого teacher/ручного уклонения или native owner labels во входах/loss нет.

Четыре own-policy раунда по24свежихбоянаветку:192native/192CUDA. Общаядинамическаяочередь48заданийнараунд,пул16,×2. ПервоначальныйTRAIN offset4; обычный collector сверяет всю физическую историю и ищет unused offsets. Обучение и численные neural checks толькоCUDA. Лимиты30actor/40critic steps иKL прежние; промежуточные checkpoint по качеству не выбираются.

## Допуск данных перед fitting

`scripts/audit_combat_projectile_training_exposure.py` исполняется после CUDA replay каждого собственного корпуса и до обновления. Native lifetimes используются только для диагностики/допуска данных, не как скрытая информация политики.

Для первого update каждой ветки заранее заданы минимум8enemy velocity rows минимумв2боях. Это минимальное покрытие механизма, не доказательство достаточного бюджета обучения. В последующих own-policy update покрытие описательное: агент может начать убивать стрелка до выстрела, поэтому уменьшение числа enemy bolts само по себе не блокирует прогресс. Каждый update требует grenade presence, отсутствие monster→monster урона и ошибку известных enemy velocities не выше10ед/с. Same-frame reuse/unmatched slots не объявляются доказанным владельцем. При провале фиксированные данные сохраняются, ресэмплирования нет.

Auditor проверен на16ранееCUDA-подтверждённыхfresh случаях(1991rows) через точный metadata adapter безкопированияданных:positive прошёл,контрольснулевымvelocity error capотклонён. Это scalar data/physics проверка, не neural CPU execution. Реальный train использует24случаясреальнымprotocolSHA, не smoke config.

## Проверка primary light soldier

Клиент наблюдает soldier model как canonical `monster_soldier`, независимо от native soldier_light. Для generated fixture после обязательного `Confirm-GeneratedCombatStart` exact light/20HP reset expectation теперь использует наблюдаемый canonical class. Native receipt по-прежнему требует фактический soldier_light, здоровье,solidиposition. Политика/декодер/нейронные признаки этим исправлением не меняются. Старый baseline source после закрытого native parity сохранён с SHA `b63b29b14fd454f098c4b926f182ad52161ece3e8d508da766c2911de14cc865` в retained-sources предыдущего опыта. PowerShell AST проверен. Новые pairs/primary conditions ещё должны пройти live first corpus до fitting.

## Оценка и запуск

После round04:40свежихnative/32CUDA,пятьвариантов×8условий. Screen включает все4карты/2точки,4пары/4тройки,4weapons/2blaster/2machinegun. Validation offset0,TRAIN не используется для оценки. Gate прежний:typed-trained минимум+2цели кneutral-trained,+1ксобственномустарту,нехужеneutral-initial/rules;смертейнебольшеneutral-trained/rules,полученныйурон≤110%обоих,model ownership. Автоматического promotion нет, положительный малый screen потребует независимого подтверждения.

Протокол `docs/combat_enemy_projectile_training_protocol_20261024.json`, SHA `85fc0325975ccdc3c12f292fa241c0fc1c7557e127bbc47cecb86be6b69486e0`. Spec `docs/combat_enemy_projectile_training_screen_spec_20261024.json`, SHA `8113318d6eb8ed87b737442e0279b4dc2347a72abc23b6b19df25d199e973697`.

Runner `scripts/run_combat_enemy_projectile_training.py`, output `workspace/artifacts/enemy-projectile-ppo-v1-20261024`. Новый runner/reporter сохраняют старые версии опыта, проверяют pre-fit exposure receipts и отдельно фиксируют новый training protocol binding существующим проверенным trainer. Начальные seal/weights/checkpoint/report SHA сверены;65inputs initializer parity и все24reward bindings проверены. Запуск выполнен, источники и JSON заморожены до завершения. Первое настоящее обновление и качество ещё не подтверждены.

## Первый live раунд

Первый общий пул закрыт48native. На19:39:46 parent PID19068жив, стадия `r01-finalize-typed`, child22648. Ошибок в просмотренных terminal receipts нет; CUDA и pre-fit exposure ещё не закрыты.

Конкретный новый pair job0 seed1404140004: native start confirmed, player80HP/light soldier20HP/gunner175HP, expected observed class monster_soldier, capture_complete/provenance_valid=true. Это live подтверждение class mapping для данного случая, а не только статический тест или разрешение неизвестного native класса. Source/model/reward не менялись.

## Остановка до первого fitting: offline class reconciliation

Около19:41 workflow завершился на `r01-advantage-typed`, до создания update directories. Первый typed корпус прошёл24CUDA replay и содержал2554принятыхперехода, но independent native credit audit нашёл43события вне принятых окон. Эта проверка не была ослаблена ради продолжения.

42события находились в40provider windows с reward unavailable `commitment_target_class_mismatch`: target commitment сравнивал canonical observed monster_soldier с exact native monster_soldier_light. Экспортёр поэтому исключал переход целиком, хотя сырые шаги и события сохранены. Это тот же известный decoder alias, но в отдельной offline reward проверке. Исправление принимает только точные native aliases light/ss при observed soldier; entity/track/map/spawncount/native-window/health доказательства сохранены, неизвестный или gunner class по-прежнему отклоняется. Коэффициенты, формула и бюджет v14 не меняются. Go scalar target-commitment/goal tests прошли.

Последнее43-е событие — incoming light bolt job16 seed1564140004 frame175. Принятый последний переход имеет verified goal handoff,bootstrap_zero=true,end174,next_health96; native goal kill174 уже закрывает учебную задачу. Supervisor подтверждает живой endpoint позже. Старый auditor относил к хвосту лишь события после supervisor observed_frame и ошибочно требовал включить попадание175. Теперь finite learning boundary определяется только по уже независимо подтверждённому living goal terminal/handoff снулевымbootstrap в интервале kill_frame..observed_frame. Последующие эффекты остаются видимыми в excluded post-goal tail и не переносятся в предыдущую награду. Три scalar regression tests проверили этот случай, не подтверждённый handoff и неверные endpoints. Причинные native окна и исходы не менялись.

До ремонта проверено отсутствие обновлений и ноль изменений исходного source fence. Копии четырёх изменённых файлов, исходный failure progress/log/native report и все старые derived exports сохранены в `offline-class-recovery-v1`. В927файлах архива сохранены SHA и origin mapping; исходные native captures не перемещались. Переэкспорт идёт в прежний рабочий processing path; старые отчёты сохраняют прежние абсолютные ссылки, а archive manifest указывает фактическое расположение сохранённых байтов.

Исходный run.json и оба JSON не переписаны. Новый `--offline-recovery` разрешает ровно объявленные4old/new source replacements и дополнительный scalar test, проверяет first-apply до fitting,48закрытыхnative captures,архив и recovery receipt. Возобновление после применения требует того же receipt; дальнейшие исходники снова заморожены. Workflow seal отдельно сохранит original_source_unchanged=false и recovery SHA.

Recovery SHA `d0fa73a2e1c9aa81c2096d091ebad6e75bb186ed7d704d95dc1d4e18cf50f981`. Возобновление запущено19:53:41,launcher8760. Новых native игр нет; коэффициенты награды, budget,model,TRAIN seedsиqualitygateпрежние. Исправленные native/advantage/exposure audits и первое настоящее GPU обновление ещё требуют подтверждения.

## Исправленный offline exporter при сохранении capture provenance

Первое восстановление вновь остановилось до fitting: audit уже правильно исключил позднее событие, но42class-mismatch effects остались, rows=2554. Причина — `q2ppo-data` по замыслу запускает замороженный `q2combat-export.exe` из capture. Пересборка batch reader не меняла код этого старого экспортёра. Оба неудачных набора derived outputs сохранены отдельно, исходные48captures и их exporter binaries не изменены.

Добавлен явный `--replay-exporter` с обязательным SHA, разрешённый только для CUDA-only batch export. Проверка исходного captured exporter по manifestSHA остаётся обязательной и исполняется до выбора нового. Метаданные сохраняют captured SHA, corrected exporter path/SHA и число восстановленных reward rows; source receipts содержат оба бинарника. Первый раунд использует этот путь явно; последующие fresh captures получат текущий исправленный exporter обычным harness build.

Шаги и server outcomes при этом обязаны остаться byte-identical captured versions. Для rewards отдельный verifier независимо повторяет scalar `SequenceReward` по тем же steps/effects и прежнему v14config. Допускаются только восстановление прежнего `commitment_target_class_mismatch` и вызванные им commitment-state/component/score изменения; посторонние identity/base component/score изменения отклоняются. Обычный exact reward replay путь не ослаблялся. Unit tests этой границы и существующего reward replay прошли, нейронных CPU операций нет.

Контрольный real capture job0 переэкспортирован:113rowsвместо112, восстановлен step66, steps/outcomes byte-identical, независимый reward replay совпал. Это native/scalar export с CUDA_pending=true, ещё не новая neural CUDA verification. Контроль с неверным corrected exporter SHA отклонён.

`offline-class-recovery-v2` сохраняет вторые927derived files сSHA/origin mapping и отдельный source revision:7изменённых существовавших файлов,3новыхscalar/helper source files. Receipt явно заменяет v1,нонепереписываетего. Protocol/spec/model/reward coefficients и бюджет прежние. Source fence теперь проверяет эту объявленную ревизию и фиксированный corrected binary SHA. Recovery v2 SHA `d3051b07fc0fc22b0ae99473da74b8e1eee438aefbb9281aa84052198a657393`.

Повторное продолжение запущено20:07:31,launcher31660;на20:08:53parent21980жив,стадияr01-finalize-typed,child19268. Повторных native игр нет. Первое настоящее обновление по-прежнему не подтверждено.

## Native credit восстановлен, corpse contact отделён от infighting

Второе восстановление успешно переэкспортировало24typed capture с исправленным бинарником и прошлоCUDA. Получены2594перехода вместо2554;40отброшенныхclass-mismatch окон восстановлены. Independent native credit теперь имеет0unassigned events,4980monster HP damage,43kills,1085received HP,6deaths. GAE/component arithmetic прошла,24segments. Это TRAIN diagnostics, не качество новых весов.

Pre-fit exposure остановил процесс до fitting из-за одного monster→monster event. Сырые поля:city1/job9/seed1548040004,networkframe154,gunner→soldier_light,MOD7,health_before−36,health_after−53,take17. Это попадание в труп. Native no-infighting gate сам защищает live SVF_MONSTER; dead bodies могут получать splash/gib damage. Никакой борьбы живых монстров этим событием не доказано.

Auditor теперь считает infighting только положительный live-health урон между разными монстрами, отдельно сохраняет corpse/self contacts. Требование нулевого live infighting не снижено, coefficients/thresholds/protocol не менялись. Три scalar теста проверили corpse event, даже1live HP и отдельные self/player случаи. Первый реальный корпус прошёл исправленный auditor:40enemy bolts/197self,31known enemy velocities в6cases,L2max1.2912ед/с,340grenade slots,unmatched/same-frame ambiguity0,live infighting0,corpse contact1.

Третий pre-fit recovery меняет только этот semantic classifier и явный source admission. Старый failed exposure report сохранён, все предшествующие source/derived архивы остаются проверяемыми. Source revision содержит8изменённых исходных файлов и4новыхscalar/helper source files; native data/модель/JSON не менялись. Recovery v3 SHA `d97e263307b192acdd2db9fd3e2c4005fe14128d12c7061b983a9182a9d70260`,запуск20:15:51,launcher29632.

## Первое настоящее обновление

На20:17:32 typed update1 закрытCUDA:24собственныхTRAINбоя,2594перехода,12actor steps/40critic steps. Weights SHA `dce80d42e71c405791a0cf9d8bb7396d99f621c24f245ccceb6a2abdab863ea3`. Новые projectile input columns иshared biasизменились,старыеbodycolumnsсохраненыточно. Checkpoint roundtrip и настоящие пробные actor/critic Adam continuations прошли; gradient новых columns0.23203,bias0.469479. JSON/native source log-probability ошибки в пределах действующих технических допусков. Преимущество в боях не оценивалось.

Parent7476жив,стадияr01-finalize-neutral,child24044. Используются те же48native captures первого пула, новыхnative запусков при ремонтах0. Neutral ещё требует CUDA/native/advantage/exposure проверок и своего первого обновления. Бюджет192TRAIN/40evalпрежний.

## Первый полный раунд завершён — около20:19МСК

| Ветка | Свежие native/CUDA бои | Переходы | Actor steps | Critic steps | Enemy bolt slots | Known enemy velocity slots / cases |
|---|---:|---:|---:|---:|---:|---:|
| typed |24/24|2594|12|40|40|31 / 6|
| neutral |24/24|2376|10|40|77|58 / 7|

Neutral weights SHA `bf30e6aaa96a92e21ae349cc7510e9c55df66133a5404a69a0dc7949297f0aad`,scope_updates1. Его native/advantage audits тоже закрыты с0unassigned effects. Pre-fit exposure всеchecks=true,live infighting0,unmatched0;known enemy velocity L2max2.21281ед/с. Native классификация использована только для аудита: neutral вход по-прежнему скрывает вид снаряда. Checkpoint roundtrip и настоящие пробные actor/critic Adam continuations обеих веток прошли; report SHA проверены по seal.

Итого48uniqueTRAINnative/48uniqueTRAINCUDA,4970переходов,поодномунастоящемуобновлению. CUDA переобработка тех же данных при ремонтах не считается новым TRAIN опытом; новые игры при восстановлении не запускались. Качество на holdout ещё не оценивалось. На20:19:07parent7476жив,стадияr02-prepare-typed,child31592. Последующие native captures будут сразу использовать исправленный offline exporter; первый пул использовал явно закреплённый replay exporter.

Для дальнейших новых классов/условий технический pilot должен включать полный native credit/advantage admission, а не только CUDA replay и projectile exposure: numerical replay сам по себе не выявляет исключённые unavailable reward windows. Эта серия обнаружила и исправила именно такой разрыв до первого fitting.
