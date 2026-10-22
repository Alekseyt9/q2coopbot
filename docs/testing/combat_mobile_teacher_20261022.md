# Подвижный rules teacher в изолированных боях

Статус:исправление short retreat проверено тестами и48/48 новыми native captures. Readiness,canonical features,native motion audit и coordinated ground labels закрыты. GPU обучения на этих метках ещё нет.

## Что мешало обучению

Первый joint teacher имел4157 alive frames,8 movement commands,0 jump/crouch. Это не подходящие данные для обучения манёврам. Config уже был campaign+combat_only=true,поэтому простое переключение campaign mode не решало проблему.

`Planner.update` при testCombatOnly устанавливает цель `combat_fixture`,сохраняя enemies/geometry и отключая дорогой поиск полного пути к выходу. Но `commandAt` разрешал короткое боевое spacing только при teammate goal либо `reach_level_exit`. Поэтому рабочий `combatRetreat` не вызывался для close threats в fixture. Более высокий результат старых rules в отдельных smoke не означает,что проверялись полноценные подвижные правила кампании.

Исправление добавляет `testCombatOnly && Goal==combat_fixture` к условию spacing. Контроллер отступления использует прежние проверки наблюдаемой угрозы,ground/BSP,leash и group traps;правка не строит маршрут через всю карту. В допустимом direct learned кадре `combatCommand` по-прежнему не вызывает `Planner.command`:правила не получают управление боем обучаемой политики.

## Проверка кода

Новый `TestIsolatedCombatKeepsShortRetreatWithoutExitRoute` на настоящем base1 BSP сначала воспроизвёл ошибку:close parasite,готовое geometry,zero movement,`no_movement_goal`. После правки тот же тест подтвердил `combat_retreat` и ненулевую команду при сохранённой цели combat_fixture,без route/hasGoal. Existing observation-retention/spacing/group-trap tests тоже прошли.

Команда проверки:

```powershell
$env:Q2_SEARCH_SCAN_ROOT='F:/SteamLibrary/steamapps/common/Quake 2/YamagiQ2/baseq2'
go test ./internal/bot -run '^(TestIsolatedCombat.*|TestCombatSpacingUsesThreatAndWeapon|TestCombatRetreatPreservesAimAndRejectsGroupTrap|TestCombatRetreat.*)$' -count=1
```

Это geometry/controller tests,без numerical NN inference/обучения на CPU. AAS в этом asset root нет;короткое отступление проверено непосредственно по BSP,не по AAS route. Shared AAS/PAK для live harness сохраняются.

## Новые данные

`combat-rules-mobile-teacher-v1-20261022`:32 train/16 development,те же8 group families на base3/jail1/city1/ware2,новые заранее закреплённые seeds,начальные offsets420/228,порт45100. Реальные offsets и historical collision check находятся в protocol. Reference target/fire BC нужен только для происхождения будущего ученика;нейросеть при rules collection не запускается. Pool и все derived outputs выполняются последовательно.

После native seal автоматически идут raw target/history readiness и canonical Go feature/native dispatch export в `combat-rules-mobile-features-v1-20261022`. Этот exporter пока не создаёт movement supervision:данные после исправления нужно проверить на реальное перемещение/покрытие и подготовить отдельный coordinated movement/aim протокол. Не выдавать ненулевые команды за полезное или успешное движение. Прыжки/приседания и сложные target priorities этим исправлением не обеспечиваются.

## Связь с предыдущим screen

Закрытый screen target/fire BC при прежнем fixture:13/24,source PPO10/24,rules12/24. Ordinary7/16,6/16,4/16;recovery6/8,4/8,8/8. Парно candidate/source4 gained/1 lost,candidate/rules3/2. Пять отчётов/72 native/48 CUDA и hashes проверены. Candidate остаётся development-only,promotion=false;coverage отложен до проверки исправленной baseline. Старые captures/seals не переписаны.

Actual MG damage contacts source256/554 (46.21%),candidate335/481 (69.65%). Large yaw9.67→4.34%,target churn34.92→11.11/100;clear-enemy standing16.27→19.34%,max stationary run45→70. Урон полученный1639→1641. Это полезный сигнал по прицелу/цели,но общего подтверждённого превосходства и решения проблемы стояния нет. Новая подвижная baseline меняет сравнение;её результаты нельзя выдавать за прямое продолжение старой неизменной матрицы.

Следующие этапы:закрыть live movement evidence,подготовить native-matched movement/aim labels с causal history,задать небольшой GPU-only BC/PPO бюджет,затем fresh сравнение с исправленными правилами. Не делать оптимизатор из старых оценочных траекторий.

## Закрытый live corpus

Все48 captures валидны,pool source unchanged. Rules goals:train24/32,development9/16. Эти seeds новые,поэтому нельзя напрямую приписывать отличие от старого teacher исправлению движения. Общее6642 alive frames,2708 movement commands,0 jump/crouch. Raw target/history audit прошёл. Feature export:4277 train/2365 development rows,сжатые features вместе2.22МБ.

`audit_registered_teacher_motion.py` дополнительно использует native matched/exclusive proof и проверяет реальный соседний переход:

| Coverage | Train | Development |
|---|---:|---:|
| Eligible grounded transitions |4223|2343|
| Requested horizontal movement |1843|860|
| Из них displacement>2 |1685|791|
| Запрос был,но next speed<10 |149|87|
| Нет горизонтального запроса |2380|1483|
| Displacement>2 без запроса (инерция) |79|36|

2476 moving transitions подтверждают,что новые команды часто реально перемещают бота. Это не доказательство оптимальных манёвров или полезности каждого перемещения.

## Готовые ground movement/aim candidates

`prepare_registered_teacher_motion.py` создал `combat-rules-mobile-motion-labels-v1-20261022`. Все context rows сохранены. Метка требует available explicit teacher target,grounded release,соседний alive кадр,native matched/exclusive. Для движения нужны displacement>2 и отсутствие move limit;для stationary примера допускается только явный `combat_spacing_hold`,а не любое стояние/тупик. Desired forward/side и yaw/pitch берутся из одного исполненного teacher action. Прицел получает fine mode при обоих |delta|<14.85°,иначе coarse180;open tanh support проверен. Это фиксированная supervised mode convention,не утверждение об оптимальной mode.

| Labels | Train | Development |
|---|---:|---:|
| Moving |1662|781|
| Explicit spacing hold |219|474|
| Fine mode |1815|1232|
| Coarse mode |66|23|
| Full context rows |4277|2365|

Будущее наблюдение используется только как quality gate меток;оно не добавлено в input features или past target intent. Нет weapon/fire labels нового решения и нет vertical supervision. Учитель всегда выбирал ближайший доступный target;сложные target priorities отдельно не проверены. Дальше требуется coordinated CUDA fit с сохранением остальных heads/critic/std и свежая игровая оценка против исправленных правил;raw corpus/labels сами по себе ещё не обученный контроллер.
