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
