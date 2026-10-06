# Fixed Machinegun pilot v1

06.10.2026. Протокол до результатов PPO/evaluation.

По запросу пользователя бот получает Machinegun и ровно100 Bullets перед
боем. Штатные recoil/spread, native ammo depletion и автоматическая смена
при исчерпании сохраняются. Сеть управляет movement/aim/attack/vertical;
learned weapon choice не добавлен. Выдача не выполняется во время боя.
Первый native pilot25800–25803:4 valid captures, confirmed Machinegun100,
native idle phase6/skill1, observed provider ownership, расход ammo и damage.
Это проверка оборудования, не training/eval набор.

MLP/feature v4 сохраняются. V4 имеет Blaster/Shotgun flags, текущие ammo/
gunframe и short history; полной weapon one-hot, inventory masks и явного
recoil input пока нет. В закрытом опыте разрешены только фиксированный MG
и штатный fallback Blaster. Перед расширением списка оружия требуется
отдельное версионированное weapon/recoil encoding; не объявлять этот опыт
системой обучаемого выбора оружия.

Новый isolated synchronous MG fixture проверяет fresh inventory100 до
barrier, native phase6 и фактический skill при release100. Direct ownership
для MG разрешён только в этом fixture. Barrel guard учитывает stock recoil
<=13.5° и spread консервативным18° cone на hitscan range8192; aim/movement
не корректирует. При наблюдаемом напарнике MG fire блокируется в этом
изолированном пилоте; general coop/spread safety здесь не приняты.

Полный go test ./... с workspace/runtime/q2go/baseq2 BSP/AAS, 75 Python
tests,4 fixture guards и native CMake build прошли. Первый Go запуск
отклонён из-за staging .go snippets и неполного assets AAS root; staging
сохранён как .go.txt, тесты повторены на полном существующем runtime.
Рабочие/live configs не изменены.

Training: same original obstacle parent53/524, composition retention bank,
constant anchor/bank1, maneuver-v4 reward, MLP810→64→64→8.4 CUDA updates,
4 instances×3 own-policy Mixed episodes, x2, seeds25100–25147, skill1,
actorHP100/stock monsters; неизменный horizon300/release100. MG выдаётся
before episode, не вводить infinite ammo или recoil removal. Bank только
исторические training observations, no-grad validation; не брать eval в fit.

Frozen final update4 evaluation, по4 seeds: MG Mixed25200/Solo25300 skill1,
MG Mixed25400/Solo25500 skill3; отдельно parent Blaster на тех же hard
seeds для сравнения оружия при фиксированных весах. Blaster preservation:
fresh Mixed25600/Solo25700 и known20800/21300/21700/22100, paired before/after.
Ожидается136 captures/26 batches сверх4 pilot. Skill3 — отдельная
оценка переноса сложности, не training skill. Считать uninterrupted full
wins, классы kills, патроны и actual weapon damage; no live promotion.

Root workspace/artifacts/combat-machinegun-v1-20261006.
