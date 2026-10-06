# Fixed Machinegun pilot v1

06.10.2026. Протокол и завершённый результат PPO/evaluation.

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

Root workspace/artifacts/combat-machinegun-v1r3-20261006.

До этого выполнены4 pilot captures25800–25803: пулемёт подтверждён
native MOD_MACHINEGUN=4, патроны действительно расходуются, возможен
штатный fallback на Blaster. У исходной политики0 kills/2 deaths на этих
четырёх Mixed episodes; выдача оружия сама по себе пока не показала улучшения.

Два первоначальных preflight сбора по12 эпизодов сохранены отдельно:
combat-machinegun-v1-20261006 и combat-machinegun-v1r2-20261006.
Первый остановлен до оптимизации из-за Blaster-only native RNG receipt
parser; второй — из-за фиксированной первой фазы9 в экспортёре.
Для MG проверяется фаза6, receipt сверяется с заявленным оружием.
Повторный полный native export второго сбора прошёл:2371 rows/5 terminals.
Эти24 preflight эпизода не потреблялись PPO; основной R3 начинается
с исходного checkpoint и единых зафиксированных исходников.

## Результат основного R3

136 valid captures/26 batches:48 training+88 evaluation,4 instances x2, свой подтверждённый seed каждого эпизода.4 CUDA updates/40 accepted actor steps,8309 fresh rows; cumulative57 updates/564 steps. Ни один из24 preflight captures не потреблялся. Source/native fingerprints: `44556496f98f7940be2405ac5db49f0a05a21b8997165a18de49925c11e21ebd` / `96bada6beb4d6242a023ce0210bde7fb8debd714729cee7d8a84f3836f103f1e`.

Полные uninterrupted победы и смерти первой жизни, по4 эпизода на условие:

| Условие | Победы до → после | Kills до → после | Смерти до → после | Incoming до → после |
|---|---:|---:|---:|---:|
| MG Mixed, skill1 | 0 → 0 | 1 → 1 | 0 → 3 | 222 → 354 |
| MG Solo, skill1 | 0 → 2 | 0 → 2 | 4 → 2 | 400 → 300 |
| MG Mixed, skill3 | 0 → 0 | 0 → 0 | 0 → 2 | 173 → 320 |
| MG Solo, skill3 | 0 → 0 | 0 → 0 | 4 → 4 | 400 → 400 |
| Blaster Mixed, skill1, fresh | 3 → 2 | 6 → 6 | 1 → 1 | 299 → 235 |
| Blaster Solo, skill1, fresh | 4 → 4 | 4 → 4 | 0 → 0 | 137 → 156 |
| Blaster Mixed20800 | 3 → 3 | 6 → 6 | 1 → 1 | 369 → 305 |
| Blaster Mixed21300 | 3 → 2 | 6 → 5 | 1 → 2 | 317 → 345 |
| Blaster Mixed21700 | 3 → 3 | 6 → 7 | 0 → 1 | 263 → 348 |
| Blaster Mixed22100 | 3 → 2 | 6 → 5 | 1 → 1 | 305 → 237 |

Parent Blaster на тех же hard seeds, без обучения: mixed 0/4 wins, 3 deaths; solo 4/4 wins, 0 deaths.

Blaster preservation20 Mixed: wins15→12, kills30→29, deaths4→6, incoming1553→1470. Потерянные прежние успешные seeds: 25601, 25602, 20802, 21303, 21702, 22101.

Stochastic MG Mixed training:48 episodes,0 kills/0 consumed kill transitions, deaths27, alive unfinished21; full wins0. Не интерпретировать меньший loss или KL как навык добивания.

MG captures: observed kills by native MOD {'1': 2, '4': 2}, first-life monster damage by MOD {'4': 4982, '1': 508}; MOD4=Machinegun, MOD1=Blaster. В 67/80 captures наблюдался MG ammo0; суммарно0 observed ammo increases (серии MG в первой жизни). Подробные per-seed ammo/weapon windows в completion-status.json. Нет программного refill/infinite-ammo/recoil override.

Movement до завершения/смерти: training proposed jump54, sent53, ground→air53; final deterministic evaluation sent jump16. Наличие движения или airborne не доказывает targeted evasion.

Проверки прошли: native resets/ammo/weapon phase/actual skill, frozen model/source lineage, exact native re-export, source rewards и eligible consumption, independent GAE closure, exact CUDA actor Adam backtracking replay без direction fallback, checkpoint/RNG/optimizer counters, composition bank и independent analytic KL. Отдельный CPU no-grad critic check совпал с Go на всех текущих values и последовательных interior next values; tails и независимый feature encoding не пересчитывались. Go tests,75 Python tests,4 negative fixture guards и native build прошли.

Вердикт: выдача пулемёта реализована и проверена, но MG policy не принята для live. Solo skill1 улучшился на этой четвёрке, skill3/Mixed успех не достигнут, Blaster preservation имеет регрессии. Четыре seeds на условие не доказывают обобщения или превосходства оружия. Original obstacle reference и live сохранены. Следующий отдельный фактор — явные наблюдаемые weapon/recoil признаки с новой версией feature encoder; learned weapon choice/GRU/изменение reward не смешивать с этим опытом.
