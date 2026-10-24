# Сопоставимое обучение на тройках противников — 24.10.2026

Новая гипотеза после первого decision-scope screen: более длительное собственное обучение решений на тройках противников улучшит полный результат боя. Предыдущий фиксированный endpoint дал5/8 против full4/8 и rules7/8; gate не пройден, широкое подтверждение прежних весов не запускается. Здесь меняются обучение и его заранее выбранные конечные веса, а не повторяется оценка прежнего checkpoint.

## Сопоставимость истории

Общий parent — cost-round01 (`bf7d90b2...`). От него обе ветки получили первый matched update на исходном собственном round02 TRAIN корпусе. Затем decisions получил ещё один собственный update для live continuation proof. Перед новым сравнением full проходит отдельные16боёв на тех же физических условиях/сидах, что этот второй decision update; source policy и его траектории свои.

Далее обе ветки получают по два новых16-case update на тройках противников. Итого endpoint каждой ветки содержит четыре update от общего parent. Количество боёв, cap30actor/40critic и KL limits сопоставимы; число фактических переходов и принятых Adam шагов может различаться из-за разных траекторий и backtracking. Они сохраняются отдельно и не выдаются за одинаковые полные шаги Adam.

Runner после обучения дополнительно читает checkpoint metadata наCUDA и проверяет `updates_completed=parent+4`, общий префикс истории и ровно четыре новых consumed rollout у каждой ветки. Receipt — `matched-update-counts.json`.

## Curriculum и сохранность модели

Восемь существующих рецептов: base3/city1/jail1/ware2, ordinary trio вsite01 и weapon-recovery trio вsite02. Все группы содержат parasite/infantry/gunner. Сохранены rewardv14cost, архитектура, коэффициенты, LR и trust region. Область обучения уfull иdecisions различается как в первом screen; новые слои/тестовый oracle не добавлены.

`fork_combat_decision_curriculum_cuda.py` явно меняет только SHA training contract и добавляет запись metadata fork. Веса, neural/Adam/RNG/history tensors и их clocks провереныCUDA на точное сохранение. Исходные протоколы и checkpoint не переписаны. Новый initializer `combat-trio-scope-contract-v1-20261024/initializer` имеет те же веса97e5cf9d..., что прежний decision update2. Проверка не является новым fitting.

Full continuation реализован отдельно в `train_combat_trio_full_ppo_cuda.py`: принимает исходный full endpoint и свои последующие checkpoints, сохраняет его actor/geometry/value optimizers и counters. Decision-scope checkpoint отвергается metadata scope guard. Canonical row optimizer новой схемы не подменяется обычным full optimizer.

Контракты: `docs/combat_trio_full_contract_20261024.json` и `docs/combat_trio_decision_contract_20261024.json`. Исследовательский протокол: `docs/combat_trio_matched_curriculum_protocol_20261024.json`.

## Фиксированный бюджет и оценка

- Full alignment:16новых TRAIN боёв, offset244, явная пара с закрытым старым decision continuation.
- Trio обучение:2этапа ×2ветки ×16боёв =64новых TRAIN, offsets246/248.
- Всего80новых native/80CUDA TRAIN; каждый корпус принадлежит своей behavior policy.
- Оценка:8новых balanced условий, offset53, parent/full/decisions/rules =32native/24CUDA. В оценке остаются ordinary/recovery и2/3противника.
- Общий бюджет112native запусков,104CUDA проверки, плюс CUDA metadata fork и fitting. Пул≤16,×2; neural training/numerical checks толькоGPU.

Сиды между ветками совпадают только в явно проверенных paired fixtures; траектории не копируются. Новые stages используют новые TRAIN seeds. Validation исходы в fitting не входят. Сравниваются только final endpoints: full-round03 и decision-round04, без промежуточного отбора по качеству.

Screen spec — `docs/combat_trio_matched_screen_spec_20261024.json`. Gate до новых исходов: decisions≥full+2победы и≥parent/rules, смерти≤full/rules, received≤110%full/rules. Движение/точность описываются отдельно; большее движение не считается успехом само по себе. Automatic promotion=false.

## Состояние на момент первого запуска (история)

Root: `workspace/artifacts/combat-trio-matched-curriculum-v1-20261024`. Metadata fork закрыт. Full alignment закрыт16native/16CUDA,1766TRAIN переходов,6actor/geometry и40critic шагов. Начат первый trio корпус full-round02. Обе ветки теперь имеют по два update от общего parent; итоговое обучение и quality ещё не закрыты.

Runner `run_combat_trio_matched_curriculum.py` сохраняет phase/PID/child, source fingerprint и закрытые шаги; `--resume` не запускает дубликат живого процесса и не перезаписывает partial output. Пока цепочка жива, существующие Go/Python/PS/native исходники не менять. Новые metadata scripts и Markdown допустимы.

`report_combat_trio_matched_screen.py --wait` ожидает закрытия цепочки и создаст `combat-trio-matched-result-v1-20261024/report.json` и `results.md`. Отсутствие результата не является улучшением качества. Основные игровые веса не заменены; общая цель превзойти правила, полная кампания и human coop остаются открытыми.

## Малый screen завершён: gate пройден

Закрыты80native/80CUDA TRAIN,10768новых переходов и32native/24CUDA оценки. Все пять update выполненыGPU. Проверка истории: у общего parent36накопленных updates; уfull иdecisions по40, то есть ровно четыре новых у каждой ветки и по четыре новых consumed rollout с общим исходным префиксом. Receipt `matched-update-counts.json` закрыт.

| Вариант | Победы / 8 | Смерти | Убийства | Получено HP |
|---|---:|---:|---:|---:|
| Общий parent | 4 | 4 | 10 | 535 |
| Full PPO | 4 | 4 | 9 | 506 |
| Decisions | 6 | 2 | 15 | 542 |
| Rules | 4 | 3 | 10 | 499 |

Противfull иrules уdecisions по2новых победы/0потерянных. Все заранее записанные условия малого gate выполнены, включая learned ownership. Полученный урон уdecisions больше на7.1%кfull и8.6%кrules; это укладывается в заранее допустимые10%, но не называется улучшением этой метрики. Две расходящиеся пары — малая выборка; sign-test противrules p=0.5. Общая победа над правилами этим не доказана.

Native action diagnostics относительноfull: standing37.43→26.39%,MG65.29→69.57%; при этом yaw≥30°8.16→9.37%,attack без target4.08→9.48%,почти обнулённые запросы20.63→23.53%. Часть лишних действий остаётся; число побед не заменяет их отдельный анализ. Все эти доли описывают разные замкнутые траектории.

Конечные веса:full `ea99c95068ba68f9816162400d82cc6e69db1044b308a8f10f3f5989f4e1d365`,decisions `4e31f0c160175fc316de7281c2a81a9d36b4b16aca296e6829f95983498a06d7`. Они не копировались в основные игровые веса. Полный sealed результат: `workspace/artifacts/combat-trio-matched-result-v1-20261024/report.json` и `results.md`.

В отчёте сохранены actor steps и суммы LR-масштабов после backtracking. Например,21принятый шагfull не равен21полноразмерному Adam шагу. Эти суммы — прозрачный учёт, не эквивалент вычислительного обучения или качества. При добавлении этой колонки до закрытия отчёта перезапущен только ожидающий metadata reporter; игры и fitting не повторялись. Reporter теперь проверяет собственный source SHA до и после ожидания.

## Независимое подтверждение запущено

После закрытия положительного малого gate зафиксирован `docs/combat_trio_matched_confirmation_spec_20261024.json`. Те же неизменные parent/full/decisions weights иrules сравниваются на24рецептах/4картах/2точках. Это отдельный зарегистрированный **confirmation split**, offset8 — первый диапазон без пересечения с замороженной историей. Старый screen был validation offset53; новые исходы/распределение ранее не оценивались этим опытом.

Набор сбалансирован:12ordinary/12recovery,12duo/12trio,по6случаев на карту ипо12наsite01/site02. Карты и общий набор геометрий уже существовали; это не24новых карты и не доказательство работы на неизвестных уровнях.

Бюджет96native/72CUDA (24на вариант),pool16×2. Ни одного нового fitting, изменения коэффициентов или выбора иного checkpoint перед confirmation нет. Gate записан до новых исходов:decisions≥full+3победы и≥rules+3,не уступаетparent по победам,смертей≤full/rules,received≤110%full/rules. Automatic promotion=false. Положительное подтверждение тоже не заменяет full campaign/human coop и более широкую проверку.

Активный root: `workspace/artifacts/combat-trio-matched-confirmation-v1-20261024`. На момент этой записи выполняются96боёв; итоговых результатов ещё нет. `report_combat_trio_confirmation.py --wait` автоматически дождётся завершения, проверит неизменность весов относительно малого screen, выполнит native audits и создаст `combat-trio-confirmation-result-v1-20261024/report.json` и `results.md`. До закрытия текущих прогонов runtime/neural helpers не менять. Общая цель остаётся активной.


## Независимое подтверждение закрыто: gate не пройден

Проверены96native/72CUDA боёв на24рецептах, confirmation offset8. Веса неизменны относительно малого экрана; повторного fitting и выбора endpoint не было.

| Вариант | Победы /24 | Смерти | Убийства | Получено HP |
|---|---:|---:|---:|---:|
| Parent |14|7|37|1332|
| Full PPO |14|9|39|1483|
| Decisions |15|8|40|1242|
| Rules |13|7|39|1167|

Gate=false: отрыв +1отfull и +2отrules вместо требуемых+3; смертей8против7rules. Урон проходит предел110%. В парных условиях decisions/full4новые победы/3потерянные, decisions/rules4/2; это не убедительное общее превосходство. Все72learned боя провереныCUDA. Основные игровые веса не заменены.

На тройках decisions6/12,full3/12,rules2/12; на парах decisions9/12,full/rules11/12. Два проигрыша противrules — base3site02recovery seed1244128008 и jail1site01recovery1254028008. В обоих decisions погиб без убийств. Наbase3 все10выстреловMG дали урон, но этого оказалось недостаточно: точностьMGсама по себе не определяет победу. Наjail1 в provider окнах не былоMGвыстрелов. Последнее HP из события попадания не считается конечным HP врага.

По всем боям full→decisions: standing31.49→38.69%,largeyaw7.11→5.24%,MGdamage71.70→70.57%,switchприживойвидимойпрежнейцели10.05→8.21%. Это описательные доли разных траекторий; причинная роль стояния/смены цели не доказана. Общее исправление лишних действий не подтверждено.

Закрытые результаты: `combat-trio-confirmation-result-v1-20261024`, новый воспроизводимый сводный разбор `combat-trio-confirmation-diagnostics-v1-20261024` (`scripts/report_combat_trio_confirmation_diagnostics.py`). Новый разбор только объединяет уже закрытые native/CUDA результаты, не выполняетNNилиновыеигры.

Следующая отдельная гипотеза — смесь пар и троек вместо только троек при одинаковом бюджете и одном checkpoint. Это новый опыт, а не повторная попытка пройти старый gate. Протокол и ограничения: `testing/combat_curriculum_balance_20261024.md`.


### Дополнительный разбор оружия в recovery (без новых игр)

Закрыт metadata-only ledger `combat-recovery-weapon-trace-v1-20261024`: шесть уже CUDA-верифицированных траекторий parent/full/decisions на двух recovery условиях,23проверенныхSHAвхода. Вjail1 seed1254028008 decisions запросилBlaster наobservation frame117, когда наблюдалсяMachinegun с40Bullets и свежийinventory содержал20Shells. Наframe122 наблюдаетсяBlaster; снаряды автомата до этого не потрачены. Это не исчерпание боезапаса. Full запросилShotgun на115 и выиграл; решения траекторий после этого различаются, поэтому причинный эффект выбора оружия не установлен. Видимый viewmodel не означает готовность стрелять: ранниеgun_frame0..3переходят в46..49после запроса следующегооружия. Вproviderокнах этогоjail1боя уdecisions нетMGвыстрелов, но естьBlasterурон50. Это отдельная диагностическая гипотеза о выборе/тайминге оружия, текущий preregistered опыт состава обучения не изменён.


Систематическийразборвсех72прежнихlearnedconfirmationтраекторийзакрыт`combat-blaster-choice-diagnostics-v1-20261024`: переходнаBlasterприfreshinventoryage0иналичииальтернативногооружияспатронамивстречаетсяуdecisionsв3/24боёв,full4/24,parent5/24(8requestframes). Blasterusageприальтернативномбоезапасе113/2822,140/2884,223/3101providerframesсоответственно. Этидолиописательны; наличиедругогооружиянеравнозначноготовностиквыстрелуилиоптимальностипереключения. Одиночныйjail1проигрышнеобосновываетобщийзапретBlaster. Новыхигр/NN/fittingнет.
