# Дообучение на legacy и checked wall исполнении

Предшествующий screen со старыми весами дал legacy8/24, checked5/24, rules10/24. Проверяется гипотеза, что PPO сможет использовать менее ограниченное исполнение после нового собственного сбора. Это гипотеза, не доказанная причина регрессии старых весов.

## Зафиксированное сравнение

- Общий parent: `combat-conditional-weapon-training-v1-20261012/training/update-conditional2`; weights SHA256 `a9ce6c976d7127026c8bd0a485ec0ee08cb5bbfbc0691970e34e01e1c5ef5216`. Обе ветки возобновляются из одного checkpoint с optimizer state.
- Registry: `combat-target-commitment-preparation-v2-20261011/registry-bonus/index.json`. Все28 семейств, train split, по2 seed, offset508. Отсутствие пересечения с найденными историческими plans проверено до запуска. Проверка не включает автоматически произвольные сторонние/nested plans вне перечисленных history receipts.
- 56 боёв на ветку,112 всего; одинаковые instances/seeds/reward, отдельные собственные траектории. В checked ветке включён `checked_wall_slide`; у legacy флаг выключен. После сбора проверяются actual manifest flags.
- Пул16, timescale2. Общие PPO config и10 actor optimizer steps, actor LR0.00003, weapon-state LR0.0003; обе ветки получают свои sequence-retention данные с одинаковым коэффициентом1. Число переходов может различаться из-за длительности боёв; равенство allocation/update count не равно идентичному числу training samples или времени GPU.
- NN replay, обучение, audit optimizer/checkpoint и input-use выполняются только CUDA. Go используется для native protocol export и планирования.

## Текущее состояние

Сбор112 боёв и CUDA дообучение обеих веток завершены, итоговые audits пройдены. Prepared protocol и две frozen plans существуют и проверены. Preflight подготовил отдельный набор offset504 без запуска серверов или обучения; рабочий набор offset508 не пересекается с ним.

Runner: `scripts/train_combat_checked_wall_ab.py`; root: `workspace/artifacts/combat-wall-slide-training-v1-20261012`. `progress.json` отражает текущую стадию; только `complete.json` с проверенными weight/checkpoint/report SHA подтверждает завершение. Частичные артефакты сохраняются при ошибке; автоматическое возобновление этого нового driver пока не реализовано.

## После завершения

Сравнить обученный legacy и обученный checked на свежих validation seeds: сначала smoke24, затем screen72 при исправном исполнении. Включить правила как третий вариант. Отдельно сопоставить gains/losses и урон, подавление движения, стояние, крупные повороты и attack без цели. Не назначать checked default и не заменять основные веса без нового результата. Старый screen492–493 уже использован для решения о гипотезе и не годится независимым confirmation.

Новый driver не реализует прыжки, human demonstrations или универсальные mover прогнозы. Эти части общего combat roadmap остаются незавершёнными.

Свежий smoke подготовлен заранее: validation offset520,8 условий на вариант (4 ordinary/4 recovery),24 native боя. `combat-wall-slide-trained-smoke-evaluation-v1-20261012` запущен в состоянии `waiting-for-updates` и ждёт seal всего training root. При upstream failed он прекращает работу; результаты/weights до завершения не используются. Старые основные веса не заменяются. Дальнейший screen72 ещё не запущен: решение требует результатов smoke и проверки механики.

Native сбор завершён: pool receipt подтверждает112/112 usable captures,56 на каждую ветку, source_unchanged=true, ошибок jobs нет. Driver проверил manifest checked flags и одинаковый model SHA для всех jobs. Стадия перешла в cuda-processing для legacy; общая training complete seal ещё отсутствует. Это завершение сбора, не завершение дообучения и не результат качества.

## Завершение обучения и оценка

Training root получил complete seal:112 native/112 CUDA, legacy6341 и checked6039 переходов; обе ветки по10 actor steps и одинаковые optimizer group LR[0.00003,0.0003]. Проверены SHA weights/checkpoint/report каждой ветки, protocol и pool. Full-checkpoint и input-use audits выполнены на CUDA. Legacy weights SHA97cff57551ca47df047b299e81ed443d9ab452b9233d18b2122997561b18489c. Оценочные победы пока не получены.

Первый smoke v1 остановился до native запуска: offset520 превысил validation count512 у части семейств. Report queue также завершилась failed без выдуманного результата. Это ошибка подготовки, не проигрыш модели. Старые roots сохранены; обучение не повторялось.

Исправление: новый `combat-wall-slide-validation-registry-v2-20261012` расширяет validation count512 до768 у16 семейств с revision+1;12 существующих cohort4096 сохранены. Геометрия, состав, оружие и reward не менялись. Начальные ranges и исходный реестр сохранены, planner проверил отсутствие пересечения диапазонов и исторических seeds. Provenance — `combat-wall-slide-validation-extension-v2-20261012.json`.

Smoke v2 (`combat-wall-slide-trained-smoke-evaluation-v2-20261012`) запущен на прежних свежих validation520 с двумя обученными ветками и правилами:24 боя,16 CUDA cases. Его результат ещё не подтверждён; назначения основных весов нет.

## Завершённый smoke новых весов

Seal smoke v2 подтверждает24/24 native и16/16 CUDA; reporter и execution limits закрылись complete. Native часть69.70с. Контроль2/8 побед (ordinary1/4,recovery1/4), checked1/8 (0/4,1/4), rules4/8 (0/4,4/4). Полученный урон631/648/476. Парно checked против control:1 новая и2 потерянные победы.

Поведение control→checked: стояние при clear enemy31.66→16.23%, yaw≥30°9.00→5.30%, attack command без цели3.04→2.72%, fully suppressed movement343/877 (39.11%)→128/905 (14.14%). Это уменьшение наблюдаемых проблем, но не рост числа побед и не причинное доказательство эффективности движения. На8 условиях нельзя выбирать основные веса.

Результаты: `combat-wall-slide-trained-smoke-report-v2-20261012/{complete.json,results.md}`, `combat-wall-slide-trained-smoke-execution-v2-20261012.json`; report queue seal с SHA inputs — `combat-wall-slide-trained-smoke-report-queue-v2-20261012.json`.

Следующий screen72 запущен: `combat-wall-slide-trained-screen-evaluation-v2-20261012`, fresh validation524–525,12 семейств,24 условия на вариант (16ordinary/8recovery). Результат пока отсутствует. Проверка seed bounds перенесена в generator make_spec, до записи spec:4 metadata tests прошли, включая last valid seed, overrun и missing confirmation. Go planner продолжает независимо проверять пересечение ranges/history; metadata tests не исполняют NN.

## Завершённый screen новых весов

Seal подтверждает72 native/48 CUDA; native часть202.65с, processed2734+3320 переходов. Reporter и execution report закрылись complete, receipts сохраняют SHA исходных seal/spec/pool/аудитов.

| Метрика | Legacy | Checked | Правила |
|---|---:|---:|---:|
| Победы |8/24|8/24|9/24|
| Ordinary |4/16|5/16|1/16|
| Recovery |4/8|3/8|8/8|
| Полученный урон |1732|1863|1522|
| Fully suppressed movement |768/2730 (28.13%)|619/3334 (18.57%)|—|
| Крупные повороты |11.87%|13.82%|—|
| Стояние при clear enemy |20.26%|19.74%|—|
| Attack command без цели |13.15%|14.75%|—|

Парно checked против control:4 новые/4 потерянные победы; против правил5 новых/6 потерянных. Малый smoke не предсказал снижение поворотов на широком наборе. По общей победности преимущества нет; по урону checked хуже, основные веса/default не заменены. Сравнивать эти числа с прежним screen492–493 как эффект обучения нельзя: условия и seeds различаются.

### Разбор восстановления оружия

Дополнительный metadata audit: `combat-wall-slide-trained-screen-recovery-v2-20261012.json`,8 recovery случаев на ветку. Все16 случаев содержат запрос смены и наблюдаемый переход на поддерживаемое оружие; задержка первого наблюдаемого supported equipment3–24 network frames от первого provider observation. Контроль4/8,checked3/8 побед. Отсутствие запросов/полная невозможность смены оружия в этой выборке не объясняет разрыв с rules8/8.

Некоторые проигрыши после перехода долго использовали Blaster; один matched случай в обеих ветках24 кадра оставался с Grenades перед Shotgun. Это описательная связь, не доказательство, что другой ствол обеспечил бы победу. Следующее исследование — фазы боя после смены, доступный боезапас, выбор между поддерживаемыми стволами, native damage и aim. Усиление общего штрафа за стояние пока не обосновано. Air control также остаётся unsupported; полноценная физика прыжка/приземления не принята.

Recovery audit теперь принимает явные learned labels и сверяет количество/идентичности случаев с frozen plans вместо исторического fixed32. Для старых вызовов сохранена after-only выборка. Живая проверка нового selection приняла8/8 случаев каждой ветки и сверила SHA исходных capture/trace.
