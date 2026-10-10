# Подготовка curriculum движения у стен

`scripts/prepare_combat_wall_curriculum.py` подготовил 80 model-independent
train условий в `workspace/artifacts/wall-curriculum-preparation-v1-20261010`.
Оба execution plan заново сгенерированы и проверены существующим Go planner
через `--verify-plan`. Никакая нейросеть, сервер или клиент не запускались.
Статическая проверка использовала mode=rules; для обучения нужны новые learned
plans выбранной модели. Это подготовленные условия, не 80 выполненных боёв.

Состав:

- 64 условия: base1 site01/site03, base2 site02/site01; в каждой точке Blaster
  и Machinegun, по восемь train seed offset176..183.
- 16 условий: четыре исходные parasite/parasite-gunner Blaster/Machinegun
  семьи, по четыре train seed offset176..179.

Выбор опирается на sealed historical development560 outcome diagnostics
m1-stochastic-a и parent3-stochastic-a. Score каждой точки — доля смертельных
боёв плюс доля полностью отменённых provider movement requests в этих боях.
Выбираются две точки каждой карты; оба оружия обязательны. base2-site01
выбран по стабильному порядку при нулевых score вместе с другими спокойными
точками base2: это сохранение разнообразия, не доказанный трудный случай.
Все score, SHA исходных диагностик/registry/planner/plans сохранены в
`preparation.json`. Отсутствие пересечения train seeds с reference validation
проверено; глобальная историческая новизна seeds не заявляется.

Исходный registry и test не изменены. Новых координат нет: используются
зарегистрированные геометрии, составы монстров и distributions. Исторический
Shotgun fallback ограничивает качество исходного ранжирования; после полного
corrected-runtime1040 сравнения проверить новые проигранные бои и при
необходимости подготовить новую версию смеси, сохранив эту как evidence.

Следующий эксперимент после выбора родителя:

1. Два CUDA continuation кандидата от одного sealed checkpoint/Adam/config.
2. Контроль: 80 own-policy боёв равномерно по исходным20 семьям.
3. Curriculum: 80 own-policy боёв по подготовленной64+16 смеси.
4. Одинаковый objective, число PPO update и параметры optimizer; отдельные
   свежие rollouts для каждой политики. Обе очереди в одном pool16 ×2.
5. Сравнение обоих кандидатов, родителя и rules на общих20 development семьях,
   с проверкой ownership, реальных sampling seeds, попаданий и движения.

Критерий — больше побед и лучше выживаемость на общем сравнении, а не больше
движения само по себе. Новые reward penalties и архитектура в этот A/B не
включаются. Independent test остаётся зарезервированным до выбора кандидата.

## Выполненная подготовка автоматического запуска

Добавлен `scripts/run_combat_wall_curriculum_ab.py`. Смешанный learned plan
проверен реальным `--verify-plan`: 12 семейств, 80 условий, существующие
stochastic m1 weights, без нейронного исполнения. Preflight сохраняется в
`wall-curriculum-learned-preflight-v1-20261010`; исправлена только import path
интерактивной проверки, сам CLI использует обычный scripts path.

Драйвер запущен с удержанием OS handle текущего evaluation PID23932.
Очередь: `wall-curriculum-ab-capture-v1-20261010/queue.json`, прогресс:
`execution.json`. Пока ждёт sealed1040 comparison, native/training не стартовали.
Проверяет core, все диагностические seals, ownership и storage SHA, неизменность
pool script и curriculum spec. Выбирает parent среди m1/parent3 до/после по
сумме побед двух stochastic arms, затем смертям и полученному урону; оба arms
должны иметь нулевой clear-target rules fallback и equip fallback.

Обе тренировочные ветки копируют одни и те же parent weights/checkpoint/report
и config/anchor/bank. Uniform получает20×4, wall —8×8+4×4, trainoffset176.
Часть общих условий намеренно совпадает. Смешанный execution plan повторно
верифицируется генератором, snapshot и allocation проверяет существующий
processor. Train outcomes разных смесей не объявляются сравнением качества.
После160 captures требуется чистый ownership, CUDA-only export/batch finalize,
по одному optimizer continuation update и точные checkpoint reload audits.
Ошибки gates сохраняются как failed; независимый test и promotion отсутствуют.

Следующий обязательный результат после этих обновлений — общий paired
development comparison parent/uniform/wall/rules. Только он позволяет судить,
помог ли curriculum. Его запуск и итоговые метрики пока не подтверждены.

## Общая оценка поставлена в очередь

Добавлен и запущен `run_combat_wall_curriculum_evaluation.py`, ожидающий actual
training Python PID29348 через удерживаемый OS handle. Каталог:
`wall-curriculum-ab-eval-v1-20261010`, queue/progress.json фиксируют ожидание.
Будет запущено560 боёв: parent/uniform/wall × stochastic-a/b ×80 плюс rules80.
Offset выбора действий20261011/20261012, общие20 validation семей24..27.
Это повторная development оценка, не независимый test. Native dispatch ещё
не начался: оба обновления должны сначала завершиться и пройти CUDA audits.

На `wall-curriculum-eval-preflight-v1-20261010` реально сгенерированы два
learned inference плана (160 условий). Оба прошли `--verify-plan`; seeds и
instances побайтно по структуре совпали с template текущего1040 сравнения.
Нейронное исполнение и native прогон для preflight не требовались.

Перед560 проверяются одинаковый родитель обеих веток, complete/checkpoint
SHA, actor/value/std/Adam CUDA exact, неизменность pool/reference и резерв6GiB.
После — индивидуальные member receipts, actual sampling config SHA/offset,
общий quality report, все9 диагностик, ownership, outcome behavior и физический
storage audit. Ownership acceptance отдельно сохраняет clean=false, если
fallback вернётся; это не будет молча считаться качеством learned управления.
Final quality и promotion пока отсутствуют.

## Парные изменения результата

`report_combat_paired_effects.py` повторно читает sealed individual native
reports, сверяет SHA с member proof и проверяет совпадение generated fixture,
engine seed и policy RNG offset для каждого сравнения. Win/death/damage берёт
из first-life/goal-stop report, итоговые wins/deaths сверяет с quality summary.
Выдаёт приобретённые/потерянные победы, death delta и средние damage delta,
с полными family/seed деталями. Разные RNG arms отклоняются до записи результата;
реальный отрицательный check на архивных a:b прошёл.

Положительная проверка на historical560: parent3-a относительно m1-a +9/-6
побед, delta+3, deaths-5; parent3-b относительно m1-b +8/-12, delta-4,
deaths+2. Архивный Shotgun fallback сохраняет ограничение этих сравнений:
это проверка обработки native данных, не доказательство лучшей модели.

Поставлен отдельный разбор за фактическим evaluation Python PID21684 с
удержанием OS handle: шесть пар uniform→wall, parent→uniform, parent→wall,
каждая отдельно по stochastic-a/b. Выход `paired-effects.json/.md` в каталоге
общей560 оценки. Arms с общими engine conditions не объединяются в160
независимых случаев; результаты development, independent test не затронут.

## A/B запущен после полного закрытия reference

Reference1040 полностью закрыт со всеми hashes/gates. Selection реально выбрал
parent3-after (133 wins/18 deaths в двух stochastic arms). `execution.json`
перешёл в collecting_own_policy, actual Python PID29348 продолжает работу.
Uniform:20×4=80; wall:8×8+4×4=80. Повторная metadata проверка подтверждает
одинаковые исходные actor weights SHA, общий checkpoint SHA и parent update2.
В момент проверки17 native задач успешно завершены без ошибок. Обновления
будут update2→3 с CUDA checkpoint audits. Общая560 оценка ждёт их завершения.

Повторный анализ уже выбранного parent3-after на corrected-runtime1040:
stochastic-a в победах стоит34.3% native alive frames, в смертях53.1%;
stochastic-b34.3%/51.9%. Полная отмена provider movement requests в смертях
62.3%/63.3%, в победах46.3%/47.0%. Это pooled descriptive association,
не выровненный причинный эффект поведения или reward.

Свежий `blocked-geometry.json` проверил оба after arms (160 боёв):
parent3-a2152 полностью остановленных команд,1322 nearest hull probes<8;
parent3-b2227 остановленных,2216 known nearest hull probes,1056<8 и11 unknown.
465/609 остановленных команд имеют nearest clearance≥32. Направленная
дискретизация не учитывает весь инерционный путь; неизвестные hull masks
сохранены, не превращены в свободное пространство. Проблема движения
сохраняется у actual selected parent, но безопасный обход не доказан по probe.
Current A/B уже собрал63/160 успешных native задач без ошибок на момент
последней проверки; native quality этого train состава не сравнивается.

## Сбор закрыт, GPU processing начался

160/160 own-policy native задач завершены без ошибок; individual verifier
принял все160. Обе ветки имеют0 rules_with_clear_target и0 equip fallback;
прочие rules кадры помечены noncombat/visibility timeout. Actual160 provider
configs проверены: parameters кроме sampling_seed совпадают с исходными
weights, actual seed соответствует engine+declared offset, повторов нет.

Execution перешёл в cuda_processing. Processor запущен с
`--cuda-only-export --cuda-batch-finalize --export-workers4`, без Go numerical
NN replay. Завершение PPO update и checkpoint audits ещё не подтверждено.
Uniform имеет4025 provider-owned native alive frames, wall5290; это не
окончательное число eligible PPO transitions. Бюджет равен по80 боям и одному
PPO update/config, а не по длине траекторий, числу transitions или секундам GPU.
Эти реальные количества нужно сохранить при интерпретации общего560 сравнения.
Processing автоматически сжимает готовые streams; свободное место после
первой стадии сжатия около9.35GiB, evaluation reserve6GiB остаётся проверкой
перед dispatch. Исходные записи не удалялись.

Uniform CUDA PPO update завершён:4020 eligible transitions,10 accepted actor
steps, checkpoint update2→3. Complete seal сверяет weights/checkpoint/report
SHA, resume SHA совпадает с общим родителем. Wall ветка пока выполняет export;
её update и финальные CUDA checkpoint reload audits ещё pending. Это успех
pipeline одной ветки, не результат общего сравнения качества.

## Оба CUDA обновления завершены, общая оценка запущена

Uniform:4020 eligible transitions; wall:5286. Обе ветки выполнили10 accepted
actor steps и update2→3. Processing report complete, SHA
`534b4dcfeb2f62448608860eec97c7d2fbab3d08436dd859d36e209e9c38e469`.
Execution complete с parent_optimizer_preserved=true. Оба complete seal
weights/checkpoint/report проверены повторно; оба CUDA checkpoint audit на
RTX5070 подтверждают actor/value/std exact и optimizer_state_exact=true.
Report аудита проверяет raw outputs на16 real sequence rows; это bounded
numerical reload audit, не native quality acceptance новых весов.

Общая `wall-curriculum-ab-eval-v1-20261010` перешла в evaluating560; actual
evaluation Python PID21684 и запущенные pool pwsh процессы подтверждены.
7 вариантов проходят одинаковые20 validation семей: parent/uniform/wall
по двум stochastic arms плюс rules. Свободно около8.16GiB на момент dispatch.
Парный effects reporter ждёт полного закрытия evaluation. Независимый test
и promotion отсутствуют; никакие quality выводы по новым весам пока не делаются.

После общего560 в очередь поставлен [stochastic selection gate](combat_stochastic_test_selection_20261010.md):
один actor выбирается по обоим RNG arms, independent test пока не открывается.
Новый gate проверен на closed1040 и отклонил допуск parent3-after66.5 против
rules68, несмотря на single-arm70. Перед будущим test нужен stochastic
adapter к ранее зарезервированному inventory; deterministic old driver
не используется для этой selection.

## Результаты core560

Все560 native captures завершены без ошибок;560 member proofs приняты.
Core quality SHA256: `9cb70283bf313b3ae7f6a3d0800717946e9265383c40e044e36bfcce1e89c0e4`.

| Вариант | Победы /80 | Смерти | Средний полученный урон |
|---|---:|---:|---:|
| parent stochastic-a |70|7|15.82|
| parent stochastic-b |63|11|22.18|
| uniform stochastic-a |66|8|15.85|
| uniform stochastic-b |65|10|18.82|
| wall stochastic-a |65|9|17.07|
| wall stochastic-b |65|12|21.84|
| rules |68|8|15.20|

Средние победы двух веток: parent66.5, uniform65.5, wall65.0 против rules68.
Один выполненный CUDA update выбранного wall curriculum не улучшил среднее
качество. Эти условия development уже использовались; это не independent
test. Итоговые diagnostics/ownership/storage и sealed selection ещё
обрабатываются. Независимый test остаётся зарезервированным; promotion нет.
Далее: парные приобретённые/потерянные победы и geometry/movement/aim
диагностика перед изменением следующего train curriculum или objective.

Дополнительный `blocked-geometry.json` принят на всех480 learned captures:
provider movement requests/cancelled parentA4169/2185, parentB4300/2131;
uniformA4311/2312, uniformB4136/2105; wallA4257/2360, wallB4309/2165.
Доли полных отмен52.4/49.6%,53.6/50.9%,55.4/50.2% соответственно.
Static hull guard отмены: parent1563/1294, uniform1452/1432, wall1680/1533.
Это pooled command counts, не независимые episodes и не причинное
доказательство trapping. Ближайший hull probe не является native collision
replay; наличие альтернативного probe не доказывает безопасный путь.

Следующий ограниченный эксперимент: `run_combat_aim_noise_evaluation.py`,
`aim-noise-eval-v1-20261010`,560 common development battles. Исходный parent,
continuous yaw/pitch standard deviations ×1/×0.5/×0.25, по двум RNG arms плюс
rules. Сеть, movement noise, attack/vertical/weapon/target/aim-mode categorical
heads не меняются; factor1 metadata равна исходной модели целиком.
Контракт log_std[-8,1] соблюдён, real JSON metadata и syntax проверены без
NN replay. До native dispatch требуется sealed diagnostics/ownership/storage
reference, повторный source SHA и отрицательный independent-test gate;
при положительном gate эксперимент пропускается в пользу independent test.
Это inference variance experiment, не GPU training и не acceptance всех
архитектур. Результатов пока нет; драйвер ждёт actual reference process21684.

Для headroom закрытый `action-sampling-offset-wide-v1-20261010` архив
дедуплицирован проверенными identical hardlinks:1260 paths сохранены,
142→2 physical inodes,754056704→17759744 unique logical inode bytes,
all_binary_sha256_preserved=true. Полный размер диска этим счётом не измеряется.
Свободно6.36GiB после операции;560-case driver требует6GiB перед dispatch.

Reference560 полностью закрыт:9 diagnostics,ownership,outcome/storage seals
проверены development selector; selected=parent,eligible=false,66.5 versus68.
Парные parent→wall effects: armA приобретена1/потеряно6 побед, delta−5;
armB приобретено6/потеряно4, delta+2. parent→uniform:1/5(delta−4) и7/5(delta+2).
Wall versus uniform:1/2(delta−1) и3/3(delta0). Две RNG ветки не удваивают
число независимых generated conditions.

Aim-noise driver уже перешёл в evaluating560: actual Python PID14228 и
работающий slot0 job0 подтверждены. Все7 plans/actor contracts проверены
registry planner, frozen fixtures/engine seeds совпали с reference. Поставлен
парный reporter за удерживаемым actual process handle14228: noise1→half,
noise1→quarter иhalf→quarter отдельно дляa/b. Результатов новой оценки нет.
