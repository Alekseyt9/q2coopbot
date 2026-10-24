# Предыдущая исполненная команда как обратная связь движения — 24.10.2026

Контракт и этапы зафиксированы в `../combat_action_inertia_protocol_20261024.json` до статистики/подбора параметров. Цель — проверить временную структуру команд движения с неизменным исходным actor, отдельно от простого снижения шума. Это гипотеза, не уже улучшенная модель.

## Текущее выполнение

`combat-action-inertia-source-cache-v1-20261024`: настоящий frozen source actor и JointAim воспроизведены наGPU из полных исходных временных историй **24TRAINмиров/3245строк**. Development исключён. Source neuralparameters и JointAim сохранены точно. Memorymaxerror9,2983e−6, LPmaxerror0,00017929в прежних допусках. Loaded Python sources архивированы рядом с кэшем, чтобы следующие изменения кода не ломали provenance закрытого результата. Sourceweights559ff0...unchanged.

Описательная статистика предложений движения:746компонент запросов развернулись относительно предыдущей исполненной команды при обеих амплитудах≥50;573компонентыmeanтоже разворачивались,329разворотов были только в sample при неразвёрнутом mean. Это не доказательство бессмысленных действий: развороты могут быть правильным уклонением, а текущее состояние меняется.

Во время2798grounded-visibleстрок у1279скорость до текущей команды<40;1175изних содержат текущий ненулевой запрос. Сравнивать эти величины как «команда не исполнилась» нельзя: скорость относится к прошлому шагу. Поэтому создан отдельный native-alignment audit.

`combat-action-inertia-motion-alignment-v1-20261024`: на3245принятых шагах2931запроснормой≥0,25,1987такихисполненных команд,881запросуменьшенexecutorдопочтинуля(<0,1). При ненулевой исполненной команде текущийgrounded-visible next-speed<40в446случаях, displacement<1в136; изпоследних58сverticalrelease. Это наблюдаемые цифры, не доказательство ошибкиgeometry/физики. Не всю стоячесть можно объяснить шумом, и инерция не объявляется решением подавления команд.

Добавлен `scripts/combat_action_inertia.py`: независимое CUDA преобразование means0:2 с feedbackpreviouscommand и отдельная projectionstd, безновыхNNпараметров. Scalartransform используетCUDAfloat64иbinary-exactclamp, выходвозвращаетсявисходныйdtype. `combat-action-inertia-core-v1-20261024` прошёл на3245реальных sourceoutputs: нулевойвариант тождественен, ненулевойменяеттолькодвижение, angularstdнеизменна, исходныетензорынеизменны, граничныеполныекомандыконечны, inputgradientsконечны/ненулевые тольконаmovement. Fitting/optimizerнет.

## Что ещё не готово

Go runtime не подключён. Нет новых nativeтраекторий, полнойпроверкиconditionalGaussianlikelihood/rng/order/Go↔CUDA дляэтойформы, новогоresumablePPOcheckpoint или доказательства качества. Default/sourceвеса не менялись. Следующийшаг — инкапсулироватьтотжепроекторв GoPPO иCUDAfullactor, проверитьρ0на4картах ивыдатьотдельныеevaluation-onlyweights дляnoise-only/inertia. Затем8свежихусловийнаsource/noise-only/inertia/rules(32native/24CUDA),16×2. Основные исходы — победа/смерть/полученныйурон; побочные — guard-suppressedmovement, движениепослеприменённойкоманды, фактическиепопадания. Только положительный nativeсигнал разрешает24-caseступень иотдельнуюмиграциюPPOдлядообучения.


## Завершённая интеграция и первая оценка — 24.10.2026

Предыдущий раздел «Что ещё не готово» описывает состояние ДО следующих проверок. Проектор теперь подключён в Go PPO Review/Decide и в полный CUDA actor. Он использует предыдущую фактически исполненную команду; rho=0 возвращает исходный результат точно. При rho=0,65 изменяются только два movement mean, а std движения отдельно умножается на sqrt(1-rho²). Noise-only использует тот же уменьшенный std с исходным mean. Направления камеры и её std напрямую не меняются. Это локальные координаты команды, не сохранение мирового направления при повороте камеры.

`combat-action-inertia-native-models-v1-20261024/report.json`: 3245 TRAIN строк полного actor на CUDA; source parameters, zero/source raw, noise-only/source raw и остальные inertia outputs совпали точно. Projection max error=0; zero/source log-probability max error=0,00017929 в существующем допуске. `go test ./internal/policy -run '^TestMovementInertiaContract$' -count=1` и сборка трёх Go CLI пройдены; этот unit test проверяет контракт/метаданные без нейросетевого CPU forward.

`combat-action-inertia-zero-native-v1-20261024/dataset/report.json`: 8 native / 8 CUDA, все четыре gate пройдены, 446 provider строк. Повтор исходной модели совпал с архивом, rho=0 совпал с новым source по полным наблюдениям, features, samples и исполненным командам первой жизни. `combat-action-inertia-probe-v1-20261024/complete.json`: 32 native / 24 CUDA закрыты, в том числе ненулевой Go/CUDA replay.

| Вариант | Победы / 8 | Убийства | Полученный урон | Смерти |
|---|---:|---:|---:|---:|
| Исходная модель | 2 | 6 | 566 | 5 |
| Меньше шума движения | 4 | 10 | 640 | 4 |
| Feedback предыдущей команды | 3 | 11 | 638 | 3 |
| Правила | 5 | 12 | 322 | 1 |

Для следующего этапа выбран noise-only: больше побед и проще изменение. Восемь условий — разведочная проверка, не доказанное улучшение и не превосходство над правилами. Изменений default/source весов и fitting нет. Для feedback-варианта обычное PPO обучение явно запрещено до отдельной миграции initializer/checkpoint/optimizer.

`combat-action-inertia-quality-audit-v1-20261024/report.json`: фактические MG выстрелы до конца первой жизни/цели, с native attribution. Source: 92 попадания с уроном по живому монстру из 200 выстрелов, 101 solid contact; noise-only: 79/131, 51 solid; inertia: 123/191, 61 solid. Это только MG и разные развившиеся игровые состояния, не общая точность оружия. Стоячие grounded-visible кадры 188/642, 238/791, 369/1066 соответственно: устранение стояния или вращений НЕ доказано.

## Расширенная проверка

`combat_movement_variance_confirmation_spec_v2_20261024.json`: 24 разных условия на base3/city1/jail1/ware2, по две точки; поровну 2/3 противника, обычные/recovery и site01/site02. Одинаковые fresh validation seeds offset44 и физические старты для source/noise-only/rules, всего 72 native / 48 CUDA, пул 16, timescale2. Это новые seeds, не утверждение о ранее невиданной геометрии.

Первый preflight v1 остановился ДО игр: восемь старых site01 descriptors не разрешали rules. В отдельном оценочном registry-v2 добавлен только этот режим, canonical registry сохранён. Сохранены effective generator seed_revision, позиции, оружие, противники и reward; preparer доказал совпадение source/rules физических validation44 instances для всех 24 условий. Это исправление метаданных, не подбор после неудачного боя. v1 не дал native исходов.

Запущен `combat-movement-variance-confirmation-v2-20261024`; при записи раздела процесс evaluator, native pool и клиенты подтверждены живыми. Результаты большого сравнения ещё не закрыты. Следующее решение — по полному сравнению с source и rules, без автоматической замены production модели или нового обучения.


24.10.2026. Movement-variance confirmation закрыта72native/48CUDA: source13/24побед,11смертей,34kills,1461HPполучено;noise15/24,8,42,1363HP;rules12/24,7,32,1078HP. Noise/source новые/потерянныепобеды3/1,noise/rules4/1; общий рост пока не доказан,ware2даёт большую часть разницы. MG live-damage fraction75,54%→72,62%,solid138→169;standinggrounded-visible33,97%→35,85%: лишние действия не устранены. Никакой замены default или fitting. Запущен неизменный повтор всех24условий с fresh offset45,72native/48CUDA,16×2; протокол записан до исходов. См.testing/combat_movement_variance_confirmation_20261024.md.
