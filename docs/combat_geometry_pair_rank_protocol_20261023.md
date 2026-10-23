# Малый оценщик различий геометрии

23 октября 2026. После отрицательных matched500 fits на20новыхtrain/6development состояниях. Immediate useful mass development2,487% уступила constant3,838%; sparse useful mass0,00264% уступила constant0,01006%. Эти прототипы не запускаются в новых играх.

Новая гипотеза: общей сети из3090параметров трудно переносить предпочтения по небольшому числу состояний. Проверить компактные относительные признаки и парное сравнение фактически выполненных действий. Это не доказанная причина предыдущих неудач и не объявление большей сети непригодной.

## Повторное использование меток

Объединить старые43native состояния из combat-native-aim-late-dataset-v1-20261023 и новые26из combat-policy-aim-dataset-v2-20261023. Сохранить все прежние split; одинаковыйseed никогда не пересекаетtrain/development. Максимум69состояний/343принятых действий,46train/23development,34учебных/16отложенныхseed. Отдельно показывать новый6-state development срез.

Разрешено смешивать только метки **немедленного native HP**. Старыйconstant baseline и обычная S4 должны иметь одинаковые всеmodel поля кроме aim_candidate_head.layers; predictor/motor/body/std/fire должны совпасть. Проверить обаsourceSHA по их dataset receipts. Состояния, sampled source actions и timing записаны явно. Return/продолжения этих двухполитик не смешивать; убрать return поля из выходных action labels. Повторных игр дляобъединения не требуется.

Минимум40train/20development состояний,30train/14development seed,≥3картывtrain. Minimum4независимыхtrainseed сHPулучшением противbaseline на≥2картах. Еслидопускнепройден,fitнезапускать. Developmentужеисследовательский,необъявлятьегофинальнойоценкойпобедности.

## Признаки и три фиксированных варианта

Длякаждогоиз25действий вычислить прежние185causal inputs, frozen motor прогноз и прежние29projected geometry признаков. Никакого native muzzle/recoil/outcome вinputs. Нормализовать29геометрических величин по frozen predictor std[185:214], вычесть baseline0 и clip±20. Добавить sigmoid(hint_k)−sigmoid(hint_0) и(yaw_offset²+pitch_offset²)/25. Итого31вход. Это относительная геометрия и прогноз; не обещание строгойэквивариантности или истинногонативногопопадания.

1. Linear31безoutputbias +baseline beta:32параметра.
2. Geometry31→8ReLU→1 +baseline beta:266параметров.
3. Прежний shared191→16→1 +baseline beta какmatchedконтроль:3090параметров.

Все500шагов наCUDA, одинаковыеsourceS4/split/teacher. Seed20261023. LinearLR0.01, остальныеnetworkLR0.003; betaLR0.05. Weight decay0.0001дляnetwork/0дляbeta,clip0.5,networkscoreclip±20,beta±20. Инициализацияlinearweights0 и последнегоMLPслоя0; beta=log(24P0/(1−P0)) по sourceTRAINmean. РазныеLR — практическийсравнительныйопыт,нечистоеизолированиеархитектуры. SourceNN/Adam/RNG/filesнеизменны. Безearlystop/подбораcheckpoint.

## Парная цель

Вкаждомtrainконтексте взятьвсеупорядоченныепарыпринятыхдействий,гдеHPwinner>HPloser. Среднийsoftplus(score_loser−score_winner) сначалапо парамконтекста,затемпоконтекстамснетривиальнымипарами. Не учитыватьtieкакпромах. Длявсехtrainсостояний добавить0.05KL(source25 || learned25). Непроверенные категории не получают вымышленныхHP; регуляризация кsource не являетсяnativeметкой. Постоянное распределение каждогоendpoint — егоTRAINmean. Показатьpair accuracy/logloss, полезную/вреднуюмассу,knownmass/unmeasuredargmax иper-contextprobabilities.

## Допуск

Основнойdevelopmentgate: полезнаяHPмасса≥source+0.05,≥своегоconstant+0.02;вредная≤source+0.05. Fresh6-stateguard:полезнаямасса≥freshsource+0.02 и≥freshconstant;вредная≤freshsource+0.05. Еслинесколькопрошли,выбратьбольшийaggregate useful−constant,затемменьшуюharmfulmass,затемимя. Не ослаблятьпороги после результата.

Толькопослепрохожденияgate — versionedruntime/JSON/CUDAcontract дляновоговидаscorer иfresh native source/trained/constant/rules screen.32native/24CUDA на8семействахчетырёхкарт,пул16×2,новыйseedoffsetдоисходов. Никакойавтоматическойpromotion. Есливсенепрошли,сохранитьотрицательныерезультатыинезапускатьqualityигрыэтихвесов. Тогдаследующийограниченныйшаг — контролируемые альтернативы движения/цели/оружия иполноебоевоеобучение;непродолжатьбесконечнооднутолькодоворотку.
