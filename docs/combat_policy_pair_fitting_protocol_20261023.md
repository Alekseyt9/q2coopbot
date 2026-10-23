# Обучение на новых policy-continuation парах

23 октября 2026. Фиксируется во время135native повторов, до чтения их альтернативных исходов и до нового fitting. Smoke5/5 уже принят, все пять его выстрелов промахнулись. Исходные32baseline, отбор28контекстов и split22train/6development уже закрыты. Старые отрицательные результаты не пересчитываются.

## Допуск и данные

Создать отдельный датасет из smoke и expansion proof. Максимум28контекстов/140вариантов; повторов(seed,frame) и категорий не допускается. Source SHA — обычная S4 cf7cd123f638945ddc49682cece27385aff55b50ad13e94ec6a4b0ab7e3b8977. Не подмешивать старые constant-continuation return labels, validation или confirmation. Сохранить заранее назначенные split/purpose.

Включать только контексты с принятойbaseline0, воспроизведением всей исходной baseline траектории и минимум3принятыми вариантами. Непрошедшие timing/physics/draw проверки явно исключить, не повторять ради квоты. Минимум16учебных и6отложенных контекстов на разных целыхсидах, минимум3карты вкаждомsplit. Для immediate fit требуются исправляющие выстрел альтернативы минимум на4независимыхtrainseed и двухкартах. Для sparse teacher требуются альтернативы с sparse delta≥1 минимум на4trainseed и двухкартах. Если конкретный допуск не выполнен, его fitting не запускать и не ослаблять порог.

Return audit прежний: от изменённого шага до последней доступной награды first-life capture, полныйscore и sparse компоненты monster_damage/received_damage/self_damage_extra/friendly_damage/death/monster_kill/tick существующегоv13. Внутренний недоступный переход запрещён; неполный post-terminal хвост явно посчитан. Prefix reward rows должны совпасть с baseline в пределах1e−6. Действия с отсутствующей return меткой исключить одновременно из обоих matched teacher наборов; если остаётся меньше3вариантов, исключить контекст. Не заменять неизвестную награду нулём. Одиночный условный rollout не является ожидаемымQ.

## Два matched fit на CUDA

Обе shared191→16→1 сети начинают заново с однойинициализацииseed20261023; S4 actor/value/std/fire/motor/hit predictor и checkpoint/Adam/RNG неизменны. Прежние входы кандидата, нормализация frozen predictor, bounds±20, фиксированные500шагов, networkLR0.003/weight_decay0.0001, betaLR0.05/weight_decay0, clip0.5. Никакогоearly stopping или выбора промежуточного checkpoint поdevelopment.

Immediate teacher: равное предпочтение максимальным известным HP при улучшении baseline, иначеbaseline. Sparse-return teacher: равное предпочтение максимальному sparse return при улучшенииbaseline минимум на1reward unit, иначеbaseline; ties1e−6. Обаtarget=0.1исходногоS4распределения+0.9предпочтения. Непроверенные действия не получают вымышленных HP/return. Показать train/development, каждуюкарту, per-context, неизвестнуюмассу/argmax. Численные/NN расчёты толькоCUDA. Prototype не считается новым training resume.

## Заранее заданные gates

Constant каждогоendpoint — его среднее25распределение наTRAIN.

Immediate: development полезнаяHPмасса≥source+0.05 и≥своегоconstant+0.02; вреднаяHPмасса≤source+0.05.

Sparse: development полезнаяreturnмасса(delta≥1)≥source+0.05,≥своегоconstant+0.02 и≥matched immediate+0.02; вреднаяreturnмасса(delta≤−1)≤source+0.05; вклад известныхкатегорийвsparse delta≥constant+0.10 и≥matched immediate+0.10. Вкладизвестныхкатегорий не равен return всейpolicy. Еслиmatched immediate недопущен, сравнениеobjectivesнеобъявляетсяготовым.

Если оба gateнепройдены, новых quality игр этихвесов не запускать. Еслиsparseпрошёл, выбратьего и включитьmatched immediateкакконтроль в новый40native/32CUDA screen. Иначе при прошедшемimmediate выбратьего для32native/24CUDA screen. Остальныеварианты:sourceS4,constantвыбранногоfit,rules. Предпочтениеsparse зафиксировано до fit, аневыбрано поудобнойпобедности. Новый runtime export/density audit обязателен; сохранятьevaluation-only checkpoint и запрет неподдержанногоPPOresume.

Native screen использует8семейств четырёхкарт нового реестра, одинаковые свежиеvalidationseeds длявариантов,16×2. Seed offset назначить штатнойпроверкойистории доигр; source/model/reward/fixtures заморозить. Допуск только кследующей24-case confirmation: выбранный goals≥source+2,≥constant+1 и≥rules, deaths≤source,received≤1.1source; дляsparse такжеgoals≥matched immediate+1. На8бояхнетpromotion. Еслиgateнепройден, не запускатьширокуюоценкутехжевесоврадиудобногоитога. Default автоматическинеизменять.
