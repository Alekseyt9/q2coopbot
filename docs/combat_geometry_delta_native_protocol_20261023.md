# Игровая проверка geometry8

23 октября 2026. Фиксируется после выбора geometry8 по frozen69-state протоколу, до новых игровых исходов. Прототип прошёл aggregate/fresh development gates, но победность не проверена. Основные веса не меняются.

## Контракт

Shared scorer kind geometry_candidate_delta31_mlp_v1,31→8ReLU→1 плюсbaseline beta. Входы:29разностей projected geometry относительноcategory0, делённых на frozen predictor std[185:214] иclip±20; разностьsigmoid(hint); сумма квадратов двухoffset/25. Выходclip±20,beta±20. Gaussian draw, target/mode/movement/vertical/weapon, final-aim fire и25-category frame-hash выбор прежние. Это обучаемая оценка категорий, безgreedyправила илиfiremask.

Sampler ABI tag frame_hash_geometry_delta31_v1: численныйhashалгоритмпрежний, но старыйruntime с неизвестнымтегом обязанотказать, анепроигнорироватьновыйscorer. Полная S4 body/value/std/fire/predictor/motor сохранена. Legacy head layers остаются дляprovenance и неиспользуются приgeometryscorer.

CUDA export проверяетвсе69состояний иJSONroundtrip противпрототипа; constant — TRAINmean25распределениевыбранногопредиктора. Checkpoints evaluation-only, безнеподдержанногоPPOresume. Нейросети наGo оцениваютсятольковигре. CUDAliveверификациягеометрическогоheadиспользуетfloat64 иисходныеJSONчисла длясопоставлениясGo float64; обучениеиэкспортпрототипаостаютсяCUDAfloat32. Проверяютсявсе25logits/вероятностей,hashкатегория,originalGaussianlatent,finalfire,actor/value/memory;порогинеослабляются.

## Свежая матрица

8семейств новогоv13реестра:base3/city1/jail1/ware2 site01,обычный бой ирежим восстановлениявооружения. Четыреварианта:sourceS4,geometry8,matchedconstant,rules. По1новомуvalidationseed/семейство,32native/24CUDA,одинпул16×2. Первыйнеиспользованныйобщийoffset≥37подоступнойистории, выбранныйдоисходов. Сохранитьучётнедоступныхисторическихссылок. Reward/skill/loadout/позициификсированы,minimum32GiBсвободно. Толькооценка,безобученияиавтоматическойpromotion.

Показатьцели,kills,deaths,dealt/received,frames/бой,gained/lost,обычные/recoveryикаждуюкарту;actualMG livingdamage/всевыстрелы,solid,targetlessshots,largeyaw,stationaryclear иattackwithouttarget. ОценкаправилпоMGэтимreporterнепроводится;неподменятьэтопридуманнойточностью.

Допусккновой24-case-per-variant confirmation:geometry8 goals≥source+2,≥constant+1 и≥rules;deaths≤source;received≤1.1source. ВсеCUDAпроверкизакрыты. Этоgateдополнительнойпроверки,недоказательствопревосходства. Еслинепройден,неувеличиватьматрицутехжевесов;сохранитьотрицательныйисходипереходитькконтролируемымальтернативамдвижения/цели/оружияиполномуобучениюбоя. Еслипройден,зафиксироватьотдельныйconfirmationпротоколдоследующихисходов.
