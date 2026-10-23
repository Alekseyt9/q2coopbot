# Следующая гипотеза: общий оценщик направлений

23.10.2026. План после закрытого200-step flat head fit: полезная probability mass наdevelopment0,268% противconstant2,003%, gate=false. Native screen не запускался. Цель — проверить более экономное обучение зависимости действия от состояния; не объявлять недостаточную ёмкость или единственную причину неудачи установленными.

## Почему этот опыт

Текущая1359→64→25голова имеет88665параметров и отдельные выходные веса для направлений. На12train состояниях она выделила полезные поправки, но на5development состояниях почти всегда сохраняет baseline. Корректирующие TRAIN примеры покрывают немного направлений и одну геометрию. Это мотивирует общую функцию score(state,candidate), а не ещё один длительный fit той же структуры.

На том же sealed17-state dataset v2, с прежними12train/5development и без новых игр, проверитьCUDA прототипы:

1. Малый калибратор: score_k=alpha×(sigmoid(hint_k)−sigmoid(hint_0))+beta×is_baseline. Alpha≥0, beta обучаемая. Это сильный простой контроль: использует контекст через frozen forecast, но не может запомнить raw сцену. Старый greedy ranker не включается в runtime; наличие beta/стохастического выбора и native supervision — отдельная проверяемая гипотеза.
2. Shared residual MLP16: одинаковая сеть для каждого направления. Inputs — выбранный action-conditioned causal185 с кандидатным итоговым прицелом, нормированные source predictor mean/std, offset yaw/pitch, baseline flag и прогнозы baseline/candidate. Предполагается около3тысяч параметров вместо88тысяч. Нет native health/outcome в inputs. Вся группа/геометрия по-прежнему доступна полной основной политике; этот маленький блок отвечает за доводку выбранной цели.
3. Flat MLP как matched500-step reference; прежний200-step отрицательный результат сохраняется. Это позволит отличить пользу новой структуры от простого увеличения числа шагов.

## Протокол до запуска

Зафиксировать отдельный protocol artifact до fit:500шагов на вариант, один и тот же TRAIN набор и teacher target0,1source+0,9native preference, никакого development gradient/early stopping/выбора промежуточного checkpoint. Flat LR3e−4; calibrator LR0,05; shared network LR0,003 и отдельная baseline bias LR0,05. Clip0,5; source actor/value/std/fire/predictor/RNG неизменны. Разные LR отражают разные масштабы параметров; это отбор практических гипотез, не чистая причинная оценка влияния архитектуры.

Исходная shared/calibrator инициализация — alpha/residual0 и baseline bias по среднемуP0 source наTRAIN. Сохранить положительную поддержку25направлений; документировать ограничения параметров/вычислений одинаково вCUDA и будущемGo runtime. Дляshared residual запечатать точный layout и нормализацию до первогоfit. Все конструкторы/forward/обучение/численные проверки толькоCUDA.

Контроль каждого fixed endpoint — постоянное распределение, равное его средним25вероятностям наTRAIN. Метрики и gate сохраняются: development полезная масса ≥source+0,05 и≥своегоconstant+0,02, вредная масса ≤source+0,05. Unknown directions не получают synthetic labels и не считаются автоматически промахом. Показать source/flat/shared/calibrator per-context результаты, известную/неизвестную массу и conditional known HP.

Development уже использован для итерации гипотез, это не независимое combat validation. Если новый scorer проходит gate — только тогда versioned Go/runtime/CUDA contract, проверка stochastic density/экспорта и свежий native source/trained/constant/rules screen. Не объявлять GPU prototype production weights до реализации и реальной приёмки. Default автоматически не менять.

Если ни один вариант не переносится, не подбирать split/порог задним числом. Использовать one-frame сбор для новых TRAIN seeds и более поздних состояний при движении/отдаче, а также разбирать неверные прогнозы motor/hit и шум single-spread native labels. Старые четыре development seed остаются вне обучения. План не устанавливает заранее, что усложнение модели или большее число боёв обязательно поможет.
