# Прогноз последствий движения и выстрела, 23.10.2026

Цель — научить систему оценивать выстрел из позы, в которой он действительно произойдёт. Текущий actor, награда, default и опубликованные веса не изменены. Новый forecaster — отдельная обучаемая модель для исследования; боевое преимущество пока не установлено.

## Исправление геометрического контракта

В `internal/quake/usercmd.go` ConnectRequest явно передаёт `hand=2` (центрированное оружие). Native `P_ProjectSource` в `yquake2/src/game/player/weapon.c` при CENTER_HANDED обнуляет боковое смещение ствола. Прежние action geometryv1/v2 и диагностический helper предполагали right-handed offset8units: это ошибка исходного контракта, а не эффект сети, который следует учить компенсировать.

Добавлена отдельная `combat_observed_action_geometry_v3` с centered muzzle. Версииv1/v2 оставлены неизменными для исторических моделей/сравнений; их predictor нельзя просто переименовать вv3. Существующий runtime ranker по-прежнему принимает только свою старуюv2; новая геометрия пока не назначает модели/default. Диагностический helper получил явный hand0/1/2, default2; старые бинарники не перезаписаны.

Проверки Go (только geometry/metadata) и CUDA прошли: centered muzzle не создаёт боковой parallax и не зависит от recoil roll; исторический right-handed контракт сохраняется. На10505реальных inputs pure Go causal185 совпал с исходными CUDA inputs с maxerror1,2e−7; centered geometry — с CUDA FP64 с maxerror1,4e−13. Это проверка вычислений, не аргумент о качестве боя. Go neural inference/replay для неё не запускался.

После корректного hand=2 на том же40-case источнике/33боях с1257MGshots: static-before-bbox57→30, selected-target hits среди таких лучей30→9. Совпадение с реальным solid contact25→21. Даже исправленный static gate не является точным условием запрета огня. Авторитетный повтор: `combat-static-occlusion-centered-v3-20261023`.

## Данные и причинная граница

`build_combat_action_effect_labels.py` связал исходные TRAIN shot recipes, pool/member manifests, эксклюзивные command windows и native hitscan events. Собрано10505shots:9248из9прежних корпусов плюс1257из двух randomized-aim captures. Всего328sourcecases,280боя/групп сMGshots;48прогонов не даютshotrows. Validation/test данные в обучение не добавлялись. Все исходные данные и совпадение actual shot/outcome проверены по hash и ключам source/job/seed/frame/index/shot.

Requests содержат исключительно observation и compact sampled action. Native start/view/recoil/aim/velocity хранятся только в labels. Будущее observation не передаётся; native game frame и server-prefix frame — разные счётчики, они не объявляются временем сетевой задержки. Fitting — условно на существовании actual MG shot; модель пока не предсказывает сам факт выстрела и не покрывает другое оружие.

Compact CPU labels/metadata и pure Go feature packing выполняются без NN. Все Tensor/NN расчёты и обучение — CUDA, существующий Python `F:/src/strat/.venv-gpu/Scripts/python.exe`. Ресурсы игры не копируются, игровые прогоны не запускаются. Requests и Go exports сжаты NTFS LZX с сохранением SHA; binary dataset/checkpoints небольшие. Новый набор: `combat-action-effect-labels-v1-20261023`.

## Задача предсказания

Пять выходов: смещение native muzzle относительно centered current-pose muzzle в yaw-local координатах, нормированное на32units, и wrapped pitch/yaw difference фактического view+recoil относительно observation+sampled aim+observed recoil, нормированное на5degrees. Истинный recoil не является входом.

Исходная192-мерная матрица: causal185, собственная observed velocity в yaw-local frame3, requested movement в том же frame2, observation_age_ms/300 и previous-command-msec/100. Угол действия связывает requested forward/side с направлением движения; это вычисляемые признаки, не новый контроллер движения. Будущая скорость и actual elapsed shot time вместо observed значений не подставляются.

Фиксированные baselines: прежний right-handed current pose, исправленный centered current pose, centered+observed velocity×100ms. Для NN заранее фиксируются300Adamsteps,LR0.001,weight_decay0.0001,seed20261023. Разбиения: целый episode/seed приseed%4==3; отдельно всеsite02. Нормализация только по соответствующему TRAIN. Gates требуют меньшую muzzle error против100ms baseline и меньшую angular error против current-pose baseline на обеих development folds. Всё это development внутри TRAIN, не боевой evaluation.

## Результаты первых архитектур

На маленьком40-case наборе linear192 и MLP192×32×5 проходили gates. На полном328-case наборе обе провалили направление наsite02:

| Полный набор, heldoutsite02 | Muzzle mean error | Direction error |
|---|---:|---:|
| Centered current pose |17,23|1,615°|
| Fixed observed velocity100ms |16,43|1,615°|
| Linear192,965params |12,33|2,369°|
| MLP192×32×5,6341params |14,33|1,956°|

Положительный маленький опыт не был выдан за устойчивое улучшение. Полные state-conditioned модели не допущены дальше. Это совместимо со spurious correlations в полной матрице, но само по себе не доказывает единственную причину.

## Разделённая модель

Следующий фиксированный опыт отделил механизмы. Position head получает76признаков тела, местной геометрии, собственного движения/action и тайминга. Direction head получает21признак собственного weapon/camera/movement состояния и тайминга. Враги, target IDs, bbox и состав группы в эти heads не входят. Это ограничение внутреннего физического прогноза; политика по-прежнему может использовать весь состав противников для выбора действий.

Сравнены factorized linear275params и MLP2949params: position76→32→3, direction21→16→2. Бюджет/folds/гейты прежние. Из них прошла только MLP; linear снова не прошла направление наsite02.

| Группа полного набора | Fixed velocity: muzzle | Factorized MLP: muzzle | Current-pose direction | Factorized MLP: direction |
|---|---:|---:|---:|---:|
| Отложенные seeds,2702shots |15,87|**6,56**|1,514°|**0,824°**|
| Отложенные site02,5322shots |16,43|**9,15**|1,615°|**1,503°**|

Четыре fits/metrics полного factorized опыта заняли около4,5с после загрузки. Сбор/проверка labels в этот замер не входят. Предсказание позиции улучшилось существенно, направления на другом месте карты — умеренно. Model selection повторяется на development folds; это не независимая оценка конечной боевой политики. Все исходные и отрицательные отчёты сохранены.

Артефакты: `combat-action-effect-probe-v1-20261023` (малый набор), `...-v2-...` (полная192архитектура), `...-v3-...` (factorized). В каждом protocol/report/complete/folds/checkpoints. Политика и runtime не обучались/не менялись; обучались только отдельные predictive regressors на GPU.

## Следующий зафиксированный шаг

Протокол `combat-action-effect-shot-probe-v1-20261023/protocol.json` подготовлен ДО fitting shot-predictor. Сравнить Linear214 с geometry29: historical right current, centered current, centered fixed100ms kinematics, learned motor+та же target velocity100ms. Исходные causal185 и бюджеты classifier прежние:200steps,LR0.001,weight_decay0.0001,seed20261022.

Внутри каждого внешнего TRAIN fold прогноз моторики для classifier TRAIN должен строиться cross-fit по трём whole-battle inner folds (seed%3); ни inner-heldout, ни внешний heldout не участвуют в fitting/normalization своего forecaster. Внешний heldout использует frozen outer checkpoint. Нельзя вычислить geometry из native muzzle/recoil и выдать её за causal input. Нельзя учить classifier на прогнозе, построенном forecaster по его же целевым меткам.

Gate: на обоих внешних development folds новый intervened-shot logloss ниже centered-current и fixed-kinematics; legacy loss не ухудшен более чем0.02 относительно centered-current. Если gate не пройдёт, runtime/новый игровой screen не добавлять. Если пройдёт — отдельный ограниченный native pilot против source/старого ranker/rules,16инстансов ×2 со свежими validation seeds и live CUDA audits. Боевые результаты новых вариантов пока отсутствуют.
