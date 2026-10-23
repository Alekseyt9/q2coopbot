# Поздние native пары и shared scorer

23.10.2026. Расширенный набор принят, один CUDA-прототип прошёл заранее заданные development gates. Игровое превосходство пока не проверено; default не меняется.

## Данные

Свежие16baseline боёв на8условиях base3/ware2 site01/02 ordinary/recovery. До чтения альтернативных исходов выбраны26поздних состояний на13seed;24состояния включают движение≥50units/s и |recoil pitch|≥1°. Seed split назначен заранее. Пул16×2 выполнил130native повторов;130CUDA приняты. Native pair audit принял129/130вариантов. Единственное исключение: seed1252025114/frame145/category14, другой head timing. Не повторялось и не возвращалось в labels.

Все26baseline воспроизводят исторические действия/features до вмешательства. Исправляющие промах альтернативы найдены в9контекстах; ухудшающие успешный выстрел — в7. Child dataset сохраняет17parent contexts/splits, добавляет26новых:43состояния/214принятых вариантов.26train contexts/14seed,17development/10seed; новыйсрез14train/7seed и12development/6seed. Вtrain7исправимыхконтекстов на7seed, вdevelopment6. Admission пройден; development не участвует в градиенте.

## Три фиксированных GPU fit

Каждый500шагов, старт от той же S4, прежние teacher/нормализация/ограничения, безearly stopping или checkpoint selection. Колонка useful — средняя вероятность **известных по native проверке улучшающих действий**, не точность и не победность. Контроль — среднее25вероятностей своегоfit наTRAIN.

| Вариант | Параметры доводки | Useful development | Constant | Fresh useful / constant | Gate |
|---|---:|---:|---:|---:|---|
| Flat |88665|2,180%|2,139%|3,079% /2,020%|нет|
| Calibrator |2|3,372%|1,881%|4,137% /1,778%|нет|
| Shared191→16→1 |3090|7,517%|2,168%|7,072% /2,022%|да|

Shared на5parent development:8,584% против2,519%constant; наTRAIN20,215%. Harmful mass development0,0845%, fresh0,1137%. В2из17development состояний argmax — непроверенное направление; неизвестные действия не объявляются промахами. Это маленький целевой development набор и single-spread labels, не независимая оценка качества боя. Старые отрицательные17-state fits сохранены.

## Последующая награда

Отдельный заранее зафиксированный протокол сравнивает те же one-frame вмешательства с остатком фактической награды. Никакого новогоfit или изменения teacher. Используются доступные first-life reward rows; incomplete_transition допускается только как явно посчитанный хвост после последней доступной строки, внутренний разрыв отвергается. Усечённый capture не объявляется полным бесконечным return.

| Срез | Контексты | Можно улучшить immediate HP | Можно улучшить полный observed reward | Immediate лучше, sparse return хуже |
|---|---:|---:|---:|---:|
| Train |14|5|11|3|
| Development |12|4|10|3|

Для полного reward есть5train/3development контекста с хотя бы одной альтернативой, улучшающей immediate HP, но ухудшающей остаток награды. Нет общего лучшего действия immediate/full/sparse в6train/5developmentконтекстах. Одна случайная реализация на действие не оценивает ожидаемыйQ; разница может включать чувствительность последующих траекторий. Это ограничение immediate teacher, а не основание обучать поdevelopment или автоматически менять reward.

## Runtime и следующий screen

Добавлен optional shared_scorer в joint aim head. Gaussian draw,25категорий, frame-hash sampler, final-aim conditional fire и базовое управление сохранены. Pure metadata unit test и сборка Go прошли. CUDA exporter воспроизводит вероятности shared prototype точно на43состояниях, constant — с максимальной ошибкой1,2e−7; JSON roundtrip проверен. Go neural forward выполняется только в игре. Live запись всех25logits позволит проверить весь вектор наCUDA.

Exported checkpoint wrappers — evaluation-only, не Adam/RNG resume; загрузчики PPO и старые joint trainers не допускают shared к неподдержанному обучению. Sourcebody/checkpoints неизменны. Runtime готов к проверке, ещё не принят по live quality.

Свежий screen закрыт:32native/24CUDA,8paired условий × sourceS4/shared/constant/rules, validation offset33,16×2. Перед запуском preflight проверил совпадение fixtures и историю6658seed. Все24CUDA приняты. Уshared313eligible frames со всеми25live logits;maxlogiterror7,3e−6впервых просмотренных случаях, полный maximum сохранён в decision report. Hash category/Gaussian latent/final-aim fire/value/memory проверены штатно.

| Вариант | Цели | Убийства | Смерти | Нанесено / получено | Actual MG living damage / shots |
|---|---:|---:|---:|---:|---:|
| Source S4 |3/8|10|4|2152 /552|158/250 (63,20%)|
| Shared |6/8|17|0|2626 /488|162/227 (71,37%)|
| Constant |3/8|11|4|2220 /585|143/246 (58,13%)|
| Rules |5/8|13|3|2401 /448|MG не оценивался этимreporter|

Shared противsource/constant:3gained/0lost;противrules2gained/1lost. Порог дополнительнойconfirmation пройден. Это8условий, не доказательство общего превосходства надправилами.

Solidcontacts source87→shared57;targetless MG shots14→8. Largeyaw9,02%→5,46%;stationary-clear23,16%→23,08%, тоестьстояние не устранено. Командыатакибезвыбраннойцели13,78%→17,41%, рост3,63п.п. сохраняется, несмотрянаменьшеtargetlessMGshots:эторазныеметрики. Frames/бой147,5source/176,5shared/114,9rules, поэтому более быстрое прохождение не доказано. Бои имеют разные длины и ранние смерти.

Отдельный frozen confirmation запущен на96native/72CUDA:по24новыхбоя на4варианта,offsets34..36,16×2. Его результаты не объединяются сscreen дляподбораcheckpoint. Критерии включаютцели,потери,MGточность,largeyaw иtargetlessattack; зафиксированы до исходов в [протоколе](../combat_shared_aim_confirmation_protocol_20261023.md). Defaultпо-прежнемубезизменений.

Артефакты: combat-native-aim-late-{baselines,selection,preparation,probe,proof,dataset}-v1-20261023; combat-native-shared-scorer-late-v1-20261023; combat-one-frame-return-diagnostic-v1-20261023; combat-shared-aim-runtime-v1-20261023; combat-shared-aim-native-evaluation-v1-20261023. Все внутриworkspace/artifacts. Новых commits/push нет.
