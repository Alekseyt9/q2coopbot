# Парный контроль совместного actor/candidate PPO

23.10.2026. Исходный head-only опыт закрыт: source5/8, initial4/8, head-only5/8, rules4/8; MG и stationary/large-yaw показатели не улучшились. Голова обучается и runtime/scorer корректны, но положительной quality-приёмки нет. Не увеличивать confirmation этой модели и не менять reward по validation исходам.

## Гипотеза и общий материал

Замороженный actor не может подстроить движение, исходный прицел, выбор цели и момент огня под новые поправки. Это гипотеза, не установленная причина проигрышей. Проверка должна отделить пользу обычного дополнительного PPO от пользы совместного обучения новой categorical-ветки.

Оба новых fork начинаются с **одинакового initial**, `combat-joint-aim-initial-v1-20261023`, а не с уже обученной головы. Тот же закрытый TRAIN `combat-joint-aim-corpus-v1-20261023`/4061 rows. Для каждого fork это первый update данного rollout от исходного on-policy behavior; head-only checkpoint и validation траектории не служат обучающими входами. Predictor одинаковый и заморожен.

## Два matched fit

1. **Base actor only**: исходные actor/std и их существующие ветки обновляются; новая categorical head полностью frozen с прежним распределением. Это не старый motor-residual `base-only` scope, где замораживали отдельную motor-ветку: здесь обучаются все исходные ветки.
2. **Joint actor + candidate**: те же исходные actor/std плюс новая голова. До 30 принятых actor steps, обычные исходные LR групп (base3e−5, branch LR из source optimizer), candidate LR3e−4, clip0,2/target KL0,005/max grad norm0,5. Guarded Adam/backtracking как в текущем PPO. Reward/GAE/entropy прежние.

Critic стартует одинаково, оптимизируется 40 steps на одинаковые returns с прежними LR/Adam. Проверить точное совпадение итоговых critic tensors/moments между двумя новыми fork и ранее выполненным head-only fit: меняется actor scope, не critic objective.

Сохранять целиком source Adam/RNG/history перед fork. Новая head имеет отдельный resumable optimizer state. У combined optimizer нельзя потерять/перенумеровать моменты старых групп; при сохранении разделить группы явно и проверить roundtrip на CUDA. У base-actor-only head tensors/moments должны остаться точными. Для совместного обучения старые actor tensors намеренно меняются; не заявлять их сохранность после fit.

Полный совместный scorer обязателен: Gaussian по original source latent, conditional fire по final action, categorical по записанной категории. Source actor raw/memory при обновлении пересчитываются по полному context. Можно кэшировать frozen hint/head inputs от записанного action; нельзя кэшировать меняющуюся исходную Gaussian/category/fire likelihood. Score-function gradients не заменять pathwise gradient через source action.

## Gates до игр

TRAIN-only provenance, reward/model/dataset SHA и consumed history; исходные tensors/moments совпадают до fit. Finite gradients/ratios/loss, принятый update, KL safeguards, scalar std clamp[-8,1]. Обычные головы и новая category: отдельно KL/entropy/ratio. После fit текущий full CUDA scorer совпадает с JSON/checkpoint replay. Frozen predictor точен; у base-only также точны head и её optimizer. Все NN checks/training CUDA; Go NN только в реальной игре.

## Следующий ограниченный screen

Фиксировать до игр **6 вариантов × 8 новых validation условий = 48 native / 40 CUDA**: original source, initial joint, уже обученный head-only, новый base-actor-only, новый joint, rules. Все восемь ordinary/recovery условий реестра base3/ware2; новые неиспользованные seeds, парные условия между вариантами; 16 инстансов ×2, один пул.

Главное сравнение — joint против base-actor-only. Дополнительно joint против source/head-only/rules и base-actor-only против initial/source. Оценивать цели, смерти/полученный урон, actual MG outcomes, stationary-clear/large-yaw/target switches. Обязательный CUDA replay каждого learned варианта, включая ненулевые heads. Малый эффект не равен превосходству; default автоматически не заменять. Если joint не отделяется от обычного actor update, не приписывать новой архитектуре выигрыш.

Протокол выполнен. Принятые fits: `combat-joint-aim-coupled-training-v3-20261023`,24base-actor/21joint steps и40critic; critic/Adam совпали с head-only reference. Screen48native/40CUDA закрыт: source5/8,initial2/8,head-only2/8,base-actor3/8,joint0/8,rules4/8. Promotion/confirmation=false. Два ранних технически отклонённых fit не вошли в игры. Подробности — [результат и аудит](testing/combat_joint_aim_coupled_20261023.md). Далее — [credit/инициализация/update budget](combat_joint_aim_reassessment_20261023.md).
