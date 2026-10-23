# Игровая проверка shared candidate scorer

23.10.2026. Протокол фиксируется после fixed500 fit на43состояниях и CUDA export, до новых validation боёв. Только shared191→16→1 прошёл основной и fresh development gates: useful mass7,5169% против constant2,1679%; fresh7,0724% против2,0218%. Это внутренний отбор прототипа, не победность.

Runtime: optional aim_candidate_head.shared_scorer kind shared_candidate_causal191_v1. Нормализация185 causal candidate inputs по первым185 mean/std frozen shot predictor, clip±20; offsets yaw/pitch делённые на5; candidate/baseline/difference sigmoid hints; baseline flag.16ReLU→1, outputclip±20, baseline beta±20. Категория25, преобразование угла, Gaussian source draw, target/mode/weapon/pose/movement, final-aim fire и frame-hash sampler прежние. Legacy head сохраняется в файле для provenance, но при наличии shared logits берутся только из shared. Диагностический one-frame override запрещён вместе с shared.

CUDA export воспроизводит все43состояния прототипа, проверяет JSON roundtrip и matched constant (mean25probabilities только наTRAIN). Тело/value/std/fire/predictor и исходные checkpoints не меняются. Новые checkpoint wrappers явно evaluation-only; продолжение PPO с таким head ещё не поддерживается и отвергается загрузчиками. Не выдавать их за training resume.

Native screen:8paired validation условий base3/ware2 site01/02 ordinary/recovery, четыре варианта sourceS4/shared/matchedconstant/rules.32native/24CUDA. Свежий общий seed offset33 при прохождении штатной проверки истории, пул16×2; иной offset допустим только после отказа preflight до игр с сохранением причины. Reward, weapons, difficulty, fixture recipes неизменны.32ГиБ minimum storage reserve. No training/автоматическая promotion.

Прежде чем считать quality: Go live исполнения shared должны пройти обычный полный CUDA audit (value, actor memory, original Gaussian latent, final-aim fire, chosen category probability, hash sampling). Offline Go NN replay не выполнять. Дополнительно проверить все25live logits/вероятностей на тех же записанных состояниях черезCUDA; не ограничиваться выбранной категорией. Новые helper edits только после закрытия текущего пула.

Показать цели/kills/deaths/dealt/received, gained/lost парные исходы; actual MG living damage/все выстрелы, solid contacts, targetless shots, large-yaw иstationary-clear. Ни native technical acceptance, ни локальнаяточность не заменяют победность.

Допуск к новой24-case-per-variant confirmation: shared целей≥source+2, ≥constant+1 и≥rules; deaths≤source иreceived≤1,1source. Это gate на дополнительную проверку, не доказательство превосходства. Если не пройден — не запускать большую confirmation и не менять порог задним числом. Если пройден — заранее зафиксировать новые72или96native paired cases и итоговый критерий до их исходов. Default остаётся прежним до отдельной широкой приёмки.

Параллельная return diagnostic не меняет этот teacher/gate: одиночное улучшение immediate HP иногда даёт худший остаток боя. Сохранить это ограничение, не выбирать checkpoint по native screen.
