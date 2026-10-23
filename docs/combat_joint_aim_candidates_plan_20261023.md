# Обучаемый выбор поправки прицела внутри PPO

23.10.2026. Эксперимент продолжает основной план learned System1; цель остаётся — полностью обучаемое управление боем и преимущество над rules. Он не объявляет отдельный predictor решением всей задачи. MG — первый контролируемый scope; дальнейшие оружие/карты и совместное обучение исходных голов остаются в основном плане.

Текущий статус: runtime/scorer/initial gate, fresh TRAIN, head-only и matched coupled fits/screens завершены. Последний результат: joint0/8,base-actor3/8,source5/8,rules4/8; новая ветка не принята. Default не меняется. Следующие действия — [credit audit, сохранение исходного поведения при инициализации и проверка update budget](combat_joint_aim_reassessment_20261023.md). Ниже сохранены исходная схема и история gates; результаты — [head-only](testing/combat_joint_aim_training_20261023.md), [coupled](testing/combat_joint_aim_coupled_20261023.md).

## Основание

Аудит `testing/combat_motor_native_error_20261023.md`: физический forecast и оценка actual shots лучше, но greedy выбор25angles дал4/8против5/8старого прицела. Причина не установлена. Проверяем другой способ выбирать действия: PPO (Proximal Policy Optimization — улучшение стратегии по результатам её боёв) должен учить поправку с учётом будущих убийств, полученного урона и завершения цели. Максимум P(попадание сейчас) перестаёт быть финальным решением.

Reward, terminal goal, карты и условия на этом опыте не меняются. Frozen motor/shot predictor — наблюдаемая модельная подсказка. Ни native truth, ни результаты validation не становятся входом или TRAIN.

## Совместное событие действия и вероятность

1. Исходный actor с temporal memory выбирает target/mode, Gaussian source latent4, pose/weapon. Его существующий RNG stream сохраняется.
2. Сохраняется ORIGINAL source latent4. По нему строятся base-action causal185 и25frozen predicted hit logits.
3. Новая candidate head выбирает категорию k из25смещений grid0/−5/−2,5/+2,5/+5° поyaw/pitch. Это stochastic learned choice, не argmax hit predictor. При отсутствии наблюдаемого MG/target/bbox категория0детерминирована, contribution logprob/entropy0.
4. Из source latent и k вычисляется physical aim: tanh→coarse180/fine15degrees→offset→clamp→atanh. Candidate0сохраняет source latent буквально. Fwd/side не преобразуются дополнительно: PPO получает реальные последствия связки движение–прицел.
5. Conditional fire вычисляется по FINAL transformed action, с прежним сохранённым fire uniform. Weapon/pose/target sampling не заменяется правилами.

Событие policy — `(source latent, k, остальные sampled categories)`. Совместная log probability содержит Gaussian density ОРИГИНАЛЬНОГО latent, исходные category terms, новый categorical term и fire term по FINAL aim. Нельзя оценивать transformed latent исходной Gaussian; нельзя вызывать старый source Review на raw latent и забыть, что fire condition должен видеть final action.

Near-boundary clipping может давать одинаковый physical output для нескольких latent/category событий. PPO работает с записанным расширенным событием, а не пытается восстановить marginal density физических углов. Обратного восстановления raw latent по final action не допускается. Стандартная replay/LP приёмка требует recorded source latent и category.

## Голова и входы

`combat_joint_aim_candidates_v1`:1359→64(ReLU)→25logits,88665params. Входы:1147полных наблюдаемых features,185source-action causal features,25frozen hit logits,2наблюдаемых timing features. Включён полный состав противников/доступных предметов из исходного feature schema; голова не ограничена только выбранным мобом. Существующий recurrent actor продолжает формировать base action. Отдельной GRU у prototype head нет.

Все sampled action/hint inputs detached для предотвращения лишнего pathwise gradient. Старый actor при совместном обучении получает обычный score-function gradient через свои log probability terms; source latent считается фиксированным при пересчёте выборки.

Начальная final layer нулевая, bias(category0)=log24:половина вероятности baseline, остальные24поровну. Это сопоставимо с predictive sampler; не заявляется команда-compatible с обычным source, у которого нет новых поправок. Независимые frame-hash gate/offset uniforms не расходуют source actor RNG. Conditional categorical sampling поддерживает произвольные learned probabilities, а не всегда uniform24.

На типичной NN precision начальное распределение соответствует50/24; finite-source compatibility проверяется по фактическим seeds/frames. Старые predictive captures остаются predictive-only. Их нельзя незаметно переименовать в on-policy PPO, даже если начальные marginals близки.

## Выполнено: CUDA math prototype

`combat-joint-aim-candidate-prototype-v2-20261023`:initial assignments совпали с existing exploration sampler на20000frames, встретились все25категорий. На200000draws произвольного неравномерного распределения частоты соответствуют probabilities. Проверены:

- сохранение source movement и literal candidate0;
- конечность coarse/fine offsets вблизи clipping;
- deterministic inactive branch0снулевыми logprob/entropy;
- joint PPO ratio сводится к categorical ratio при frozen source;
- nonzero candidate-head gradient и toy optimization step уменьшают заданный surrogate;
- отсутствие pathwise gradient в исходные observation/hint tensors;
- необходимость final-aim condition у fire term.

Все NN/math checks выполнены CUDA. Toy optimizer step выполнен на копии головы с искусственными данными: это НЕ обучение боевой модели. Initial head checkpoint сохранён отдельно; он не установлен в actor и ещё не принят runtime/PPO processor.

## Дальнейшая реализация и gates

Обновление 23.10: runtime, dedicated CUDA scorer и начальная миграция реализованы. Native gate закрыт 8/8 боями и 4/4 CUDA replay, 626 transitions; 627 provider steps на вариант с точным совпадением source latent и native commands. Полный LP max error 0,00006390. Это начальная совместимость, не обучение/выигрыш качества. Следующий этап — fresh 32 TRAIN и отдельный head-only trainer. Подробности: [начальная приёмка](testing/combat_joint_aim_initial_20261023.md).

### 1. Runtime и scorer с отказом от неподдерживаемого формата

Добавить явные `action_distribution`/head version и sample metadata: raw source latent4, k, categorical LP. До создания первого migratable PPO файла все старые scorers должны явно отвергать неизвестный joint distribution; неизвестные поля нельзя игнорировать. Старые inference-only execution flags и predictive registries остаются неподходящими для стандартного PPO.

Go live Decide/Review и CUDA scorer должны одинаково упаковывать head inputs, применять offsets и пересчитывать весь joint LP. Увеличивать81основных raw outputs не нужно: новая dense branch отдельная. JSON/checkpoint export, optimizer groups и dataset schema версионируются совместно. Source actor/value/std/Adam/RNG/history сохраняются при миграции; predictor frozen. Тест с synthetic NONZERO conditional fire обязателен: source initial fire-zero не покрывает этот порядок.

Go unit checks — только metadata/pure geometry. Neural Go проверяется только в реальной игре с CUDA replay; никакого offline Go neural benchmark или `go test ./...` с NN fixtures. Все обучение/numerical NN checks — CUDA.

### 2. Ограниченная native compatibility

Четыре условия ordinary/recovery с разной геометрией; existing predictive explorer и initial joint candidate policy,8native/4CUDA. Fresh seeds,16×2. Сверить реальные actions/samples с explorer и полный новый joint LP с CUDA; старые counterfactual explorer LP не выдаются за standard PPO. До успешной приёмки не запускать длительное обучение.

### 3. Fresh on-policy TRAIN и head-only PPO

Обновление: head-only fit завершён30actor/40criticsteps CUDA, frozen actor/std/Adam/RNG точны. Новый screen32native/24CUDA закрыт: source5/8,initial4/8,trained5/8,rules4/8;MG иstationary/large-yaw не улучшились. Promotion иexpandedconfirmation отклонены. Это не доказательство превосходства головы. Следующая отдельная гипотеза — [matched joint/base actor control](combat_joint_aim_coupled_plan_20261023.md). Подробный [отчёт обучения](testing/combat_joint_aim_training_20261023.md).

Fresh корпус закрыт32native/32CUDA,4061TRAINtransitions/4076contextrows,trainoffset92,все8условий. Dedicated head-only trainer затем реализован и проверен; standard trainer намеренно новый distribution отвергает. См.testing/combat_joint_aim_initial_20261023.md.

Новый32-case TRAIN corpus через существующий queue pool16×2. Explicit initial joint behavior model/hash и samples; старые40predictive-onlyбоёв в этот PPO не подставлять. Первый matched опыт: head-only actor update до30steps,LR3e−4,critic40steps,existing clip/target-KL safeguards. Остальные actor параметры/logstd и frozen hit/motor predictor должны сохранить точные tensors/moments. Настоящий callback/score используется для likelihood ratio, не transformed Gaussian proxy.

До native screen: finite losses/gradients/ratios, ненулевое изменение candidate head, current-policy replay passed, correct eligibility/LP mask, old actor tensors/moments unchanged. Target entropy/ratio/KL отдельно по новой category и по исходным головам. Прежний reward не правится по validation исходам.

### 4. Небольшое парное сравнение

По8freshvalidationбоёв для source, frozen initial candidate policy, head-trained policy и rules; при необходимости historical ranker отдельным пятым control. Итого32или40native, обязательный CUDA replay всех ordinary stochastic PPO вариантов. Размер/состав зафиксировать до запуска. Победы/полученный урон/actual MG outcomes и лишние действия оценивать отдельно. Малый успех расширять до24боёв на вариант; default автоматически не заменять.

### 5. Совместное обучение и покрытие

Если новая голова доказывает пользу, следующий отдельный fork обучает её совместно с основным actor, затем переносится по основному реестру оружия/карт. Если она не даёт выигрыша, сохранять отрицательный результат и пересматривать action representation/credit assignment, а не бесконечно доучивать greedy classifier. Емкость, GRU/attention и learned dynamics остаются средствами, не критериями завершения задачи.

Предохранитель формата сохраняется: `validate_precision_model` отвергает action_distribution/aim_candidate_head в старом Python scorer. Go loader использует DisallowUnknownFields и теперь явно валидирует поддержанный joint header. Dedicated runtime/scorer принимают только свой versioned contract; старый scorer не должен молча обрабатывать новый distribution. Положительная поддержка принята сначала на initial, затем на trained ненулевой голове в реальной игре/CUDA replay.

23.10: независимый native/credit audit завершён на 32 TRAIN боях: 4061 переход, 33 убийства, 19 смертей; потерь native effects и преждевременных handoff не найдено. GAE и spacing проверены отдельно. Новая near-identity 99% инициализация прошла 8 native / 8 CUDA: source и near по 1/4 цели, 213 команд до первой поправки совпали. Боевого обучения этой версии ещё нет; это техническая приёмка, не quality gain. Следом свежий общий TRAIN 32 и фиксированный опыт B/G/S с четырьмя обновлениями на ветку. Протокол combat_joint_aim_near_training_protocol_20261023.md; отчёт testing/combat_joint_aim_credit_near_20261023.md. Default не меняется.

