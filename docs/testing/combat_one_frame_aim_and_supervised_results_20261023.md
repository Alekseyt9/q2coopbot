# One-frame сбор и первый supervised fit головы

23.10.2026. Диагностическое вмешательство в заданном кадре реализовано и принято реальными играми/CUDA. Новый17-state набор прошёл admission;200-step head fit завершён, но development gate не пройден. Новые веса не направлены в native quality screen, default не менялся.

## Произвольный кадр: работающий контракт

`diagnostic_joint_aim_probe`, version combat_joint_aim_one_frame_probe_v1: объявленные25logits применяются только в заданном observation frame и только при обычной eligibility. В другие кадры сохраняется constant baseline head. Actor/value/std/fire/predictor/RNG и neural параметры не меняются. Conditional fire видит окончательный прицел; Gaussian/category draws прежние. Все направления имеют положительную поддержку.

Контракт разрешён только с нулевыми весами последнего слоя head. Go actual live path получает logits из metadata либо прежних biases, без бесполезного вычисления hidden/hints для известной константы. Нормальные модели остаются на прежнем neural path. Численная проверка полной original-Gaussian/final-fire/categorical density, value и recurrent state выполняется CUDA с recorded frame tensor.

Обычные Python constructors/trainers и quality evaluator отвергают probe; Go IsStochastic eligibility для обычного PPO также false. CUDA-only native exporter разрешает его явно, а finalized metadata получает отдельный diagnostic rollout version и ppo_training_ready=false. Обычный merged PPO pipeline запрещён. Это acquisition инструмент, не ограничение production обучаемого боя.

## Native приёмка и новые метки

Выбран заранее TRAIN seed1172065112, ware2 site01 ordinary, frame110. Бот движется около300units/s, recoil pitch−9°. Этот кадр нельзя было выбрать первым вмешательством прежним record-high методом.

Первый smoke:2native/2CUDA. Baseline0 промахнулся; category4 (yaw0°,pitch+5°) нанесла8HP выбранному живому монстру. До кадра110 совпали physical/features/memory/source draws/команды, timing в110 одинаков. Это локальный causal результат, не оценка победности.

Затем полный5-way повтор:0,4,3,14,1. Все5native/5CUDA и5prefix сравнений приняты. Category4/3/14 дали по8HP, baseline0/category1 попали в геометрию. Дополнительно baseline воспроизвёл прежний бой целиком:133provider frames с теми же actions/features. Обычный PPO exporter отказал probe до NN evaluation; receipt сохранён. Go unit test выполнял только metadata validation, без CPU NN; другие NN проверки только CUDA.

Первый5-way запуск остановился **до игр** из-за UDP45301, занятого PnkBstrB. Ничего не завершалось/не останавливалось в этой службе. После проверки процессов/портов выбран свободный диапазон46500, новый root v2. Неудачный preflight сохранён отдельно; это не повтор провалившихся боёв.

CUDA preparer теперь принимает явные --seed/--frame/--holdout-dataset и2..25категорий; проверяет actual MG window, first life и запрет development seed. Другой seed/frame (base3 seed1164065112/frame99) прошёл CUDA API preparation; новых native игр для этого дополнительного API примера не было. Полная native приёмка здесь относится к ware2/frame110.

## Набор v2 и fixed fit

Создан новый dataset v2:17контекстов/85native вариантов, прежние16записей и split сохранены без изменения полей (проверено сравнением записей).12train состояний на7seed,5development на4seed. Новый frame110 добавил второй независимый TRAIN seed с полезной поправкой; на каждомseed не больше2состояний. V1 admission=false не переписан.

Серия этого этапа:2smoke +5label native/CUDA. Объединённый label набор использует прежние80вариантов +5новых. Повторные smoke контроли не считались дополнительными независимыми состояниями.

Обучена только существующая1359→64ReLU→25голова:88665параметров,200полных GPU шагов Adam LR0,0003,weight decay0,0001,clip0,5. Target=0,1source distribution+0,9native preference: baseline при не худшем наблюдённом HP, иначе лучшие принятые альтернативы. Непроверенным действиям не присваивались фактические отрицательные метки. Actor/value/std/fire/predictor и их Adam/RNG/history/PPO counters сохранены; head optimizer сброшен явно из-за новой задачи. Constant control имеет средние25вероятностей финальной head поtrain.

Loss train1,258→0,224. Это fitting показатель, не качество боя.

| Средняя масса на проверенных улучшающих действиях | Source | Trained MLP | Constant |
|---|---:|---:|---:|
| Train12состояний |0,013%|14,978%|2,503%|
| Development5состояний |0,015%|0,268%|2,003%|

Trained вернулась к baseline примерно99,0% наdevelopment, хотя наtrain baseline≈84,9%. Это признаки слабого переноса/запоминания малого набора. Нельзя установить единственную причину по17состояниям. Масса известных вредных действий не выросла, но рост полезной массы против source всего0,253п.п. вместо требуемых5п.п.; против constant −1,735п.п. вместо+2п.п. Gate=false. Native32-case screen не запускался.

Conditional known-action HP наdevelopment:source4,800,trained4,821,constant4,969. Эти значения нормированы только по проверенным действиям; constant имеет больше неизвестной массы. Поэтому это не оценка ожидаемого реального урона полной политики и не доказательство превосходства constant в бою.

## Следующий ограниченный опыт

Не наращивать вслепую тот же88k MLP fit. Проверить наCUDA малые shared candidate scorers: одинаковая функция оценивает каждое направление по его offset/прогнозу и общему состоянию. Это гипотеза о более экономном использовании данных и переносе между направлениями, не доказанный результат.

Сначала небольшой калибратор score/предпочтения baseline и shared MLP с низкой размерностью action-conditioned inputs. Сравнить с закрытым flat MLP и постоянным распределением на том же фиксированном train/development split, с заранее заданными бюджетами. Development здесь — инструмент итерации гипотез, не независимый combat validation. Реальные native labels остаются источником истины; нельзя заменять их output forecaster.

Если offline результат проходит критерий — отдельная versioned runtime реализация и свежий native source/trained/constant/rules screen. Если нет — расширять one-frame TRAIN сбор поздними движущимися/отдачными состояниями и разбирать ошибки forecaster/меток, сохранив development seed. Production head не заменяется диагностическим расписанием, rule aim или greedy predictor.

## Артефакты

- `combat-one-frame-aim-preparation-v1-20261023`, `combat-one-frame-aim-smoke-v1-20261023`, `combat-one-frame-aim-smoke-proof-v1-20261023`.
- `combat-one-frame-aim-label-preparation-v1-20261023`, `combat-one-frame-aim-labels-v1-20261023` (preflight failure), `combat-one-frame-aim-labels-v2-20261023`, `combat-one-frame-aim-label-proof-v1-20261023`.
- `combat-native-aim-pairs-dataset-v2-20261023`:parent preservation/fixed split/admission.
- `combat-native-aim-supervised-v2-20261023`:trained иconstant веса/чекпоинты, fixed endpoint/gate=false.
- `combat-one-frame-aim-generic-api-cuda-v1-20261023`:отдельнаяCUDA API preparation для base3/frame99.

Протоколы: `../combat_native_aim_one_shot_smoke_20261023.md`, `../combat_native_aim_one_shot_label_extension_20261023.md`, `../combat_native_aim_supervised_protocol_v2_20261023.md`. Планы/первые выводы не заменены задним числом. Все процессы этапа завершены; коммиты/push/default без изменений.
