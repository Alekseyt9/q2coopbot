# Приёмка диагностической поправки в одном кадре

Протокол23.10.2026 до новых игр. Контракт combat_joint_aim_one_frame_probe_v1. Это acquisition smoke, не fitting, validation или проверка победности.

Источник — запечатанная S4 baseline копия combat-native-aim-pairs-baseline-model-v1-20261023. Меняется только metadata diagnostic_joint_aim_probe; actor/value/std/fire/predictor и все neural параметры неизменны. Контракт допустим только при нулевых весах последнего слоя constant head. Во всех кадрах, кроме выбранного, logits равны прежним biases. Декларированный override не использует native labels.

Две копии: baseline с прежними logits в frame110 и альтернатива category4 (yaw0°,pitch+5°) с полной положительной поддержкой25категорий и прежним frame-hash sampler. Категория выбрана заранее как технический ненулевой вертикальный сдвиг; полезность неизвестна. Native conditional fire продолжает учитывать итоговый прицел.

TRAIN seed1172065112, ware2 site01 ordinary, намеренный повтор известного TRAIN случая. В frame110 по старой записи движение около300units/s, recoil pitch−9°, baseline выстрел в геометрию. Этот кадр не является record-high gate, прежний constant-head метод не мог сделать его первым вмешательством. Пример не принадлежит development seeds.

До игр CUDA проверяет обычный baseline head против probe logits вне frame110 на записанных inputs, ожидаемые категории на всей исторической траектории и отказ ordinary constructor/precision trainer от probe metadata. Код Go офлайн не исполняет NN: только build и pure metadata validation. В native smoke — два боя, существующий пул до16, ×2, новый planner/runtime из текущего checkout.

После игр два CUDA replay должны подтвердить original Gaussian/full augmented LP, value/recurrent state, category transform и frame-hash choice с явным recorded frame tensor. Finalized probe corpus получает отдельный diagnostic version, ppo_training_ready=false. Обычные Python trainers и обычный Go PPO export отвергают probe; CUDA-only exporter допускает его только для этой проверки. Никакого обычного PPO fitting на этих данных.

Приёмка: baseline воспроизводит прежние source actions/features; пара имеет одинаковые physical/features/memory/source draws/command prefix, первое отличие category ровно в110, head timing одинаков в этот момент. ID объектов сопоставляются только стабильной bijection, остальные поля не удаляются. Снять actual native MG outcome обоих вариантов. Не требовать попадания или победы для технической приёмки, не менять кадр/offset ради нужного исхода. Последующие действия/миры могут разойтись.

Первый screen/fit из старого16-state протокола остаётся не допущенным; это отдельная инженерная приёмка нового способа сбора. Рабочие веса/default не меняются.
