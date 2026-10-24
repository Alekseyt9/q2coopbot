# Калибровка выбора цели — 24.10.2026

Реализован optional target_sampling: temperature0,5 применяется только к masked categorical target probabilities для sampling и log probability. Raw actor outputs, inputs движения/прицела/огня, log_std и neural parameters сохранены. Исходная настройка1 и отсутствие header дают прежний путь вычислений. Новых RNG draws нет. Жёсткая фиксация на враге и скрытые HP не используются.

Go contract test и сборка q2coopbot/q2ppo-data/q2episode пройдены. Полные реальные temporal histories32TRAIN/4298rows проверены наCUDA: raw source/unit/calibrated одинаковы; unit full density/entropy точны; изменение density ограничено target factor; RNG при scoring не расходуется. Source/native LP maxerror0,00011158, target-factor delta error7,15e-7 в прежнем допуске. Loaded Python helpers сохранены в source snapshot.

Native unit compatibility закрыта8native/8CUDA на4картах: все4гейта пройдены,646provider строк. Fresh source точно воспроизвёл архивную первую жизнь; header1 точно воспроизвёл fresh source по exposed observations/features/samples/commands. Это техническая совместимость, не качество новой настройки.

## Свежий боевой screen

Validation offset48,8условий на вариант,24native/16CUDA закрыты, native pool73,71с. Проверены реальные ненулевые температурные log probabilities. Нейросетевое обучение не выполнялось, source/default не менялись. Header явно запрещён для обычного PPO до отдельной migration; это inference-only artifact без выдуманного checkpoint.

| Вариант | Победы / 8 | Смерти | Убийства | Получено HP |
|---|---:|---:|---:|---:|
| Source/noise-only | 2 | 5 | 7 | 553 |
| Target temperature0,5 | 3 | 5 | 9 | 619 |
| Rules | 6 | 0 | 15 | 180 |

Заранее заданный gate=false: нужна прибавка минимум2победы, не меньше побед, чем уrules, и не выше110%уронаsource; эти условия не выполнены. Ветку не расширяем до24условий и не принимаем вdefault.

Переключения при ещё видимой предыдущей цели уменьшились:56/669→24/700. Native MG live-monster damage contacts132/183(72,13%)→136/179(75,98%); solid contacts40→40. Но standing grounded-visible260/659(39,45%)→357/692(51,59%). Состояния и длины боёв различаются, поэтому частные доли не доказывают причинную связь и не заменяют итог боя. Меньшая случайность цели сама по себе не решила задачу.

## Свидетельства

- combat-target-sampling-cuda-preparation-v1-20261024: реальные full-actor TRAIN проверки, модели unit/calibrated и snapshot кода.
- combat-target-sampling-zero-native-v1-20261024/dataset: строгая совместимость source/unit.
- combat-target-sampling-evaluation-policy-v1-20261024: evaluation-only weights SHA88d07e907ce30f9e6fa431f0ffb3bd3cbbb9ff79ff928e364399bd0927482a09.
- combat-target-temperature-screen-v1-20261024:24native/16CUDA и matched сравнения.
- combat-target-temperature-quality-v1-20261024 и combat-target-temperature-damage-v1-20261024: реальные выстрелы, movement и target switching.
- combat-target-temperature-decision-v1-20261024: заранее заданный gate, hashes и totals.

Следующий отдельный эксперимент — более длительное on-policy обучение на24разных рецептах из реестра с обоими участками каждой карты; он начинается с прежнего noise-only4b109703, а не с непринятой температурной модели. См.combat_diverse_registry_ppo_20261024.md.
