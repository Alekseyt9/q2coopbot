# Прогноз по замороженному внутреннему состоянию actor

Статус: причинный cache, два training seeds и нормализационный контроль закрыты
на CUDA. Новых боёв и изменений рабочей policy не было.

## Что проверялось

После [raw-feature predictor](combat_future_prediction_20261022.md) нужно
отделить полезность дополнительной информации от возможности извлечь её из
уже обученной модели. Source — закрытый `control-source` из
`combat-sequence-hitscan-pair-v2-20261022`, actor195 cumulative PPO steps.

У source trunk два выхода: current encoder64 и causal attention64.
Base logits складывают `head(encoder)` и `residual(attention)`, поэтому
одного attention output недостаточно для описания всего trunk.
Spatial/weapon branches дополнительно читают raw observations; их информацию
этот frozen-trunk probe целиком не описывает.

`prepare_combat_future_latent_cuda.py` построил
`combat-future-latent-cache-v1-20261022/cache.pt`. Сохранены current encoder
и результат attention на всех 3350 TRAIN / 1926 development rows из
проверенного binary dataset v3. Воспроизведены **полные** canonical histories,
включая provider context, не попавший в future label mask. Source weights/checkpoint,
dataset и sequence hashes связаны; Go memory tokens проверены на CUDA.

Для всех 48 cases выполнены два теста причинности: возмущение будущих
features и вычисление на обрезанном prefix не меняют прошлые representations.
Сумма head/residual воспроизводит base logits. История window32 и resets
сохранены; cached row keys точно совпадают с source dataset.

## Сравнение декодеров

Каждый вход дополнен теми же 6 текущими applied action channels.

| Вход | Размер | Обучаемые параметры decoder |
|---|---:|---:|
| Raw features | 1147+6 | 168739 |
| Encoder | 64+6 | 30115 |
| Attention | 64+6 | 30115 |
| Encoder + attention | 128+6 | 38307 |

Все decoder — MLP2×128, residual к явному observed constant-velocity baseline.
Таким образом, CV bypass предоставляет текущую velocity **каждому** варианту,
и это не попытка восстановить всю динамику исключительно из latent.
Сжатие входов и разное число trainable parameters указаны явно; одинаковой
ёмкости со всей policy или raw MLP здесь не утверждаем.

Output35 = motion27 + track8. Вырожденный conditional clear-shot head исключён
из обучения; clear metrics в новых reports — только persistence baseline.
Поэтому новый raw comparator обучен заново с теми же 35 outputs/loss,
а не взят из прежней 43-output пары.

Внутри каждого опыта: одинаковые TRAIN minibatches, 2000 updates, batch256,
AdamW lr0.001/weight_decay0.0001, motion SmoothL1 beta0.1 +0.1 track BCE.
Output initialization zero: initial motion forecast ровно CV.
Encoder/attention decoder имеют полностью одинаковые initial weights;
все варианты имеют одинаковые middle/action weights и output initialization.
Development не выбирает checkpoint: всегда fixed final.

Два seeds20261022/20261024, затем такой же фиксированный повтор с TRAIN-only
centering/std normalization (floor0.05, action не нормализуется).
Batch draws совпадают между normalized/non-normalized вариантом того же seed.
Всего 16 decoder fits; каждый блок из четырёх занял около41s на CUDA.

## Development результат

Enemy RMSE — ошибка на ось в Quake units за100ms, тот же набор16 cases.

| Вход | Seed22, без нормализации | Seed24, без нормализации | Seed22, нормализован | Seed24, нормализован |
|---|---:|---:|---:|---:|
| CV | 5.883 | 5.883 | 5.883 | 5.883 |
| Raw | 5.389 | 5.340 | 7.158 | 6.852 |
| Encoder | 5.885 | 5.872 | 5.876 | 5.903 |
| Attention | 5.878 | 5.870 | 5.880 | 5.891 |
| Combined | 5.880 | 5.857 | 5.893 | 5.921 |

Ненормализованный raw comparator даёт около8–9% улучшения к CV.
Combined остаётся близко к CV: pooled improvement0.05%/0.44%, case-weighted
RMSE5.952/5.928 против CV5.960. Простая нормализация разрыв не устранила;
она ухудшила и raw comparator. Это не исчерпывающий поиск всех оптимизаторов
или доказательство отсутствия forecast information при любом decoder.

Self RMSE без нормализации: raw4.385/4.275, encoder5.001/4.984,
attention5.477/5.415, combined5.061/5.092, CV7.093.
Действующий физический PMove predictor на своём subset остаётся значительно
лучше; оснований переучивать известную self physics нет.

Полного collapse констант не видно: у encoder42/64 channels std>1e-4,
у attention64/64. Это проверка вариативности, не доказательство полезности
для конкретной задачи. Нельзя утверждать, что вся боевая модель лишена нужной
информации: она также использует direct raw-feature branches.

Артефакты:

* `combat-future-readout-v1-20261022` и `...readout-replica-v1-20261022`;
* `combat-future-readout-normalized-v1-20261022` и `...normalized-replica-v1-20261022`;
* `combat-future-latent-cache-v1-20261022/readout-audit-v2.json`.

Audit закрыл receipts, budgets, common draws, исходный actor, causal cache,
initial CV equivalence и точное вычисление normalization stats **только TRAIN**.
Source/default не менялись, native quality не оценивалась.

## Решение и следующий прямой тест

Frozen-trunk readouts не извлекли из этих states полезного улучшения enemy
forecast относительно CV, сопоставимого с raw predictor. Гипотеза auxiliary
обучения representation остаётся уместной, но игровая польза не доказана.
Дополнительные sweep decoder sizes/seeds/normalizers автоматически не назначены.

Следующий тест — matched PPO control/auxiliary pair, затем свежие native
source/control/aux/rules бои. Это уже изменение actor training, а не ещё один
декодер на неизменных признаках. Сохранить одинаковую runtime architecture;
future head используется только в обучении и не исполняет игровые команды.

Исходные данные допустимы лишь после проверки on-policy hashes и отсутствия
их rollout SHA в consumed source checkpoint. Development никогда не fit.
Source config проверен: actor scope all, actor_lr3e-5, weapon residual_lr3e-4,
actor step cap10, value steps40, target KL0.005. Общий objective reward SHA
совпадает с корпусом. Не увеличивать эти caps только в aux arm.

Предварительный фиксированный fork: control coefficient0, aux coefficient0.1
для masked known-velocity enemy-motion residual loss. Self/conditional-clear
задачи в actor auxiliary не включать. Source/cases/initial actor/value/Adam/RNG
и начальный future decoder одинаковы. Decoder initialization — заранее
фиксированный seed22 unnormalized combined, не выбор лучшего по dev.
Decoder optimizer отделён от actor; контрольный decoder получает detached
actor features. В candidate auxiliary gradient проходит в encoder/attention.

KL guard, accepted step counts, decoder updates и wall/CUDA cost показывать
для обеих сторон; равные caps не выдавать за одинаковое число accepted updates.
Сохранять точную objective-fork/resume metadata и не переносить future head
в runtime weights. После CUDA pair audit — frozen32native/24CUDA smoke,
новые seeds, pool16 ×2. Только воспроизводимый игровой результат оправдает
больший бюджет, многошаговый forecast или JEPA/EMA/planning.


Последующий matched PPO auxiliary тест выполнен: два пилота не дали улучшения боя. См. [результаты и аудит reward/credit](combat_future_auxiliary_ppo_20261022.md). Default не менялся.
