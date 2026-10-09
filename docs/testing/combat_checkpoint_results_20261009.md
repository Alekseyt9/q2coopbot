# Итог Attention128 checkpoint continuation, 2026-10-09

Четыре дополнительных CUDA PPO раунда дали небольшой прирост побед: seed07 **46→48/80**, seed08 **45→46/80**. Исходный FireBC получил49/80, обычные rules69/80. Ни один кандидат не продвигается: преимущества перед controls и независимого final-test результата нет.

## Покрытие и продолжение обучения

Серия `workspace/artifacts/attention-continuation-v1-20261009`:640/640 новых тренировочных боёв, восемь continuation updates. Каждая модель получила320 дополнительных own-policy battles, вместе с исходным update —400,5 updates,50 actor optimizer steps и200 critic steps. Adam state и история consumed rollout сохранены; CUDA tensor audit всех переходов и seals описан в [протоколе](combat_checkpoint_continuation_20261009.md). Численные policy проверки выполнялись на GPU; Go policy replay/tests не запускались.

Оценка480/480 завершена, ошибок нет;80 одинаковых условий для каждой из шести variants,20 семейств, пул16/×2. Исходный driver прервался после237 queue receipts. Recovery проверил и сохранил238 terminal members, включая один без queue receipt, и завершил242 оставшихся боя. Незавершённые captures сохранены отдельно. Все480 member reports/fixtures/seeds/frame budgets/source/runtime/binary hashes проверены; общий source fingerprint сохранился. `report.json` серии закрыт по sealed four-round reports и полному member proof. Завершение pipeline не означает достижения превосходства.

Первичные доказательства: `evaluation/quality-report.json`, `quality-episodes.json`, `recovery/verified-members.json`, `trace-snapshot-all.json` (480/480,80 matched conditions). Before — веса после первого80-battle PPO, after — после всех400 собственных боёв. FireBC — исходные frozen weights, а не его регрессировавший PPO вариант.

## Результаты полной оценки

| Вариант | Победы /80 | Смерти | Средний урон монстрам | Средний полученный урон |
| --- | ---: | ---: | ---: | ---: |
| Attention128 seed07 before | 46 | 33 | 67,7 | 50,0 |
| Attention128 seed07 after | 48 | 29 | 67,4 | 47,2 |
| Attention128 seed08 before | 45 | 35 | 69,4 | 55,0 |
| Attention128 seed08 after | 46 | 30 | 67,5 | 51,6 |
| Исходный FireBC | 49 | 30 | 61,7 | 46,8 |
| Rules | 69 | 8 | 72,5 | 14,8 |

Парно seed07 приобрёл9 побед и потерял7, seed08 приобрёл8 и потерял7. Смертей стало меньше на4/5, но средний нанесённый урон не вырос. Это небольшое изменение результата, не устойчивое доказательство превосходства. Контрольные outcomes также варьируют между повторными сериями; reused validation не является независимым тестом.

## Прицел и движение

| Вариант | Средняя yaw ошибка при атаке,° | Атаки с yaw ошибкой>20° | Ground stall proxy |
| --- | ---: | ---: | ---: |
| Seed07 before | 17,65 | 28,96% | 1,19% |
| Seed07 after | 19,30 | 31,46% | 0,63% |
| Seed08 before | 31,65 | 58,48% | 5,93% |
| Seed08 after | 28,31 | 58,06% | 10,31% |
| FireBC | 24,52 | 37,60% | 0,58% |
| Rules | 0,07 | 0,00% | 0,00% |

Yaw metric — минимальный горизонтальный угол до любого clear observed enemy, усреднённый по эпизодам. Он не измеряет точность попаданий и не учитывает pitch, упреждение или recoil. Ground stall — малая фактическая дистанция между grounded movement кадрами, диагностический proxy; он не доказывает причину остановки. Guard counters относятся ко всему capture и не равны числу физических столкновений или уникальных кадров.

У seed07 yaw ухудшился, хотя побед стало больше; у seed08 средний yaw улучшился, но доля больших ошибок почти не изменилась и stall proxy вырос. Поэтому простого продолжения того же loss недостаточно для уверенного закрытия разрыва. Следующий эксперимент проверяет связь прицела и движения на существующих BC branches.

## Следующий запущенный эксперимент

`workspace/artifacts/coupling-eval-v1-20261009`:400 paired battles, FireBC + joint aim/movement без coupling + coupling weights1/16 + rules. Все three branches от одного FireBC parent,150 CUDA BC epochs, одинаковые data/config кроме world-input weight. Trainer hashes различаются и сохранены; это сравнение исторических sealed branches, а не новая серия обучения. Предыдущий joint-result30/80 на validation28 не переносится автоматически на текущие validation24 условия.

Новый CUDA audit подтвердил equality actor/critic/std с branch checkpoints, неизменность encoder/attention/value/fire/pose/weapon/std параметров относительно FireBC и явный BC reset actor optimizer; inherited PPO counters29/279 не выдаются за150 новых PPO updates. Existing native pool16/×2 запущен. Качество coupling ещё не установлено. После результата выбирается дальнейшая коррекция labels/loss/visitation; размер сети сейчас не увеличивается.

## Победы по семействам

| Семейство | Seed07 до | Seed07 после | Seed08 до | Seed08 после | FireBC | Rules |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| parasite-blaster-generated | 3/4 | 2/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| parasite-gunner-blaster-generated | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| parasite-machinegun-recoil | 3/4 | 4/4 | 2/4 | 2/4 | 4/4 | 4/4 |
| parasite-gunner-machinegun-recoil | 2/4 | 2/4 | 1/4 | 1/4 | 1/4 | 0/4 |
| campaign-base1-site-01-blaster | 2/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| campaign-base1-site-01-machinegun | 3/4 | 2/4 | 1/4 | 2/4 | 3/4 | 4/4 |
| campaign-base1-site-02-blaster | 4/4 | 4/4 | 3/4 | 3/4 | 3/4 | 4/4 |
| campaign-base1-site-02-machinegun | 4/4 | 4/4 | 3/4 | 3/4 | 3/4 | 4/4 |
| campaign-base1-site-03-blaster | 3/4 | 4/4 | 2/4 | 0/4 | 1/4 | 4/4 |
| campaign-base1-site-03-machinegun | 0/4 | 0/4 | 1/4 | 3/4 | 0/4 | 3/4 |
| campaign-base1-site-04-blaster | 2/4 | 1/4 | 3/4 | 1/4 | 1/4 | 4/4 |
| campaign-base1-site-04-machinegun | 2/4 | 1/4 | 0/4 | 1/4 | 0/4 | 4/4 |
| campaign-base2-site-01-blaster | 4/4 | 4/4 | 3/4 | 3/4 | 4/4 | 4/4 |
| campaign-base2-site-01-machinegun | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| campaign-base2-site-02-blaster | 2/4 | 3/4 | 0/4 | 0/4 | 0/4 | 4/4 |
| campaign-base2-site-02-machinegun | 1/4 | 0/4 | 1/4 | 0/4 | 1/4 | 4/4 |
| campaign-base2-site-03-blaster | 3/4 | 3/4 | 3/4 | 4/4 | 4/4 | 4/4 |
| campaign-base2-site-03-machinegun | 0/4 | 0/4 | 2/4 | 3/4 | 4/4 | 4/4 |
| campaign-base2-site-04-blaster | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 | 3/4 |
| campaign-base2-site-04-machinegun | 1/4 | 2/4 | 4/4 | 4/4 | 4/4 | 3/4 |

## Повторяемость контрольных исходов

| Контроль с теми же весами | Предыдущие победы | Текущие | Новые | Потерянные |
| --- | ---: | ---: | ---: | ---: |
| attention128-seed-20261007-before | 46 | 46 | 0 | 0 |
| attention128-seed-20261008-before | 45 | 45 | 0 | 0 |
| firebc-baseline | 50 | 49 | 0 | 1 |
| rules-baseline | 68 | 69 | 2 | 1 |

Это повторные validation условия, не новые независимые эпизоды. Изменение исходов controls ограничивает интерпретацию небольшого прироста кандидатов.
