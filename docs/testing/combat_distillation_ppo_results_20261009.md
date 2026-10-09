# Own-policy PPO после исправленной дистилляции, 2026-10-09

Полная оценка завершена: **560/560 native battles**, 0 ошибок, source unchanged, успешная агрегация. Root `workspace/artifacts/distill-ppo-eval-v1-20261009`; supervisor terminal, `evaluation-continuation.json` подтверждает SHA итогового quality report. Это reused validation offset24, не независимый final test.

Три модели обучены только на CUDA, каждой выделены свои 80 fresh train battles offset64, общие 20 семейств и условия между вариантами. Всего 240/240 valid training captures. Fresh optimizers, один PPO update/10 accepted actor steps на модель, retention/bank loss отключён. Eligible transitions: FireBC 6050, Attention128 seeds07/08 6635/7388. Исторический опыт различается: FireBC — ранее обученный учитель, Attention128 — его новые дистиллированные priors; равен только добавочный бюджет. Checkpoint/weights/report completion seals всех трёх проверены.

| Вариант | Победы / 80 | Смерти | Средний урон монстрам | Средний полученный урон |
| --- | ---: | ---: | ---: | ---: |
| Исходный FireBC | 50 | 29 | 63,0 | 45,7 |
| FireBC + PPO | 46 | 32 | 61,4 | 48,9 |
| Attention128 seed07 до PPO | 43 | 36 | 63,6 | 48,3 |
| Attention128 seed07 + PPO | 46 | 33 | 67,7 | 49,9 |
| Attention128 seed08 до PPO | 41 | 37 | 62,9 | 58,5 |
| Attention128 seed08 + PPO | 45 | 35 | 69,4 | 54,9 |
| Rules | 68 | 8 | 72,3 | 14,2 |

Парные изменения: seed07 +3 (12 новых/9 потерянных побед), seed08 +4 (8/4); FireBC −4 (4/8). Attention128 улучшили собственные старты, но остались ниже исходного FireBC и rules. Новые weights **не продвинуты**. Отрицательная FireBC ветка не заменяет прежний checkpoint.

## Трассы и воспроизводимость

`trace-snapshot-all.json`: полное покрытие всех 80 условий × 7 вариантов, SHA закрытых traces/server/report. Средняя эпизодная минимальная горизонтальная ошибка при огне относительно любого clear observed enemy:

| Модель | Before → after yaw | Доля clear firing frames с yaw >20° до → после | Ground stall proxy до → после |
| --- | ---: | ---: | ---: |
| FireBC | 24,50° → 23,84° | 35,36% → 33,85% | 0,58% → 1,29% |
| Attention128 seed07 | 19,21° → 17,66° | 29,63% → 28,93% | 1,09% → 1,19% |
| Attention128 seed08 | 32,73° → 31,65° | 62,62% → 58,48% | 3,33% → 5,94% |

Yaw не является hit accuracy: pitch, recoil и lead не учитываются. Малое улучшение этой геометрической метрики не объясняет результат целиком. Guard events остаются частыми; низкий stall proxy не подтверждает самостоятельное избегание стен. Полный quality report содержит capture-level guard/provider/frame-gap/latency diagnostics. Frame gaps всех вариантов 0. Lockstep ждёт команды, поэтому выбор Attention128 около33–34ms влияет на throughput, но не доказывает причину поражений в обычном realtime.

Повторные before-условия относительно предыдущей 400-battle серии: победы FireBC и обоих Attention128 совпали во всех 80 условиях каждого варианта. У rules изменились 4 исхода (2 gained/2 lost), общий счёт сохранился 68/80. Поэтому полная детерминированность native/rules поведения не утверждается; это не независимые новые условия и не final-test подтверждение.

Геометрические слабости сохраняются: base1 site03 machinegun Attention128 seed07 1→0/4; base2 site03 machinegun 1→0/4. Одновременно seed07 улучшился на base1 site04 machinegun 0→2/4 и parasite+gunner machinegun 0→2/4. Seed08 base1 site03 blaster 0→2/4, base1 site04 blaster 1→3/4. Положительные и отрицательные изменения зависят от семейства; среднее число побед скрывает эту неоднородность.

## Следующее действие

Продолжить обе Attention128 с их запечатанных PPO checkpoints на четырёх свежих own-policy раундах по80 battles на модель (640 дополнительных боёв, train offsets72/80/88/96), сохраняя optimizer state и счётчики consumed rollout. Существующий processor перед этим нужно расширить для явной checkpoint continuation; нельзя выдавать новый fresh-optimizer запуск за продолжение. Пул16/x2, только CUDA, весь реестр20семейств. Исходный FireBC50/80 остаётся замороженной baseline, регрессировавший FireBC+PPO не является новым основным checkpoint. После раундов — полная before/after + FireBC/rules validation; независимый final test только после выбора кандидата.

Одного80-battle PPO update недостаточно для заключения об архитектуре. Если более длинная серия не закрывает разрыв, следующий эксперимент должен проверять согласование aim/movement, recoil/target selection и вмешательства guards на проблемных семействах; увеличение сети само по себе пока не обосновано. Уже обученные coupling-loss варианты требуют отдельной native оценки, их превосходство не установлено.

Связанные данные и история: [исправление переноса](combat_distillation_repair_20261009.md), [первый архитектурный эксперимент](combat_architecture_results_20261009.md).
