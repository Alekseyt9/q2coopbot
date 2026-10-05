# Curriculum Parasite 60 HP с canonical test start

06.10.2026. Цель — получить опыт более длительного сопровождения и добивания, которого почти не было в HP20 width pilot. Архитектура остаётся MLP 386→64→64, actor 8/value 1; GRU/attention не смешиваются с изменением fixture.

Root: `workspace/artifacts/combat-ppo-curriculum-hp60-v1-20261006`. Start weights/checkpoint — mainline HP20 fourth update SHA256 `9920296b170acaa1f184f61ea900d6579af88b4999978ce1e76d9bbbd76cb01b`; более поздние width candidate checkpoints не выбирались по eval. Reward v2 и PPO config сохраняются, optimizer/RNG/consumed-rollout history resume без дополнительного reset. Старый опыт не используется повторно.

Замороженный протокол: четыре fresh batch × четыре инстанса, x2, отдельный seed каждого эпизода 15100–15115; 300 **post-barrier** игровых кадров; release game_frame 100, Blaster READY gunframe 9, post-frame seed reset. HP60 применяется один раз на release; никаких изменений здоровья/AI/aim во время боя. HP и native receipts не поступают в policy features. Full world reset не доказан: [диагностика повторений](learned_combat_reset_clock_v1.md) сохраняет отрицательный seed 14902.

Fixed fourth update, затем четыре deterministic eval до/после на обычных 175 HP, seeds 15200–15203. Eval не поступает в trainer. Показатели — включённые PPO kill reward windows, actual outgoing/received health damage, first-life deaths/kills и diagnostics. Legacy rules-specific harness acceptance не используется как learned-policy приёмка. Native provenance, phase effects, command dispatch, reward SHA и Go/PyTorch replay остаются обязательными.

## Завершённый цикл

Четыре fresh batch: 897/913/576/1094 переходов, всего 3480. Принятые actor steps 10/10/10/8, KL 0.007004/0.005815/0.007030/0.009978. CPU быстрее CUDA по benchmark всех четырёх batch. Checkpoint содержит 25 накопленных updates и 248 actor steps. Во всех 24 эпизодах native provenance/capture complete/dispatch/seed подтверждены.

Один positive kill window самой политики: seed **15110**, step index **116**, observation server_frame **213** (примерно 11.5 игровых секунд после первого provider observation). Component monster_kill=+5, полный score=5.099. Owner=provider; reward score совпал с включённой строкой PPO rollout. Это более длительный успешный бой, чем прежний HP20 kill за три кадра, но один пример не доказывает освоение сопровождения цели. Аудит с SHA исходных rewards/steps/reports — `audit.json`.

| Seed | Урон монстру 175 HP до → после | Полученный урон до → после | Kills до → после |
| --- | ---: | ---: | ---: |
| 15200 | 30 → 20 | 9 → 27 | 0 → 0 |
| 15201 | 30 → 20 | 69 → 9 | 0 → 0 |
| 15202 | 30 → 20 | 73 → 11 | 0 → 0 |
| 15203 | 30 → 20 | 0 → 25 | 0 → 0 |

Итого исходящий урон 120→80, входящий 151→72, смертей 0→0. На полном HP добивание не освоено, автоматической боевой приёмки нет. Снижение incoming damage не является достаточным улучшением боя: запрос движения без фактического перемещения вырос 642→926 среди 1204 provider observations. Горизонтальная ошибка >15° при видимой атаке: 1024/1188→956/1180; metric не включает pitch и не является shot accuracy. По-прежнему нужны длительное удержание прицела и освобождение от остановок у геометрии.

Полная engine repeatability не доказана; четыре eval пары одного fixture не устанавливают статистическое преимущество или перенос на другие условия. Отличия post-barrier бюджета от прежних циклов описаны отдельно; сравнение 60 HP с 20 HP по одному cap не является одинаковым бюджетом боя.

Финальные веса `iteration-4/update/weights.json`, SHA256 `e85ff19b8000f4bb173b6ab81221f772769a7399efaf940c67780851bc405763`. Сохранён optimizer/RNG checkpoint, родитель не изменён. Go tests, native build и синтаксис PowerShell прошли; после завершения порты 33100–33103 освобождены, пользовательский live сохранён.
