# Доставка результатов learner во время прогона

Дата: 05.10.2026. `combat_feedback_v1` доставляет проверенный reset, переходы, награды и окончание первой жизни внешнему процессу. Это асинхронный поток результатов существующего харнеса; обучение весов ещё не реализовано.

## Устройство

Новый Go-инструмент `q2learning-relay` читает только дописанные строки client capture и native server log. Он сохраняет неполную строку до следующего чтения и отвергает усечение/замену файла. Незакрытый native pulse не публикуется. Применяется тот же Assembler, парсер фаз, проверка точного usercmd, DamageJoiner и reward v1, что в offline-экспорте.

Для каждого эпизода relay открывает отдельную session канала `combat_bridge_v1`, с теми же worker/episode/seed и явно ожидаемой версией peer. Request имеет `kind=feedback` и JSON-поле `feedback`. Ответ обязан подтвердить `feedback_ack=true` и совпасть по session/request/policy version. Подтверждённое событие записывается в `feedback.jsonl`; отсутствие ACK или чужой ответ завершает relay с ошибкой. Автоматических повторов доставки нет, поэтому потерянный ACK не считается успешной доставкой и не вызывает скрытую повторную метку.

| Событие | Содержание |
| --- | --- |
| `reset` | Первый пригодный observation, проверка полей fixture, свежего инвентаря и native barrier с seed. Это подтверждение состоявшегося старта, не команда reset полного сервера. |
| `step` | Полный Step: исходное наблюдение, предложенное и применённое действия, отправленный usercmd, exact dispatch/native proof, next observation, owner, interventions, terminal/truncated. Отдельно Reward и фазовые ServerOutcome. |

Серверные эффекты являются обучающими метками. UDP-клиент и его боевой observation не читают этот канал. Внешний процесс может использовать метки для обучения; диагностический peer только подтверждает и журналирует их. Сессии решений и feedback независимы, связываются по episode/seed и identity игрового шага. Reset поступает после проверки первого перехода, а не обязательно до первого решения. Решения не ждут доставки reward: это ещё не блокирующий Gym `Reset/Step` API.

Публикация заканчивается на первом terminal или truncated. Первая смерть приходит как terminal с native-доказательством; респавны не продолжают этот учебный эпизод. Контрольный handoff, разрыв/усечение и лимит кадров приводят к truncated. При известном лимите relay закрывает хвост сразу по `relative_frame`; worker completion marker служит резервным сигналом. Неполный хвост сохраняет `reward.score=null`; смерть не подставляется вместо тайм-аута. Full-world reset equivalence и reward за победу по-прежнему не доказаны.

## Запуск и аудит

```powershell
./scripts/run_learned_combat_bridge.ps1 -Workers 4 -EpisodesPerWorker 1 -Timescale 2 -GameFrames 300 -Seed 12000 -Port 33004 -BridgePort 33008 -Mixed -HealthKit -Feedback
```

Основной baseline runner строит один relay binary и запускает по одному relay на независимый эпизод перед игровым клиентом. Отчёт relay проверяется отдельно от gameplay acceptance. В `finally` runner завершает собственные процессы. `audit_learning_feedback.ps1` сравнивает каждый доставленный Step/Reward/ServerOutcome и initial reset с offline-экспортом, включая global damage indexes. Проверка включена в capture acceptance при `-Feedback`; результат сохраняется как `feedback-worker-*/audit.json`.

## Проверки

Unit-тесты: неполные строки, только новые байты, усечённый лог, недописанный pulse, немедленный урон ClientThink, exact dispatch, global damage indexes, повторный end/перекрытие фаз, обязательность release в offline-парсере и ACK с неверной identity. `go test ./...` прошёл.

Первый live-прогон: `workspace/artifacts/learner-bridge-20261005-183931-177/`, четыре независимых seed 12000–12003, x2/300 игровых кадров. Learner получил четыре reset и 198 step-событий (50/50/41/57), каждое с доступной reward. Все четыре завершились подтверждённой первой смертью; после смерти новые метки не публиковались. Всего 202 feedback-сообщения и 997 внешних решений. Все captures и native synchronous proof прошли; доставленные Step/Reward/Effects совпали с offline-экспортом без расхождений. Последнее terminal-событие каждого relay доставлено раньше последнего client capture: результаты поступали во время прогона. Peer диагностический, gameplay acceptance — 0.

Короткий прогон: `workspace/artifacts/learner-bridge-20261005-184527-700/`, четыре отдельных seed 12100–12103, x2/60 кадров. В каждом relay — 34 step-события, из них 33 с наградой; последний хвост без next имеет reward null и `truncated=true`, reason `game_frame_limit`. Terminal — false. Все четыре автоматических аудита и native synchronous proof прошли, расхождений с offline-данными — 0. Всего 140 feedback-сообщений, включая четыре reset, и 136 внешних решений.

Полный повтор с автоматическим аудитом: `workspace/artifacts/learner-bridge-20261005-184627-766/`, четыре worker, по два холодных эпизода, независимые seed 12200–12207, x2/300 кадров. Доставлены восемь reset и 455 step-событий (44/58/50/61/63/71/59/49), все с наградой; каждый эпизод закончен первой смертью. Всего 463 feedback-сообщения и 1975 внешних решений. Все восемь capture, native proof и feedback audit прошли; mismatch — 0. Скорость 19.51–19.55 игровых кадров/реальную секунду, batch 50.98 секунды вместе со стартами и аудитом. Gameplay acceptance — 0: транспорт и метки проверены, политика остаётся диагностической.

Следующая работа: отбор демонстраций и условий train/validation/test, поддержка безопасного управления в воздухе, затем реальные BC/PPO. Cold reset выполняет существующий runner; внешний learner пока получает подтверждение старта и асинхронные результаты.
