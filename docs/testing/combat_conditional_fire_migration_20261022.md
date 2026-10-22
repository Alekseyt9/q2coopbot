# Conditional fire: optimizer, retention и checkpoint migration

22.10.2026. Нулевая миграция проверена на CUDA на 4205 реальных TRAIN rows с полными causal histories. Обучения новой модели пока не было; игровой default сохранён.

## Реализация

- `combat_weapon_state_residual.optimizer` добавляет четвёртую группу `conditional_fire` после base/std, weapon и motor. Старые parameter IDs и порядок групп сохраняются; отдельный LR по умолчанию 0.0003.
- `combat_conditional_fire.freeze_scope` позволяет обучать только новые 6369 параметров. Прежние actor/std и их Adam states восстанавливаются проекцией, включая fallback, который очищает momentum.
- `combat_sequence_retention.policy_kl` включает Bernoulli KL при фиксированном полном behavior action: target, mode, latent movement/aim, vertical и weapon. Это Monte Carlo expectation по записанному действию, не точное суммирование всех остальных действий. Контекст не пересэмплируется при backtracking; teacher branch заморожена. Остальные terms прежние.
- Контракт expectation: `fixed_behavior_complete_action_conditional_fire_v1`. Он записывается в sequence-retention metadata; смена objective требует существующего явного fork.
- Recurrent PPO принимает scopes `all` и `conditional-fire`, отдельный `--conditional-fire-lr`, требует resume и положительный train-only sequence retention. Likelihood, checkpoint export/resume и отчёт включают новую branch.
- Исторические feedforward PPO и conditional SAC нового формата пока не поддерживают и явно отказывают. Fire не ограничивается правилом «нет цели — не стреляй».

## Доказательства миграции

Команда:

```powershell
& F:/src/strat/.venv-gpu/Scripts/python.exe scripts/migrate_combat_conditional_fire_cuda.py --source workspace/artifacts/combat-motor-feedback-v1-20261022/round-2-full-update --data workspace/artifacts/combat-ppo-learning-curve-v1-20261022/round-1-full-corpus/processing-train/merged --out workspace/artifacts/combat-conditional-fire-zero-v1-20261022
```

Закрытый receipt: `workspace/artifacts/combat-conditional-fire-zero-v1-20261022/report.json`; weights/checkpoint/report закреплены SHA в `complete.json`.

- Нулевые raw outputs и joint likelihood точно совпали с исходными на всех 4205 rows.
- Максимальная ошибка CUDA/native исходной likelihood: 0.00011349 при прежнем допуске 0.003; memory error 0.00000954 при допуске 0.00002.
- Нулевой conditional KL равен 0; при ненулевой fire bias отличие от ожидаемого Bernoulli KL равно 0. Градиент новой ветки ненулевой.
- Прежние actor/value/std, Adam states/IDs, RNG, training config, consumed-rollout history и counters сохранены. Новая optimizer group пуста. Сериализованный checkpoint повторно загружен, optimizer state повторно восстановлен и сопоставлен.
- Старая sequence-retention metadata сохранена в migration receipt; активный objective сброшен для явного нового fork. Старые raw streams использованы только для численной проверки миграции; перед fit нужны свежие captures с текущими helpers.
- `scripts/test_combat_conditional_fire_cuda.py`: **8 tests, OK**. Добавлены fixed-context KL, teacher-gradient isolation, append-only optimizer groups и forced-fallback preservation старых Adam states.

Вся NN математика выполнена на CUDA. CPU deserialization сохраняет byte RNG tensors и не исполняет сеть.

## Native gate

`combat-conditional-fire-zero-compat-v1-20261022/evaluation`: source/fire-zero comparison закрыт на 4 fresh условиях, 8 native / 8 CUDA replay, seed offset752, base3/ware2, ordinary/recovery, max16, ×2. CUDA/native likelihood checks прошли. **Строгий full-trace audit не прошёл**: у ware2 recovery seed1252026752 source содержит один дополнительный кадр234 после достижения цели. Остальные три пары имеют идентичные canonical histories.

Отдельная диагностика `audit_combat_conditional_fire_native_parity.py --diagnose-post-goal-tail` подтвердила **598 общих provider frames**: canonical inputs/memory, samples и proposed/applied/sent commands точны, кроме объявленной model version и wall-clock packet-age jitter до2ms в одном допустимом freshness class. Это **диагностика общего префикса**, не замена проваленного full-trace gate. Актуальный receipt: `common-prefix-diagnostic-v2.json` рядом с evaluation.

Последний kill в проблемной паре одинаков: frame230; здоровье15 и native damage/kills совпадают. `goal_stop.observed_frame` различается:232/231. Единственный непарный frame234 находится после обеих terminal observations, observed enemies пусты. В `Wait-CombatGoalOrExit` receipt строится по последней прочитанной строке при опросе процесса каждые50ms; это объясняет зависимость observed boundary от момента опроса. Сами post-goal captures сохранены без изменений.

Последующее [исправление границы](combat_goal_boundary_repair_20261022.md) завершено: earliest observed terminal не зависит от poll timing. Fresh v2 comparison8native/8CUDA прошёл semantic episode audit **и отдельный strict full-trace audit**:612 provider frames exact,611/611CUDA rows. Старое v1 расхождение сохранено как исторический результат. [Первый matched fit](combat_conditional_fire_training_20261022.md) закрыт:32 TRAIN боя/5058rows,full/fire-only по10actor steps. Обаresume audits иshared-critic comparison прошли; независимое32-case сравнение закрыто: start/full/fire-only по3/8, rules4/8. Promotion=false; далее train-only reward-signal diagnosis.
