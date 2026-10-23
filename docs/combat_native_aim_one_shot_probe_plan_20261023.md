# One-frame probe для плотных native меток прицела

23.10.2026. Следующий этап после admission failure16-state набора. Это план реализации и технической приёмки, пока не работающий runtime контракт. Цель — снять record-high ограничение, которое допускает только28из379подходящих выстрелов.

## Выбранный подход

Добавить явно диагностическую metadata `diagnostic_joint_aim_probe` к существующей augmented stochastic policy. Поля: собственная version, один положительный absolute provider frame,25конечных logits для этого кадра. Допускается только на модели с постоянной baseline head (нулевые веса последнего слоя); до/после кадра head и её sampler прежние. Изменять actor/value/std/fire/predictor/RNG запрещено. Все25вероятностей положительны. Никаких native исходов или скрытой информации в выборе действия.

В Go `jointAimLogits` после обычной проверки eligibility в заданном кадре возвращает metadata logits; остальные frames работают как прежде. Gaussian/discrete draws не меняются; conditional fire по-прежнему видит окончательное направление. Существующая AimCandidateSample сохраняет source latent,category,LP,timing без изменения структуры. Это одно объявленное диагностическое вмешательство, не production mask или новая обученная политика.

Новый контракт требует явного CUDA понимания frame: `JointAim`/`probabilities` получают необязательный batch frame tensor для probe, native finalizer передаёт recorded frames и проверяет полный augmented LP. Обычный конструктор/тренеры probe-модель отвергают; только native diagnostic scorer разрешает её явно. Нельзя молча считать старую neural head вероятностями расписания или отправлять такие повторы в обычный PPO. Исходный v1 dataset и его split не переписываются.

Вначале оставить прежнюю сетку25направлений, чтобы не смешивать action versions. Если в поздних состояниях с большой отдачей все±5°варианты бесполезны, это отдельное основание проверить более широкий диапазон в новом контракте, не незаметно переопределять старые категории.

## Техническая приёмка

1. CUDA fork baseline S4: модель отличается только диагностической metadata, source checkpoint остаётся запечатанным; scope запрещает её продолжение прежним budget trainer.
2. Pure metadata/geometry проверки не исполняют Go NN offline. Все NN численные проверки CUDA; Go NN только в реальных играх.
3. Native smoke: TRAIN seed1172065112, ware2 site01 ordinary, frame110, baseline и один заранее выбранный ненулевой offset. Это намеренное повторение известного TRAIN случая; исход альтернативы заранее неизвестен. Новый planner/runtime бинарник собрать из текущего checkout, не использовать старый executable без поддержки metadata.
4. Проверить первый изменившийся кадр ровно110, одинаковый physical/features/memory/source draws prefix, head timing на вмешательстве и ожидаемую категорию. Факт выстрела/его исход взять из эксклюзивного native window. Последующие состояния могут расходиться. Сравнить baseline с прежним CUDA-verified capture отдельно.
5. После закрытого smoke собрать несколько заранее выбранных более поздних TRAIN моментов. Разделение development seeds1172165113,1244025112,1244125112,1252025113 сохраняется; их нельзя добавлять в обучение. Для расширенного fit предпочтительны свежие TRAIN seeds и до2состояний на каждый; include good baseline и реальные промахи при движении/отдаче.

Пул максимум16, ×2. Ни smoke, ни новые labels сами по себе не устанавливают качество боя. После расширения нужен новый fixed-budget fit protocol с сохранением прежних неудачных результатов; исходный 16-state опыт не объявляется прошедшим допуск задним числом.

## Точки реализации

- `internal/policy/ppo.go`: PPOFile metadata и путь Decide.
- `internal/policy/aim_candidates.go`: validation и live logits override.
- `scripts/combat_joint_aim_policy.py`: явное probe permission, frame-conditioned logits и density.
- `scripts/finalize_combat_cuda_rollout.py`: передача recorded frame и dedicated probe acceptance.
- Диагностический CUDA fork/пул/paired native auditor: переиспользовать existing queue и prefix/data proofs.

Старый `geometry_aim_explore_v1` изучен, но не выбран: его sampler/аудит жёстко привязаны к50%hash вмешательствам и не выполняют нынешний полный augmented CUDA replay. Расширение текущего joint контракта сохраняет уже проверенный source-Gaussian/final-fire/categorical путь. Contract metadata должна быть явно отклонена всеми обычными train entrypoints.
