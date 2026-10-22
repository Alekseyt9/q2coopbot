# Условное решение об огне: резервный опыт, 22.10.2026

Статус: [conditional-fire core](combat_conditional_fire_prototype_20261022.md) подключён к Go runtime, CUDA likelihood, recurrent PPO, optimizer и retention. [Zero checkpoint migration](combat_conditional_fire_migration_20261022.md) проверена на 4205 TRAIN rows; 8 CUDA tests passed. После исправления supervisor fresh v2 comparison8native/8CUDA прошёл semantic и strict full-trace gates:612 provider frames точны. [Первый fit](combat_conditional_fire_training_20261022.md) закрыт:32 TRAIN боя/5058rows,full/fire-only по10CUDA actor steps;fresh32-case evaluation закрыта: start/full/fire-only по3/8, rules4/8; promotion=false. Четырёхраундовая [кривая обучения](combat_ppo_learning_curve_20261022.md) завершена без прироста: 10/24 для весов 0/2/4, у rules 13/24. Игровой default не менялся.

## Основание

В закрытом40-case feedback probe full3/8→4/8 побед, но native MG live damage126/199→118/237, targetless solid shots28→47, крупные yaw команды6.58→10.38% provider frames. Полная победность и эффективность огня могут двигаться в разные стороны.

`internal/policy/ppo.go` выбирает attack по `x[4]`; `Review` считает Bernoulli likelihood по тому же logit. `scripts/combat_precision_head.py` передаёт в общий likelihood `raw[:,4:20]`, `combat_weapon_head.py` использует `Bernoulli(logits=raw[:,4])`. Поле fire может учитывать наблюдаемую сцену и историю через encoder/attention, **но не текущие фактически выпавшие target/mode/move/aim/vertical/weapon**. В частности, выбранная сейчас категория «нет цели» не меняет его вероятность сама по себе. Это ограничение факторизации, не доказанная причина каждого выстрела в стену.

## Предлагаемый контракт

Сохранить все выходы81и feature-v10. Добавить optional zero-output learned residual к fire logit, принимающий наблюдения и полный уже выбранный action context: target slot с маской/наблюдаемыми свойствами, mode, actual sampled movement/aim, vertical, weapon request и observed equipped weapon/phase. Ни исход физического выстрела, ни будущая позиция, ни reward не входят в runtime.

Новая факторизация: `P(other actions | observation) × P(attack | observation, actual other actions)`. Не ставить постоянный запрет стрельбы без цели: soft learned Bernoulli остаётся решением модели, включая свободное направление/прочие объекты.

Для сохранения zero compatibility Go должен вытянуть uniform fire draw в прежнем месте RNG-потока, но применить его после окончательной доводки aim/vertical и weapon sample. Нулевая ветка должна дать точно те же commands, samples и продолжение RNG, включая следующий кадр. Deterministic path использует тот же окончательный logit без RNG.

`Review`, CUDA replay, PPO/BC likelihood, entropy и sequence retention должны использовать один и тот же actual action context. Нельзя добавить ветку только в sampler или оставить старый unconditional Bernoulli в replay. Action context при PPO считается фиксированным behavior sample; через него нельзя протянуть лишний reparameterization gradient в другие heads.

Для retention потребуется явная новая версия expectation: условный fire KL на фиксированных actual behavior target/mode/aim/vertical/weapon, с frozen teacher и полной causal history. Остальные categorical/aim terms сохраняют прежнюю семантику. Это Monte Carlo expectation по behavior action contexts, а не точное аналитическое интегрирование новой continuous conditional policy. Не называть его прежним exact unconditional fire KL.

## Проверки до сравнения качества

1. CUDA zero/nonzero migration: старые actor/critic/std/Adam/RNG/consumed history сохранены; новая optimizer group пустая; old-policy log probabilities совпадают в zero case.
2. Nonzero probe при неизменном observation/raw меняет fire logit при изменении actual target/aim, остальные outputs не меняет. Invalid/missing context отвергается явно. Защищённые ветки и их Adam проверяются отдельно.
3. Native source/zero paired compatibility, включая vertical-conditioned final aim, deterministic/stochastic sampling, следующий RNG кадр и replay. Go NN numerical tests на CPU не запускать; CUDA для численных проверок, реальный Go runtime для native compatibility.
4. Только после этого fresh own-policy TRAIN и отдельное matched сравнение с исходным вариантом. Награду в первом опыте не менять одновременно с факторизацией. Endpoint выбирать по заранее заданному бюджету, не по удачной записи.

Супервизия по actual native shots может быть отдельным опытом, но отсутствие выстрела не даёт отрицательную hit-label. Классификатор вероятности попадания сам по себе не задаёт оптимальную вероятность стрелять; не приравнивать их без отдельной проверки задачи.
