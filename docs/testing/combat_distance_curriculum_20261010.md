# Дальность боя: подготовка расширения реестра

Дальность уже входит в observation каждого противника и напрямую в shared spatial fine-aim branch. Разбор завершённой400-battle оценки выявил отсутствие устойчивого far покрытия: Before/Instant не имеют измеренных firing frames свыше512 units, Postmove имеет лишь12. Это геометрия выбранной цели, а не фактические bullet hits; нельзя считать такой набор проверкой дальнего боя.

Подготовлен отдельный модельно независимый draft registry: `workspace/artifacts/combat-far-loadouts-draft-v1-20261010/registry/index.json`. Канонический реестр не изменён, действующая480-battle оценка продолжает работать с прежними hash bindings.

| Рецепт | Карта / противник | Loadout |
| --- | --- | --- |
| campaign-base1-site-02-far-blaster | base1 / Infantry | Blaster |
| campaign-base1-site-02-far-machinegun | base1 / Infantry | Machinegun |
| campaign-base2-site-03-far-blaster | base2 / Soldier | Blaster |
| campaign-base2-site-03-far-machinegun | base2 / Soldier | Machinegun |

`prepare_combat_far_registry.py` перебрал105 кандидатов на восьми прежних campaign sites, radius640/768 и восьми направлениях. Два старта прошли существующий Go BSP planner: grounded hull, teleport headroom, inline start reservations, overlap и primary visibility. `assemble_combat_far_registry.py` собрал четыре рецепта и заново сгенерировал и verify-plan проверил32 условия: четыре train и четыре validation seeds на рецепт. Начальные расстояния636.64..648.00 units. Plan hashes, parent recipe/registry hashes, planner binary hash и source provenance сохранены в report.json. Ни модель, ни inference сети в этих проверках не исполнялись. Test/confirmation условия не генерировались.

Это статическая пригодность, не native acceptance: не проверены settling, состояние дверей/движущихся объектов, первая наблюдаемая цель и валидный live capture. `site.wall_distances` унаследованы как данные старого положения игрока; они явно не объявляются rays из нового старта. Перед канонической регистрацией эту метаинформацию нужно пересчитать или явно разделить в версии схемы.

Последовательность следующих действий:

1. Дождаться полного закрытия480-battle оценки и проверить source/member/native seals; выбрать BC/PPO candidate по paired outcomes, без automatic promotion.
2. Через прежний16-slot refill pool выполнить native smoke дальних рецептов для rules и выбранной модели, ×2, каждый seed самостоятельный процесс. Проверить reset pose/loadout/visibility и завершение цели.
3. Сохранить failures; непригодные дальние условия исправить по native evidence. Обновить wall rays у нового player start после окончания активного fingerprint-bound пула.
4. Зарегистрировать прошедшие условия с near/medium/far curriculum, обеими loadouts и составом монстров; дальний subset нужен дополнительно к прежним геометриям. Два far sites не доказывают универсальность на всех типах/составах.
5. Собрать свежие stochastic own-policy train seeds и выполнить CUDA PPO continuation из выбранного PPO checkpoint. Сравнить до/после с rules по исходам боёв, native hits и distance strata; наблюдаемые дистанции во время боя могут отличаться от начальных.

Подготовка draft не является выполнением этих пяти шагов. Новое обучение на дальних боях ещё не запускалось.

## Поставленная native оценка

`scripts/run_combat_far_evaluation.py` подготовил шесть frozen evaluation plans по16 боёв, всего96: четыре far рецепта ×четыре validation seeds для Instant/Postmove до/после PPO, FireBC и rules. Root `workspace/artifacts/combat-far-eval-v1-20261010/`, process receipt `workspace/build/combat-far-eval-v1-20261010-process.json`, PID25884 при запуске. Подготовка прошла; процесс подтверждён живым, stage `waiting_for_previous_evaluation`, удерживает OS handle PID25796. Серверы дальнего прогона ещё не запущены.

После завершения предыдущего процесса требуется complete progress, quality hash и verified-members protocol seal. Ошибка upstream завершает очередь failed с сохранением артефактов. Затем прежний16-slot refill pool ×2 запускает96 отдельных native runs, source/model/binary proofs и отчёты quality/strata/selected-target/native-blaster/modes/distance. Предел16 общий для последовательных пулов; параллельный второй пул не запускается. Это evaluation новой геометрии на validation conditions, без обучения, final-test использования и automatic promotion. Те же шесть вариантов проходят дальние условия независимо от исхода прежнего сравнения, чтобы избежать выбора по его результату.
# Native evaluation completed, 2026-10-10

The queued 96-battle evaluation is now sealed complete. All 96 members passed
the strict capture verifier. The four draft far recipes have native validation
captures; canonical registry promotion remains pending because the stored wall
rays still describe the parent location. Training and final-test conditions
were not exercised in this validation cohort.

| Frozen variant | Wins / 16 | Deaths | Mean received health damage |
| --- | ---: | ---: | ---: |
| Instant BC | 13 | 1 | 33.00 |
| Instant PPO update1 | 13 | 1 | 34.13 |
| Postmove BC | 12 | 4 | 34.69 |
| Postmove PPO update1 | 13 | 3 | 38.25 |
| FireBC | 13 | 3 | 44.50 |
| Rules | 14 | 0 | 0.56 |

Both spatial branches win all eight Blaster cases. Machinegun remains uneven
across the two sites. PPO shows no overall superiority over rules and the
received damage difference remains large. Applied selected-target firing-ray
error beyond 512 units is 6.64 to 7.23 degrees for Instant before/after PPO,
and 10.30 to 10.01 degrees for Postmove. Rules/FireBC do not expose explicit
selected-target identity, so that metric is unavailable for those controls.
These ray errors are not measured hit probabilities and do not include muzzle,
projectile lead or recoil correction.

Quality hash: `e2d9d0cf168af9827d5ba9c6c9f45be2d92512c79bfe079da8a7f784a150407a`.
Artifacts: `workspace/artifacts/combat-far-eval-v1-20261010`.
