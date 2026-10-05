# Данные каждого врага и состав видимой группы

06.10.2026. По уточнению пользователя learned combat получает отдельные параметры каждого наблюдаемого монстра и состав группы. Реализация в Go: `internal/quake/decode.go`, `internal/policy/contract.go`, `typed_features.go`, `history.go`. Python используется для offline PPO и миграции весов.

Optional observation fields: `observed_model`, `observed_skin`, `observed_animation_frame`, `observed_angles`; composition `visible_monster_composition` содержит пары type/count. Позиция относительно бота, расстояние, observed velocity, motion direction, bbox solid, clear-shot и track уже присутствовали. UDP skin теперь декодируется для всех8/16/32-bit форм и сохраняется при delta omission; facing берётся из entity angles в градусах. Скорость и направление движения вычисляются из двух последовательных видимых кадров с компенсацией движения бота и game time10Hz, независимо от x2 wall time. Gap/occlusion/model replacement обрывают motion track. Неизвестная скорость отличается от наблюдаемой нулевой.

Восемь ближайших видимых врагов имеют индивидуальные slots, сортируемые по расстоянию. Состав считается по всем доступным clear-shot enemies до ограничения slots; при >8 сохраняются counts и overflow mask. Это группа, наблюдаемая клиентом в рамках decoder range/PVS/BSP visibility, не все монстры уровня и не скрытые позиции за стенами. Hidden HP, server monster intent, seed, identity не поступают в feature vector. Состояния выстрелов/ближних предметов/BSP geometry из предыдущего контракта сохранены.

Feature version `combat_features_v4`: unchanged466-value v3 prefix +8×40 entity values +24 group values = **810**. Actor/value64→64, три affine layers, ReLU, actor8 outputs/value1; вместе112717 параметров включая4 log_std. GRU/attention пока не реализованы.

| Новые значения каждого slot | Число | Представление |
|---|---:|---|
|Тип|21|one-hot20 известных категорий +unknown|
|Движение|6|velocity mask, speed/400, direction mask, unit direction3 в yaw-local coordinates|
|Facing|5|angles mask, sin/cos relative yaw, sin/cos pitch|
|Внешнее состояние|4|skin mask +skin/8, animation mask +frame/512|
|Габариты|4|valid bbox mask, radius/bottom/top, scale64|

Group24: full-composition-known mask,21 counts/8, total/8, overflow mask. При чтении старого observation без composition используются только counts сохранённых slots и known=0. В новом observation пустой известный список отличается от неизвестного. Scalars bounded[-4,4], nonfinite/invalid counts/duplicate categories rejected. Position и vector velocity остались в прежнем prefix. Movement direction и facing — разные признаки; animation frame не считается скрытым состоянием AI.

Фиксированный порядок taxonomy: berserk, boss1, boss2, brain, bitch, flipper, float, flyer, gladiatr, gunner, hover, infantry, insane, medic, mutant, parasite, soldier, tank, jorg, makron, unknown. Full model path различает Jorg `boss3/jorg/tris.md2` и Makron `boss3/rider/tris.md2`, хотя legacy class у обоих monster_boss3. Неизвестный model остаётся unknown. Legacy class aliases принимаются при отсутствии model. Soldier/Tank variants не объявлены отдельными доказанными классами: наблюдаемые skin/model/animation доступны сети, но скрытое native classname/HP не раскрывается.

Миграция `scripts/fork_combat_features.py`: v3→v4 добавляет344 нулевых input weights для actor/value. Сохранены objective/std/RNG/consumed history/counters; оба Adam состояния явно сброшены из-за изменения shapes. Parent — fixed maneuver eighth update. Fork root `workspace/artifacts/combat-entity-types-v4-fork-20261006`, weights SHA `18f38979ec9028d3702662171b3bbb42bc99b16fa90d585ef437c1b45612d40f`. На1165 parent rollout rows с произвольными ненулевыми новыми inputs max actor/value output difference=0. Старые feature versions1–3 поддерживаются.

Live integration: existing harness, **4 независимых server/client instances x2**, seeds16100–16103, synchronous Mixed Parasite+Gunner, Blaster,300 game frames. Root `workspace/artifacts/combat-entity-types-v4-mixed-20261006`. Capture/provenance/dispatch valid4/4. В clear observations одновременно оба типа присутствовали223/152/209/265 frames; параметры facing присутствовали у всех1862 enemy records, observed velocity у1813. Проверка охватывает также более поздние жизни; PPO использует только initial first-life confirmed segment. Mixed fixture не поддерживает fixed release100/HP60, поэтому это отдельная integration condition, не paired продолжение одиночного curriculum.

Go q2ppo-data повторно проверил native proof и экспортировал353 fresh first-life rows,810 inputs. Первый PPO update:10 actor steps,40 critic steps, final KL0.00942; replay logprob error5.25e-5/value2.39e-7. CPU3.30ms/step быстрее CUDA5.64ms. Новые input weights действительно обучаются:1366 actor/958 value appended weights ненулевые, включая parasite/gunner one-hot и group counts. Updated weights SHA `1a3c07712f93d7e1980a99bba15d1fb39862cca470535b9138d30b67514d9e44`; checkpoint cumulative46 updates/456 actor steps. Это доказательство прохождения данных и градиентов, **не доказательство выученной тактики для разных групп**. До обновления бот погиб во всех четырёх mixed first lives; отдельно held-out eval обновлённого checkpoint ещё не выполнен.

Проверки: `go test ./...`,20 Python unittest checks; tests cover taxonomy, unchanged prefix,810 width, unknown/zero masks, yaw-local direction, ID exclusion, group>8/occlusion/nearest slots, pointer clone independence, nonfinite/duplicate rejection, UDP skin width/alignment/truncation/delta retention, zero-weight fork parity. Пользовательский live runtime сохранён; experimental models не назначались для live. Следующий этап — разнообразные состава/расстановки и отдельная отложенная оценка без подбора checkpoint по eval, включая добивание и сохранение навыка одиночного боя.

Продолжение того же дня: [Mixed PPO/held-out evaluation](learned_combat_mixed_ppo_v4.md). Первое обновление проверено на seeds16200–16203: deaths4→4/kills0→0. Ещё4 updates/1477 fresh rows, fixed final eval16400–16403: deaths4→4/kills0→0, outgoing120→80. Новые признаки работают, но успешная тактика для группы пока не выучена.
