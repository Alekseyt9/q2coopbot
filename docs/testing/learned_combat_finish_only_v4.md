# Finish-only curriculum v4

06.10.2026. Продолжение [group/range-band опыта](learned_combat_group_threat_v4.md).
General range-band не принят: ухудшает Mixed и увеличивает barrel suppression.

## Протокол до результата

Parent — group final `c5b04e538238ad6be39ee8334d282241fb7f2d51f105f1ba1f6f34d9ae00d808`,
checkpoint `1aea95a92c54ab4ef255525eb3fcc0fd3f19a8723bfa612832561e4d8d8b9811`.
53 PPO updates/524 actor steps. Новая supervised ветка, не новый PPO update.

Offline разбор прежнего eval18700 выявил: после Gunner kill Parasite остаётся
видимым на дистанции около378–379; бот упирается в стену, стреляет мимо и не
завершает бой. Eval служит диагностике, его states не входят в обучение.

`--finish-only` ставит метку только при ровно одном видимом typed Parasite
дальше320 units. Два видимых врага, Gunner, близкий Parasite и неизвестная
цель остаются под distillation parent. Единственная видимая цель не доказывает
смерть остальных врагов: скрытый server truth не используется при labels.
Используются прежние bbox aim, known BSP clearance/drop/barrels и projectile
velocity masks; локальный прогноз0.25s не является engine simulation.

Один fixed2000 epoch fit на прежних6691 verified group training states:
17800–17815,18100–18115,18600–18615. Только109 states отвечают дальней
sole-Parasite метке. Solo retention1720 states16800–16815, evaluation excluded.
Активные movement/aim labels имеют вес100; inactive states дополнительно
distill весом5, retention movement10, aim40, прочие outputs10.
Сохранить gates solo movement≤0.05, aim≤2°, attack≤0.05, vertical KL≤0.02;
новый gate для inactive training states movement≤0.05 и aim≤2°.
Это mean training-state gates, не гарантия сохранения runtime поведения.
Adam reset, critic output zero, std/RNG/history/counters сохраняются.
CPU/CUDA benchmark выбирает фактически быстрый backend, caches на F.

После gate один fixed candidate сравнивается с group parent на новых парных
Mixed19100–19103 и Solo19200–19203:4 independent instances x2,300 frames,
Blaster, synchronous, deterministic; solo175HP/release100, Mixed stock/release0.
Основной критерий — оба Mixed монстра убиты без смерти; отдельно native
damage attribution, guard interventions и solo provider kill/handoff.
Нет best checkpoint selection или обучения по eval. Live policy не меняется.

17 focused tests прошли, включая отказ от меток для двух угроз/Gunner/close
solo, unknown velocity и неизменность parent lineage. Артефакты:
`workspace/artifacts/combat-finish-only-v4-fork-20261006` и
`workspace/artifacts/combat-finish-only-v4-eval-20261006`.

## Fixed fit и проверка lineage

97 movement labels и109 aim labels;6582 inactive states под distillation.
GPU2.48ms/step против CPU13.95ms/step, выбран CUDA. Этот замер состоялся
до требования пользователя тренировать дальше только на GPU.
Movement MAE0.446/0.370→0.0636/0.0584; aim18.941/4.528→1.588/0.505°.
Inactive drift movement0.0254/0.0190, aim0.724/0.229°; solo movement
0.0208/0.0181, aim1.021/0.421°, attack0.000203, vertical KL2.26e-6.
Оба training-state gates пройдены. Это не доказательство runtime retention.

Weights SHA256 `2eaf8ee9c9d6ba18bbf957cc473a05004e9005da40d9a8008ea67781927a7ccf`,
checkpoint `a51cc9666a1c2e999cb866dd4932f8f6e93ce167d36e042443a45042fa64b84b`.
Независимый audit проверил source proofs, отсутствие eval seeds, tensor parity,
std/RNG/history,53/524 counters и reset optimizer/critic. Frozen trainer лежит
в fork root, поскольку позднее в текущих scripts изменён выбор устройства.

## Парный боевой результат

| Условия | Group parent | Finish-only |
|---|---:|---:|
| Mixed19100–19103: kills | 4 | 4 |
| Mixed: deaths | 0 | 2 |
| Mixed: оба монстра убиты без смерти | 0/4 | 2/4 |
| Mixed: incoming health damage | 21 | 327 |
| Mixed: outgoing health damage | 780 | 850 |
| Solo19200–19203: kills/deaths | 4/0 | 4/0 |
| Solo: incoming health damage | 119 | 146 |

На19101 и19103 новая policy убила Gunner и Parasite. Все четыре kill rows
принадлежат provider, reward kill component+5. Вторые kills frame248/285,
handoff249/286. У parent Gunner убит во всех четырёх, Parasite остаётся.
Новые полные победы — ограниченный сигнал на четырёх сидах, не общая приёмка.

На19100 новая policy получила33 grenade damage от Gunner и67 barrel damage
с attribution player; на19102 —100 barrel damage с attribution Gunner.
Native `MOD_BARREL=26`, `MOD_GRENADE=6` подтверждены в engine local.h.
Обе смерти связаны со взрывоопасными бочками. Aggregate incoming по классам:
Gunner139, Parasite121, player67. Guard barrel suppression0→19,
hull696→141, unsupported0→2. Applied shots1165→526, bbox error>15°
относительно любой доступной цели474/1165→3/526; это не hit accuracy.

Solo provider kill frame178→183, handoff179→184, hull268→284.
Нет смерти, но дополнительный входящий урон27 и замедление убийства5 frames
учтены как регрессия, несмотря на пройденный mean retention gate.
Все16 captures valid; paired fingerprints/native provenance/dispatch/seeds
проверены. Независимый reward audit max component error4.44e-16.
Live promotion нет: улучшение завершения сопровождается опасной регрессией
и проверено только на четырёх Mixed seeds. Checkpoints сохранены.

Следующий опыт: учитывать blast proximity наблюдаемых barrels в offline
movement labels и reward/diagnostics, не ослабляя runtime guard; сравнивать
полное завершение и barrel/self damage на новых парных seeds. Нужно отдельно
проверить перенос изменений в состояние двух угроз: mean training retention
не сохранил безопасность и траекторию всего боя. Fresh stochastic rollout
нужен перед PPO; evaluation нельзя повторно использовать для обучения.

## Новое требование: только GPU

Пользователь потребовал дальнейшее обучение только на GPU во время eval.
После завершения captures все действующие PyTorch BC/PPO/aim/maneuver/group
fit используют общий `training_devices()`: только CUDA, без CPU training
benchmark и без fallback. При недоступном CUDA явная ошибка до optimization.
CPU остаётся для Go game inference, подготовки данных, export и проверки
checkpoint; это не обучение.54 focused Python tests прошли, включая CUDA-only
selector/ошибку при отсутствии CUDA и actual CUDA curriculum fits в тестах.
Харнес остаётся4 independent server/client instances x2 с разными seeds.
