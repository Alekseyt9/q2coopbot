# Fresh GPU PPO на окончательном occlusion runtime

06.10.2026. Продолжение [occlusion continuity](learned_combat_occlusion_continuity_v4.md).

## Протокол до результата

Frozen barrel parent weights
`91697086e5c57d21f85ad393e0674e044ccd08fb54f77e91485e98ee3df7fd6b`,
checkpoint `6a46f68ed40db4fbd2f9b5a5e1b6b34db9c814c4188eb00ec987c7b9c1f32151`.
Не наследуются диагностические веса предыдущего update до budget fix.
Архитектура/810 features/reward v4/300-frame horizon/std неизменны.

Runtime включает fixed100 world preparation hold, bounded30-frame learned
occlusion ownership и отсутствие local wall-clock budget fallback в lockstep.
Невидимые позиции не подставляются в observation. Передача rules после
timeout остаётся явной границей и не принимается за самостоятельный бой.

Сначала frozen policy дважды на21500–21503; совпадение uninterrupted
learned prefix обязательно до начала обучения. Это4 независимых условия,
не8. Затем4 fresh stochastic Mixed batches21200–21215 со своей текущей
behavior policy,4 CUDA-only updates с pinned anchor beta1,2/3,1/3,0.
Старые/eval transitions не используются для PPO. Fixed final fourth update,
без выбора лучшего checkpoint по eval.

Парный deterministic Mixed21300–21303 до/после; отдельный Solo21400–21403
до/после,175 HP. Все4 server/client instances x2,Blaster,synchronous,
release100,stock monster HP, native seed/dispatch/world/weapon receipts.
Training/eval/repeat seeds не пересекаются. Root:
`workspace/artifacts/combat-occlusion-final-ppo-v4-20261006`.

Проверить свежесть/lineage/Adam/re-export/reward scores, actual consumed
empty-enemy и kill rows. Основной результат — убийство обоих без смерти и
без предшествующей помощи rules; incoming, timeout/held frames и solo
регрессия отдельно. Captures с timeout или иным возвратом rules до конца
боя не объявляются uninterrupted acceptance. Live promotion отсутствует.
