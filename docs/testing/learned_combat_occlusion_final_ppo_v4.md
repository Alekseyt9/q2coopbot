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

## Результат и проверка

Завершены48 captures:8 повторов,16 training,8 Mixed eval,8 Solo eval
и8 дополнительных regression captures. Везде4 instances x2, отдельные
подтверждённые seeds. Frozen learned prefixes на21500–21503 совпали4/4;
совпали также наблюдаемые полные траектории и outcomes этих повторов.

| Update | Fresh rows | Empty-enemy rows | Kill reward rows | Accepted actor steps | Anchor beta | KL к behavior |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 911 | 181 | 1 | 10 | 1 | 0.005284 |
| 2 | 889 | 378 | 1 | 10 | 2/3 | 0.009525 |
| 3 | 743 | 38 | 1 | 10 | 1/3 | 0.009993 |
| 4 | 1098 | 178 | 3 | 0 | 0 | 0.0000000053 |

Всего3641 fresh transitions,775 с пустыми enemies,6 kill reward rows.
Последний update изменил только critic: его3 убийства не обновили actor.
Итого30 принятых actor steps,57 cumulative PPO updates/554 actor steps.
Adam counters10/20/30/30 и critic40/80/120/160 проверены независимо.
Ни один rollout не переиспользован. Native re-export совпал побайтово;
reward components пересчитаны, max error6.66e-16. Проверены checkpoint
lineage, optimizer state, CUDA-only training/benchmark, pinned anchor и
расписание, source/native fingerprints, фактическое владение kill windows.

Final weights SHA256
`a4dfbb3180f51d63480fd424be2689445295650a5a5ea23b7e5a89e32d3d3b8b`,
checkpoint `36c809f49161cd83006ae3e1566e73346eecb48740ec4f02565d3d2a7ab335ec`.
Actor update4 совпадает с update3; используется заранее назначенный final
checkpoint, без выбора весов по оценке. Архитектура MLP810→64→64→8,
reward/horizon/std сохранены, GRU/attention/выбор оружия не добавлялись.

| Проверка | Полные победы без rules до завершения, before→after | Kills | Deaths | Incoming health damage |
|---|---|---|---|---|
| Fresh Mixed21300–21303 | 2/4→3/4 | 6→6 | 1→0 | 288→235 |
| Fresh Solo21400–21403 | 4/4→4/4 | 4→4 | 0→0 | 124→109 |
| Known regression20800–20803 | 2/4→1/4 | 5→3 | 1→1 | 243→241 |

Fresh evaluation не встретил empty-enemy held frames у обеих policies:
это не проверка освоенного поиска за препятствием. Поэтому после основной
оценки дополнительно повторены ранее диагностированные20800–20803.
Этот набор — post-hoc regression, не новый holdout; не использован в fit,
выборе checkpoint или изменении параметров. Seeds всех training/eval
наборов разделены; regression seeds уже известны из прежних проверок.

На20800 parent удерживает provider168 суммарных empty-enemy frames,
дважды достигает visibility timeout, не убивает врагов и выживает.
Final policy там не теряет видимости, но погибает без убийств. На20801
раньше убивались оба, теперь бой не завершён за horizon;6 held frames
без timeout. На20802 смерть сменяется выживанием с одним убийством;
на20803 обе policies завершают бой. Сокращение held frames168→6
не считается освоением поиска: сама траектория и исход изменились.

Успех на fresh Mixed сопровождается регрессией на известном наборе.
Четыре seeds в каждом наборе недостаточны для общего заключения.
Ветка не принята как улучшение и не включена в пользовательский live.

## Почему четвёртый actor update остановился

Отдельный CUDA numerical probe на consumed update4, без экспорта весов,
воспроизводит объективную функцию, gradient clipping, Adam proposals,
backtracking и ограничения. При сохранённых Adam moments направление
шага повышает loss: grad·delta положительно даже при сильном уменьшении
шага. Retry7 даёт loss0.004629 против исходного0.004312, KL0.00001749;
предложение правильно отклоняется из-за роста loss. Это не KL rejection.

Контрольный probe с пустыми moments на том же batch/actor/gradient при
retry7 получает отрицательное grad·delta, loss0.001584 и KL0.0006691;
такое предложение проходит текущие gates. Это локальное свидетельство
проблемы направления накопленного Adam шага на данном batch, не готовая
новая policy и не обоснование сбрасывать optimizer каждый update.
Все формальные checkpoints сохраняют штатный resume и моменты.

Следующий шаг: контролируемый fallback при подтверждённом uphill Adam
направлении, с rollback/counter/CUDA regression tests и явным логированием
применения. Затем fresh PPO и заранее закреплённые Mixed/Solo/regression
проверки. Не увеличивать одновременно horizon, reward и архитектуру.

## Артефакты

В root сохранены `training-audit.json`, `consumption-audit.json`,
`evaluation-audit.json`, `known-occlusion-audit.json`,
`rejection-diagnostic.json`, `prefix-repeat-audit.json`;
в `retention/` — `native-reexport-audit.json`, `shaping-audit.json`.
Go/runtime/trainer исходники в этом цикле не менялись; новые Go tests
не запускались. Captures используют существующий финальный runtime.
Пользовательский live marker и config не изменены.
