# Constant против linear anchor retention

06.10.2026. Продолжение [obstacle PPO](learned_combat_obstacle_ppo_v4.md).

## Протокол до результата

Добавлен explicit `--retention-mode constant`/runner `-RetentionMode`.
Linear по умолчанию сохраняет legacy checkpoint spec без новых полей.
Constant spec включает mode; смена mode/anchor/weight/horizon при resume
отклоняется, constant без anchor тоже. Initial weight1 остаётся1 после
горизонта, linear даёт1,2/3,1/3,0.66 Python tests прошли, gradient tests
только CUDA. Никакого автоматического нового retention state bank.

Обе ветки стартуют от одного obstacle parent weights
`3c5cd86df29e73b1709a02b5e969e35cfa262ff4464c6cded022d403653a8d7d`,
checkpoint `969b8a9b174e193081a444f1444f726148c08a2687d0f3e3cc9f4da6925ed39c`.
Одинаковые конфиг/reward/MLP810→64→64→8/horizon300/initial std/Adam/RNG,
4 fresh updates каждой ветки, pinned anchor исходный parent. Отличается
только расписание beta. Actor repair/rollback тот же. Final update4
назначен до eval; checkpoint selection по оценке отсутствует.

Paired training seeds22600–22615, но каждая ветка получает own-policy
rollout от своей версии после предыдущего update; branch rollout не
переиспользуется другой веткой. Повторные seeds между branches задают
paired условия, не дополнительные независимые seeds. First batch при
одинаковом behavior/β ожидается совпадающим в непрерывной learned части;
любое различие после rules handoff или в trainer inputs явно учитывать,
не объявлять весь engine snapshot exact deterministic.

Парные parent→final каждой ветки на fresh Mixed22700–22703,
fresh Solo22800–22803 и known20800–20803/21300–21303/21700–21703/
22100–22103. Known regression не fresh holdout и не training.64 captures
на ветку/128 total, максимум4 simultaneous server/client instances x2,
индивидуальные подтверждённые seeds в каждом batch. Последовательные
ветки linear затем constant; это не8 simultaneous workers. Blaster,
stock HP/fixed100 synchronous; Solo175 HP. Sources frozen до всех eval.

Основной показатель — обе fixture classes убиты без смерти/предшествующих
rules. Incoming/blockage/held/timeouts/Solo и sample coverage отдельно.
Не приписывать изменение боевого качества одному guard metric. Проверить
native provenance/re-export/rewards/consumption/Adam/RNG/pinned schedule
и exact CUDA actor replay в обеих ветках. No live promotion.

Root `workspace/artifacts/combat-constant-retention-v4-20261006`.
