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

## Результаты и границы сравнения

Все32 batches/128 captures завершены,4 инстанса x2. Individual seeds
подтверждены; source/native fingerprints одинаковы во всех manifest.
Обе ветки получили4 CUDA updates/40 accepted actor steps,57 cumulative
updates/564 actor steps. Adam clocks10/20/30/40, critic40/80/120/160,
RNG/consumption/native re-export/reward/точный CUDA actor+Adam replay
прошли независимые аудиты. Linear repair сработал один раз на update4;
constant repair не требовался. Модель и reward/horizon не изменялись.

| Обучение | Linear | Constant |
|---|---:|---:|
| Новые переходы | 3682 | 4112 |
| Переходы без видимого врага | 1182 | 1657 |
| Строки награды за убийство | 2 | 3 |
| Максимальная ошибка reward audit | 6.66e-16 | 4.44e-16 |
| KL к parent на собственных states update4 | 0.02061 | 0.01235 |

Все kill reward rows включены. KL измерен на разных own-policy states,
поэтому это диагностическое отклонение, не парная оценка эффективности.

Полная победа ниже требует убийства Gunner и Parasite без смерти и без
предшествующей передачи rules; native class attribution проверена.
Solo требует одного убийства. Before — исходный obstacle parent, after
— заранее назначенный update4. Точки оценки не использовались в обучении.

| Набор / seeds | Parent wins | Linear wins | Constant wins | Linear deaths | Constant deaths |
|---|---:|---:|---:|---:|---:|
| Fresh Mixed22700–22703 | 4/4 | 0/4 | 4/4 | 4 | 0 |
| Fresh Solo22800–22803 | 4/4 | 4/4 | 4/4 | 0 | 0 |
| Known20800–20803 | 3/4 | 2/4 | 3/4 | 2 | 1 |
| Known21300–21303 | 3/4 | 0/4 | 0/4 | 4 | 3 |
| Known21700–21703 | 3/4 | 0/4 | 2/4 | 4 | 1 |
| Known22100–22103 | 3/4 | 1/4 | 1/4 | 2 | 3 |

Описательно20 Mixed условий: linear16→3 побед,2→16 смертей,
32→18 kills,received1467→1850; constant16→10 побед,2→8 смертей,
32→25 kills,received1503→1596. Повторные parent captures между ветками
не дополнительные независимые seeds. Fresh Mixed received282→400
linear и282→264 constant. Solo received185→144 и185→118.

У linear blocked provider frames1162/4474→473/4159 (26.0%→11.4%),
у constant1175/4466→723/4157 (26.3%→17.4%). Уменьшение blockage
не означает безопасный бой: known21300 constant теряет3 победы при
23.0%→1.5% blockage. Окна имеют разную длину до смерти/финального
убийства; агрегат не причинная оценка навыка обхода.

### Обнаруженный confound

Первый uninterrupted learned prefix совпал по всем4 training seeds;
все24 parent evaluation prefixes также совпали. Но после handoff
на22603 правила и последующий возврат provider дали разную траекторию:
PPO input первого батча1150 против1094 rows,22603 отдельно298 против242.
Остальные seeds293/289/270 rows одинаковы. Различие не скрыто:
первые exported weights уже отличаются при одинаковом beta1.
Повторные parent full outcomes тоже не полностью равны (received на
known21300 разный), хотя непрерывные начальные provider траектории равны.
Фиксированный старт не делает rules-assisted capture полным engine snapshot.

Дополнительный изолированный CUDA audit повторил первый linear rollout
с constant mode от того же parent/RNG/Adam: weights bytes точно совпали
с исходным linear update1. Отличие режима при beta1 не меняет обучение
на одинаковых данных. Этот audit не новый capture и не кандидат на
продолжение; его checkpoint не подключался к игровым веткам.

Поэтому результаты обеих итоговых политик проверены, но причинное
утверждение «только decay вызвал регрессию» не подтверждено. Constant
на этой выборке лучше linear, однако тоже хуже parent на3 known наборах.
Обе ветки отклонены как замена исходной reference; live не менялся.

### Следующий ограниченный опыт

Начать от original obstacle parent. Зафиксировать один fresh initial
on-policy rollout и использовать одинаковые bytes для первого update
обеих независимых branches; одинаковая behavior policy делает такой
первый input on-policy для обеих. После первого update собирать только
собственные новые rollouts. Отдельно проверять resumed segments после
rules: paired seeds не гарантируют идентичные inputs.

Если дальнейшая постоянная retention снова теряет навыки, отдельным
фактором добавить фиксированный retention bank из fresh training-only
observations parent: обе угрозы, одиночное добивание, стены/бочки и
краткая потеря видимости. Eval traces и seeds исключить, веса банка
зафиксировать до eval. Это пока план, банк в данном опыте не реализован.
Архитектуру/reward/horizon одновременно не менять; недостаток полных
успешных stochastic episodes остаётся открытым ограничением.

Артефакты root: `comparison-audit.json`, `prefix-repeat-audit.json`,
`matched-first-update-audit.json`, `completion-status.json`; в каждой
ветке `training/direction/consumption/evaluation/blockage-audit.json`,
`retention/native-reexport-audit.json`, `retention/shaping-audit.json`.
Final linear weights SHA256
`18a8392ad1ac3d928fc52f5b07a7a7376456467ad52d1c2a4bb6383e0df75d27`,
checkpoint `942c957c8c71ddeb8f741ed0d49d7937d1fe6c27d5660e62141aeb136b77a9ed`.
Final constant weights SHA256
`5914285e86c660c787acf9134e4efacc86b20c1fd4e5a62ce91a9952efc83585`,
checkpoint `4a4454b291e9e3a5c3f0210b44a6cb3d463317858461c4eafe43db44911c25f3`.
