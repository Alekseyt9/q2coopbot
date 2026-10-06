# Matched initial rollout: constant против linear retention

06.10.2026. Продолжение [предыдущего сравнения](learned_combat_constant_retention_v4.md).

## Протокол до результатов

Original obstacle parent, прежние reward/horizon/MLP/Adam/RNG и fixed
4 updates каждой ветки. Единственное отличие — beta1,2/3,1/3,0 против
constant1. Первый fresh stochastic batch22900–22903 собирается один раз
от общей behavior policy. Runner `-InitialBatch` проверяет model/reward,
условия харнеса, confirmed seeds/dispatch/capture до создания output;
каждая ветка самостоятельно re-export native data, затем делает update1
из одинаковых bytes и своего independent parent checkpoint. Сравнить
actor/value/std/Adam/RNG update1; retention spec отличается mode.
First batch является on-policy обеим веткам, его не считают двумя
независимыми captures. Trainer duplicate-consumption check сохраняется.
С update2 — новые собственные rollout каждой ветки22904–22915, никогда
не переиспользовать data после divergence. Никаких копий PAK/runtime.

Final update4 выбран до eval. Fresh Mixed23000–23003/Solo23100–23103,
known regression20800/21300/21700/22100 по4seeds, before/after обеих веток.
31 batches/124 уникальных captures (4shared+24own train+96eval), максимум
4 server/client instances x2, CUDA-only обучение. Собирать serial arms.
Sources frozen до окончания. Проверить first update parity, native
provenance/reward/consumption/Adam/RNG и exact CUDA replay. Parent eval
uninterrupted prefix сравнивается отдельно от resumed/rules outcomes.
Общая успешность — class-confirmed kills обеих угроз без смерти/раннего
handoff; Solo отдельно. Нет выбора checkpoint по eval/live promotion.
Training-only retention bank пока отдельный будущий фактор.

Root `workspace/artifacts/combat-matched-retention-v4-20261006`.

## Результаты

Все31 batches/124 уникальных captures завершены. Native/runtime source
fingerprints одинаковы; каждый эпизод подтвердил seed, max4 instances/x2.
66 Python tests прошли, PowerShell AST без ошибок; negative runner check
отклонил неверный seed batch до создания output/collection. Shared batch
переиспользуется ссылкой на каталог и pinned receipt, assets не копируются.

### Контроль начальных данных

First batch857 rows с SHA256
`7d90bed90ac79f16e92475ea3593e6cb89aa18c87e514330fefef3e8e24009f3`.
Update1 weights SHA256
`b99df68ae2df338879f489fca3223efb4364c40a8f7164a114dac698e6b46c79`
совпали точно. Независимый tensor audit подтвердил равенство actor,
value, std, обоих Adam states, CPU/CUDA RNG, consumed history/counters.
Checkpoint retention metadata отличается только mode.

Independently collected update2 batch22904–22907 тоже совпал:
875 rows, одинаковая behavior и rollout bytes. Различие весов впервые
после update2, где beta linear2/3 против constant1. Update3/4 получили
собственные новые данные от уже разных behavior policy. Таким образом,
ранний confound прежнего опыта устранён, а различие следующих own-policy
траекторий ожидаемо и не исправляется переиспользованием чужих данных.

Обе ветки прошли точный CUDA actor/Adam replay, counters10/20/30/40,
critic40/80/120/160, pinned retention, native re-export, consumption и
reward audit.4 updates/40 accepted actor steps каждой, cumulative57/564,
ни одного direction repair. Все доступные kill reward rows включены.

| Учебные данные | Linear | Constant |
|---|---:|---:|
| Переходы, включая один и тот же initial batch | 3644 | 3281 |
| Без видимого врага | 1213 | 1559 |
| Награды за убийство | 3 | 1 |
| Максимальная ошибка reward audit | 6.66e-16 | 6.59e-16 |
| KL к parent на собственных states update4 | 0.02027 | 0.01183 |

Эти sums не независимые unique transitions: первые857 rows общие;
вторые875 независимо собраны, но совпали. KL на разных state distributions
не causal combat quality metric. В первых двух батчах kill rewards нет;
успешные stochastic исходы крайне редки в этом ограниченном бюджете.

### Игровая оценка

Победы требуют обоих class-confirmed kills без смерти и без handoff
перед окончанием боя; Solo — одного kill. Проверка native ownership
прошла, raw и uninterrupted win totals совпали.

| Seeds / набор | Parent wins | Linear wins | Constant wins | Linear deaths | Constant deaths |
|---|---:|---:|---:|---:|---:|
| Fresh Mixed23000–23003 | 1/4 | 3/4 | 1/4 | 1 | 3 |
| Fresh Solo23100–23103 | 4/4 | 4/4 | 4/4 | 0 | 0 |
| Known20800–20803 | 3/4 | 2/4 | 3/4 | 2 | 1 |
| Known21300–21303 | 3/4 | 2/4 | 3/4 | 2 | 0 |
| Known21700–21703 | 3/4 | 2/4 | 1/4 | 2 | 2 |
| Known22100–22103 | 3/4 | 2/4 | 3/4 | 1 | 1 |

На20 Mixed conditions обе ветки дали11 побед против13 у parent.
Linear kills26→28, deaths6→8, incoming1645→1698; constant kills26→28,
deaths5→7, incoming1612→1568. Повторный parent на21300 получил different
rules-assisted death/received, но все24 paired parent uninterrupted
prefixes совпали. Repeated baseline copies не независимые samples;
все известные наборы — regressions, не fresh holdout.

Fresh Mixed received391→317 linear и391→353 constant. Solo received
146→144 linear,146→161 constant. Linear проиграл по одной known победе
в каждом квартете, constant сохранил три квартета, но21700 потерял две
победы. Constant не является универсальным средством от forgetting.
В этом paired pilot decay не объясняет все регрессии; general statistical
вывод о превосходстве какого-либо режима не следует из одного цикла.

Blocked actual provider frames linear952/3936→1011/4174 (24.2%→24.2%),
constant968/4040→820/4278 (24.0%→19.2%). Окна до смерти/final kill
разной длины; снижения guard metric недостаточно для принятия политики.
Обе final ветки отклонены как замена original obstacle training reference.
Live не менялся, архитектура/reward/horizon не изменялись.

## Следующий шаг

От original obstacle parent отдельным фактором проверить pinned
training-only retention bank. Собирать новые parent observations,
включающие две угрозы, добивание одиночной, walls/barrels и отсутствие
видимости. Делить по episodes/seeds до fit; не использовать eval traces
или known regression seeds для банка. Фиксировать банк/веса до eval,
сравнить constant on-policy-state retention без банка и с банком с общим
initial rollout и own-policy continuations. Банк только для distillation,
не примешивать off-policy actions/rewards в PPO surrogate. Отдельно
логировать coverage/success scarcity. GRU/attention/reward/horizon не
менять одновременно. Банк здесь ещё не реализован.

Артефакты: `completion-status.json`, `first-update-parity-audit.json`,
`training-pair-audit.json`, `comparison-audit.json`; branch audits
`training/direction/consumption/evaluation/blockage-audit.json`,
`retention/native-reexport-audit.json`, `retention/shaping-audit.json`.
Final linear weights
`80bd004a0db8fde35e4481dad1197b9f4ed35ffffb5187b66304ac20dfda0534`,
checkpoint `fab600900d4172f5f91eb5391118b85df470b47201b049023133552bf2ab0851`.
Final constant weights
`0f085e4ca3e4b8d8c0f47e1c3ce20430829aa059815f84d6ec690f650816d489`,
checkpoint `af0197831b28e965debdf5030d9bd2bdb97d7f90e5d488972703e2697668b553`.
