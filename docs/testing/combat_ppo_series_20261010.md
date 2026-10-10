# Последовательное GPU PPO продолжение parent

После отрицательных uniform/wall,continuous-noise и categorical-temperature
проверок исходный parent2 остаётся рабочей базой. Победа над обычным ботом
пока не доказана. Для следующей проверки увеличивается собственный опыт
этого actor, без смены архитектуры,наград или inference confidence.

`run_combat_ppo_series.py`, root`parent-ppo-series-v1-20261010`:
два последовательных раунда по80 физических own-policy боёв, все20 registry
семейства,train offsets200..203 и204..207,16 refill slots,×2. Блоки train
не пересекаются внутри серии; это не утверждение глобальной исторической
новизны сидов. Independent test не используется.

Старт: sealed `pickup-ppo-processing-v1-20261010/parent3/update`,CUDA update2,
weights SHA`85101a72bb06fc640fef12ffbc4e8bdab9f976e826ff6e70923dd8847ea72267`,
checkpoint SHA`799e00ba5688676ce0a0176c8e68e3cdd281c615d6a10003ddd796c9636fd4d1`.
Actor/value/std и Adam продолжаются точно; исходный config/reward,anchor иbank
сохраняются. Калиброванные temperature/quarter-noise веса не используются.
Round2 собирает опыт уже из результата round1; старые fixed trajectories
не выдаются за fresh on-policy.

Каждый раунд: проверенные actual planner contracts,terminal-member LZX/SHA,
80 native member proofs,training outcomes (не quality comparison),zero
clear-target rules/equip fallback,CUDA-only native export with batch finalize,
GPU PPO continuation,CUDA checkpoint reload actor/value/std/Adam audit.
Численных CPU/Go NN проверок нет. Actual series Python PID28952 подтверждён.
Round1 уже собрал80/80 без ошибок и перешёл вcuda_processing;update ещё
не принят на момент этой записи.

`--resume` сверяет frozen protocol,пропускает только sealed completed rounds
после проверки weights/checkpoint/report/audit и checkpoint lineage. Existing
terminal native pools переиспользуются,processing вызывает проверенный
`process_combat_architecture_pool.py --resume`. Незавершённый native pool
нужно сначала восстановить по его actual state: автоматической повторной
отправки уже исполненных боёв нет. Resume outer series ещё не проверен живым
прерыванием; прежний processor/checkpoint resume проверен ранее.

За удерживаемым actual series process handle28952 поставлен
`run_combat_ppo_series_evaluation.py`,root`parent-ppo-series-eval-v1-20261010`.
После двух accepted CUDA updates начальные и конечные веса проходят общие
validation32..35 условия:parent/after по двум stochastic RNG arms,80 условий
каждая ветка плюс rules80,всего400 боёв,16 slots,×2. Новые conditions отличаются
от прежних validation24..27:будущий rules score не сравнивается напрямую
с прежними68/80. Это development,не untouched independent test.

Queue dispatch требует complete training,sealed final checkpoint/current
source bindings и3GiB headroom. Это сокращённая400-case очередь с shared
immutable binaries иterminal LZX; запас проверяется непосредственно перед
dispatch. Training dispatch требовал4GiB,каждый раунд требует2GiB. Если место
не позволяет evaluation,надо compact closed archives,не ослаблять proof.

Добавлен общий `combat_native_evaluation_closure.py`:member proofs,actual
sampling configs,quality,9 diagnostics,ownership,outcome/storage. Syntax и
diff checks пройдены;новый400-case live evaluation ещё не запускался. Никакого
автоматического promotion,test dispatch или superiority claim.

## Фактическое завершение серии и восстановление

Серия завершена: 160 native боёв, два последовательных CUDA PPO обновления,
update 2 → 4. Round1 использовал 4608 переходов и принял 5 actor steps;
round2 использовал 3633 перехода и принял 10 actor steps. Итоговые веса:
`7254788a2d17ba01787601599278fd98313db5b23f12e546f7a09367f6f934ec`,
checkpoint:
`e2effcfb7dfa8a0c1e688e48711f5898d645c12cba6075daaaa399e8a4d6dc2d`.
Root progress подтверждает `parent_optimizer_preserved: true`.

Первоначальный запас диска оказался недостаточным: round1 остановился
на merge JSONL после завершения всех 20 CUDA exports. Native бои не
перезапускались. После прозрачного LZX сжатия closed streams processor
проверил SHA исходных данных и завершённых CUDA exports и переиспользовал
их. Добавлено сжатие закрытого export сразу после выхода его producer и
до merge; свежая batch CUDA обработка может сжимать каждый завершённый
корпус отдельно. Resume серии проверен фактически: принят round1, затем
при следующем запуске он пропущен с проверкой seals, round2 собран новым actor.

Старая очередь `parent-ppo-series-eval-v1-20261010` завершилась ошибкой вслед
за первым неудачным запуском обучения; 400 оценочных боёв она не запускала.
Следующая очередь должна иметь новый output root и пройти запас 3 GiB.
До её завершения улучшение качества этих новых весов не установлено.

Для освобождения места объединены только побайтно одинаковые native/final
sequence копии закрытых processing roots, SHA проверен до и после замены
на hardlinks. Сохранены оба пути, веса, checkpoints и все данные. Отчёты:
`closed-sequence-dedup-v1-20261010.json` (152 пары),
`closed-sequence-dedup-round2-v1-20261010.json` (20 пар),
`closed-sequence-dedup-extra-v1-20261010.json` (60 пар).
Также объединены старые sealed shared-evaluation binaries (82 → 2 inode)
и очищен только воспроизводимый локальный Go build cache при отсутствии Go
процессов. Эти меры не являются оценкой качества модели.

## Текущая development оценка

`parent-ppo-series-eval-v3-20261010` запущена: 400 боёв, 16 refill slots,
×2, parent/after по двум stochastic RNG arms и rules. Actual Python PID25788
подтверждён живым; native servers и Go clients запущены, slot0 running.
Результат качества пока отсутствует.

V2 остановилась до native dispatch: planned offset32 превышает validation
count32 у исходных generated recipes. Training protocol не переписывался.
Для v3 задан явный `--validation-offset 28`, все пять планов с одинаковыми
условиями проверены штатным compiler. Queue/protocol сохраняют planned32
и actual28 и причину коррекции. Это development validation28..31, без
утверждения исторической новизны сидов; independent test не затрагивается.

Дополнительная закрытая дедупликация exporter copies:
`closed-processing-exporter-dedup-v1-20261010.json`, 258 замен, 260 → 2 inode;
все binary и manifest SHA сохранены. Перед оценкой выполнен запас3GiB,
логические пути, checkpoints и traces сохранены. LZX compression старых
закрытых aproc/pickup archives завершена штатно.
