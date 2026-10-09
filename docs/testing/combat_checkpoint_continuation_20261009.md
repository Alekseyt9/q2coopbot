# Checkpoint continuation Attention128, 2026-10-09

Основание: [560-battle PPO comparison](combat_distillation_ppo_results_20261009.md). После первого own-policy update Attention128 seeds07/08 получили 46/80 и45/80 против исходного FireBC50/80 и rules68/80. Кандидаты не продвинуты. Проверяем более длинный бюджет при той же архитектуре.

## Реализация и проверка

`process_combat_architecture_pool.py` принимает в model binding явные `resume_checkpoint`, checkpoint/report SHA и `parent_updates_completed`. До обработки проверяет связь родительских весов, CUDA report, config/anchor/bank SHA. Передаёт checkpoint в существующий `ppo_recurrent.py --resume`; ожидает инкремент update count и соответствующий resume SHA. Проверенный pipeline resume сохраняет completed updates по seals; ожидание update count теперь учитывает родительскую историю.

Численное сравнение загруженных actor/critic/log_std при resume переведено на CUDA. Сериализация checkpoint и чтение метаданных не являются обучением. Optimizer state и CPU/CUDA RNG восстанавливает существующий trainer; retention/bank weights остаются0.

`prepare_combat_checkpoint_training.py` замораживает sealed parent weights/checkpoint/report и компилирует fresh train планы через существующий registry/harness. `verify_combat_checkpoint_cuda.py` на RTX5070 подтвердил для обеих моделей точное совпадение actor/critic/log_std с CUDA checkpoint tensors, конечные ненулевые optimizer histories и уникальные consumed rollout. Updates=1, accepted actor steps=10, consumed rollout=1. Проверка пройдена как на preparation smoke, так и повторно на фактических frozen inputs первого раунда. Это preflight continuation, а не подтверждение успешного следующего PPO update.

Go policy parity/replay/tests не запускались. Модели и численные проверки только CUDA; native Go harness/exporter используются для игры и закрытого корпуса.

## Запущенная серия

Root `workspace/artifacts/attention-continuation-v1-20261009`. `run_combat_checkpoint_series.py` ведёт четыре последовательных раунда, каждый — отдельный native pool16/x2 на всех20семействах, затем две CUDA continuation updates. Train seed offsets72/80/88/96; по80 собственных боёв/model/round, всего640 новых training battles. Общий дополнительный бюджет после первого80-battle update станет320 на модель, итоговый own-policy budget400 на модель. Validation captures не используются для gradients.

Python sources фиксируются в series protocol и сверяются перед каждым раундом; per-round input checkpoints копируются и проверяются поSHA. Новый rollout не должен присутствовать в `consumed_rollouts`. У каждого боя отдельный native сервер/клиент и seed; между моделями paired условия совпадают. Свободный slot немедленно получает следующую задачу очереди. Round1 подготовлен, CUDA preflight прошёл, native сбор запущен. Результатов новой серии пока нет.

После четырёх раундов автоматически480 validation battles: две пары initial-after-first-PPO → after-four-more-rounds плюс frozen исходный FireBC и rules. `prepare_combat_architecture_evaluation.py --before-processing-root` выбирает sealed weights первого checkpoint, а не ошибочно старт только последнего раунда. Validation offset24, same80 conditions, pool16/x2. Это reused validation; final test отложен. Promotion требует результата качества.

Авторитетный process receipt: `workspace/build/attention-continuation-v1-process.json`; stdout/stderr рядом. Состояние раунда — series `progress.json`, parent lineage и update receipts — per-round capture/processing reports. По истёкшему observation timeout новые процессы не запускаются. Завершённые checkpoint и первичные captures сохраняются; weights/checkpoints остаются ignored Git artifacts.

## Первая фактическая CUDA continuation

Round1 native pool завершён:160/160 jobs без ошибок, source unchanged; все member receipts отдельно проверены. Attention128 seed07 продолжил checkpoint на5287 eligible transitions:10 новых accepted actor steps, total20, update count1→2, approximate KL0,004189. Весовой/checkpoint/report seal проверен.

Отдельная проверка checkpoint tensors на CUDA подтвердила **реальные** optimizer counters: у всех сохранённых actor parameter states step10→20, critic step40→80. `consumed_rollouts` сохранил предыдущую запись и добавил ровно один новый уникальный rollout (1→2); report resume SHA совпал с frozen parent checkpoint. Это подтверждает optimizer continuation, а не fresh-optimizer update с вручную увеличенным номером. Proof `round-1/optimizer-continuation-proof.json` фиксирует CUDA device, counters, KL и output weights/checkpoint SHA. Seed08 в момент проверки ещё проходит export/compression; round2 ещё не начался. Роста качества по одному train update не утверждаем.

**Round1 полностью завершён:**обе CUDA continuation updates complete. Seed08:6279 transitions,10 новых accepted actor steps,total20,updates2,KL0,004918. Его checkpoint отдельно проверен наCUDA: actor optimizer10→20,critic40→80,consumed rollout1→2 с сохранением предыдущего SHA и новой записью, совпадающей с текущим rollout report. Seals weights/checkpoint/report и parent resume SHA проверены; proof `round-1/optimizer-continuation-proof-seed08.json`.

Серия автоматически перешла к round2,train offset80. Новый CUDA preflight frozen inputs обеих моделей прошёл:updates2,total actor steps20,consumed rollouts2. Это первая проверенная передача между раундами с сохранением истории; native сбор второго160-battle корпуса начался. Полная серия640train+480validation и её результаты качества ещё впереди.

**Сбор round2 завершён:**160/160 jobs без ошибок. Повторная совокупная проверка round1+round2 подтвердила320/320 native receipts, frame budgets и единый source fingerprint; frozen Python sources не изменены. Snapshot `closed-corpus-snapshot.json` в series root фиксирует частичное320/640 покрытие. Начался export/compression второго корпуса перед следующими CUDA updates; update count3 и round3 на момент этой проверки ещё не подтверждены.

**Round2 processing завершён:**seed07/08 соответственно5788/6183 eligible transitions,по10 новых accepted actor steps,updates3,total actor steps30. KL0,005734/0,005619. Seals обоих output checkpoint/report/weights проверены. CUDA tensor audit подтвердил actor optimizer20→30,critic80→120,consumed rollout2→3 с сохранением родительской истории. Proofs `round-2/optimizer-continuation-proof-seed07.json` и `...-seed08.json`.

Round3 начался автоматически,train offset88; его CUDA input preflight подтвердил updates3,total30,consumed3 для обеих моделей. Native сбор идёт, всё ещё без новой quality оценки; нельзя объявлять улучшение по одному training KL или увеличению update count.

**Сбор round3 завершён:**160/160 без ошибок. Совокупно480/640 planned training battles проверены по native receipts и frame budgets, source fingerprint общий, frozen Python sources не изменены. `closed-corpus-snapshot.json` обновлён до480. Export/compression третьего корпуса продолжается; его CUDA updates и новая оценка качества в момент проверки ещё не завершены.

**Round3 processing завершён:**seed07/08 соответственно6830/6540 eligible transitions,по10 новых actor steps,updates4,total actor steps40. KL0,007169/0,007862. Output seals обоих checkpoint/report/weights проверены; CUDA audit подтвердил actor optimizer30→40,critic120→160,consumed rollout3→4 с сохранением предыдущих SHA. Proofs `round-3/optimizer-continuation-proof-seed07.json` и `...-seed08.json`.

Автоматически начался последний round4,train offset96. Его CUDA input preflight прошёл для обеих моделей с updates4,total40,consumed4. После ещё160 training battles и двух CUDA updates запланирована полная480-battle validation; промежуточная train статистика не заменяет проверку качества.

**Сбор всех четырёх раундов завершён:**640/640 jobs, без ошибок. Отдельный аудит всех native receipts подтвердил capture/provenance/dispatch/seed validity и frame budgets. Повторный CUDA tensor audit всех шести завершённых updates round1–3 подтвердил сохранение optimizer counters, consumed rollout prefix и output seals; proof `cuda-continuation-audit.json` в series root. Round4 проходит export/merge/compression перед последними двумя GPU updates; результаты качества ещё не получены.

## Обучение завершено; полная оценка запущена

Все четыре continuation rounds complete:640/640 новых native battles, восемь GPU updates. Вместе с исходным PPO update каждая модель обучалась на400 собственных боях. Совокупный аудит `closed-corpus-snapshot.json` подтвердил общий native source fingerprint и неизменность frozen Python sources.

| Модель | Eligible transitions round4 | Всего transitions за5 updates | Updates | Actor optimizer steps | Critic optimizer steps | Round4 KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention128 seed07 | 6553 | 31093 | 5 | 50 | 200 | 0,008026 |
| Attention128 seed08 | 5782 | 32172 | 5 | 50 | 200 | 0,007495 |

`cuda-continuation-audit.json` повторно проверил все восемь continuation updates на CUDA: реальные optimizer counters, конечные Adam moments, сохранённый consumed rollout prefix, уникальность всех пяти rollout SHA и seals checkpoint/report/weights. Последний переход обеих моделей: actor40→50, critic160→200, updates4→5, consumed rollout4→5. Proofs `round-4/optimizer-continuation-proof-seed07.json` и `...-seed08.json`. Проверка не заменяет оценку качества поведения.

Driver автоматически запустил480 validation battles, pool16/×2. `evaluation/protocol-audit.json` подтвердил совпадение всех80 условий для каждой из шести variants, plan hashes и происхождение весов: before — checkpoint после первого80-battle PPO, after — после400 own-policy battles, FireBC — исходный frozen baseline, rules — обычный контроллер. Validation повторно используется, независимый final test ещё не выполнен. На момент фиксации отчёта оценка идёт; победы и promotion пока не подтверждены.

## Возобновление прерванной оценки

При следующей проверке исходный driver и его native процессы отсутствовали; журнал очереди содержал237 завершённых jobs без ошибок, aggregate report ещё не существовал. Причина остановки по этим данным не установлена. Состояние `evaluating` в JSON само по себе не является доказательством живого процесса.

Добавлен `scripts/resume_combat_architecture_evaluation.py`: OS file lock, проверка отсутствия исходных drivers/занятых портов, SHA plans/registry/model/reward/runner и source/native records, actual hashes клиентских/exporter/runtime файлов, подтверждённые seeds/fixtures/frame budgets. Для finished jobs первичные отчёты сохраняются. Незавершённые каталоги перед новым запуском перемещаются в проверенный подкаталог `evaluation/recovery/interrupted`; данные не удаляются. Новые бои запускает прежний `run_registered_combat_pool_episode.ps1`,16 независимых slots,×2; native Go/PowerShell source fingerprint сохранён. Обучение и численные policy checks при восстановлении не выполняются.

Dry-run подтвердил238 валидных members:237 queue receipts плюс один terminal native report, чью queue receipt прерывание не сохранило. Этот job82 восстановлен по primary report. Остальные242 jobs повторно поставлены в очередь. После их завершения existing member verifier проверит полный480-battle корпус, затем existing reporter выполнит paired comparison через отдельный member proof; незавершённые исходные aggregates не подменяются.

Новый авторитетный process receipt `workspace/build/attention-evaluation-resume-v1-process.json`; stderr/stdout рядом. Фиксация повторного запуска — `evaluation/recovery/resume-*.json`, прогресс — `resume-progress.json`. Файл lock может остаться после выхода процесса: блокировку удерживает ОС, наличие файла не означает живой driver. На момент записи восстановление запущено; итоговые победы ещё не подтверждены.

**Восстановление и полная оценка завершены:**480/480 без ошибок, полный member proof подтвердил runtime/binary/source/fixture/seed/frame budget hashes. Driver terminal, series `report.json` и `progress.json` закрыты отдельным recovery seal; четыре исходных training reports и их SHA сохранены. Seed07 46→48/80,seed08 45→46/80,FireBC49/80,rules69/80. Новые checkpoints не продвинуты. Full trace diagnosis480/480 показал сохраняющиеся yaw errors и выросший stall proxy seed08. [Полный результат, все семейства и следующий coupling experiment](combat_checkpoint_results_20261009.md).
