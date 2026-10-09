# Independent episode pool, 2026-10-07

## Изменение

Старый пул имел4слота по4инстанса. Общая ConcurrentQueue освобождала слот только после всей группы, поэтому быстро закончившийся инстанс мог ждать остальных. Теперь default Scheduler=episode:16independent slots, один job=один seed/сценарий/model/mode. После завершения native capture/provenance checks тот же slot/port сразу TryDequeue следующего job. Нет barrier между четырьмя seed, сценарием или моделью. End-of-queue tail и startup/cleanup/export overhead остаются; постоянную100%CPU/GPU загрузку это не гарантирует.

`run_registered_combat_episode_pool.ps1` смешивает jobs по seed_index,task_index,plan_index,mode. Parent frozen plans проверяются целиком до старта; child получает ровно одну instance из frozen generated fixtures. `run_learned_combat_baseline.ps1` поддерживает Workers1. Per-seed manifests/reports/native outputs остаются отдельными. Case report объединяет results и SHA receipts всех members для совместимости с зарегистрированным curriculum/exporter. Все seeds, reports, registry/model hashes и source fingerprint проверяются, failure не считается usable.

Default применяется ко всем20 текущим combat-baseline training families. Полные campaign recipes другого runner требуют explicit `-Scheduler cohort`; отдельный адаптер их single-episode запуска пока отсутствует. Текущие isolated base1/base2 combat sites входят в episode pool. Existing cohort scheduler сохранён.

## Live smoke

Root `workspace/artifacts/ep-smoke-v1-20261007`.32train diagnostic battles:8fresh distilled architecture priors ×4seeds одного generated Parasite/Blaster recipe,offset44, pool16/x2. Native captures32/32 usable, failed0,source unchanged. Все8case reports и member SHA receipts проверены. Это инфраструктурный smoke, не comparison побед/architecture superiority.

Повторных назначений слота16; max gap0.074535s,mean0.012959s между завершением job и назначением следующего. Интервалы на одном port не пересекаются. Это время queue assignment, не zero-delay нового gameplay после cold startup.

## Полная партия

Root `workspace/artifacts/apool-v1-20261007`. Завершены640own-policy training episodes:20families×4seeds×8models,training offset48, pool16/x2.8frozen plans и models.json сохраняют архитектуру и initialization seed. Все640jobs завершились, source_unchanged=true. Job464 не прошёл capture proof; исходный pool state=failed,usable_captures=636, поскольку aggregate исключил всю группу из4seed. Обработчик отдельно проверил остальные639members, повторил только job464 с тем же seed1680048/model/task; первая retry прошла. `workspace/artifacts/aproc-v1-20261007/selected-captures.json` содержит640выбранных валидных записей. Исходная невалидная запись сохранена и исключена. Actor weights не менялись внутри партии; PPO чужих моделей/old captures не переиспользуется. Final test не запускается.

Полный аудит очереди: `workspace/artifacts/apool-v1-20261007/pool/queue-audit.json`.624повторных назначения, mean gap0.011075s,median0.003795s,p950.042786s,max1.168424s; пересечений jobs одного слота0. Пик16active jobs; занятость слотов98.908%, все16одновременно заняты97.761% времени окна jobs1940.158s. Это занятость заданиями с запуском/проверкой/очисткой, а не процент загрузки CPU/GPU и не непрерывное время gameplay. Tail очереди включён.

## Обработка корпусов и CUDA

После прерывания процесса между m5 и m6 добавлено явное `--resume`: проверяются frozen protocol, Python snapshots, выбранные640member receipts и completion seals готовых updates. Завершённые6моделей не обучаются повторно; незавершённый каталог m6 сохранён под `interrupted-m6-*`, затем сборка продолжена. Источник обучения остаётся собственным80episode корпусом каждой модели. Файл `resume-*.json` фиксирует пропущенные готовые модели и hash обработчика. Resume следует запускать только после подтверждения отсутствия прежнего процесса.

Агрегатор episode pool теперь сверяет exporter SHA каждого member и сохраняет проверенную копию `q2combat-export.exe` перед публикацией сводного manifest/report. Устаревший запрет MLP weapon head в зарегистрированном training launcher удалён после CUDA/native smoke. PowerShell syntax проверен; следующий live validation pool проверит публикацию сводных файлов на новых captures.

`scripts/process_combat_architecture_pool.py` обрабатывает завершённую партию: проверяет полный набор jobs и frozen receipts, повторяет невалидные captures отдельно, собирает собственный корпус каждой модели и выполняет CUDA PPO. Невалидные исходные попытки сохраняются, в обучение не входят. Экспорт запускается максимум в4потока, обновления моделей последовательно на GPU. Пул игровой симуляции не гарантирует непрерывную загрузку GPU между фазами сбора и обучения.

Полный путь проверен на32 диагностических боях: `workspace/artifacts/aproc-smoke-v1-20261007/report.json`, state=complete,8/8моделей получили по одному CUDA update и10actor steps. Eligible transitions по m0..m7:363,366,442,689,401,550,587,567. MLP64,attention64,attention128,GRU128 проверены с двумя initialization seeds, включая V6 weapon head. Диагностические веса не заменяют frozen priors полной партии640. Это проверка обработки и обновления, а не доказательство улучшения игрового качества.

Полная обработка завершена после возобновления: `workspace/artifacts/aproc-v1-20261007/report.json`, state=complete,8/8CUDA updates,80allocated episodes/model,10actor steps/model. Eligible transitions m0..m7:7359,7603,8204,9620,8077,8348,7798,6805. Веса и checkpoint seals сохранены отдельно; исходные priors не заменены. Равенство allocated episodes не означает равенство числа переходов или полного prior knowledge; общий FireBC distillation prior объявлен в protocol. Игровая эффективность после обновления ещё не оценена.

`scripts/prepare_combat_architecture_evaluation.py` готовит18парных вариантов:8models×before/after плюс FireBC и rules. Каждый получает одинаковые20families×4validation seeds offset24 и exact generated fixtures, всего1440battles,pool16/x2. Validation seed повторно используется для настройки; final test остаётся отложенным. Проверяются registry SHA, completion seals и равенство starts. Подготовка предварительно проверена на diagnostic smoke:18plans/72battles; их live pool запущен в `workspace/artifacts/aeval-plan-smoke-v1-20261008`. Полная evaluation root: `workspace/artifacts/aeval-v1-20261008`.

Live smoke завершён:72/72valid captures, source_unchanged=true; проверены18/18aggregate exporter binaries против manifest SHA. Исправленная публикация сводных корпусов подтверждена реальным запуском. Затем запущен основной pool1440battles в `workspace/artifacts/aeval-v1-20261008/pool`,16slots,ports34400..34415. До полного terminal report и анализа результаты эффективности не объявляются.
