# Общая пространственная ветка прицеливания, 2026-10-10

Расстояние уже присутствует во входах каждого наблюдаемого противника. Новая ветка даёт ему прямой путь к точной доводке: observed distance, относительная позиция/скорость, bbox, угловые ошибки, обратная дальность, угловые размеры и относительная угловая скорость. Она также получает текущую собственную скорость и планируемые средние forward/side команды старого actor. Дальность и угловой размер связаны с точностью и параллаксом; обучение использует геометрическую цель от центрального muzzle бластера, а не от камеры.

Общая для восьми enemy slots сеть 119→64→32→4 имеет 9892 обучаемых параметра. Выходы корректируют fine yaw/pitch и coarse/fine mode. Старый encoder, первые45 выходов, value и std заморожены. Базовый actor сохраняет доступ к геометрии мира; новая ветка не получает отдельные ray probes. Фактическое sampled движение не является её входом: используется планируемое среднее. Будущие позиции и результаты native выстрелов не подаются в модель.

Два варианта обучены на CUDA RTX5070, по500 эпох на корпусе80 train боёв сильного FireBC prior:

| Метрика validation | Instant center muzzle | Postmove teacher |
| --- | ---: | ---: |
| Fine query RMSE, до→после | 8.083→0.970° | 8.241→2.407° |
| Mode accuracy, до→после | 61.34→94.17% | 63.59→90.59% |

Postmove использует следующий client displacement только для offline labels; признаки остаются прежними. Это factual движение parent policy, а не предсказание последствий произвольного нового действия. Ни один вариант не обучался по корректным recoil labels machinegun: перенос на него проверяется отдельно.

Завершено80/80 native прогонов: пять вариантов, каждый на16 одинаковых условиях validation28 (четыре семейства × четыре seeds). Пул16, ×2, автоматический refill. Строгая проверка source/native/binary/member proof завершилась успешно.

| Вариант | Победы | Смерти | Попадания бластером в живого монстра | Applied firing-ray error |
| --- | ---: | ---: | ---: | ---: |
| Before81 | 7/16 | 9 | 100/142 =70.42% | 11.363° |
| Spatial instant | 10/16 | 6 | 89/109 =81.65% | 8.352° |
| FireBC | 7/16 | 9 | 100/142 =70.42% | target не объявлен |
| Обычные правила | 8/16 | 5 | 94/94 =100% | target не объявлен |
| Spatial postmove | 10/16 | 6 | 98/120 =81.67% | 10.410° |

Instant получил четыре новые победы и потерял одну относительно before. Postmove получил три новые победы без потерянных относительно FireBC. Unknown projectile outcomes0; знаменатель — реальные native blaster shots через corrected ordered-window join, не firing frames. Ray error — диагностический угол к наблюдаемой цели без lead/recoil, не hit rate.

Разбивка по зарегистрированному стартовому loadout уточняет перенос: в blaster условиях все три learned варианта имеют4/8 побед (rules4/8), в machinegun условиях spatial instant/postmove6/8 против3/8 FireBC и4/8 rules. Все дополнительные победы относятся к machinegun условиям. Улучшение native blaster hit fraction пока не увеличило число побед на blaster сценах. Loadout обозначает начальный инвентарь, а не оружие каждого выстрела. Новый `quality-strata.json` повторно сверяет победы/смерти/урон с sealed native report каждого участника, затем группирует по loadout/map/site.

Instant выбирает fine mode для бластера на187/198 medium frames и243/344 near frames; postmove —172/182 и268/408 соответственно. Это объявленный candidate mode; guards могут менять применённую команду. В этих данных нет достаточного far subset для вывода о большой дальности.

CUDA аудит всех восьми базовых архитектур проверил миграцию, shared branch, recurrent/attention context, likelihood и gradients. Это не обучение всех восьми новых моделей. Для двух обученных вариантов отдельно проверено точное восстановление checkpoint actor и выходов из JSON на CUDA. Go/CPU численные сравнения не запускались.

Сравнение сохранённых MLP PPO checkpoint параметров с JSON в `ppo_combat.restore_checkpoint` также переведено на CUDA. Проверен реальный исторический `aproc-v1-20261007/m0/update` checkpoint (один update,10 actor steps): совпадающие параметры приняты, намеренно изменённый actor отклонён. Receipt: `spatial-muzzle-v1-20261010/ppo-resume-cuda-audit.json`. Это проверка загрузки прежнего PPO checkpoint, не новый spatial PPO update и не доказательство сохранения всех optimizer/RNG свойств.

Это положительный пилот, не доказательство общего превосходства: validation28 ранее использовался для разработки. Правила пока имеют лучшую точность бластера и меньше смертей. Следующий запуск —400 боёв на20 семействах, validation offset24, обе spatial модели, before, FireBC и правила. Эти условия не входили в текущие train/validation query labels; прежние эксперименты могли использовать их как development checks. Final test остаётся отложенным.

## Подготовка свежего PPO корпуса

После расширенной оценки следующий шаг — собственные stochastic trajectories, затем native-reward PPO на CUDA. `scripts/prepare_combat_spatial_ppo_pool.py` подготовил и проверил через существующий pool DryRun160 боёв: две модели ×20 семейств ×4 seeds, train offset108. Это новые seeds относительно исходного query корпуса (train104..107 и validation28..31). Замороженные stochastic копии, модели и планы перечислены в `workspace/artifacts/spatial-ppo-capture-v1-20261010/{models,plans,preparation}.json`. Фактический сбор и PPO update на момент подготовки не запускались.

Начальный config использует прежнюю pinned reward recoil-v5, actor_lr0.00003, target_kl0.005. BC checkpoint не является PPO optimizer: первая итерация начинает свежий optimizer, последующие обязаны возобновлять PPO checkpoint. В PPO обновляется весь actor/value/std, включая spatial branch; это отдельный эксперимент от замороженного parent в BC. Target-aware retention пока не реализован, поэтому retention/bank weights0. Используется существующий `process_combat_architecture_pool.py`, без второго харнеса. Новый сбор запускается после авторитетного завершения текущего400-battle процесса, затем после CUDA update требуется новая paired validation; promotion заранее не выполняется.

Запущен последовательный coordinator `scripts/run_combat_spatial_ppo.py`: он держит Windows process handle текущей оценки, ждёт его завершения, требует sealed complete progress/quality/member proof, затем использует тот же16-slot pool и существующий CUDA processor. Состояние `spatial-ppo-capture-v1-20261010/execution.json`, process receipt `workspace/build/spatial-ppo-v1-20261010-process.json`; PID16712 при запуске. На момент проверки coordinator жив и находится в `waiting_for_evaluation`, оценки PID24636 ещё выполняются. Сбор160/PPO ещё не начат. При ошибке coordinator сохраняет failed и частичные артефакты, не объявляет обучение завершённым. После двух updates обязательна новая paired native оценка; automatic promotion отсутствует.

Следующая native проверка также поставлена в очередь (`scripts/run_combat_spatial_ppo_evaluation.py`, PID26668 при запуске). Она ждёт handle coordinator16712 и требует его complete плюс hash/complete seals обоих CUDA updates и связь behavior weights с каждой исходной моделью. Затем шесть вариантов — instant/postmove до и после PPO, FireBC, rules — проходят480 боёв на20 семействах ×4 validation seeds, offset24. Это повторно использованные development условия, не final test. Отчёты включают sealed member proof, wins/deaths/damage по каждому loadout/map/site, observed selected-target ray diagnostic, actual blaster shots/hits и declared aim modes. На момент постановки очередь находится в `waiting_for_ppo`; новая оценка ещё не запускала серверы. Root `workspace/artifacts/spatial-ppo-eval-v1-20261010/`, process receipt `workspace/build/spatial-ppo-eval-v1-20261010-process.json`. Ошибка upstream не считается окончанием обучения и блокирует эту оценку с явным failed, сохраняя артефакты.

Артефакты: `workspace/artifacts/spatial-muzzle-v1-20261010/`, `spatial-postmove-v1-20261010/`, `spatial-eval-v1-20261010/{quality-report,selected-target-aim,blaster-projectile-hits,aim-modes}.json`, `recovery/verified-members.json`. Расширенная оценка: `spatial-broad-eval-v2-20261010/`. Веса не добавляются в Git.
