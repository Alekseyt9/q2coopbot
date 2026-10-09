# Spatial CUDA PPO после геометрического BC, 2026-10-10

## CUDA bootstrap prototype

`scripts/combat_cuda_bootstrap.py` и `scripts/audit_combat_cuda_bootstrap.py` проверяют расчёт next critic value на CUDA по реальным последовательностям исходных temporal-attention моделей. Instant: 5788 проверенных строк из5877, max error0.000010014; Postmove: 6117 из6206, max error0.000018597. В каждой выборке89 boundary/reset/handoff строк исключены. Receipts: `workspace/artifacts/spatial-ppo-process-v2-20261010/{instant,postmove}/cuda-bootstrap-prototype.json`. Reference — ранее сохранённые значения; нового Go исполнения сети в этом аудите нет.

Это прототип, не замена экспортёра: next features пока взяты из соседнего provider context. Для production нужны точные Step.Next features, reset и bootstrap-zero метаданные от native exporter, CUDA проверка всех границ и остальных архитектур. Native command/reward/source proofs должны сохраниться. До завершения текущей480-battle оценки Go/PS source не меняется. Время GPU вычисления прототипа не является измеренным ускорением всего pipeline.

Дополнительный `audit_combat_cuda_bootstrap_boundaries.py` прошёл на обеих исходных temporal-attention critic моделях:42 real-prefix locations Instant и40 Postmove. Synthetic next features сравнивались с независимым full-prefix replay на CUDA при продолжении, empty-prefix reset и смешанном reset батче, в том числе вокруг window32 и конца контекста. Максимальная ошибка0.000003100 для обеих моделей; изменение будущего padding не повлияло на результат. Receipts: `{instant,postmove}/cuda-bootstrap-boundaries.json` внутри processing root. Это подтверждает вычислительную обработку reset/window для этих двух моделей; реальные Step.Next/reset/handoff метаданные и другие архитектуры этим не проверены. Новых Go model calculations, CPU model tests и training updates нет.

В действующем харнесе обнаружен отдельный сборочный overhead: каждый single-episode worker вызывает два `go build` в `run_learned_combat_baseline.ps1` для client/exporter. При текущем запуске CPU занят примерно95..100%, число непосредственно работающих native instances меняется, поэтому низкая GPU загрузка во время сбора не означает свободный общий pipeline. Это source evidence, не измеренная доля времени. После закрытия fingerprint-bound пулов стоит подготовить один immutable hash-pinned bundle бинарников на source fingerprint и переиспользовать его, сохранив per-capture binary/native/source proof. Такой bundle пока не реализован; активный harness не изменён.

## Дистанция выбранной цели

Расстояние уже присутствует в observation для каждого противника и прямо входит в shared spatial branch вместе с inverse range, угловыми размерами и относительной угловой скоростью. Геометрические BC labels учитывают центральный muzzle бластера; наличие признаков не доказывает освоение параллакса моделью.

`scripts/report_combat_aim_distance.py` разбирает только завершённые оценки с verified-members и quality seals. Диапазоны согласованы с прежним muzzle-query audit: near до128, medium128..512, far свыше512 world units. Отдельно all/firing provider frames для явно выбранной наблюдаемой цели. Результат `selected-target-distance.json` — геометрия applied ray к observed bbox, без ballistic lead/recoil/muzzle correction; это не bullet hit rate. Rules без declared target не подменяются ближайшим противником. План следующего сбора — проверить представленность дальних условий перед добавлением distance-balanced registry curriculum; изменение реестра и повторное обучение пока не выполнены.

Обе spatial temporal-attention модели получили первый native-reward PPO update на GPU RTX5070. Это обновление всего actor/value/std, включая spatial branch, на собственных stochastic игровых траекториях; не дополнительное BC по геометрическим подсказкам. По80 train108..111 боёв на20 семейств для каждой модели. Reward/config SHA pinned, actor_lr0.00003, target_kl0.005, target-aware retention/bank weights0. Успех боя ещё не оценён.

| Модель | Eligible transitions | Принятые actor steps | Approx KL | CUDA old log-prob error | Update time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instant | 5877 | 10 | 0.00499901 | 0.00009453 | 2.52s |
| Postmove | 6206 | 10 | 0.00485874 | 0.00008345 | 2.07s |

Update time — время оптимизации из trainer report, без сбора, native proof export, подготовки и проверки данных. Actor95946 параметров, critic75650. Torch2.10.0+cu128, CUDA обучение. Model quality не выводится из времени, KL, loss или количества шагов.

Первые160 попыток содержали16 невалидных member captures: первая волна завершилась по60s wall лимиту раньше300 game frames. Source/native proof144 остальных сохранён; причина замедления не установлена. Первый план одиночных retries отвергнут до запуска серверов из-за frozen4-seed contract.64/64 повторных боёв через16-slot pool завершены без ошибок; заменены только16 failed members с проверкой прежних seed/fixture/model/source. Исходные failed и retry receipts сохранены. Recovered native corpus содержит160 подтверждённых members, затем Go exporter сформировал frozen rollout/sequence для CUDA trainer. Native proof и legacy exporter работают на CPU; обучение и новые checkpoint numerical audits — на CUDA. Отдельные Go model tests не запускались.

После sealed update Instant processor остановился на отсутствующем метаполе `initialization_seed`. Исправлен orchestration driver: неизвестный seed записывается null, не выдумывается. Resume восстанавливает verified contiguous prefix sealed updates, включая окно между durable complete.json и summary write. Уже обученный Instant не обучался второй раз. Проверены triple hashes, behavior model, update counter и rollout hash. Сохранены прежние report/execution в before-resume файлах, trainer snapshots и protocol inputs не менялись. Затем Postmove завершён теми же frozen CUDA trainers. Это возобновление processing после прерывания, не новый PPO update из прежнего optimizer на другой rollout.

Для обоих новых checkpoint отдельный CUDA аудит восстановил actor/value/std и Adam state, сравнил параметры и raw outputs точно на16 настоящих sequence feature rows. RNG continuation и качество поведения этим аудитом не доказаны. Receipts: `instant/update/checkpoint-cuda-audit.json`, `postmove/update/checkpoint-cuda-audit.json`. Trainer consumed rollout counters — по одному на модель.

| Артефакт | SHA256 |
| --- | --- |
| Instant weights | 26c34b8f4552560c8db3778956d01d3761de61821993f9f763ef17966e996a03 |
| Postmove weights | b9479af1aac3faee02002a34ab8de1eaebd878245e30d9da685f387d74f541f9 |
| Processing report | f76d1d4bbf4115e42b2f54bff007c842b3b6f3b711b5da4314e74d5015ee5d55 |

Запущена оценка `workspace/artifacts/spatial-ppo-eval-v3-20261010/`:480 боёв, шесть вариантов (оба BC parents до/после PPO, FireBC, rules),20 семейств ×4 validation seeds, offset24,16slots ×2. Это development условия, ранее использованные в400-eval, не final test. Результаты в момент запуска отсутствуют, baseline не заменён. Failed eval-v1/v2 остановились из-за незавершённого upstream; сохранены отдельно.

Processing root `workspace/artifacts/spatial-ppo-process-v2-20261010/`; recovered capture root `spatial-ppo-repair-v2-20261010/`; process receipts в `workspace/build/`. Связанные результаты: [широкая BC оценка](combat_spatial_broad_evaluation_20261010.md), [архитектура spatial branch](combat_spatial_aim_20261010.md).
