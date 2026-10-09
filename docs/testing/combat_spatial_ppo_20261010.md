# Spatial CUDA PPO после геометрического BC, 2026-10-10

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
