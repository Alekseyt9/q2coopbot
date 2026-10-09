# Coarse/fine aiming: 10.10.2026

Общий контракт `combat_target_coarse_fine_aim_v1`, features V7/854, actor81. Работает через существующие MLP, GRU и temporal attention исполнители. Предыдущие45 outputs сохраняются;45..62 содержат девять пар fine yaw/pitch (none+8 observed targets),63..80 — девять пар coarse/fine logits. Сначала выбирается observed target, затем mode, затем conditional Gaussian latent. Coarse команда:180×tanh; fine:15×tanh. Движение, attack, vertical и masked weapon выбираются прежними outputs. Новый дискретный `sample.aim_mode` участвует в PPO likelihood; legacy policy отклоняет ненулевой mode. Кэш повторного recurrent кадра хранит тот же sample/action.

Миграция задаёт mode bias+5/−5, fine weights0: deterministic before выбирает coarse. CUDA аудит всех восьми MLP/GRU/attention priors подтверждает точное сохранение прежних actor/value outputs, finite likelihood/entropy, gradient routing только в выбранную пару и выбранные mode logits, отказ unavailable target и диапазон fine±15°. Это GPU проверка контракта, не численная проверка Go модели. Go companion и episode compiler собраны; Go модельные тесты не запускались по указанию пользователя.

## Обучение

Источник — сильный FireBC target-before из `target-refresh-v2-20261010`, не отвергнутая intent BC модель. Corpus80 own-policy train боёв/16 validation,20 train families; сжатый JSONL. Train/validation seeds разделены.100 epochs CUDA/RTX5070; Adam новый, PPO optimizer не возобновляется.

Все45 прежних head rows, encoder, recurrent/attention параметры вне новых output rows, value и log_std фиксированы. Обучаются только36 новых rows. Blaster unexecuted intercept queries: fine label при обоих углах≤15°, иначе coarse. Fine MSE нормирован на15°, mode cross entropy×0.1. Не добавлены guessed machinegun recoil labels. Выходы для machinegun могут обобщаться из blaster данных; это риск, который проверяет native trial. Нет утверждения об оптимальности firing/movement/target order.

| Метрика queries | До | После |
| --- | ---: | ---: |
| Train fine RMSE,° | 8.263 | 6.470 |
| Validation fine RMSE,° | 6.878 | 6.048 |
| Train mode accuracy | 61.47% | 60.80% |
| Validation mode accuracy | 49.58% | 55.13% |

Fine labels1481train/536validation; mode labels3844/1063. Fine RMSE — подмножество geometric queries, не hit rate и не общий RMSE предшествующих experiments. Сохранение весов подтверждено точными CUDA сравнениями. Parent SHA256 `8424becef2d7c1541cea96fc41f6b1970f58c0d70ae7128042275ccc6a70684e`; after `e85b86cea5d1751e39a9f3602a9555151c6f6a6cda6f7f7d7d5d3be508df91fb`.

## Native trial

Завершено64/64 paired боя: precision-before/precision-after/legacy FireBC/rules,4 validation families×4seeds. Native/source/binary proof проверен, ошибок нет. Existing harness,16 независимых инстансов,×2, очередь сразу заполняет освобождённый слот. Seed повторяется только в парных controls на одинаковых условиях; каждый member — отдельный процесс/прогон. Validation28 уже использована для разработки; final test отложен. Рабочий baseline не заменён.

| Вариант | Победы | Смерти | Средний урон нанесён / получен |
| --- | ---: | ---: | ---: |
| Precision before | 7/16 | 9 | 176.8 / 67.8 |
| Precision after | 7/16 | 9 | 180.6 / 67.5 |
| FireBC legacy | 7/16 | 9 | 184.3 / 67.2 |
| Rules | 8/16 | 5 | 116.9 / 45.5 |

Все16 before/after исходов совпали. Before81 воспроизвёл прежний45 parent по исходам, нанесённому/полученному урону и selected-target диагностике. After firing-frame applied-ray error вырос11.371°→13.649°, доля ошибок>10°41.10%→49.53%. Переключений цели по18. Ray диагностируется без intercept/recoil коррекции; это не hit rate. **Precision after не принят как улучшение.** Улучшение offline fine RMSE не перенеслось в бой; причина не установлена. Среди проверяемых объяснений — недостаточная точность выбора режима, ограниченность frozen encoder и изменённое распределение текущих наблюдений.

Следующий приоритет: разделить выбор режима и точность fine aim по weapon/дистанции/скорости; добавить подтверждённый blaster shot/hit feedback и собственные corrected trajectories, затем проверять firing timing совместно с aim. Для machinegun сначала нужны наблюдаемые burst-phase признаки и подтверждение recoil labels; нынешний blaster-query corpus не даёт основания заявлять обучение отдаче. Не расширять массовое обучение всех моделей до положительного paired результата сильного baseline.

## Уточнение: дистанция и параллакс

Distance/512 и relative position уже входят в каждый из восьми observed enemy slots; отсутствие дистанции в input не подтверждается. Однако aim queries до сих пор используют eye origin, а не фактический muzzle. `ConnectRequest` задаёт hand=2 (center), поэтому `P_ProjectSource` убирает боковые8units, но вертикальное смещение остаётся: blaster muzzle24forward, viewheight−8; machinegun0forward, viewheight−8. `aimfix` по умолчанию0; при включении engine дополнительно меняет forward по eye trace. Текущее значение конкретного server run отдельно не подтверждено. Не применять стандартную боковую компенсацию для right hand к center-hand боту.

До следующего обучения: подтвердить native muzzle/direction и server aimfix для выстрелов; исправить offline blaster intercept queries с учётом muzzle, зависящего от конечного угла, сохранив unknown masks. Затем отдельной абляцией дать aim head явные observed selected-slot признаки: дальность до aim point, обратную дальность с ограничением, angular bbox size и transverse angular velocity. Дистанция влияет и на параллакс, и на lead time, и на угловую скорость; сама угловая команда не должна произвольно умножаться на расстояние. Сравнить близкие/средние/дальние случаи на парных seeds. Все эти изменения пока план, не выполненный muzzle-aware training и не установленная причина предыдущей регрессии.

Артефакты: `workspace/artifacts/precision-head-v1-20261010/{report.json,checkpoint.pt,weights.json,cuda-architecture-audit.json}`; evaluation `workspace/artifacts/precision-eval-v1-20261010/`; процесс `workspace/build/precision-eval-v1-20261010-process.json`. Веса/бинарники остаются вне Git.
