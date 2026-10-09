# Результаты парного сравнения архитектур, 2026-10-09

Завершены и проверены все 1440 боёв: 18 вариантов, по 80 боёв на одинаковых 20 семействах × 4 validation seeds. Native captures, sid/dispatch/frame proofs, состав и стартовые позиции подтверждены. Каждая новая модель до этого получила один CUDA PPO update на собственных 80 train боях; бюджеты эпизодов одинаковы, число переходов различается.

Лучший новый кандидат после PPO — MLP64, initialization seed 20261008: 19/80 побед. FireBC получил 50/80, обычные rules — 70/80. Ни одна новая архитектура не превзошла эти baselines. Новые checkpoints не продвигаются вместо FireBC. Увеличение сети само по себе не дало выигрыша в этом эксперименте.

## Победы до и после CUDA PPO

| Модель | Архитектура | Initialization seed | До PPO | После PPO | Изменение | Новые / потерянные победы |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| m0 | MLP64 | 20261007 | 6/80 | 9/80 | +3 | 5 / 2 |
| m1 | Temporal Attention64, 4 heads | 20261007 | 5/80 | 2/80 | −3 | 0 / 3 |
| m2 | Temporal Attention128, 4 heads | 20261007 | 2/80 | 2/80 | 0 | 1 / 1 |
| m3 | GRU128 | 20261007 | 2/80 | 2/80 | 0 | 0 / 0 |
| m4 | MLP64 | 20261008 | 14/80 | 19/80 | +5 | 6 / 1 |
| m5 | Temporal Attention64, 4 heads | 20261008 | 9/80 | 5/80 | −4 | 0 / 4 |
| m6 | Temporal Attention128, 4 heads | 20261008 | 5/80 | 3/80 | −2 | 0 / 2 |
| m7 | GRU128 | 20261008 | 9/80 | 10/80 | +1 | 3 / 2 |
| Baseline | FireBC, исходная temporal attention policy | — | 50/80 | — | — | — |
| Baseline | Обычные rules | — | 70/80 | — | — | — |

«Новые / потерянные» считаются попарно по episode ID и engine seed. Победа требует проверенного native goal-stop; валидный capture проигрыша не считается победой. Temporal attention и GRU здесь отдельные архитектуры, не совместная attention+GRU.

## Ограничение исходной инициализации

Новые модели уже ДО PPO получили только 2–14 побед, тогда как FireBC — 50. Одинаковый бюджет обучения не означает одинаковую исходную силу: перенос поведения через дистилляцию не сохранил качество учителя в живой игре. Поэтому эксперимент показывает качество именно этих стартов и одного update, а не общую непригодность attention или GRU.

В CUDA distillation report MLP64 seed 20261008 после 300 epochs: validation movement RMSE 0,3301, aim RMSE 6,388° и fire probability RMSE 0,0963. Эти ошибки согласуются с неполным переносом действий, но сами по себе не доказывают единственную причину провала. Следует отдельно проверить heads, temporal/reset semantics и распределение посещаемых состояний.

Следующий порядок: (1) CUDA-диагностика переноса действий FireBC→новые модели на сохранённом корпусе; (2) сохранить FireBC как рабочую learned baseline и добавить контроль без переинициализации сети; (3) исправить перенос и проверить исходное поведение короткими парными живыми прогонами; (4) продолжать PPO только с сопоставимых стартов; (5) повторить сравнение, затем использовать нетронутый final-test split. GPU-only обучение сохраняется. Этот validation cohort уже использовался для настройки и не является независимым final test.

## Восстановление анализа после сбоя агрегации

Все 1440 job receipts завершились без ошибок, но исходный агрегатор остановился на несовпадении SHA экспортёра. Причина: первые бинарники собраны с `vcs.modified=false`, последующие — `true`, после изменения документации; mod version получил `+dirty`. Исходные Go/PowerShell и native source fingerprints одинаковы.

`verify_combat_evaluation_members.py` проверил все actual exporter/client SHA, frozen weights/plan/registry/reward, runtime file SHA, исходные source records и native proofs. Обнаружены четыре binary hashes: две версии metadata для клиента и две для экспортёра. `go version -m` совпадает по compiler/settings/revision/time и отличается только dirty stamp. Captures и их manifests не переписаны; сохранён исходный `pool.stderr`, общего успешного pool report не создавали задним числом.

Итог рассчитан непосредственно по проверенным individual captures. Отдельный proof: `workspace/artifacts/aeval-v2-20261009/recovery/verified-members.json`. Аналитический report state=complete, episodes=1440. Для будущих прогонов `run_learned_combat_baseline.ps1` использует `go build -buildvcs=false`; native source/binary provenance по-прежнему записывается отдельно. Проверена сборка экспортёра с отключённым VCS stamp; Go model parity/tests не запускались, нового обучения в этом анализе не было.

Полные метрики, deaths/damage, family breakdown и per-episode receipts: `workspace/artifacts/aeval-v2-20261009/quality-report.json`, `quality-episodes.json`, `quality-report.md`. По всей матрице модели в этом эксперименте слабее правил.
