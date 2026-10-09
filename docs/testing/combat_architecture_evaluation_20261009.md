# Возобновление сравнения архитектур, 2026-10-09

Завершённый анализ: [все результаты и проверка 1440 боёв](combat_architecture_results_20261009.md). Исходная групповая агрегация остановилась из-за VCS dirty metadata в бинарниках; анализ восстановлен отдельной проверкой individual captures. Новые модели получили 2–19/80 побед после PPO, FireBC 50/80, rules 70/80. Ни один новый checkpoint не продвинут.

Все 8 CUDA updates из `aproc-v1-20261007` сохранены; каждой модели выделено 80 собственных боёв. Перед новым сравнением проверены completion seals весов, checkpoint и отчёта каждого update.

Прежнее сравнение `aeval-v1-20261008` было прервано: найдено 84 завершённых job receipts, общего terminal report нет. После добавления автоматического сжатия и общего AAS хранилища fingerprint харнеса изменился. Старые captures сохранены, в новый cohort не включаются.

Новый cohort: `workspace/artifacts/aeval-v2-20261009`. `rebind_combat_architecture_evaluation.py` проверил исходные plan/weights/registry/reward SHA и создал новые планы, сохранив байты deterministic weights и точные условия всех сцен: состав, позиции, здоровье, оружие, skill, map geometry и seeds. Обновлены только output paths и привязка к текущему runner. Не проводилось новое обучение или изменение action/reward.

Сравнение: 18 вариантов × 20 семейств × 4 validation seeds = 1440 боёв, 16 независимых слотов, timescale 2, порты 34400–34415. Освободившийся слот получает следующее задание очереди. Восемь моделей сравниваются до/после CUDA PPO; отдельно FireBC и rules. Веса и сцены одинаковы с предыдущим планом. Validation seeds уже использовались для настройки, поэтому это не финальный независимый test.

| Модели | Архитектура | Initialization seeds |
| --- | --- | --- |
| m0, m4 | MLP64 | 20261007, 20261008 |
| m1, m5 | Attention64 | 20261007, 20261008 |
| m2, m6 | Attention128 | 20261007, 20261008 |
| m3, m7 | GRU128 | 20261007, 20261008 |

`run_combat_architecture_evaluation.ps1` запущен скрытым самостоятельным процессом. PID receipt: `process.json`; состояние: `progress.json`; stderr/stdout: `pool.stderr`, `pool.stdout`; individual receipts: `pool/jobs/job-*-result.json`. Не менять Go/PowerShell источники во время capture: source fingerprint проверяется на завершении.

После успешного terminal pool report автоматически запускается `report_combat_architecture_evaluation.py`. Он проверяет native capture/dispatch/seed/frame proofs, generated starts, exporter SHA, общий fingerprint и полноту 80 боёв на вариант. Отчёт содержит победы с verified goal-stop, смерти в первой жизни, native damage/kills, а также попарно приобретённые/потерянные победы before→after. Полнота и эффективность различаются: валидный проигрыш учитывается как проигрыш.

Генератор итогового отчёта проверен на предыдущем завершённом smoke cohort: 72/72 captures обработаны успешно. Эти 72 боя проверяют путь расчёта; не используются вместо основного сравнения. Финальные файлы нового cohort: `quality-report.json`, `quality-episodes.json`, `quality-report.md`. До их успешного создания результаты качества нового cohort не объявляются.

Первая живая проверка нового запуска: 47 завершённых боёв, 47/47 native proof валидны, один общий source fingerprint, текущие источники ему соответствуют. Процесс comparison продолжает работу. Это промежуточная проверка запуска, не полный итог 1440 боёв. Receipt: `live-audit.json`.
