# Продолжительное сравнение MLP, GRU и attention

06.10.2026. Протокол зафиксирован до запуска. Результаты добавляются после проверки capture, rollout и checkpoint lineage.

Первый [пилот](learned_combat_architecture_v1.md) дал новым блокам только два обновления. Его результат не позволяет считать MLP лучшей архитектурой: общая MLP база уже обучена, GRU и attention инициализированы заново. В этом опыте все четыре варианта снова начинают с одного obstacle reference и нового Adam; прежние два обновления не наследуются.

Фиксированный бюджет: 20 PPO updates на вариант, по четыре свежих Mixed skill1 эпизода на update, не более 10 actor steps и ровно 40 critic steps. Всего 80 training episodes на вариант, 320 на четыре варианта. Это в десять раз больше первого пилота; достаточность бюджета ещё предстоит оценить по кривым качества. Новые residual projections исходно нулевые, начальные действия всех вариантов должны совпасть.

Состав вариантов: исходная MLP; residual GRU64 с BPTT32; temporal causal attention64 с четырьмя головами и окном32; current-entity attention64 с четырьмя головами и масками присутствия. Параметры не уравнены. Все модели используют v4/810, прежние восемь action logits, Blaster, maneuver-v4 reward и stock HP. Выбор оружия в этот опыт не входит. Anchor и исторический composition bank имеют постоянный weight1; банк без историй проверяет retention только при нулевой памяти.

Промежуточные оценки после updates5 и10: Mixed skill1, Solo skill1, Solo skill3, четыре эпизода на условие. Их seeds28100–28103 /28200–28203 /28300–28303 повторяются между checkpoints для кривой. Они не используются для обучения, ранней остановки, подбора настроек или выбора checkpoint. Финал после update20 оценивается на новых seeds28400–28403 /28500–28503 /28600–28603; на тех же финальных условиях оценивается исходный reference. Training seeds28000–28079, парные между архитектурами и отдельные для каждого worker/эпизода.

Во всех batch работают четыре независимых server/client instances x2, synchronous native harness, horizon300 и release100. Обновления и initialization выполняются на CUDA RTX5070, без CPU fallback. Инстансы собирают опыт параллельно; GPU updates разных архитектур идут последовательно. Порядок обучения чередуется по update между четырьмя вариантами. PAK использует существующий общий runtime cache.

Всего запланировано 476 captures в119 batches: 320 training,96 промежуточных evaluation и60 финальных evaluation. Качество сравнивается по uninterrupted full wins, kills по классам, deaths, incoming, eligible consumed kill transitions; отдельно учитываются реальные accepted actor steps, rows и p95 inference с guards. Не выбирать архитектуру по одному четырёхэпизодному результату. При parity/provenance ошибке цикл останавливается, исходные данные сохраняются.

Root: `workspace/artifacts/combat-architecture-v2-20261006`. `protocol.json` содержит hashes исходных model/config/reward/bank; `progress.json` показывает этап. Checkpoints, native traces и export сохраняются по update. Финальный audit проверяет все updates, Adam counters/resume, модель каждого evaluation, sequence replay, eligible coverage и независимое GAE. Live policy не меняется.

Запуск: `scripts/run_combat_architecture_compare.ps1 -Iterations 20 -CurveIterations 5,10 -Seed 28000 -Port 34200 -OutputRoot workspace/artifacts/combat-architecture-v2-20261006`.

Статус: цикл выполняется. До финального audit итоговое улучшение не подтверждено.

Промежуточная оценка после update5 прошла native/reset/seed/kill ownership audit; receipt `curve-partial-audit.json`. Это monitor seeds, не финальная отложенная оценка. Каждая ячейка: uninterrupted wins/4, native kills, deaths, incoming первой жизни.

| Вариант | Mixed skill1 | Solo skill1 | Solo skill3 |
|---|---|---|---|
| MLP | 2/4, 6, 2, 324 | 4/4, 4, 0, 176 | 4/4, 4, 0, 278 |
| GRU64 | 3/4, 6, 1, 340 | 4/4, 4, 0, 176 | 4/4, 4, 0, 278 |
| Temporal attention | 4/4, 8, 0, 313 | 4/4, 4, 0, 208 | 4/4, 4, 0, 300 |
| Entity attention | 3/4, 7, 1, 324 | 4/4, 4, 0, 198 | 4/4, 4, 0, 300 |

Перестановка наблюдаемого качества относительно первого двух-update пилота показывает, почему нельзя объявлять архитектуру победителем после короткого прогрева. Сиды между v1 и v2 отличаются; сравнивать цифры двух опытов как чистый эффект добавленных updates нельзя. Здесь все четыре варианта имеют одинаковый новый бюджет, а итоговые20 updates оцениваются на отдельных final seeds. Четыре monitor episodes недостаточны для статистического превосходства; budget не меняется по этой таблице.
