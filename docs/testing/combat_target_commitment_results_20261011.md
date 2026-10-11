# Награда за устойчивость цели: первая закрытая оценка

Закрыты784 native боя без ошибок,336 stochastic CUDA cases/41374 rows и8 парных factor comparisons. Все28 семейств, по4 свежих confirmation сидов offset384 на контроллер, разные точки base3/ware2/jail1/city1. Episode/fixtures/сиды и native environment bindings одинаковы; weights/checkpoints и source pin проверены. Deterministic outcomes проверены нативно, без заявления о stochastic PPO replay. Обучение каждой ветки до этой оценки:112 собственных train боёв,10 actor/40 value steps на CUDA; общие начальные параметры и моменты, различается objective. Promotion отсутствует.

| Контроллер | Победы /112 | Ordinary /80 | Recovery /32 | Смерти | Полученный урон |
|---|---:|---:|---:|---:|---:|
| Control stochastic |27|24|3|77|8594|
| Cost stochastic |27|21|6|73|8067|
| Bonus stochastic |34|27|7|67|7822|
| Control deterministic |36|36|0|65|7323|
| Cost deterministic |33|33|0|64|6935|
| Bonus deterministic |33|33|0|66|7226|
| Rules |55|26|29|47|6467|

Stochastic bonus/control:13 прежних поражений стали победами,6 побед потеряны, итог+7/112 (+6,25 п.п.). Описательный family-bootstrap95% интервал [-0,89;13,39] п.п. включает0. Cost/control gains8/losses8. Deterministic bonus/control gains3/losses6, −3/112; bonus/cost gains4/losses4. Это один training seed на линию и один holdout; bootstrap28 семейств не устраняет общие карты и не является автоматическим significance/promotion gate.

## Поведение после release, первый life

| Метрика | Control stochastic | Cost stochastic | Bonus stochastic |
|---|---:|---:|---:|
| Смены цели /100 provider кадров |39,95|37,58|37,80|
| Смены, обе цели остаются clear /100 кадров |35,27|33,21|33,59|
| Applied yaw ≥30°, доля кадров |11,21%|9,75%|9,80%|
| Стояние при clear-shot противнике, доля кадров |19,49%|18,56%|18,35%|
| Attack commands без выбранной цели, доля attack frames |9,42%|17,21%|14,78%|
| Бои без attack command |12|12|11|
| Бои без native health damage монстрам |15|14|13|

Стабильность выбора/поворотов и стояние немного улучшились, но команды атаки без цели стали чаще. Это смешанный результат; уменьшение переключений не означает устранения лишних действий. Команды атаки не равны выстрелам/попаданиям, стояние и большой поворот не объявляются бесполезными автоматически. Короткие/длинные траектории меняют denominators, поэтому частоты и native победы сообщаются отдельно.

Deterministic bonus/control: смены9,79→8,98/100 кадров, большие yaw5,40→4,52%, stationary-clear37,13→38,85%; побед36→33. Recovery0/32 у всех deterministic моделей при29/32 у rules. Награда устойчивости цели не устранила проблему initial Grenades/weapon recovery; общий результат хуже правил. Ordinary bonus27/80 и deterministic control36/80 сами по себе не доказывают общей готовности к кампании/кооперативу.

Поведение пересчитано по правильному network release clock в audit v4. Ранний v3 сравнивал observation frame с native game_frame, пропуская2 первых кадра каждого боя. Native outcomes, обучение и CUDA likelihood/state checks не менялись. Старые sealed reports не перезаписаны; новый receipt `combat-target-commitment-network-clock-audit-v1-20261011/complete.json`. Во всех6×112 случаях release/counts/trace/server-log SHA совпали с независимым execution audit; добавлено224 provider кадра на контроллер. Таблица выше использует исправленный отсчёт. Summary запрещает смешивать v3 и v4.

Контекст targetless attack commands: control753, из них693 без текущих наблюдаемых enemies и60 при clear enemy; cost1480 =1418+62; bonus1202 =1130+72. У bonus94,0% таких команд возникают без текущего наблюдаемого enemy; ни в одной линии не обнаружен отдельный context с observed enemies, но без clear-shot. Это не доказывает отсутствие remembered threats или физические выстрелы в стены. Следующий анализ должен связывать этот контекст с actual fire/impact и weapon animation; простой общий штраф по attack flag недостаточно обоснован. Первые attack command delays (только бои с attack): control15,98/cost15,96/bonus17,80 network frames; нельзя смешивать их с native damage delay или временем setup/выбора оружия.

## Источники и продолжение

- `workspace/artifacts/combat-target-commitment-quality-analysis-v1-20261011/complete.json`, SHA `2baaaaf0d6f13fae4f2f7139dfca20ece72732fa1d1d0d0527b9598fa7fc0f64`.
- Native pool SHA `53863ddb1c9445a3aaf36cb7f44cb0a1e19802128f6ac4378f35b013b27c992a`.
- `combat-target-commitment-factor-analysis-v1-20261011/complete.json` и results.md:8 matched comparisons, ordinary/recovery, описательные интервалы.
- `combat-target-commitment-quality-analysis-v1-20261011/action-efficiency.json`:post-release first-life audits, capture/manifest/trace SHA join к native damage timing.

Первый из двух дополнительных train336 циклов уже запущен на fresh offset256; второй использует260. Награды, архитектура и learning rates сохраняются, checkpoint/Adam resume проверяется на CUDA. Затем отдельный smoke12 общего registry comparison runner и paired1456 before/after на новых confirmation448. [Условия эксперимента](combat_target_commitment_plan_20261011.md), [общий runner и очередь](combat_registry_model_comparison_20261011.md). Все будущие результаты ещё отсутствуют. При сохранении recovery провала следующий отдельный фактор — exploration weapon-head в наблюдаемом проблемном equipped state, с ordinary контролем; прежняя global temperature2 ухудшала обычные бои.

## Восстановление дополнительного обучения

Первый continuation pool закрыл336 валидных боёв. Обработка control завершила112 CUDA cases/15026 rows и merge, затем остановилась на NTFS-сжатии sequence.jsonl. Повторное сжатие прошло; hashes CUDA receipts, rollout/sequence каждого case, merged streams, plans/model bindings и pool повторно проверены без neural replay. Receipt `combat-target-commitment-continuation-v1-20261011/resume-reuse-proof.json` подтверждает336 native/112 CUDA reused без повторного обучения. Продолжение `resume-target-commitment-training-20261011.py` возобновило round1 с control checkpoint; control update и CUDA audits завершены, далее обрабатывается cost. Новые quality результаты ещё отсутствуют.

Compression helper теперь потоково вычисляет SHA и делает максимум3 попытки с сохранением текста ошибки; неизменность байтов проверяется при каждой попытке. Проверка transient retry и permanent failure прошла на subprocess mocks (`compression-retry-test-20261011/test-result.json`); реальное успешное сжатие восстановленного корпуса подтверждено отдельно.

## Плотность обучающего сигнала

Отдельный audit первого закрытого train корпуса связал38821 eligible rows с исходными available reward labels и сверил SHA112 reward streams на каждую ветку, score и сумму компонентов. В bonus positive hold credit встречается на283/12938 rows (2,19%), в76/112 боёв; A-B-A cost —237/12938 rows (1,83%), в58/112. В cost —237/12946 rows и58/112 боёв. На bonus4421 eligible rows удержание уже длилось≥3 кадров,892 rows имели selected-target damage; бонус требует обоих условий и остатка episode budget. Control не получает обоих компонентов.

Доля абсолютных величин компонентов награды: cost0,221%, bonus вместе с churn0,701%. Это масштаб reward labels, а не доля градиента/причинный эффект; малый сигнал может быть одной из причин слабого изменения поведения, но это гипотеза. Существующие objective/LR не меняются до paired cumulative evaluation. Receipt `combat-target-commitment-credit-audit-v1-20261011/complete.json`; анализ не выполнял relabeling, neural inference или обучение.

## Первый дополнительный цикл закрыт

Continuation round1:336 native/CUDA cases,45657 eligible rows (control15026, cost14863, bonus15768), три GPU updates. Независимо повторно сверены weights/checkpoint/report seals, CUDA full-checkpoint Adam clocks/consumed-rollout audits и input-use model bindings; по10 actor steps, LR groups[3e-5,3e-4]. Round2 native capture начался на fresh train offset260 после fixture/model SHA verification; pool16×2. Это новые учебные веса, не новая held-out статистика качества и не promotion.

Второй continuation native pool также закрыт:336/336, по112 на control/cost/bonus, ошибок0 и source_unchanged=true. Pool SHA `20800adb95f37f3dd307af2d076b8feb25158cf84b63ca50dce07a295e84640b`. Отдельно сверены все336 capture/manifest/seed/model bindings (`round-2/native-binding-verification.json`); coordinator перешёл к CUDA processing. Всего дополнительные native captures672 готовы; round2 GPU updates и последующая quality evaluation пока не завершены.

Round2 control GPU update закрыт и независимо сверены seals/Adam/input-use:13146 rows,10 actor steps, LR[3e-5,3e-4]. Продолжается processing cost. Credit audit round1 (после исходного обновления) также закрыт:45657 rows точно связаны с reward labels. Bonus313/15768 rows (1,99%),80/112 боёв; churn323 rows в68/112, абсолютная доля двух компонентов0,706%. Ранее bonus283/12938 rows (2,19%),76/112. Плотность сигнала остаётся низкой; новые сиды/собственные траектории и различная длина боёв запрещают считать эту разницу причинным эффектом награды или held-out улучшением. Receipt `combat-target-commitment-credit-audit-round1-v1-20261011/complete.json`. Инвентаризация закрытых capture streams обоих циклов не обнаружила крупных несжатых файлов, дополнительное сжатие не выполнялось.

## Контекст выбора цели у бонуса

Reward rows дополнительно разбиты по существующей availability mask целевой головы426:466:5 (8 наблюдаемых bbox slots, не число native monsters и не clear-shot count). Initial bonus:при multiple128/8476 rows (1,51%), one155/3469 (4,47%), none0/993. После первого обновления:multiple135/9200 (1,47%), one178/3703 (4,81%), none0/2865. Все237 initial/323 following churn rows приходятся на multiple. Соответственно178/313 (56,87%) положительных bonus rows следующего корпуса получены при одном доступном выборе. Это описывает плотность существующих меток, не доказывает причинную бесполезность бонуса. Receipt `target-credit-choice-context-20261011.json`:pinned existing rollout/reward SHA, полное покрытие eligible rows, без inference/relabel/training.

Гипотеза после закрытия текущей cumulative evaluation: отдельно проверить вариант productive-hold shaping только в состоянии реальной конкуренции нескольких наблюдаемых допустимых целей; сравнивать с текущим bonus на новых own-policy данных. Сначала offline calibration должна подтвердить, что критерий использует тот же observed contract и не награждает idle/недопустимые цели. Не менять objective нынешних frozen веток и не объявлять уменьшение stochastic переключений улучшением побед/прохождения.

Round2 cost GPU update и независимые seals/Adam/input-use проверки закрыты:13126 rows,10 actor steps. Продолжается бонусная ветка.

Offline competition-only calibration завершена на двух112-case корпусах с pinned full histories:original hold credit точно восстановлен (включая spent episode budget); простая фильтрация готовых bonus rows не использовалась. Initial:same-scale multi-only128 bonus rows/54 cases, sum3,996 вместо283/76/sum10,352. После initial update:135 rows/58 cases,sum3,688 вместо313/80/sum9,468. Удвоение coefficient/budget0,004/0,4 оставило те же128/135 положительных кадров и дало7,992/7,376 суммы. Это лишь калибровка меток на неизменных историях, не обучение/игровой эффект; новые trajectories нужны для честного опыта. Receipt `target-multi-choice-credit-calibration-20261011.json`. Конкурентное условие повышает смысловую направленность, но делает сигнал ещё реже; нельзя предполагать, что оно автоматически улучшит выбор целей.

## Дополнительное обучение полностью закрыто

Continuation complete:672 native/CUDA cases, два цикла и6 GPU updates (3 ветки×2),85757 eligible rows (45657+40100). Все6 weights/checkpoint/report seals и CUDA Adam/consumed-rollout/input-use bindings независимо сверены после закрытия. В round2:control13146/cost13126/bonus13828 rows, каждый10 actor steps; objective/архитектура/LR не менялись. После первого reward update каждая ветка теперь имеет ещё два собственных обновления; качество свежих весов не объявляется по train outcomes.

Smoke12 общего comparison driver начал native evaluation на старых известных control stochastic/deterministic и rules, offset480 (это runtime smoke driver, не оценка новых моделей). После полного smoke seal автоматически запускается cumulative quality1456 на13 контроллерах со свежими confirmation448; в ней сравниваются настоящие before/after веса трёх наград в обоих режимах и rules. Promotion отсутствует.

Smoke driver полностью закрыт:12/12 native,4 CUDA cases/454 rows и полный reporter CLI результат, original control stochastic/deterministic/rules на одном base3 family. Источники/seals/processing SHA сверены; это ограниченная runtime acceptance общего driver, не качество новых weights. Большая cumulative evaluation1456 уже начала native capture на confirmation448:13 variants,28 families,112 cases each,12 learned model bindings, все fixture/model/plan SHA независимо сверены. Native pool16×2. Её новые outcomes/CUDA replay ещё не закрыты. Source receipts: `combat-registry-comparison-driver-smoke-v4-20261011/complete.json`, `combat-registry-comparison-driver-smoke-v4-20261011-report/complete.json`, `combat-target-commitment-cumulative-quality-v2-20261011/fixture-proof.json`.

12.10, отдельный экспорт кандидатов сохранён в `models/combat/target-commitment-candidates-20261012/`:control/cost/bonus после third reward update, все CUDA closed/audited. Три byte-identical inference weights16,559MiB; ZIP3,858MiB,5 entries, CRC и SHA каждого содержимого совпали. Manifest фиксирует source checkpoint/report/complete hashes; сами Adam checkpoints остаются в original artifact folders. Экспорт явно помечен experimental/evaluation pending, не promotion; это три reward variants одной Temporal Attention архитектуры, не три разные архитектуры. Старый latest-20261010 не изменён, Git commit/push не выполнялись. Новая оценка ещё идёт.
