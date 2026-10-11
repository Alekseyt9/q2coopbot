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
