# Fire BC → mixed PPO, 2026-10-07

Раздельные BC вмешательства выявили полезную ветку: только fire head улучшает Update29 с36/80 до50/80 на общей validation offset20. Aim-only даёт24/80. Обновления encoder/attention и остальных action heads в этих двух контролях запрещены и проверены на точное совпадение. [Полная таблица BC](combat_sequence_bc_20261007.md).

Fire-only добавила17 парных побед, потеряла3, сохранила33;27 исходных проигрышей остались проигрышами. Blaster18/40→26/40, Machinegun18/40→24/40. Native goals подтверждены, infighting не включался. Это результат выбора на validation, не независимый final-test и не доказательство превосходства rules в кампании.

Следующий этап выполняется через существующий training runner: `scripts/scenarios/combat-training-suites/mixed-fire-bc-resume-v1.json`. Модель fire29, все20 семейств, train offset40,4 боя/семейство, один mixed PPO update на CUDA. Policy получает только собственные новые stochastic transitions; BC observations и старые captures не объявляются on-policy PPO опытом этой ветки. Начальный actor optimizer свежий после BC, critic/log_std и PPO counters сохранены. Resume начинается с29 completed updates и279 actor steps.

Root активного эксперимента: `workspace/artifacts/fire-bc-rl-v2-20261007`. Pool16, x2. Оценка before/after: новый validation offset24,4 боя/семейство,160 deterministic captures. Подготовлен отдельный Update29 reference на том же offset24; его80 captures выполнятся после освобождения пула. Все пары generated fixtures будут сверены. Ни один из этих validation cohorts не является final-test.

## Исправление Windows runtime paths

Первая попытка `workspace/artifacts/combat-mixed-fire-bc-resume-v1-20261007` остановилась до PPO:40/80 usable captures. Для длинных путей `Install-RuntimeImmutable` добавлял к AAS filename GUID и `.tmp`; временный путь превышал MAX_PATH. Ошибка hard-link ошибочно переходила к32 shared shards. Это сбой подготовки runtime, не проигрыш модели; эта неполная партия не использована для обучения.

В `scripts/prepare_runtime.ps1` временное имя создаётся в том же каталоге как самостоятельный `.r<GUID>.tmp`; сохранена атомарная замена и immutable hash verification. Добавлен ранний отказ для overlong temporary/hard-link path, чтобы такая ошибка не порождала32 копии shared shards. Архивы по-прежнему hard-linked; private PAK копии не добавлены.

Проверен реальный Windows boundary: target229 characters, старый temporary266; новая установка и повторная атомарная замена прошли с SHA equality. Корень нового эксперимента и model id сокращены. Новая полная партия собирается с теми же заранее запланированными train seeds; повторённые captures старой попытки не считаются дополнительным PPO опытом. Старые отчёты сохранены для диагностики.

`go test ./cmd/... ./internal/...` прошёл. CUDA обучение двух BC контролей завершено, export invariants проверены. Результат текущего PPO update и новой оценки пока ожидается; модель не назначена основной. Counterfactual aim-query corpus и сравнение temporal128/GRU128 остаются дальнейшими этапами плана.

После исправления runtime paths полная новая партия80/80 captures проверена. На CUDA выполнен Update30:6986 PPO-eligible transitions,6987 context rows,85 sequences,10 actor steps; итог30 completed updates и289 total actor steps. PPO resume BC checkpoint реально прошёл. Сейчас выполняется160 paired validation captures; новый Update29/rules reference будет собран после них. Прежняя неполная попытка не использована.


Update30 before/after validation на offset24 завершена: fire BC51/80→BC+PPO48/80; deaths28→31, kills52→49, incoming health damage3527→3706. Этот PPO шаг не улучшил ветку; обе модели сохранены, Update30 не назначен основной. CUDA update seal: weights SHA c4911dd5e67438c943bc558cbb4061099fd3deae4bf0f4e2dc679a613fc8f5a8, checkpoint SHA ee79e3ecc2661a6a730e8cb1fdc53f20df4f9a5d882ca21fbed6e2bdd263ee06, report SHA9742ac1c349813d5b77a57b80f3b9fbfb13e800db55f8d9b30b6c68d3ff0087e.

Обычный rules-controller теперь разрешён и в двух старых generated MG recipes: parasite-machinegun-recoil revision1→2 с seed_revision1; parasite-gunner-machinegun-recoil revision2→3 с seed_revision2. Stock gear/HP/positions/geometry/reward не менялись. Перед запуском нового reference-pool сверены все80 instances обеих reference plans с before-plan уже завершённой оценки: совпали.

В отдельном ignored prototype разработана Go nominal aim query на observed bbox/current eye. Unit tests покрывают большой промах, wrap179→-179, crouched eye, неизвестный bbox, скрытую ближайшую цель и stale observation. Prototype не подключён к runtime и ещё не экспортирует verified training corpus. Источник source fingerprint активного reference-pool не менялся.


## Сравнение всех четырёх вариантов

| Метрика | Update29 | Fire BC | Fire BC+PPO30 | Rules |
|---|---:|---:|---:|---:|
| Победы |45/80|51/80|48/80|66/80|
| Смерти в первой жизни |32|28|31|8|
| Убийства в первой жизни |46|52|49|68|
| Полученный health damage |3976|3527|3706|1207|

Полная проверка320 validation captures завершена: batches complete/provenance valid, pool source_unchanged=true, reports SHA совпали с bindings. Generated fixtures и multiset native release class/pose/health/solid всех80 условий совпали. Engine entity IDs иногда различаются из-за allocation; они сопоставляются внутри каждого прогона и не являются physical reset conditions. Raw entity IDs не входят в policy feature vector. Полная эквивалентность скрытого engine state не заявлена.

Fire BC улучшила reference Update29 на6 побед в этом новом cohort; PPO30 снизил результат на3 относительно Fire BC. Rules остаётся лучше:66/80 против51/80. Это известные20 семейств и validation seeds, не unseen-geometry или full base1/base2 acceptance. Превосходство ordinary controller ещё не достигнуто; основная runtime policy не заменена.

| Семейство | Update29 | Fire BC | Fire BC+PPO30 | Rules |
|---|---:|---:|---:|---:|
|parasite-blaster-generated|4/4|4/4|4/4|4/4|
|parasite-gunner-blaster-generated|0/4|0/4|0/4|0/4|
|parasite-machinegun-recoil|4/4|4/4|3/4|4/4|
|parasite-gunner-machinegun-recoil|1/4|1/4|1/4|0/4|
|campaign-base1-site-01-blaster|4/4|4/4|3/4|4/4|
|campaign-base1-site-01-machinegun|2/4|3/4|2/4|4/4|
|campaign-base1-site-02-blaster|4/4|4/4|3/4|4/4|
|campaign-base1-site-02-machinegun|4/4|3/4|4/4|4/4|
|campaign-base1-site-03-blaster|1/4|1/4|2/4|4/4|
|campaign-base1-site-03-machinegun|0/4|0/4|0/4|3/4|
|campaign-base1-site-04-blaster|4/4|2/4|2/4|4/4|
|campaign-base1-site-04-machinegun|1/4|0/4|2/4|4/4|
|campaign-base2-site-01-blaster|4/4|4/4|4/4|4/4|
|campaign-base2-site-01-machinegun|3/4|4/4|4/4|4/4|
|campaign-base2-site-02-blaster|0/4|0/4|0/4|4/4|
|campaign-base2-site-02-machinegun|0/4|1/4|0/4|4/4|
|campaign-base2-site-03-blaster|1/4|4/4|2/4|4/4|
|campaign-base2-site-03-machinegun|4/4|4/4|4/4|4/4|
|campaign-base2-site-04-blaster|4/4|4/4|4/4|2/4|
|campaign-base2-site-04-machinegun|0/4|4/4|4/4|1/4|

Authoritative comparison: workspace/artifacts/fire-bc-rl-v2-20261007/comparison.json. Update seals и source SHA сохранены. Следующий этап — native-verified learned-state corpus с отдельными nominal aim annotations; отдельная ветка от Fire BC без принятия неулучшившего PPO30.

