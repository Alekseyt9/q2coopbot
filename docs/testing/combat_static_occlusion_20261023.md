# Статическая линия огня и расхождение позы, 23.10.2026

После закрытой 96-case проверки aim-ranker выполнена ограниченная диагностика на уже собранных TRAIN боях. Никакое обучение по validation, новые игры, изменение reward/default и neural inference не выполнялись.

## Входы и реализация

Источники содержат40randomized TRAIN боёв; в33изних были1257actual MG shots, ещё7не дали MG shots. Источники: источники `combat-predictive-aim-smoke-v1-20261023` и `combat-predictive-aim-corpus-v1-20261023`. Все исходные SHA, split и recipe проверены. Dataset members связываются по source/job/index/frame/shot с эксклюзивным command window. Для дополнительной проверки повреждение связывается с объявленными entity/track выбранной цели, не с ближайшим противником.

Новый `cmd/q2shot-world` выполняет только упаковку наблюдения и геометрию. Использует исходный client observation, sampled action, observed recoil, правосторонний MG muzzle offset; выбирает видимую bbox по тому же185-input контракту. По направлению фактического sampled action строится forward ray, проверяется пересечение bbox, затем существующий `MapInfo.TraceProjectile` по static MASK_SHOT brushes до входа в bbox. Препятствие должно находиться раньше bbox более чем на0,125unit; trace epsilon BSP0,03125. BSP base3/ware2 прочитан из единственного существующего `F:/src/quake2/assets/baseq2`, файлы ресурсов не копировались.

`cmd/q2shot-world` не читает веса и не оценивает neural model. Pure geometry tests прошли: параллельная ось, начало внутри bbox, yaw-local/world преобразование, translated muzzle и unknown target; существующий тест projectile brush trace также прошёл. Фокус-команда:

```powershell
go test ./cmd/q2shot-world ./internal/quake -run '^(TestObservedShotRayCoordinatesAndParallelSlabs|TestProjectileTraceBrushEntry)$' -count=1
```

`diagnose_combat_static_occlusion.py` отделяет запрос от native labels: в Go поступают только observation и compact sample. Native muzzle/shot/selected-target damage добавляются к результату только для диагностики. Runtime, actor и версии feature/weights не изменены. Большие exports сжаты NTFS LZX с сохранением SHA.

## Результаты

Авторитетный завершённый запуск: `workspace/artifacts/combat-static-occlusion-diagnostic-v2-20261023/complete.json` и `report.json`. Более раннийv1 не различал damage выбранной цели и damage другого монстра, поэтому для вывода используетсяv2.

| Срез | Все карты | Base3 | Ware2 |
|---|---:|---:|---:|
| Actual shots |1257|683|574|
| Наблюдаемая выбранная bbox известна |1136|577|559|
| Расчётный луч пересекает bbox |727|369|358|
| Статическое препятствие до bbox |57|40|17|
| Из них фактический solid contact |25|18|7|
| Из них фактический урон выбранной цели |30|21|9|

Из453solid contacts при известной выбранной bbox лишь25(5,52%) одновременно показывают пересечение bbox и препятствие до неё. При этом30из57предполагаемо перекрытых выстрелов всё же повредили именно выбранную цель;ещё2повредили другого живого монстра. Следовательно, текущий статический расчёт нельзя выдавать за точную причину промаха или использовать как runtime firemask. Измерение не доказывает, что BSP не нужен; оно показывает ограниченность расчёта при исходной наблюдаемой позе.

Реальные native MG traces сообщали0muzzle_blocked. Наблюдаемый current-pose muzzle probe также не показал startsolid. Динамические двери/монстры, движение в command window, новая отдача, смена позы, integer-angle truncation и spread этим расчётом не моделируются.

## Поза до команды отличается от позы выстрела

`pose-error-summary.json` вычислен по sealed `members.jsonl.gz`: евклидово расстояние между рассчитанным muzzle из текущего наблюдения и native shot start, отдельно горизонтальная/вертикальная компоненты. Native start — диагностическая метка, не вход predictor.

| Срез | Shots | Средняя ошибка muzzle | Горизонтальная | Вертикальная абсолютная |
|---|---:|---:|---:|---:|
| Все с известной bbox |1136|18,91|13,32|9,53|
| Расчётное препятствие до bbox |57|16,39|12,58|6,64|
| Solid contact при выбранной bbox |453|19,18|14,35|8,67|
| Урон выбранной цели |627|18,50|12,46|9,97|

У742/1136shots горизонтальная ошибка не меньше8units, у362 — не меньше16. Это не погрешность float и не единичный grazing case. Разложение причин ещё не выполнено: движение игрока, смена duck/eye height и recoil могут вносить разные доли. Нельзя по одному расстоянию объявить доказанным конкретный сетевой лаг или исправлять его универсальным offset.

## Решение и следующий опыт

BSP gate/маску выстрела не добавлять. Статический расчёт не заменяет предсказания того, к какой позе приведёт выбранное действие до момента выстрела.

1. Из сохранённых TRAIN захватов получить compact action-effect labels: фактический muzzle displacement и native recoil/view при выстреле, target-relative ошибки. Только causal observation/sample и доступные данные тайминга становятся входами; future/native truth только targets. Validation исходы не использовать для fitting.
2. Разделить ошибку происхождения луча, направления, bbox цели и баллистики; сравнить с current-pose и заранее заданной constant-velocity базой на whole-battle seed/site development folds. Не выбирать время упреждения на validation.
3. Если нужен learned forward predictor, обучать/сравнивать его только CUDA; raw source и engine seeds сохранить в реестре. Проверять action sensitivity на randomized TRAIN, а не только качество классификации попавших выстрелов.
4. Использовать только прогнозируемую геометрию в небольшом shot/action predictor с cross-fitting и явной проверкой утечки. Нативную позу на inference не подставлять. Моделировать совместно action движения/прицела и sampled pose.
5. Лишь после подтверждённых development gates — ограниченный native pilot против неизменного source, старого ranker и rules,16инстансов ×2; стандартные PPO и отдельные нестандартные execution audits сохранять. Успешный pilot расширять, не назначать веса автоматически.

Предыдущие результаты96native:source9/24,old/newrankers11/24,rules16/24. Задача превзойти rules остаётся нерешённой. Текущая диагностика объясняет конкретный пробел fixed-pose геометрии; улучшения боевого качества она сама по себе не доказывает.
