# Эффективность действий и режим выбора политики

## Закрытая оценка 560 боёв

Новый `scripts/summarize_registered_action_efficiency.py` объединяет принятые post-release behavior audits с native damage timing по job/episode/seed/map, capture/manifest SHA и plan index. Все сравниваемые варианты обязаны иметь одинаковые условия. Выход: `workspace/artifacts/combat-weapon-exploration-action-efficiency-v2-20261011.json`, четыре обучаемых варианта, по112 боёв. Инструмент не выполняет нейросеть и не обучает модель.

| Метрика | Control до | Control после |
|---|---:|---:|
| Победы /112 |29|34|
| Кадры с горизонтальной скоростью <10 |18,92%|20,13%|
| Кадры стояния при наличии clear-shot противника |16,66%|18,30%|
| Большие applied yaw команды (≥30°), доля кадров |10,72%|9,83%|
| Атака без выбранной цели, доля attack-command кадров |20,04%|16,73%|
| Смены цели на100 provider кадров |34,27|35,73|
| Смены при обеих ещё наблюдаемых clear-shot целях на100 кадров |29,47|30,27|
| Бои с непрерывным stationary-clear отрезком ≥20 кадров |14|10|
| Бои без attack command |11|11|
| Средняя задержка первого attack command, кадров |15,89|15,35|
| Бои без native monster-health damage |13|14|
| Средняя задержка первого native monster-health damage, native кадров |19,27|17,91|

Вывод ограничен: некоторые признаки улучшились, другие ухудшились; лишние действия не устранены. Standing/turn/target-switch сами по себе не обозначены бесполезными. Задержки усредняются только по боям с соответствующим событием; отсутствие события указано отдельно. Provider-command и native-damage clocks не смешиваются. Counter intent_changes включает переходы к none; отдельный счетчик обеих clear целей исключает этот случай.

Temperature2 давала более раннюю команду атаки (около3,5 кадров против15,3), но после обучения лишь6/112 побед против34/112 у control. Раннее нажатие attack не равно эффективной стрельбе; универсальный штраф за задержку без учета возможности попадания может закрепить плохое поведение.

## Проверка вклада случайного выбора действий

Веса текущей confirmation имеют `deterministic=false`. В Go PPO target, aim mode, weapon, attack и vertical выбираются случайно из распределений, непрерывные действия получают шум. Это правильно для сбора собственного PPO-опыта, но такая оценка не заменяет проверку режима игрового исполнения. Высокая частота смен целей может частично объясняться sampling; причинный вклад пока не измерен.

Подготовлены копии before/after с единственным измененным полем `deterministic=true`; все остальные JSON поля/параметры проверены на точное сохранение. Исходные веса и checkpoints не менялись. Seal: `combat-weapon-state-deterministic-models-v1-20261011/complete.json`.

Confirmation224 сравнит обе копии на тех же28×4 offset256 fixtures, что и stochastic confirmation336. Правила из первого пула используются повторно как matched контроль, без дополнительных112 боёв. Второй пул начинается только после полного закрытия первого native пула, чтобы одновременно работали максимум16 инстансов. Timescale2 сохраняется. Нет обучения на evaluation и нет automatic promotion.

Координатор native анализа: `workspace/artifacts/analyze-weapon-state-deterministic-20261011.py`. Он считает before/after и stochastic/deterministic пары, native damage и behavior. Его `native-complete.json` явно не является CUDA action replay: детерминированные traces не выдаются за stochastic PPO samples. Полная кампания/кооператив и численная CUDA проверка такого исполнения остаются необходимыми.

## Ограничения и следующий эксперимент

Текущая native hitscan telemetry описывает Machinegun (mod4), projectile telemetry — Blaster. Shotgun/SSG попадания в стены нельзя посчитать из attack commands или количества damage events: отдельные дробинки и выстрелы имеют разные единицы. Для точного диагноза этой части требуется отдельная native fire/impact telemetry с идентификатором выстрела и дробинок; в ходе текущих capture движок не меняется.

После сравнений: выбрать дальнейший эксперимент с согласованием target/movement/aim/fire или устойчивостью выбранной цели. Изменять обучение по подтвержденным ошибкам, не вводить постоянное принудительное удержание цели и не считать каждый маневр/промах ошибкой. Исходные требования полного управления боем моделью сохраняются.

## Закрытый native deterministic224

Stochastic336 закрыта вместе с224 CUDA cases/26764 rows. Deterministic224 закрыта нативно без ошибок; CUDA action replay детерминированного режима пока не выполнен. На одинаковых offset256 условиях before/after: stochastic31→31 побед, deterministic36→38, rules58 из112. Смерти after71 stochastic /62 deterministic /44 rules.

Для одних after весов: target changes35,90→6,62 на100 кадров; large-yaw11,16%→3,45%; attack commands без выбранной цели15,42%→3,01%. Однако stationary доля17,15%→43,11%, never-attack15→35/112, максимальный stationary-clear отрезок95→292 кадров. Recovery: stochastic4/32, deterministic0/32 (native dealt0), rules29/32. На обычных80 боях deterministic38 против stochastic27 и rules29; общего превосходства нет из-за recovery. Уменьшение sampling лечит часть смен/поворотов, но обнажает неподходящее удержание оружия/бездействие. Не продвигать режим по одному общему росту побед.

Sources: `combat-weapon-state-quality-analysis-v1-20261011/{complete,action-efficiency}.json`, `combat-weapon-state-deterministic-analysis-v1-20261011/{native-complete,quality-strata,action-efficiency}.json`. Начат отдельный fresh-own-policy reward-v12 эксперимент, см. `combat_target_commitment_plan_20261011.md`.
