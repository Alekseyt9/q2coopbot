# Граница победы и хвост остановки харнеса

22.10.2026. Исправлен supervisor `scripts/combat_goal_stop.ps1`. Модель не получает native kill events: они используются только для остановки/разметки эпизода, как и раньше.

## Причина

В source/zero compatibility v1 последний kill одинаков — frame230. Опрос лога каждые50ms выбирал последнюю доступную строку: `goal_stop.observed_frame` получился232 у source и231 у zero. Полные записи различались одним post-goal frame234, хотя 598 общих provider frames имели одинаковые canonical inputs, memory, samples и commands.

Строгий full-trace audit v1 остался непройденным. Его receipt не заменяется новым утверждением о совпадении всей записи.

## Исправление

Для всех goal-stop сценариев используется существующий cursor чтения только полных JSONL-строк. Супервизор сохраняет компактные identity/health поля, а не feature/action tensors. После поступления native kill evidence выбирается **самое раннее полное наблюдение после последнего kill**. Отложенная запись damage event больше не заставляет выбирать более поздний observation.

При проверке выбранной строки учитываются события только до её frame. Более поздняя смерть после уже завершённой цели не отменяет предыдущую победу; смерть до достижения цели по-прежнему её запрещает. First-death supervision сохранена.

`Get-CombatGoalKillEvidence` проверяет actor/map/spawncount, release frame, ожидаемые классы, distinct target IDs, killed/health-before и исключение MOD21. Окончательную native evidence по-прежнему независимо проверяет Go exporter.

## Проверки

- `pwsh -NoProfile -File scripts/test_combat_goal_stop.ps1`: passed. Помимо прежних guard tests проверяются delayed events, batch independence, earliest row, future post-goal death, pre-goal death и отсутствие закрывающего observation.
- `test_combat_goal_boundary_real_traces.ps1` повторно обработал сохранённую проблемную пару: обе стороны выбирают frame231 после kill230. Старые captures/receipts не изменены. Receipt: `workspace/artifacts/combat-goal-boundary-repair-v1-20261022/report.json`.
- Fresh source/zero comparison v2 закрыт: seed offset753, 8 native / 8 CUDA, base3/ware2 ordinary/recovery, max16, ×2. **Semantic episode audit и отдельный прежний strict full-trace audit прошли**. Все 612 provider frames, canonical histories, samples и proposed/applied/sent commands совпали. CUDA eligible rows611/611; packet-age metadata может различаться, но freshness class одинаков и допустим.

## Контракт приёмки

`audit_combat_conditional_fire_native_parity.py --episode-boundary` вводит явный `native_episode_terminal_parity_v1`: сравниваются **все кадры игрового эпизода вплоть до первого проверенного terminal observation включительно**, а не случайный общий префикс. Обе стороны должны иметь одинаковые kill frame, observed terminal frame, classes и health; для этой проверки требуется закрывающий frame=kill+1. Вплоть до границы canonical history и actions совпадают полностью. Для эпизодов без goal receipt сравнивается вся запись, как раньше.

Асинхронный shutdown tail после terminal сохраняется и перечисляется отдельно; его команды не входят в игровой эпизод. Это отдельный семантический контракт, **не full-trace parity**. Прежний строгий режим и диагностический common-prefix режим сохранены, автоматически новый режим не включается.

Receipts: `workspace/artifacts/combat-conditional-fire-zero-compat-v2-20261022/episode-parity.json` и `full-trace-parity.json`. Старый v1 gate остался непройденным; новое подтверждение получено на новых сидax с исправленным supervisor.

Начат сбор нового train-only корпуса: `combat-conditional-fire-training-v1-20261022/corpus`, 32 боя на8 family/site/loadout комбинациях base3/ware2, pool16 ×2, CUDA replay. Заранее зафиксированы два fit на одном корпусе: full policy и fire-only, каждый до10actor/40critic steps. Затем separate8-case/variant start/full/fire-only/rules smoke. Ни обучение, ни эта будущая малая проверка не означают автоматической замены default или доказанной общей победы над rules.
