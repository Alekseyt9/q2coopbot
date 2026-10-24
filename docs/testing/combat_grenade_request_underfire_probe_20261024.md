# Проверка отменённого движения под огнём на city1

## Итог на 21:51 МСК

Закрыт `workspace/artifacts/combat-grenade-request-underfire-probe-v1-20261024`:4native/4CUDA,source_unchanged=true,mechanical_gate_passed=true,16/16checks. Report `analysis/report.json`, SHA `ff19798aa56d4744e0e2858cf3c5250de3c80c37ceb7664731cb3558643e9cf9`. Протокол/source/trace/model SHA повторно проверены, исходные и оба cold prefix совпали, entity IDs не нормализованы. Неверные бюджеты legacy/underfire режимов отдельно отклонены до output creation/игр.

Запрос Forward156/Side289/Up0 действительно выполнен на134; Pitch1556,Yaw−29678,Buttons1,Light61 и прочие поля сохранены. Следующий кадрgrounded/неcrouched,health45→43 в обоих режимах. Строгие position/velocity thresholds не ослаблялись:

| Режим, оба повтора | Position error,units | Velocity error,units/s | Смерть network frame |
|---|---:|---:|---:|
| Control | 0.0540833 | 0.0901388 | 137 |
| Request | 0.0951305 | 0.1145655 | 137 |

### Боевой эффект — отрицательный для этого вмешательства

В полном наблюдаемом окне3кадра после134каждыйвариантполучил45HP(изних39HPотгранаты)ипогиб. Окна5/10кадров неполные из-за first-life остановки, в отчёте это отмечено, они не объявляются полными отрицательными/положительными исходами.

Уcontrol:parasite2HPна135,136,137;gunner grenade на137,health39→−1. Уrequest:parasite2HPна135,136;gunner grenade на136,health41→2;parasiteна137,health2→0. Requested движение сдвинуло попадание гранаты на один кадр раньше, не спасло бота. Поздняя причина смерти различается; это не улучшение и не повод продвигать guard/веса.

Дальнейшаяpolicyтраекториячастичноменяется:request135имеетненулевуюappliedкоманду,а136сноваостанавливаетсяunsupported_motion_guardпослеурона;control135/136двигается. Нельзя приписывать весь последующий исход только первому displacement без учёта закрытого policy loop и атак двух врагов.

Положительный результат ограничен механикой: в одном реальном состоянии простая проверка отклонила модельный запрос, detailed physics/path допустили его, а native исполнение совпало с прогнозом даже под огнём. Это не доказывает, что все отмены ошибочны, что данный манёвр полезен или что default следует изменить. Base3/jail1/ware2 unknown случаи из предшествующего разбора не затронуты. Promotion/default/weights остаются прежними.

### Следующая ветка

Проверить более ранние альтернативы движения и короткое окно манёвра вместо повторения того же позднего single-frame восстановления. В коде уже есть diagnostic movement mean shift/window и continuation fork, но их admission сейчас ограничен старым `EquippedWeaponFeatureVersion`; текущий projectile-input endpoint ими ещё не покрыт. До нового сравнения нужны совместимость без изменения обычной policy, CUDA проверки draw/logprob и zero-shift/native prefix controls. Затем заранее заданные альтернативы на собственных TRAIN состояниях и короткие native damage/life окна, без выбора по confirmation и без выдачи одного состояния за общее превосходство. Нового fitting пока нет.

## Исходная постановка до игр

24.10.2026. Только диагностический повтор собственногоTRAINсостояния, не новая оценочная выборка и не обучение. Исходные веса neutral round02 `5dcf0d640d28394817bc105a58b379e4e6723eee747867d16a0cfbd87535ea0c` соответствуют поведению записанного round03 случая. Seed1468045009,frame134,city1/index37; grenade impact записан на137. Выбор состояния и подробная геометрия зафиксированы до новых игр в `combat_grenade_preimpact_20261024.md`.

## Условия до запуска

Новый протокол `combat_checked_ground_underfire_probe_v1`:однасцена,control/request×2coldповтора=4native/4CUDA. Старый трёхсценовый v1 с12проверками и no-injury gate остаётся отдельным режимом. Runner/auditor проверяют размер бюджета; исходники старых версий сохранены в `workspace/artifacts/combat-grenade-request-probe-preflight-v1-20261024`.

В request на единственном кадре134существующий Go diagnostic helper может восстановить только выбранные моделью Forward/Side/Up, если detailed Pmove/path checks допускают их и единственная movement-причина=`unsupported_motion_guard`. Ни новый вектор движения, ни прыжок, ни обход неизвестной опоры не добавляются. Остальные команды, включая Light,attack,yaw,pitch,impulse,msec, сохраняются из guarded control. Default helper выключен; новых Go/runtime изменений нет.

Все варианты должны совпасть с исходной записью до root observation/proposal/sample и по integer commands до root. Cold repeats сравниваются до следующего кадра. Нормализуется только spawncount; entity IDs не нормализуются. На следующем кадре требуются grounded/нужная поза, positionerror<=0.25,velocityerror<=1.0. В данной under-fire сцене здоровье уже в исходном следующем кадре45→43; отсутствие урона не является критерием этого нового режима. HP/armor сохраняются в отчёте отдельно. Известный риск: текущая Pmove prediction может не покрывать knockback от попадания; ошибку скорости не скрывать и не ослаблять порог после опыта.

Отдельно публикуются first-life native события урона и окна1/3/5/10networkframes после вмешательства; неполные окна отмечаются. Они не заменяют mechanical gate и не доказывают общее качество, эффективность новой модели или достаточность одного случая. Все4capture,включаяcontrols,диагностические и не должны попадать в TRAIN; request export дополнительно помечается diagnostic-only и отвергается обычным PPO fitting/merge.

Сбор через существующий harness,×2,неболее16инстансов(здесь4задания);CUDA replay только наGPU. Веса/reward неизменны. Протокол `docs/combat_grenade_request_underfire_probe_protocol_20261024.json`, SHA `58f86433e2849f7273b381076509aaee80c1aaf5c531035283ecf666418a17fe`. В нём зафиксированы source SHA runner/auditor/основных harness и CUDA helper файлов. До игр проверяются model/trace/reward/prerequisite/source SHA, после — source/протокол снова. Promotion/default changes отсутствуют.

Output: `workspace/artifacts/combat-grenade-request-underfire-probe-v1-20261024`. PowerShell/Python AST прошли. Новый live результат ещё не установлен.
