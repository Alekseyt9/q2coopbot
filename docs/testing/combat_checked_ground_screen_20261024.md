# Наземный запрос модели в целом бою — 24.10.2026

Предпосылка: `combat-checked-ground-native-probe-v1-20261024` закрыл12native/12CUDA и46/46checks на трёх фиксированных TRAIN состояниях. Максимальные ошибки позиции≈0.09497units, скорости≈0.07263units/s. Это проверка исполнения одного кадра, а не улучшение качества боя; подробности `combat_checked_ground_probe_20261024.md`.

## Режим и сравнение

`test.checked_ground_request_probe_all_frames` применяет существующий `diagnosticCheckedGroundRequest` в первом life после release. Он может восстановить только выбранные моделью Forward/Side/Up после полного stop=`unsupported_motion_guard`, если подробный Pmove и прежние path checks допускают команду. Остальные поля проверенной команды сохраняются. Затем выполняется тот же stopped-crouch helper, что и в crouch-only контроле, если горизонтальная команда осталась отменённой. Поэтому сравнение с crouch-only не лишает новый вариант прежнего сохранения позы.

Режим exclusive, local, captured, synchronous, learned combat-only; default отключён. Registry train/rules и несовместимые diagnostic flags отвергаются. Сквозные config/manifest/native/CUDA metadata сохраняют `diagnostic_checked_ground_request_all_frames`; версия rollout отдельная. Обычный PPO fitting/merge запрещён, processor проверяет каждый diagnostic случай отдельно и не пишет merged training corpus. Это эксперимент исполнения, без обучения.

Спецификация `../combat_checked_ground_screen_spec_20261024.json` зафиксирована до исходов: validation offset57,8pairedrecipes, source/crouch/request/rules. Все три learned варианта используют одинаковые веса4e31f0c1…; источник не выбирается по новому результату. Те же четыре карты base3/city1/jail1/ware2, пары/тройки, обычный/ограниченный боезапас. История сидов включает предыдущий screen56 и более ранние планы. Разные варианты получают одинаковые сгенерированные сцены и seed внутри случая.

Бюджет32native/24CUDA, очередь existing harness≤16,×2, остановка на цели/первой смерти. GPU используется для всех NN проверок; Go tests только физики, config и registry metadata. Целевые тесты, Python/PowerShell AST и diff check перед запуском пройдены.

## Критерии до запуска

Expansion gate: минимум+2победы относительно crouch-only; побед не меньше source/rules; deaths не больше crouch/rules; received≤110% каждого crouch/rules. Дополнительно: есть реальное восстановление горизонтальной команды, все8prefix request/crouch совпадают до первого ground-request вмешательства и на root меняется только Forward/Side/Up, каждое восстановление повторно проходит существующий detailed Pmove/path check.

Для следующих native наблюдений без изменения health/armor считаются ошибки позиции≤0.25 и скорости≤1.0; требуются on_ground и ducked при отрицательном Up. Соседний кадр извлекается также из terminal/non-provider записей первого life, если доступен, чтобы завершение цели не скрывало проверку физики. Кадры с native damage/armor effects не доказывают точность свободного движения. Нет сопоставимых кадров — нет mechanical gate. Entity IDs не нормализуются, float prefix tolerance1e−9.

Native action audit отдельно показывает standing, большие yaw, отменённые команды, выстрелы и попадания. Эти описательные метрики не заменяют победы/смерти. Прохождение малого screen потребует нового независимого подтверждения; автоматического продвижения нет.

## Выполнение

Runner `scripts/run_combat_checked_ground_screen.py`, reporter `scripts/report_combat_checked_ground_screen.py`. Root `workspace/artifacts/combat-checked-ground-screen-v1-20261024`. Цепочка evaluation→native-quality→result завершена:32native/24CUDA, все capture валидны, source_unchanged=true. Source fence включает Go/Python/PowerShell/native sources и frozen specification. Seal результата и все входные SHA проверены после завершения. Default и веса сохранены; цель устойчивого превосходства над rules остаётся открытой.

## Результат

| Вариант | Победы / 8 | Смерти | Убийства | Полученный урон HP |
|---|---:|---:|---:|---:|
| source, обычный guard | 3 | 4 | 10 | 496 |
| crouch, сохранение приседания | 3 | 5 | 9 | 554 |
| request, движение + fallback | 2 | 5 | 10 | 547 |
| rules | 5 | 1 | 14 | 389 |

**Expansion gate=false.** Нет требуемых+2побед к crouch; побед меньше source/rules, deaths/received хужеrules. Широкое подтверждение и default promotion не запускаются. Это новая paired партия с offset57, а не повтор предыдущих8условий screen56; сравнение абсолютных итогов между57и56 не измеряет эффект изменения.

Все8prefix request/crouch совпали до первого вмешательства. Выполнено13ground-request восстановлений; все повторно проходят detailed Pmove/path checks. Есть8сопоставимых соседних native наблюдений без health/armor effects: maxposition≈0.13469,maxvelocity≈0.16957, всё в заранее заданных пределах0.25/1.0. Другие5вмешательств не дают такого чистого соседнего сравнения. Источник наблюдаемого боя — модель, без rules fallback при видимом противнике.

Относительно crouch: новых побед0, потеряна1. Ware2site02 trio seed1172166057: crouch3килла/65HPполучено/goal, request2килла/100HP/смерть. Это описательная локализация заранее выбранного случая, а не новый отбор для fitting.

Native quality: standing среди grounded-visible кадров source225/695≈32.37%,crouch206/694≈29.68%,request318/747≈42.57%. Request reduced near zero126/127/102кадра соответственно, но уменьшение отмен не стало меньшим стоянием. Доля MG live damage source139/202≈68.81%,crouch147/198≈74.24%,request140/203≈68.97%. Больших yaw65/54/55кадров, solid contacts54/46/53. На разных closed-loop траекториях эти счётчики описательные: устранение верчения/пустых выстрелов не доказано.

## Вывод и следующий шаг

В проверенных состояниях физически допустимое движение теперь можно исполнить, но простое переключение guard у готовых весов не улучшило исходы. Продолжать расширять разрешения без обучения или повторять screen до удачной партии нельзя считать прогрессом качества.

Следующая содержательная гипотеза — обучение при согласованном контракте исполнения действий: собственные TRAIN бои, fitting и evaluation используют один и тот же явно объявленный режим; обычный и новый режимы сравниваются с одинаковым бюджетом обучения. Предположение, что ухудшение связано именно с переносом весов между разными режимами исполнения, пока не проверено. Потребуются отдельный версионированный training contract, защита от смешивания корпусов/возобновления с неверным контрактом и новая заранее зафиксированная held-out партия. Текущие diagnostic validation записи по-прежнему не подходят для fitting; их запрет не снимается. Default guard остаётся прежним. Нужна проверка обученной модели, а не только более свободного исполнения старых весов.
