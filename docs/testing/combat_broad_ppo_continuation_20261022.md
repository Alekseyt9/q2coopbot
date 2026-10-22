# PPO продолжение на полном наборе групповых учебных семейств

Статус22.10: все пять раундов160 принятых train боёв завершены на GPU,19979 переходов и50 actor updates. Полный checkpoint/Adam audit пройден. Итоговый smoke24 native/16 CUDA закрыт без выигрыша над исходной моделью; новые веса не продвигаются.

## Родитель и неизменные условия

Источник `combat-sampled-motion-ppo-fit2-v1-20261021/update`: temporal attention и sampled-motion aim v1, ранее закрытые64 собственных учебных боя и coverage56 условий. Его coverage:25/56 против BC22/56 и правил28/56. Это проверенный development parent, не доказанный победитель над правилами. Основные/default веса не переключаются.

Дополнительные вертикальные входы и center teacher не включаются в это продолжение. Их отдельные эксперименты не дали достаточно устойчивого выигрыша для замены родителя. Здесь проверяется более длительное обучение полного actor/critic на более широком распределении геометрии и loadouts. Reward/objective сохраняется `combat_reward_v12`,SHA `af9b9e4c…`; actor/value/std и Adam восстанавливаются из checkpoint. В одном раунде10 actor/40 critic steps по исходному config,sequence retention1.0. Все численные NN проверки и обучение только CUDA.

Новый generated registry `combat-bodypoint-ppo-registry-v1-20261021/index.json` содержит28 групповых семейств. Скрипт `prepare_combat_objective_registry.py` изменил только привязку reward/revision, сохранив геометрию, состав, оружие и split ranges. Receipt расположен рядом с registry, не внутри него. Исходный реестр не переписан.

## Зафиксированный бюджет

`combat-broad-ppo-pilot-v1-20261021/protocol.json` создан до запуска. Во всех раундах карты base3/city1/jail1/ware2,по4 seeds на каждое из8 семейств:4 ordinary+4 recovery. Все32 сцены раунда имеют разные seeds. Используется существующая очередь16 инстансов ×2, без второго native pool.

| Раунд | Ordinary | Recovery | Минимальный train offset | Принятых боёв |
|---|---|---|---:|---:|
|1|site01,parasite+infantry+gunner,weapons|site01|392|32|
|2|site02,parasite+infantry+gunner,weapons|site02|396|32|
|3|site01,parasite+infantry+gunner,weapons-ssg|site01|400|32|
|4|site02,parasite+infantry,weapons-ssg|site02|404|32|
|5|site02,parasite+infantry+gunner,weapons-ssg|site01|408|32|

В сумме28 уникальных семейств,160 принятых новых боёв. Recovery начинает с grenades и доступными поддерживаемыми альтернативами. Предварительно проверено отсутствие пересечения160 seeds с историческими планами; collector повторяет проверку перед каждым раундом и закрепляет фактическую выдачу. Неполные попытки после прерывания отдельно архивируются и не выдаются за завершённые train cases.

После каждого закрытого corpus/CUDA fit следующий rollout собирается именно его новыми весами. Нельзя повторять обучение на прежних on-policy transitions после изменения политики. Промежуточная validation/выбор checkpoint запрещены протоколом; итоговые веса оцениваются отдельно после полного фиксированного бюджета. Training complete не означает quality acceptance.

## Исполнитель и защита возобновления

`scripts/run_registered_combat_ppo_pilot.py` последовательно вызывает существующие `collect_combat_aim_feedback_corpus.py --ppo-ready --train-only` и `fit_registered_combat_ppo_cuda.py`. Сохраняет source/checkpoint/corpus/fit связи, проверяет уникальность seeds и закрепляет Go/Python/PowerShell/native source snapshot. Нельзя менять закреплённый исполняемый код во время пилота. Документацию и новые независимые отчётные scripts можно добавлять отдельно.

OS byte lock защищает от двух исполнителей одного пилота. Живой отрицательный тест21.10 отклонил второй `--resume` до collector/fitter; receipt `duplicate-launch-audit.json`. Обычный `--resume` переиспользует только полностью закрытые фазы после проверки их хешей. Неполный corpus/fit сохраняется и требует восстановления; автоматического повторения всего частичного раунда нет.

## Восстановление первого раунда22.10

Исходный handle больше не существует, Python/native процессы отсутствовали. Это подтверждено процессами и свободным UDP диапазоном, а не только stale progress/lock файлом. Сохранились19 scheduler results без ошибок и20 independently closed capture reports. Веса ещё не обновлялись.

`scripts/recover_registered_combat_corpus.py` получил OS lock пилота, проверил все закреплённые исходники, model/reward SHA, frozen plan, source/native fingerprints и exporter hashes20 завершённых членов. У20-го scheduler result отсутствовал: он восстановлен из закрытого capture с явным отсутствием точных scheduler timestamps. Длительности не выдаются за первоначальный queue timing.

Двенадцать незавершённых каталогов и связанные fixture/log файлы перенесены в `combat-broad-ppo-round1-corpus-v1-20261021/recovery-v1/interrupted-members`. Оба абсолютных пути проверены внутри этого корпуса; данные не удалялись, assets не копировались. Повторяются только12 недостающих jobs с исходными plan/task/seed через штатный episode runner и16 slots.

Recovery PowerShell script находится в artifacts, чтобы добавлением нового top-level PS файла не менять fingerprint старых captures. Он использует исходный pool finalizer без изменения его проверок, дополнительно проверяет совпадение source/native fingerprint всех старых и новых членов. Перед CUDA повторно сверяются хеши20 сохранённых captures. Время pool wall_seconds помечается как recovery-session-only; старые per-job времена и новая сессия не смешиваются в исходную производительность.

Восстановление закрыто:20 retained/12 rerun,все32 native captures валидны и прошли обычный CUDA replay,3511 train rows. В `recovery-v1/native-recovery.json` закреплены preparation/pool SHA,в `complete.json` корпуса — реальный recovery script/provenance. Не утверждается, что прерванный collector самостоятельно дописал этот seal. После recovery повторно проверены сохранённые capture/report/manifest/exporter/server/bot hashes. `--resume` пилота уже переиспользовал этот закрытый corpus и перешёл к `combat-broad-ppo-fit1-v1-20261021` на GPU. Поддержка частичного native восстановления не означает проверенное продолжение произвольно оборванного optimiser step.

Fit1 полностью завершён:10/10 принятых actor steps, стандартный update в `combat-broad-ppo-fit1-v1-20261021/update`. Отдельный `round1-checkpoint-cuda-audit.json` на NVIDIA RTX5070 подтвердил изменение30 actor/12 value tensors, finite CUDA значения, неизменность config/objective и optimizer parameter groups. Веса SHA `632f24b3…`. Это реальное PPO обновление после восстановленного корпуса, не только сохранение файла или проверка метаданных. Пилот автоматически перешёл к round2 с этим update как source. Новые победы требуют итоговой независимой оценки.

## Заранее закреплённая итоговая оценка

До первого нового PPO fit сохранены `combat-broad-ppo-smoke-template-v1-20261022.json` и spec того же имени. После всех пяти rounds запускается smoke24 native/16 CUDA:исходный PPO parent,итоговый update5,правила. Validation offset592,пул16 ×2,порт44200. Семейства/seed не выбираются по промежуточным training outcomes.

Текущая последовательная команда после training complete автоматически запускает evaluator,пять bundle reports и дополнительный actual MG outcomes report. Второй native pool одновременно не создаётся. Если любая фаза завершится с ошибкой, последующие стадии не выполняются; частичные результаты сохраняются. Итоговый `complete.json` пилота означает только закрытое обучение; улучшение модели требует отдельного сравнения.

## Что остаётся проверить

Нужны все пять закрытых собственных rollout/update, принятые actor steps, восстановление конечного checkpoint и независимое сравнение с исходным родителем/правилами на новых seeds. Кривая train исходов на разных сценариях не заменяет парную оценку. Остались недоказанными общее превосходство, устранение слепой стрельбы и campaign/coop acceptance.

## Завершение обучения и независимая оценка

| Раунд | Принятых боёв | Train rows | Принятых actor steps |
|---|---:|---:|---:|
|1|32|3511|10|
|2|32|4523|10|
|3|32|3555|10|
|4|32|4547|10|
|5|32|3843|10|
|Всего|160|19979|50|

`complete.json` пилота закрывает учебную фазу. `scripts/audit_registered_ppo_pilot_cuda.py` независимо проверил все5 checkpoint на CUDA:JSON model точно совпадает с actor/value/log_std; config/objective/retention неизменны; формы/группы/число состояний Adam сохранены; каждый из31 actor states увеличил step на число принятых updates,каждый из12 value states — на40 в каждом раунде. Конечный `final-checkpoint-cuda-audit.json` подтверждает все160 случаев. Это не заменяет оценку качества.

Smoke `combat-broad-ppo-smoke-evaluation-v1-20261022` завершён:24 native/16 CUDA,1099 parent/1316 candidate replay rows. Bundle закрывает пять отчётов,их SHA и входы проверены. Дополнительные native outcome/ownership отчёты сохранены отдельно,старые seals не переписаны.

| Вариант | Победы | Ordinary | Recovery | Полученный урон | Смены цели /100 кадров | Повороты ≥30° | Стояние при clear enemy | Attack без цели |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Исходный PPO parent |2/8|2/4|0/4|529|31.55|7.27%|15.64%|26.03%|
| После160 новых боёв |2/8|1/4|1/4|525|32.17|13.13%|22.08%|13.47%|
| Правила |6/8|2/4|4/4|392|—|—|—|—|

Парно parent/candidate:1 новая/1 потерянная победа,разница0,exploratory95% интервал[−37.5;+37.5]п.п. К правилам0 новых/4 потерянных победы. Меньше команд attack без цели не стало общим выигрышем. Native MG fires120→170,с уроном живому монстру76→81,доля63.33%→47.65%;solid contact44→84,candidaт также дал5 corpse contacts. Это разные closed-loop траектории,не фиксированная вероятность попадания,но положительного сигнала достаточного для расширения оценки нет.

`decision.json`:candidate_not_advanced,promotion=false,screen не запущен. Исходный parent/default сохранён. Не утверждается,что PPO в принципе непригоден:это один training seed/зафиксированный бюджет и малый fresh smoke. Дальше наращивать evaluation этого checkpoint без новой гипотезы не назначено. Следующий ограниченный опыт — [vertical conditioning при калиброванном обучении center teacher](combat_center_vertical_calibrated_20261022.md).

## Дополнительные диагностические результаты

`audit_registered_train_fire_rewards.py` проверяет реальные MG events из accepted train rows и сохранённые reward components,без NN inference/изменения rewards. В round1 все877 MG shots имели хотя бы один known visible bbox;535 дали урон живым монстрам,323 solid,19 corpse,off_target_attack сработал73 раза. В round2 из1020 shots62 не имели known bbox:57 solid и5 damageable other contacts,off_target_attack=0 во всех62. Эти5 other contacts принадлежат func_explosive/func_door,поэтому нельзя объявлять каждый выстрел без видимой цели бесполезным. Мгновенный outcome не доказывает стратегическую избегаемость. Текущий objective намеренно не менялся во время пяти раундов.

Найдена ошибка clock cutoff старого ownership audit:network frames98–99 ошибочно считались setup при game release100. Отдельный canonical-release audit вернул64 таких кадра в32 SSG/recovery cases round3 и16 в каждом из8 parent/candidate quality cases. Visible-target rules frames/attacks и missing observations/actions равны0. Rules вне видимого боя посчитаны отдельно; полное управление кампанией/coop этим не доказано.

После terminal всех процессов этого пилота исправлен основной `audit_registered_combat_ownership.py`,версия v3. На8 реальных parent cases его totals точно совпали с независимым canonical audit,включая16 ранних released frames. Исправление отчёта не меняет старые captures/weights/seals. Старые v2 результаты сохраняются как исторические; новые запуски учитывают network release.
