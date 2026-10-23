# Ближайший bounded опыт: near-identity и бюджет обновлений

Протокол фиксируется до нового TRAIN/fitting. Независимый native/credit audit32старыхTRAIN закрыт; новое99% initial прошло8native/8CUDA,213команд до первой поправки совпали с source, встретились2nonzero события. Обе модели выполнили1/4цели; это только приёмка инициализации, не quality gain.

Начальные weights — `combat-joint-aim-near-identity-v1-20261023`, source actor/value/std/Adam/RNG точны к исходной модели. Predictor frozen, reward v13 и gamma0,99/lambda0,95 не меняются. Новый rollout обязательно собирается этой policy; старый50% corpus не используется как PPO data нового behavior.

## Три TRAIN ветки

- **B, base-budget control:** новая categorical head frozen, исходные actor/std обучаются до15steps с base KL budget0,0025.
- **G, global joint:** до30steps всех actor/head параметров с прежним общим KL safeguard0,005.
- **S, split joint:** сначала до15steps candidate при frozen source actor, затем до15steps source actor/std при frozen candidate. Отдельные per-factor budgets0,0025, весь full joint KL остаётся≤0,005.

LR исходных групп сохранены: base3e−5, branches3e−4; новая head3e−4. Entropy0,001, clip0,2, max grad norm0,5, critic40steps/прежний LR и Adam, legacy GAE. B сопоставим с base-частью S; различие вычислительного/итерационного бюджета G явно учитывается. Сравнение оценивает метод распределения updates, не только ёмкость сети.

Тренеру нужны независимые deep-copied optimizer states, сохранение frozen tensors/moments, versioned scope на resume и current-policy CUDA replay. В S неактивный optimizer не меняется при backtracking активного. Старые `train_combat_joint_aim_*` не должны молча принимать новый scope. Sparse tail не требует всех25категорий в каждом batch; число реальных interventions всегда показывается.

Точное определение factor guards до fitting: base — средний sampled approximate KL исходных факторов с fire на окончательном прицеле; category — точный categorical KL, усреднённый по eligible TRAIN events. У B/S guarded step принимает только `max(full_KL, 2*base_KL, 2*category_KL) <= 0,005`. Все значения считаются относительно policy, собравшей текущий корпус; для S budget не обнуляется между двумя фазами. У G сохранён только прежний full approximate KL guard. Это ограничение измерено на TRAIN; оно не обещает такой же KL или боевое качество в новых состояниях.

## Объём и собственные траектории

Фиксировано **4 updates на ветку**, без промежуточного выбора по validation. Первый32-case TRAIN общий, потому что B/G/S стартуют одной policy. Далее3раунда по32собственных fresh on-policy случая на каждую из3веток. Итого320уникальных TRAIN captures,320CUDA acceptance; каждый fork потребляет свой rollout один раз. Первое общее чтение одного on-policy corpus тремя независимыми fork допустимо и явно записывается.

Все8условий base3/ware2 ordinary/recovery,4seeds/условие/корпус. Сиды новые на каждом раунде, между ветками — заранее сопоставленные условия; после расхождения policies нельзя подставлять чужие траектории. Очередь16instances×2. Native/credit checks и CUDA full-LP/context acceptance обязательны перед каждым update; ошибки останавливают зависимую ветку, а не запускают скрытый restart. Веса/чекпоинты/протоколы и итоговые отчёты сохраняются, дисковый reserve32GiB.

## Один endpoint screen

После4updates: **6вариантов×8freshvalidation =48native/40CUDA**: source, frozen near initial, финальные B/G/S, rules. Условия и seeds парные, новые и зафиксированные до запуска. Основное архитектурное сравнение SпротивB; дополнительноG/Sпротивsource/rules. При равенстве S и B выигрыш нельзя приписывать новой head.

К расширенной24-case confirmation допускается только явный малый сигнал: endpoint выигрывает у source и rules минимум по2целям из8, смертей не больше и received damage не выше лучшего из двух controls более чем на10%. Это правило фиксируется до игр. При нескольких прошедших кандидатах подтвердить их на новых seeds, не выбирать по тем же8боям окончательную production policy. Actual MG outcomes/solid contacts/target switches/standing/turning показать отдельно; выигрыш в одном proxy не заменяет боевой результат.

Статус перед первым fitting: общий корпус собран и проверен — 32 native / 32 CUDA, 4179 переходов, девять ненулевых поправок. Независимые native/GAE audits пройдены. Добавлен отдельный CUDA trainer `train_combat_joint_aim_budget_cuda.py`; первое обновление и дальнейшие три раунда пока не выполнены. Default не меняется. Этот протокол остаётся неизменным на протяжении четырёх раундов; ход опыта записывается в отдельных отчётах.
