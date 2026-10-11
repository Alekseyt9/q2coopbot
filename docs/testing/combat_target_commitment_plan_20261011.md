# Эксперимент: устойчивость выбранной цели

Запрос пользователя: поощрять удержание цели, чтобы уменьшить метания. Реализованная выбранная-цель reward v8 уже поощряет улучшение прицела на стабильной цели, но не сам факт её удержания. Отдельная v7 SequenceReward штрафует только короткие A-B-A циклы с большими поворотами, без промежуточного урона, новых угроз или потери видимости. В текущих train recipes используется v8, поэтому v7 target_churn в них не действует. Добавление одной строки target_churn в v8 запрещено текущим валидатором; требуется явная версия/композиция objective.

## Требуемое поведение

Удерживать доступного противника достаточно долго для результативной атаки; переключаться при его смерти/потере линии огня, появлении новой опасности или необходимости уклониться. Никакого постоянного runtime запрета на смену цели. Reward/history офлайн, скрытые server outcomes не становятся входом политики.

## Два раздельных варианта обучения

1. Сохранить v8 доводку и добавить ограниченный штраф за подтвержденные наблюдениями A-B-A метания. Условия защиты существующей v7 сохраняются; после native/CUDA acceptance сравнить частоту срабатывания, иначе нулевая активация даст бессмысленный эксперимент. Не штрафовать каждую смену цели.
2. Поощрять продуктивное удержание: небольшая ограниченная добавка только при подтвержденном уроне удерживаемому противнику. Без положительной награды за время/простое удержание, без награды за урон другой цели или трупу. Нужна привязка target entity+observed track и native damage к одной жизни/окну исполнения; текущий агрегат MonsterHealthDamage недостаточен для такой атрибуции. Параметры выбрать на train данных, не на оценочных исходах. Ограничить вклад относительно основной награды за урон/убийство; длинный бой не должен позволять бесконечно накапливать бонус.

Второй вариант — намеренное изменение предпочтений objective, а не автоматически policy-invariant shaping. Требуется оценить, не мешает ли он спасаться от ближнего противника или добивать другую цель. Нельзя выдавать sparse damage/kill бонус за гарантированное средство устранения метаний.

## Порядок интеграции и проверки

Сначала закрыть выполняющиеся stochastic336/deterministic224, чтобы отделить sampling от качества политики. Действующие исходники Go/PS/native runtime во время этих capture не меняются. Затем ввести отдельный версионированный objective и evidence, сохраняя прежние recipes/checkpoints. Для возобновления — явный objective fork с архивированным предыдущим reward/config SHA; старые корпуса не считать собственными данными новой политики после обновления.

Проверки контракта reward: непрерывное удержание без урона даёт0 бонуса; урон другой цели/трупу —0; смена life/world/track или разрыв кадров сбрасывает историю; потеря цели и новая угроза не получают штрафа за уход; bonus budget не растёт от повторного переключения/reset; обычный damage/kill credit сохраняется. NN обучение и численные проверки — CUDA; боевое исполнение — Go.

Fresh own-policy сбор на разных картах/составах/оружии, 16-slot pool×2, независимые сиды. Контроль v8 и экспериментальные варианты получают одинаковый бюджет GPU, но каждый свой опыт. Выбор масштабов/лимитов по train, затем отдельные confirmation seeds.

Критерии: больше побед и не больше received/death; меньше switches при обеих доступных целях и больших разворотов; не хуже реакция на новую ближнюю угрозу. Отдельно ordinary/recovery и группа/оружие. Детерминированные evaluation trajectories не использовать как PPO опыт. После улучшения — полноценные карты и реальный coop, демки в нормальной скорости.

## Реализована v12 и запущен собственный сбор

В `internal/learningenv` добавлена `combat_reward_v12`: v8 selected-aim shaping, существующий ограниченный A-B-A cost и опциональный productive-hold bonus. Наблюдаемый entity/track/previous-intent должен непрерывно совпадать минимум3 кадра; новая наблюдаемая clear-shot угроза сбрасывает streak. Native `ServerOutcome.Events` уже содержит target/actor/map/spawncount/frame/class и health damage, поэтому новый input или скрытый серверный state в политику не добавлен. Evidence и история reward существуют только при офлайн-экспорте.

Бонус: `0,002 × подтвержденный health damage выбранной цели`, максимум0,2 за episode/life суммарно для всех целей. Переключение, отсутствие урона, разрыв исполнения и смена track не пополняют бюджет; урон другой цели/трупу не даёт бонуса. Неверная атрибуция/сумма native damage отклоняет reward. Death/handoff бонуса не дают; обычный damage/kill credit сохраняется. Cost-only включает target_churn=-0,02 и bonus0; control сохраняет v8. Для v12 добавлены строгие recipe allowlist, reward coefficient bounds и привязка objective SHA/gamma в CUDA trainer.

Изолированный stage позволил проверить код, пока прежние capture/verification были активны. Focused проверки нового reward, старой v7 и selected-aim v8 прошли; NN inference в этих тестах не выполнялся. После native224 и полного CUDA закрытия confirmation336 изменения интегрированы в основной source. На16 закрытых train боях сохранены все прежние компоненты:43 cycle costs (-0,86 суммарно),49 положительных productive-hold шагов (1,358 суммарно), idle bonus0. Источник `target-commitment-calibration-v1-20261011/complete.json`; это label calibration, не обучение или доказательство качества.

Три warm objective fork подготовлены только на CUDA: actor/critic/std/Adam/RNG/history совпадают с одним parent; источники config/reward запечатаны отдельно. Исправлен перенос отдельного weapon residual LR:0,0003 сохраняется, не заменяется общим actor LR0,00003. Fork probe использует реальные observed train inputs: случайные непрерывные значения не являются допустимыми bbox masks. Неуспешная подготовка v1 сохранила лог, тренировок/шагов optimizer в ней нет. Закрытая подготовка: `combat-target-commitment-preparation-v2-20261011/complete.json`.

Live smoke трёх индивидуально закрытых свежих train captures прошёл native export и CUDA likelihood/context checks: по90 rows на control/cost/bonus,270 суммарно; reward versions v8/v12 и objective SHA/gamma совпали. Receipt: `combat-target-commitment-live-cuda-smoke-v2-20261011/complete.json`. Это не seal всего ещё работающего пула.

Train336 и все три GPU updates закрыты:28 семейств ×4 fresh train seeds ×3 линии, offset248 (проверено против1900 исторических сидов), разные точки base3/ware2/jail1/city1, обычные/recovery и разные loadouts. В текущих pinned registry минимальный train cohort512, поэтому offset512 +4 не помещается; рабочие планы v2 используют248. Fixtures и все условия кроме reward recipe совпали; каждая линия получила собственные captures. Пул16×2, ports40400:336 usable captures, source_unchanged=true. CUDA coordinator завершил контроль:112 боёв/12937 rows; cost:112/12946; bonus:112/12938. Каждая линия получила10 full-actor steps и прошла checkpoint/input-use audits; actor LR0,00003, отдельный weapon residual LR0,0003. Общий train seal: `combat-target-commitment-training-analysis-v1-20261011/complete.json`,38821 rows суммарно. Качество новых весов ещё не подтверждено; продвижение отсутствует.

Дополнительно проверены первые eligible train transitions всех112 matched fixtures: frame/next_frame,1147 наблюдаемых features, полный записанный sample с memory и interventions совпадают у control/cost/bonus. Входные weights одинаковы; recipe fixtures совпадают, потоки привязаны SHA к native export reports. Receipt: `workspace/artifacts/combat-target-commitment-training-start-audit-v1-20261011.json`. Это сравнение записанных данных без NN inference; оно не доказывает совпадение всей последующей траектории и не заменяет CUDA numerical verification. Native preflight для confirmation offset384 проверил все112 новых сидов против2012 исторических, без запуска боёв: `combat-target-commitment-quality-analysis-v1-20261011/preflight.json`.

Coordinator `workspace/artifacts/evaluate-target-commitment-20261011.py` дождался общего completion seal трёх CUDA updates, проверил hashes weights/checkpoint/report, одинаковый бюджет actor10 и LR групп. Запущены784 новых confirmation боя:3 награды ×2 режима выбора действия плюс rules, по112 боёв на контроллер,28 семейств. Offset384 помещается в минимальный confirmation cohort512; перед capture дополнительно проверены отсутствие исторических сидов и точное совпадение episode/instances/seeds/split. Все оценочные линии используют одну reward recipe; оценочные данные не обучают модель. Пул16×2, ports40600; guard32GiB свободного места пройден.

Сравнение включает native победы/урон/смерти по strata, частоту переключений/больших поворотов и простои с видимым противником. Обычный stochastic выбор и deterministic выбор оцениваются отдельно: меньше переключений само по себе недостаточно, если растёт простой или исчезает восстановление оружия. CUDA likelihood verification выполняется только для трёх stochastic линий; deterministic outcomes имеют native evidence, без заявления о PPO replay. Coordinator находится в native-evaluation; итоговые результаты784 пока отсутствуют.

## CUDA probe на одинаковых train histories

Все новые модели повторно прогнаны на одном закрытом control corpus12937 eligible rows с12985 context rows; attention-history каждого нового actor пересчитывается полностью. Parent проверен по записанным likelihood/state, masks и previous-target one-hot фиксированы. Это отдельная CUDA-проверка без обучения и без подстановки сохранённой parent-memory в новый actor. Receipt: `workspace/artifacts/combat-target-commitment-common-history-probe-v1-20261011.json`.

На8272 состояниях с несколькими доступными clear-shot целями и ещё доступной прежней целью вероятность выбрать её вновь:

| Вариант | Средняя вероятность |
|---|---:|
| Parent |45,3534%|
| Control после обучения |45,4046%|
| Cost после обучения |45,4034%|
| Bonus после обучения |45,4079%|

Bonus выше control лишь примерно на0,0033 процентного пункта. Это слабый сдвиг распределения после одного обновления, а не подтверждение уменьшения метания в игре. Probe условен на поведении parent и не является held-out оценкой. В зависимости от native784 следует рассмотреть несколько свежих циклов обучения либо отдельный бюджет target-branch; не увеличивать бонус за простое ожидание и не вводить принудительное удержание цели.

Дополнительный coordinator `workspace/artifacts/analyze-target-commitment-factors-20261011.py` запущен и ждёт полного quality seal. Он создаст8 парных сравнений: cost/control, bonus/control, bonus/cost, rules/control отдельно для stochastic и deterministic. Bonus/cost проверяет добавление productive-hold credit в условиях обучения; обучение выполнялось на собственных captures, поэтому это не доказательство чистого причинного эффекта одной метки на одной неизменной траектории. Все парные capture/environment/model bindings повторно проверяет существующий comparator. В отчёте будут ordinary80/recovery32, gained/lost победы, поведение и описательные интервалы разницы побед через bootstrap28 семейств. Сиды внутри семейства не считаются независимыми; интервалы не являются автоматическим promotion gate и не устраняют общие карты/единственный training seed каждой линии. Аналитически известные крайние случаи (+100/−100/0 п.п.) и синтаксис helper проверены; игровых результатов это не добавляет. Итоговый Markdown будет в `combat-target-commitment-factor-analysis-v1-20261011/results.md`.
