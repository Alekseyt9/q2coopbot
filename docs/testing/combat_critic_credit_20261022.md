# Подготовка критика перед PPO, 22.10.2026

## Гипотеза и границы опыта

В [предыдущем аудите](combat_future_auxiliary_ppo_20261022.md) все 3363 behavior value estimates исходного TRAIN корпуса были нулевыми. Actor обновлялся до критика. Проверяем, помогает ли предварительное обучение baseline связывать ранние действия с отложенной наградой. Награда, архитектура, условия боя и правила исполнения не меняются. Это пока гипотеза, а не установленная причина проигрышей.

В `scripts/ppo_recurrent.py` добавлены явные режимы `--critic-credit legacy|mc-after|mc-before` и отдельный `--fork-critic-credit`. Режим и переход сохраняются в checkpoint; незаявленная смена режима при resume запрещена. Default — прежний `legacy`.

Три обучения на одном существующем TRAIN корпусе, без новых данных и без development fit:

1. `legacy`: прежний PPO actor, затем critic на прежних GAE targets.
2. `mc-after`: прежний PPO actor, затем 40 critic updates на возвратной награде gamma=.99, lambda=1.
3. `mc-before`: те же 40 critic updates на тех же targets до actor; затем пересчёт baseline на полных causal histories и PPO с прежними gamma=.99/lambda=.95.

Все получают одинаковые source actor/critic/std/Adam/RNG, 3363 rows, sequence retention, actor scope all и предел 10 actor updates. Ветвям mc-after/mc-before выделен одинаковый бюджет critic. Исходная холодная оценка поведения проверяется до warmup; записанные samples/log probabilities не подменяются. Для bootstrap используются фактические next context/features/reset, terminal и bootstrap_zero. Truncation не склеивается с последующим боем.

`scripts/combat_critic_credit.py` содержит отдельные функции fit и causal baseline replay. `scripts/test_combat_critic_credit_cuda.py` проверяет causal successor, terminal/zero-bootstrap, неизменность samples и отказ при несовпадающих next features. Вся нейросетевая математика — CUDA.

## Проверенные результаты обучения

Артефакты: `workspace/artifacts/combat-critic-credit-pair-v1-20261022`.

- Все три fit завершены: по 10 actor / 40 critic updates.
- `scripts/audit_combat_critic_credit_cuda.py` проверил исходные границы actor/critic/Adam/RNG, hashes и отсутствие повторного потребления TRAIN rollout в исходном checkpoint.
- Default legacy runtime weights побайтно совпали с прежним закрытым PPO control.
- Actor/std/Adam у mc-after точно совпали с legacy: смена цели критика после actor не изменила обучение действий.
- Итоговые critic weights и critic Adam у mc-before/mc-after точно совпали. Их TRAIN MSE одинаково снизился с 32.725 до 6.265. Это TRAIN fit, не независимая точность baseline.
- Warmup оставил actor/std неизменными; последующий actor fit с новым baseline дал отличающиеся веса. KL legacy/mc-after 0.0049273, mc-before 0.0038983, в пределах прежнего guard 0.005.

Fresh paired smoke завершён: source / mc-after / mc-before / rules, по 8 боёв, 32 native и 24 CUDA replay, seed offset692, пул16 ×2. Legacy отдельно не прогоняется: его actor/std точно совпадают с mc-after. Spec проверил отсутствие пересечения seeds с предыдущими планами.

| Метрика | Source | MC после actor | MC до actor | Rules |
|---|---:|---:|---:|---:|
| Победы | 2/8 | 2/8 | 4/8 | 4/8 |
| Убийства | 7 | 6 | 10 | 10 |
| Нанесённый урон | 1681 | 1513 | 1949 | 1886 |
| Полученный урон | 548 | 540 | 509 | 325 |
| Смерти | 6 | 5 | 4 | 2 |

MC-before парно получает 2 gained / 0 lost против source и mc-after. Против rules 2 gained / 2 lost: одинаковая итоговая победность скрывает разные выигранные случаи. Rules заметно лучше сохраняют здоровье. Малый smoke поддерживает следующий опыт, но не доказывает превосходства над rules или устойчивости результата.

`smoke-decision.json` закрепил переход только к независимому screen без refit. Независимый screen тех же фиксированных source / MC-before / rules закрыт: 72 native / 48 CUDA, 12 ordinary +12 recovery на вариант, offsets696–698, пул16 ×2.

| Screen | Source | MC до actor | Rules |
|---|---:|---:|---:|
| Победы | 12/24 | 11/24 | 13/24 |
| Убийства | 29 | 29 | 31 |
| Нанесённый урон | 5291 | 5837 | 5711 |
| Полученный урон | 1500 | 1613 | 693 |
| Смерти | 9 | 11 | 4 |

К source: 0 gained /1 lost; к rules:5 gained /7 lost. Smoke-сигнал не подтвердился. `decision.json`: warm_critic_not_advanced_after_screen, default сохранён. Никакого дальнейшего promotion или coverage этой ветки не назначено. Возможность будущего critic warmup остаётся исследовательской функцией, а не принятой заменой baseline.

Фактические MG shots на native-exclusive окнах: source218/459 (47.49%) дали урон живым монстрам, MC-before271/429 (63.17%). Targetless MG shots без видимого bbox83→19. Стрельба стала продуктивнее, но общий бой хуже; одна точность не является критерием продвижения. Подробности `screen-aim.json` и `screen-outcomes.json`.

## Диагностика base3 после screen

Обе learned ветки выиграли0/6 на base3, rules4/6. `scripts/report_combat_elevation_losses.py` проверяет только фактически исполненные native-exclusive alive frames и клиентские наблюдения; hidden positions не используются. `screen-elevation.json` связывает trace/reward/behavior hashes.

- Source опускался на192 units ниже исходной площадки в3/6 боёв base3, MC-before в4/6. Само снижение не доказывает ошибку: спуск может быть намеренным манёвром.
- В recovery seed1244126696 source имеет220 кадров одновременно ниже старта минимум на96, без видимых противников и с последним наблюдением врага минимум на64 выше. У MC-before тот же режим71 кадр; в ordinary seed1164166697 —166 кадров. При10Hz это22,7.1,16.6 игровых секунд соответственно.
- Эти наблюдения поддерживают отдельную гипотезу потери позиции/невосстановленного контакта. Они не объясняют все смерти: часть проигрышей происходит без глубокого спуска.
- Нужные исходные сведения уже существуют:8 ground-drop probes на40units, ground mask/range32; remembered threats с XYZ/velocity/age до200frames. В fixture navigation goal=`combat_fixture`, без route waypoint. Это не основание объявлять отсутствие геометрии во входах.
- Spacing potential использует горизонтальную дистанцию до видимого Parasite (`internal/learningenv/reward.go`), поэтому нельзя утверждать, что модель прямо получает spacing bonus за увеличение вертикальной дистанции.

Следующая проверка: по существующим TRAIN captures отделить начало спуска на земле от прыжка/воздушной фазы; измерить предупреждающие ground probes и применимость уже имеющегося PMove diagnostic. Он проверен для dry-ground движения и не покрывает прыжок; неизвестные случаи нельзя выдавать за безопасные/опасные counterfactual labels. Только после этой проверки — training-only движение/сохранение контакта как отдельная гипотеза, с обычными боями как защитой от запрета полезных спусков. Runtime actor сохраняет полное управление; blanket запрет падений/подмена движения правилами не добавлялись.

## Хранение

После закрытия всех соответствующих pipelines проверены192 крупных JSONL файлов трёх завершённых экспериментальных roots (5.88GiB логического размера). Применена прозрачная NTFS LZX compression с SHA256 до/после каждого файла; пути и bytes сохранены, удалений0. Многие файлы уже были сжаты: заметного дополнительного освобождения не получено. `ntfs-compression-audit.json` фиксирует глобальное изменение свободного места; параллельная запись на диск не позволяет приписывать его целиком операции. Для следующего native запуска по-прежнему действует storage guard32GiB.

## Ограничения

Дополнительно закрыт read-only CUDA diagnostic `development-value-report.json` на существующих отложенных source-policy траекториях, без fit/selection. Для gamma=.99/lambda=1 observed-return targets MSE: исходный нулевой critic 27.599, legacy 21.512, MC critic 19.840. Explained variance соответственно 0%, 0.88%, 6.96%. MC baseline лучше по общей ошибке, но объясняет небольшую часть вариативности; MSE не следует трактовать как точное предсказание исходов. Legacy обучался на более коротких GAE targets, поэтому эта таблица измеряет именно качество длинного return, а не сравнивает одинаковые training objectives. Development data принадлежит исходной политике; это не проверка калибровки на новых действиях MC-before. Скрипт: `scripts/report_combat_critic_credit_cuda.py`.

Baseline обучается и применяется на одном on-policy TRAIN батче. Это отдельная проверка порядка/credit assignment; она не доказывает обобщение критика и может усилить переобучение. Полная возвратная награда при lambda=1 всё ещё использует записанные значения на реальных bootstrap boundaries. Поведение на новых боевых seeds решает, заслуживает ли гипотеза следующей проверки. Совпадение critic/Adam изолирует данный pilot; продолжение на собственных новых траекториях потребует новых paired corpora и отдельной проверки resume.
