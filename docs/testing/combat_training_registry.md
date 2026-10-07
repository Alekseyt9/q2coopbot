# Реестр генераторов боёв и общий учебный цикл

Дата: 07.10.2026. Реестр: `scripts/scenarios/combat-training/index.json`. Это учебные рецепты и генераторы стартов; результаты каждого исполнения сохраняются отдельно в `workspace/artifacts`. Старый [реестр регрессий](episode_registry.md) применяется для воспроизведения ошибок и остаётся отдельным инструментом.

## Доступные семейства

| ID генератора | Враги | Оружие на старте | Контроллеры |
| --- | --- | --- | --- |
| `parasite-blaster-generated` | Parasite | Blaster | rules, learned |
| `parasite-gunner-blaster-generated` | Parasite + Gunner | Blaster | rules, learned |
| `parasite-machinegun-recoil` | Parasite | Machinegun, 100 bullets + Blaster | learned |
| `parasite-gunner-machinegun-recoil` | Parasite + Gunner | Machinegun, 100 bullets + Blaster | learned |

Machinegun использует штатные отдачу, разброс, cadence, расход ammo и physics; скрипт не компенсирует отдачу за policy. Kick angles входят в текущие наблюдения модели. Имя упражнения не доказывает, что модель уже научилась компенсации. Фактическое оружие может смениться по решению поддерживаемой weapon head; оно записывается в трассу. Rules для этих MG-рецептов пока не поддержан существующим фиксированным teacher/harness.

Кроме четырёх генераторов сохранены четыре прежних фиксированных Parasite/Gunner учебных рецепта, две естественные карты только для оценки и три planned рецепта Soldier/Infantry. Planned не исполняются и не засчитываются как покрытие обучения.

## Что генерируется

В JSON каждой новой записи есть `generator.version=1`, `generator.kind=base1-ground-combat-v1` и `distributions` для четырёх splits. Каждый split задаёт `player`, `primary`, опциональный `flank` через `min/max` координат и список допустимых `health`. Выбор выполняется в Go; координаты квантуются до 1/8 unit. Generator seed вычисляется из ID/revision/split/engine seed без зависимости от модели. У сравниваемых моделей одинаковая спецификация и cohort дают одинаковые выбранные условия.

Текущая область — штатная стартовая площадка base1, исходный BSP. Игрок: x16..48, y−240..−208; Parasite: y−248..−200. X Parasite в train140..164, validation176..188, test200..212, confirmation224..236. Gunner в revision2: x80..104, y−184..−152. Z24.125 сохраняет hull над плоскостью пола. Здоровье игрока80/100; здоровье обоих монстров штатное175, skill1. Четыре диапазона engine seeds разделены глобально между записями; диапазоны первичной позиции train и отложенных splits также не пересекаются.

Число/тип монстров и loadout выбираются семейством, позиции и здоровье — генератором. Вариации угла старта, произвольного ammo, карт, новых стен, предметов и скрытого противника пока не реализованы. Отложенные оценки здесь проверяют новые позиции/дистанции внутри той же площадки; перенос на другую геометрию этим не доказан.

Каждый кандидат проверяется по реальному BSP: консервативный hull объём для игрока и монстров, опора пола, запас места для начальной teleport фазы, отсутствие пересечения акторов и линия до primary. Для дополнительного монстра в entity lump резервируются штатные точки появления игрока: монстр существует ещё до teleport и иначе может погибнуть от setup telefrag. Текущий reset exporter требует наблюдаемого primary; поэтому старт за укрытием пока нельзя честно подключить этим адаптером. Максимум128 попыток; причины отклонений и число попыток сохраняются. Если допустимое состояние не найдено, подготовка завершается ошибкой до запуска серверов.

## Проверка и сохранение условий

План содержит версию/hash записи, seed генерации и engine seed, выбранные player/monster позиции, HP, loadout, skill, источник и SHA256 BSP. Перед запуском `q2episode --verify-plan` повторно генерирует условия и сверяет весь план, bindings, reward и модель. Изменённые условия, рецепты и assets не исполняются молча.

Baseline передаёт каждому из четырёх worker его fixture JSON; дополнительный Gunner размещается в собственном entity lump до старта. Основной противник и игрок размещаются штатными test командами существующего харнеса. При fixed release100 серверные `g_test_entity_start` подтверждают фактические позиции, здоровье, solid и точный состав; reset exporter дополнительно проверяет первую пригодную наблюдаемую позу, оружие/ammo и primary. Отдельные seed и synchronous barrier/RNG proof сохраняются.

Статический BSP preflight и native pose receipt не являются полной проверкой динамических hull/movers или восстановлением всего engine состояния. Штатная геометрия и исключение overlap проверены; произвольные новые динамические препятствия требуют дополнительного native collision адаптера.

## Команды

Из `F:\src\quake2\q2coopbot-src`, используя PowerShell7:

```powershell
go run ./cmd/q2episode --list

go run ./cmd/q2episode --episodes parasite-blaster-generated `
  --split validation --mode rules --count 4 `
  --out workspace/artifacts/my-blaster-plan.json `
  --artifacts workspace/artifacts/my-blaster-run

./scripts/run_registered_combat_episodes.ps1 `
  -Plan workspace/artifacts/my-blaster-plan.json -DryRun

./scripts/run_registered_combat_episodes.ps1 `
  -Plan workspace/artifacts/my-blaster-plan.json -Port 34300
```

И план, и runtime output должны быть новыми путями. Повтор можно запустить в другой output root с тем же cohort для сравнения; для новых учебных эпизодов менять `--seed-offset` внутри train split. Каждый seed означает самостоятельный запуск сервера/клиента, четыре инстанса работают параллельно, x2. Пакеты игры подключаются штатным immutable runtime механизмом; реестр не копирует PAK для каждого эпизода.

Для learned добавляются `--mode learned --model <weights.json>`. Train требует stochastic PPO weights; evaluation использует deterministic копию в общем training runner. Go inference adapter должен поддерживать архитектуру/feature/action contract. Маски выбора оружия и guards остаются видимыми в capture.

## Новая архитектура и продолжение модели

Общий конфиг suite задаёт `episodes`, `evaluation_episodes`, `models`, `epochs`, `episodes_per_scene`, `evaluation_count`, `training_seed_offset`, версии PPO objective и optional checkpoint. Пример продолжения Update26 на Blaster/MG: `scripts/scenarios/combat-training-suites/generated-blaster-machinegun-resume-v1.json`.

```powershell
./scripts/run_registered_combat_training.ps1 `
  -Config scripts/scenarios/combat-training-suites/generated-blaster-machinegun-resume-v1.json `
  -OutputRoot workspace/artifacts/my-registry-training -DryRun

./scripts/run_registered_combat_training.ps1 `
  -Config scripts/scenarios/combat-training-suites/generated-blaster-machinegun-resume-v1.json `
  -OutputRoot workspace/artifacts/my-registry-training-live -Port 34400
```

`comparison_kind=continuation` возобновляет checkpoint с optimizer/RNG/counters и проверяет совпадение весов/конфигурации. `comparison_kind=architecture` требует одинаковых observation/action contracts, общего объявленного initialization group и нулевого prior environment budget; прошлое обучение всё равно следует подтверждать происхождением исходных моделей. Проверка декларации не устанавливает автоматически, что веса были случайными. Несколько независимых training seeds нужны для полноценного вывода о качестве архитектуры.

Каждая эпоха собирает все выбранные семьи текущими замороженными весами. Каждый native batch отдельно проходит `q2ppo-data` replay; `--merge` объединяет только совместимые свежие экспорты с одинаковыми behavior/reward/features и непересекающимися episode seeds. Все исходные native receipts, исходные rollout и sequence файлы остаются закреплены hash. Память, indices и seed не перенумеровываются; последовательности разных эпизодов не склеиваются. После объединения выполняется один CUDA PPO update, затем следующий epoch получает новые веса.

Готовый конфиг всех четырёх семейств: `scripts/scenarios/combat-training-suites/generated-blaster-machinegun-curriculum-v1.json` —4 эпохи,8 эпизодов на семейство в эпоху,16 validation эпизодов на семейство до/после. Его preflight прошёл; полный этот бюджет ещё не запускался. Smoke-конфиг выше обучает только Solo Blaster/MG и оценивает все четыре семейства.

Доли сцен пока задаются одинаковым числом выделенных эпизодов; доли PPO samples зависят от принятых переходов. Перевзвешивание loss по желаемому curriculum и бюджет по равному фактическому опыту пока не реализованы. В отчёте есть вклад каждой сцены и фактические rows. Число allocated episodes нельзя выдавать за равный объём опыта моделей.

Оптимизация только CUDA; CPU используется для Go inference и диагностического чтения checkpoint. Поддерживаемые адаптеры: MLP с8 действиями, GRU, temporal/entity attention при совместимых контрактах. MLP с20-output weapon head и произвольная новая сеть требуют отдельного trainer/export/inference адаптера. Активная retention loss в continuation требует своего адаптера; текущий suite использует нулевые retention/bank weights.

## Оценка и артефакты

До/после оцениваются одинаковые validation условия с deterministic policy, по каждому семейству; rules включается там, где рецепт его поддерживает. В `report.json` выводятся живые завершения, смерти, убийства и полученный урон; native damage/capture trace остаются источниками подробной диагностики прицела, weapon use и kick angles. Финальные test/confirmation нельзя многократно использовать для подбора модели.

Выходы: замороженный protocol, plan каждого сбора, sampled fixtures, native start receipts, capture provenance, отдельные и общий PPO rollout с sequence histories, CUDA report/checkpoint/completion receipt и validation leaderboard. Live/default контроллер автоматически не меняется. Маленький smoke-run подтверждает связность pipeline, но не превосходство над rules.

После завершения можно получить дополнительную оружейную диагностику:

```powershell
./scripts/analyze_registered_combat.ps1 `
  -RunRoot workspace/artifacts/my-registry-training-live `
  -Output workspace/artifacts/my-registry-training-live/weapon-diagnostics.json
```

Диагностика разделяет native projectile ledger для Blaster и damage contacts для hitscan. Наличие kick angles не доказывает компенсацию: они содержат и отдачу оружия, и реакцию на входящий урон. Наблюдаемый расход bullets — нижняя граница по соседним кадрам с MG; он не используется как точный знаменатель accuracy.

## Подтверждённый smoke07.10.2026

Завершён [общий отчёт](../../workspace/artifacts/combat-generator-resume-v2-20261007/report.json):8 training captures и32 validation captures,4 инстанса x2, все native start/provenance проверки пройдены. Сбор8 train распределён между Solo Blaster и Solo MG, одна замороженная Update26 policy. В joint PPO вошли429 и344 перехода, всего773. Один CUDA update продолжил checkpoint26→27,249→259 actor steps; final approximate KL0.00983191. [CUDA report](../../workspace/artifacts/combat-generator-resume-v2-20261007/temporal26/epoch-1/update/report.json), [completion receipt](../../workspace/artifacts/combat-generator-resume-v2-20261007/temporal26/epoch-1/update/complete.json).

| Семейство,4 paired validation seed | Живые победы до→после | Смерти первой жизни до→после | Убийства до→после | Полученный урон до→после |
| --- | --- | --- | --- | --- |
| Solo Blaster | 4→4 | 0→0 | 4→4 | 218→220 |
| Parasite/Gunner Blaster | 0→0 | 4→4 | 0→1 | 380→380 |
| Solo Machinegun | 4→4 | 0→0 | 4→4 | 225→186 |
| Parasite/Gunner Machinegun | 1→1 | 2→2 | 2→2 | 181→181 |

Всего9/16 побед до и после,6 смертей до и после. Есть меньший урон Solo MG на этой маленькой когорте, но устойчивый прирост качества не доказан. Групповая Blaster сцена остаётся нерешённой. Это проверка реестра/смешанной партии/resume, не architecture benchmark и не основание менять live/default.

[Оружейная диагностика](../../workspace/artifacts/combat-generator-resume-v2-20261007/weapon-diagnostics-v2.json) содержит32 результата. После update в Solo MG зарегистрирован наблюдаемый расход285 bullets по четырём эпизодам; ненулевой kick pitch есть в281 provider кадре из327. В группе MG расход260, kick157 из483 кадров. Это подтверждает реальную стрельбу и передачу kick в наблюдение, а не освоенную компенсацию.

Ранее первый опыт `combat-generator-resume-v1-20261007` прошёл8 train captures и CUDA update874 rows, но общий цикл остановился на native проверке группового старта: startup Gunner пересекал native spawn клиента и погибал от telefrag до боя. Этот цикл сохранён как failed и не считается завершённой оценкой. Исправлены spawn exclusion и revision2 диапазон Gunner; повтор использовал новые training seeds600004–600007/620004–620007. Ошибка генерации не засчитана как боевое поражение модели.

Дополнительный rules probe Solo Blaster:4/4 подтверждённых старта и живых завершения в `combat-generator-probe-20261007`. Оценку на новых составных сценах rules/MG это не заменяет.

Проверки: `go test ./cmd/... ./internal/...` прошёл; покрыты повторяемость условий, независимые seed, domains leakage, изменение frozen plan, setup telefrag, смешение policy/objective, повторение episode seed и изменение native receipts при merge. `go test ./...` не проходит из-за прежних нескольких `main` в двух игнорируемых папках артефактов (`temporal-mixed-demo-20261006`, `combat-mobile-recoil-aim-v1-20261007`); все продуктовые Go packages прошли. PowerShell parser и live UDP checks пройдены; runtime PAK hardlinks убраны штатным cleanup после завершения captures.

## Развитие

Добавлен [общий пул разных сцен и моделей](combat_instance_pool_20261007.md). Проверены лимиты 16/24/32; рабочий — 16. Поле `pool_instances` в training config включает сбор всех моделей текущей эпохи через общую очередь; CUDA updates, опыт и checkpoints остаются отдельными. `pooled-two-models-smoke-v1.json` прошёл полный цикл с двумя версиями Temporal: 16 train captures, 1640 проверенных PPO transitions, два CUDA updates и 32 validation captures. Это проверка инфраструктуры, а не сравнение архитектур.

Добавлены исполняемые Soldier/Infantry и 16 рецептов на 8 разных участках base1/base2: [точки кампании](campaign_combat_sites_20261007.md). Для этих изолированных боёв отключён поиск выхода с карты, наблюдения боя сохранены.

Следующие адаптеры: составные группы Soldier/Infantry; углы и инвентарь; отложенная геометрия/укрытия с поддержкой reset без видимого primary; проверка native hull/достижимости; scene loss weights и равный опыт архитектур; участки естественных карт с PPO export. Увеличение реестра без исполняемого рецепта и проверки native старта не считается добавленной тренировкой.
