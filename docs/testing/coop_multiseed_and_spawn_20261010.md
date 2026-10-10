# Кооператив: закрытый batch16 и подключение companion

`natural-companion-bolt-multiseed-v1-20261010` закрыт:16 native scenes,
16 серверов ×2 клиента, timescale2,1800frames,skill1. Source unchanged;
все client runs завершились штатно. Новые веса не обучались.

| Карта | Companion | Переходы | Смерти companion | Combat friendly damage | Telefrag damage |
|---|---|---:|---:|---:|---:|
| base1 | rules | 3/4 | 1 | 0 | 200 |
| base1 | learned | 3/4 | 0 | 0 | 300 |
| base2 | rules | 2/4 | 3 | 0 | 0 |
| base2 | learned | 2/4 | 2 | 0 | 0 |

Пять смертельных событий mod21 — native KillBox при появлении второго
клиента в занятой стартовой точке. По first client snapshot и damage frame
это startup telefrag. Проверенный пример: actor1 на frame34 находится
[42.875,-238.125,24.125], actor2 появляется на первом frame35 в
[32,-224,34]. Это не промахи learned fire и не повтор проблемы barrel.
Все raw damage сохранены; они не вычеркнуты из отчёта.
`friendly-closure.json` содержит классификацию событий и SHA исходного report.

Эти пять сцен нельзя считать чистой проверкой совместного боя с человеком:
ведущий был убит подключением. Combat friendly damage0 в данном batch
не доказывает общую безопасность или превосходство модели. Модель нанесла
мало собственного урона; часть убийств — добивания после ведущего.

Server RNG seeds в batch различались, однако learned action RNG использовал
общий sampling_seed20261010. В harness исправлено: каждый stochastic task
получает отдельный provider file с sampling_seed=server seed и записанным
SHA. Параметры сети не меняются; model identity исключает sampling_seed.

## Проверка ожидания стартовой зоны

Harness ждёт три свежих наблюдения живого ведущего вне радиуса512 от
начальных spawn locations, прежде чем запускает companion. Карта, spawn
entities, уязвимость и движение игроков не подменяются. Startup telefrag
отдельно маркируется infrastructure failure; combat и telefrag damage
считаются отдельно, общий damage также сохраняется.

`natural-companion-spawn-seed-smoke-v5-20261010`:4 scenes,1200frames,
timescale2,seeds185011/185111, штатные веса. Source unchanged=true.
Три сцены стартовали корректно, combat friendly и telefrag damage0.
Все три остановлены frame budget, уровни не завершены.
Четвёртая, base2 rules, не стартовала: ведущий за25 wall seconds не вышел
из стартовой зоны. Harness завершился с ошибкой инфраструктуры; этот run
не является отрицательным результатом combat модели.

Ожидание512 слишком мешает сценам, где ведущий сражается около spawn;
уменьшать радиус без ограничения задержки model load → protocol begin
нельзя считать доказанным исправлением. Следующий шаг — загрузить provider
до разрешения подключения и освобождать protocol begin по свежей проверке
занятости стартовой точки. Затем повторить полный batch без startup failures.

Human server1976 и bot3556 оставлены; все процессы обоих batch закрыты.

## Предварительная загрузка provider и разрешение подключения

Добавлены необязательные `run.connect_ready_file`/`connect_release_file`.
После полной проверки и загрузки provider, до открытия UDP и отправки
пакетов, Go пишет ready с PID и случайным token. Затем ждёт совпадающий
token release не более30s, учитывая отмену context. Старые paths не
переиспользуются: ready создаётся exclusively, release обязан отсутствовать.
Gate разрешён только на loopback; обычные запуски без этих paths прежние.
Это startup coordination, не остановка физики и не изменение команды боя.

Harness заранее запускает companion и ждёт ready. После трёх свежих
наблюдений живого ведущего вне192units от начальных spawn locations
отдаёт token. Нет задержки загрузки модели после проверки зоны;
protocol handshake остаётся, поэтому это не абсолютная гарантия для
произвольной задержки или возвращения игрока. Startup telefrag по-прежнему
делает run infrastructure failure. Проверяется freshness trace менее1s.

Тесты актуального token, stale paths/release, cancellation, loopback,
campaign leader и live config прошли (`connect-gate-regressions.jsonl`).

`natural-companion-ready-gate-smoke-v6-20261010`:4/4 infrastructure_ok,
source_unchanged=true,1200frames,timescale2,seeds185011/185111.
Все4 имеют0 telefrag и0 combat friendly damage. Ведущий освобождает
зону на frames42/44 для base1 и80/82 для base2. В отличие от v5,
base2 rules запускается. Learned provider frames407 на base1 и545 на
base2; base1 learned завершён, остальные сцены остановлены frame budget.
Это проверка запуска и наличия боя, не доказательство преимущества policy.
Веса сети не менялись; stochastic sampling_seed отдельный для каждой карты.

Следующая оценка — полный повтор16 сцен с готовым provider и отдельными
seed. Результаты старого batch со startup telefrags не заменяются.

## Закрытый повтор16 без startup failures

`natural-companion-ready-multiseed-v2-20261010` завершён. Все16 scenes
infrastructure_ok=true,source_unchanged=true; telefrag и combat friendly
damage0 в обе стороны.1800frames,timescale2,skill1,16 slots ×2 clients.
Server seeds185010..185013 для base1,185110..185113 для base2;
каждый learned task имеет соответствующий sampling_seed.

| Карта | Companion | Переходы | Смерти companion | Его убийства | Его monster health damage |
|---|---|---:|---:|---:|---:|
| base1 | rules | 4/4 | 0 | 1 | 40 |
| base1 | learned | 4/4 | 0 | 1 | 50 |
| base2 | rules | 1/4 | 3 | 11 | 325 |
| base2 | learned | 0/4 | 2 | 7 | 200 |

Прохождение — результат всей пары. Урон/убийства в таблице принадлежат
только companion, не ведущему. Все companion использовали только Blaster.
Преимущество learned не установлено: меньше смертей, но хуже завершение
и меньше собственного урона на base2. Четыре seeds не дают устойчивого
статистического вывода. Веса не менялись и не promoted.

Добавлен streaming report_natural_companion_batch.py: читает весь закрытый
trace, сохраняет SHA/размер, пропускает duplicate snapshots, считает
provider decisions, proposals/sent fire, suppressions, скорость и ожидания.
Provider counts совпали с исходным native harness report во всех16 scenes.
Artifact behavior-summary.json содержит оба клиента каждой сцены.

Base2 learned:1935 attack proposals,1193 sent attack frames,742 отмены
(38.3%). Только51 из1193 sent attack frames имеют XY speed<10 (4.3%).
Это команды/наблюдаемая скорость, не фактические выстрелы или попадания.
Основная отмена — bolt_partner_launch_guard; убирать его по одному проценту
нельзя, поскольку предыдущая native сцена показала настоящий friendly damage.

Потеря ведущего тоже существенна: в base2 seed185112 есть741 последовательный
wait_for_teammate frame (74.1 game seconds) после потери наблюдения. На
первом frame1094 последняя позиция [-205,2037.5,-167.875] имеет age201;
бот находится [516.25,2366.75,-231.875]. Это истечение существующего
20-second bounded long-search, а не доказательство смерти ведущего.
На base1 три learned сцены также ожидали24.0..28.8 game seconds.

Следующие приоритеты: восстановление сопровождения по доступным наблюдениям,
естественный подбор оружия и обучение бою с напарником в observation.
Сохранять ограничение на stale positions и безопасные search routes;
не подставлять истинную позицию второго клиента в observations модели.
Natural companion captures пока evaluation-only: штатные registry recipes
для full-map evaluation имеют ppo_trainable=false. Их нельзя автоматически
приравнять к validated isolated on-policy training corpora.

Закрытые JSONL обоих16-scene batches сжаты штатным scoped maintenance
скриптом в прозрачный Windows WOF LZX:843744872→185434112 bytes и
802192139→155664384 bytes. Освобождено примерно1.3GB без удаления строк,
изменения путей и логических размеров. Новый batch имеет32 SHA проверки
после сжатия, совпадающие с behavior-summary.json;
compression-verification.json сохраняет результат. Человеческая сессия
не входит в область maintenance и продолжает работать прежним бинарником.
