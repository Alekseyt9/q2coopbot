# Бои в разных точках кампании — 07.10.2026

Реестр содержит 8 пространственно разных участков: 4 на base1 и 4 на base2. Для каждого зарегистрированы Blaster и Machinegun (16 рецептов). Источник точки — настоящий monster entity исходного BSP; ближайшие принятые точки одной карты разнесены минимум на 512 игровых единиц. Это изолированные тренировочные бои на исходной геометрии: остальные монстры удалены, двери, платформы, предметы и триггеры сохранены. Полные прохождения кампании остаются отдельной оценкой.

| Карта / точка | Исходная встреча | Координаты исходного монстра | 8 дистанций до стен от позиции игрока |
|---|---|---|---|
| base1 / 01 | monster_infantry, entity 7 | -1824, 1288, 120 | 504, 512, 64, 102, 416, 102, 72, 102 |
| base1 / 02 | monster_infantry, entity 8 | -1240, 1320, -24 | 32, 45, 464, 362, 96, 67, 240, 339 |
| base1 / 03 | monster_soldier, entity 307 | -1496, 1824, -16 | 176, 68, 88, 136, 360, 509, 320, 368 |
| base1 / 04 | monster_soldier_light, entity 309 | -614, 866, -16 | 512, 119, 271, 203, 467, 150, 106, 41 |
| base2 / 01 | monster_infantry, entity 10 | 96, -96, 8 | 86, 157, 47, 59, 42, 150, 81, 256 |
| base2 / 02 | monster_soldier_light, entity 11 | -616, 320, -128 | 160, 226, 200, 204, 232, 328, 312, 204 |
| base2 / 03 | monster_soldier_light, entity 12 | 424, 1688, 24 | 24, 34, 40, 57, 104, 215, 216, 396 |
| base2 / 04 | monster_soldier_light, entity 16 | -112, 2144, -160 | 422, 50, 39, 156, 217, 308, 74, 281 |

Дистанции измерены статическими горизонтальными лучами на высоте глаз с шагом 45°, предел 512. Они описывают разницу окружения, но не являются категорией комнаты или доказательством проходимости динамических дверей. Все стартовые hull, пол, запас высоты для teleport и видимость primary проверены по BSP; нативный старт перепроверяется по server.log. Стартовые hull дополнительно резервируют границы inline brush объектов, а primary должен иметь reset-линию прицела без пересечения дверей.

Для каждого рецепта заданы четыре непересекающихся seed диапазона и разные primary position domains для train/validation/test/confirmation. Каждый инстанс получает отдельный engine seed и детерминированно сгенерированные позиции/HP. Monster HP штатный; Soldier light/ss в этих рецептах заменены на shotgun Soldier, исходный тип сохранён в site.source_class.

Важно: held-out здесь отделяет seed и стартовые позиции внутри участка; обобщение на совершенно неизвестную геометрию требует отдельного location holdout. Углы с невидимым primary пока не включены — текущий reset export требует clear shot.

## Повторная генерация

```powershell
go run ./cmd/q2episode --campaign-sites 4 --root . --out workspace/artifacts/new-campaign-sites.json
```

Команда пишет новый каталог, не заменяя действующий index. Изменение исходного BSP отклоняется по SHA256. Проверки генератора покрывают воспроизводимость, разнос точек, все split и запрет подмены карты/геометрии.

## Общая партия обучения

Готова конфигурация `scripts/scenarios/combat-training-suites/campaign-sites-resume-v1.json`: все 16 рецептов, resume Temporal Update27, одна CUDA эпоха, 64 train captures и 128 validation captures до/после, 4 инстанса x2. Preflight этой партии успешно выполнен: `workspace/artifacts/campaign-sites-training-preflight-v2-20261007`. Конфигурация не является результатом обучения; новый GPU update в этой задаче не выполнялся.

## Живая проверка

Все 8 участков прошли rules-проверку: 32/32 native starts и captures. Base1: `campaign-sites-rules-v3-20261007` (первые четыре завершённые сцены; общий ранний цикл остановился на base2). Base2: завершённый `campaign-sites-base2-rules-v7-20261007`. Завершён Temporal Update27 на всех 16 рецептах: 64/64 captures, все native starts и provenance подтверждены. Всего 96 проверенных живых эпизодов вместе с rules probes. Отчёт: [acceptance-summary.json](../../workspace/artifacts/campaign-sites-learned-v4-20261007/acceptance-summary.json).


Сохранены диагностические неуспешные попытки base2: первоначальное пересечение inline стены; затем блокированная reset-линия прицела через дверь. Эти старты не считаются боевыми поражениями. Исправлены выбор точки игрока (revision3 site01), reserve inline bounds и проверка линии по той же верхней точке прицела, что используется Go. Зависавший поиск campaign exit заменён режимом combat_only только для этих изолированных синхронных рецептов. Старые естественные прохождения используют прежний режим.


capture_valid подтверждает условия старта, seed, контракт наблюдений/команд и native receipts. Старый harness_accepted worker дополнительно требует специфическое отступление от Parasite и переключение с Shotgun; для Soldier/Infantry это не критерий качества. Новые результаты не выдаются за прохождение этих прежних требований. Сравнение качества до/после обучения следует запускать общей конфигурацией с одним adapter/source snapshot.


PPO-export probe: base1 site01 Blaster — 412 проверенных переходов; base2 site01 Blaster — 237; base2 site01 Machinegun — 408. Экспортер заново сверил native receipts и карту/тип цели из замороженного bot-config. Эти captures относятся к validation; данные probe используются только для проверки адаптера и не отправляются на обучение. Следующая общая GPU партия должна собрать свежий train cohort из реестра.


Проверки кода: все cmd/internal Go packages прошли; PowerShell parser прошёл. Полный go test ./... по-прежнему упирается в старые дубли main в игнорируемых папках temporal-mixed-demo-20261006 и combat-mobile-recoil-aim-v1-20261007. Продуктовые пакеты в этом запуске прошли. Изменения game assets и новый native build не требовались.


## Проверочный прогон Temporal Update27

Это stochastic smoke совместимости, без GPU update и без сравнения архитектур. Общие seed cohort совпадают для воспроизводимых условий; прирост качества не измерялся.

| Рецепт | Captures | Цель выполнена живым | Смерти первой жизни | Native kills | Полученный урон |
|---|---:|---:|---:|---:|---:|
| campaign-base1-site-01-blaster | 4 | 3 | 1 | 3 | 233 |
| campaign-base1-site-01-machinegun | 4 | 1 | 1 | 1 | 189 |
| campaign-base1-site-02-blaster | 4 | 3 | 1 | 3 | 170 |
| campaign-base1-site-02-machinegun | 4 | 3 | 1 | 3 | 181 |
| campaign-base1-site-03-blaster | 4 | 3 | 1 | 3 | 142 |
| campaign-base1-site-03-machinegun | 4 | 0 | 4 | 0 | 380 |
| campaign-base1-site-04-blaster | 4 | 2 | 1 | 2 | 242 |
| campaign-base1-site-04-machinegun | 4 | 1 | 3 | 1 | 330 |
| campaign-base2-site-01-blaster | 4 | 4 | 0 | 4 | 81 |
| campaign-base2-site-01-machinegun | 4 | 4 | 0 | 4 | 176 |
| campaign-base2-site-02-blaster | 4 | 2 | 2 | 2 | 204 |
| campaign-base2-site-02-machinegun | 4 | 1 | 3 | 1 | 284 |
| campaign-base2-site-03-blaster | 4 | 4 | 0 | 4 | 0 |
| campaign-base2-site-03-machinegun | 4 | 1 | 3 | 1 | 352 |
| campaign-base2-site-04-blaster | 4 | 3 | 1 | 3 | 100 |
| campaign-base2-site-04-machinegun | 4 | 2 | 2 | 2 | 216 |


## Продолжение через реестр

Партия `campaign-sites-resume-v1-20261007` завершена: 64 свежих train episodes, 4938 проверенных on-policy transitions, один CUDA update 27→28 (10 actor steps, общий счётчик259→269). SHA weights: `29bda0e0a71b5e626de7eda8bfecaab1cd681a5f31f5238a9b5ef2053560b63c`; weights/checkpoint/report completion receipt проверен.

Полная deterministic validation: по64 captures до/после, одинаковые16 рецептов и validation seeds. Победы30/64→33/64; первые смерти31→29; убийства30→33; полученный урон3314→3366. Это небольшой диагностический результат на повторно использованных validation seeds. Превосходство над rules, перенос на полные карты и устойчивое улучшение не доказаны; live/default не переключён.

Первоначальная оценка остановилась на границе победного control handoff. Ошибка экспортёра исправлена, update повторно не выполнялся. Оценки продолжены через [общий пул](combat_instance_pool_20261007.md) с лимитом16. Одна группа base1 site03 MG была отклонена из-за дополнительной аварийной stop-команды при задержке синхронного кадра. Для test lockstep отключён этот повтор без нового кадра; обычная live safety stop сохранена. Сцена повторена для обеих моделей в пуле8. Неудачный запуск и заменённая старая парная группа исключены; итог собирает32 полных проверенных группы/128 captures с повторной проверкой binding SHA и native receipts. Исходные failed reports сохранены.

[Полный результат и все ссылки на captures](../../workspace/artifacts/campaign-sites-resume-v1-20261007/report.json).

| Рецепт | Победы до→после | Смерти до→после | Полученный урон до→после |
|---|---:|---:|---:|
| campaign-base1-site-01-blaster | 1→1 | 3→3 | 288→288 |
| campaign-base1-site-01-machinegun | 0→2 | 4→2 | 360→247 |
| campaign-base1-site-02-blaster | 4→4 | 0→0 | 17→17 |
| campaign-base1-site-02-machinegun | 4→3 | 0→1 | 42→128 |
| campaign-base1-site-03-blaster | 1→2 | 3→2 | 374→336 |
| campaign-base1-site-03-machinegun | 0→0 | 4→4 | 380→380 |
| campaign-base1-site-04-blaster | 4→3 | 0→1 | 120→270 |
| campaign-base1-site-04-machinegun | 0→0 | 3→3 | 330→332 |
| campaign-base2-site-01-blaster | 4→4 | 0→0 | 102→104 |
| campaign-base2-site-01-machinegun | 2→4 | 1→0 | 113→28 |
| campaign-base2-site-02-blaster | 0→0 | 4→4 | 360→360 |
| campaign-base2-site-02-machinegun | 0→1 | 3→2 | 280→220 |
| campaign-base2-site-03-blaster | 1→1 | 3→3 | 240→240 |
| campaign-base2-site-03-machinegun | 4→4 | 0→0 | 0→0 |
| campaign-base2-site-04-blaster | 4→3 | 0→1 | 24→140 |
| campaign-base2-site-04-machinegun | 1→1 | 3→3 | 284→276 |



Дополнительная диагностика предыдущего stochastic smoke: [выбор оружия](../../workspace/artifacts/campaign-sites-learned-v4-20261007/weapon-choice-smoke-diagnostics.json). Во всех MG сценах provider большую часть кадров удерживает MG; присутствуют переключения в обе стороны. Например, base1 site03 MG: 433 кадров MG и 64 Blaster при 0/4 побед. Это не доказывает причину поражения, но не поддерживает объяснение исключительно неправильным выбором оружия. Нужны дальнейшее обучение на этих условиях и оценка прицела/движения.
