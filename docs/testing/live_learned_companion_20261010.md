# Обученный бот в кооперативе с игроком — 10.10.2026

Живая сессия: `workspace/runtime/live-coop-20261010-133308`,
loopback UDP29110, coop1,skill1,cheats0,timescale1.
Сервер и бот продолжают работать; игрок подтвердил закрытие клиента. Актуальные PID/пути в
`workspace/artifacts/processes.json` и `current-live-coop.txt`.
Yamagi подтвердил windowed OpenGL3 drawable1920×1080; WASD и MOUSE2 jump.
Исходный пользовательский config.cfg сохранен в human-config-backup.cfg,
малый отдельный override меняет лишь оконный размер и запрошенные клавиши.

Первый запуск бота не подключился: старый direct-mode guard допускал
только isolated fixture либо finite campaign evaluation. Добавлен явный
`combat.live_companion=true`: локальный PPO, frame pacing и capture,
обычный companion navigation без fixtures/LLM. Следование игроку не заменено
автономным прохождением. Общие проверки natural combat сохранены;
новые тесты live companion и существующие campaign isolation прошли.
Поддержанные естественные вооружения совпадают с campaign learned mode.

Игрок сообщил, что бот стоит и не стреляет. В первой live-записи для
396 уникальных ранних frames188 были provider-owned, лишь5 candidate attack,
184 static_hull_blocked. Ограничение столкновений останавливало движение
в стену, а модель почти не выбирала огонь. При старте был установлен
deterministic=true; исходный опубликованный attention64 update4 имеетfalse.
Только бот штатно отключен stop_file и переподключен с неизмененной копией
исходных stochastic weights. Сервер/человеческий клиент не перезапускались.

Последующий live-срез357 записей map/frame:159 provider frames,
148 candidate attack,147 sent attack commands,230 movement commands,
20 static_hull_blocked,13 component clipping,33 dynamic door stops,
1 friendly line-of-fire block. Это команды, не число реальных попаданий;
набор условий/карт отличается от раннего среза, причинное superiority-сравнение
отсутствует. Контроль friendly fire и столкновений сохраняется. Общая надежность
навигации и боя с реальным напарником пока не доказана.

Текущая trace: `bot-stochastic.jsonl`, config `bot-stochastic-config.json`.
Живую сессию не останавливать для обучения/оценочных прогонов.
Параллельно запущен отдельный A/B опыт reward v8
с выбранной целью; текущий бот использует прежние опубликованные веса,
v7 не принят. Продолжить v8 отдельно от человеческой сессии.

После выхода игрока последняя запись base3 показывает `wait_for_teammate`,
пустой список наблюдаемых врагов и нулевые команды. Этот простой ожидаем
при отсутствии напарника и не характеризует поведение модели в бою.
Клиент автоматически повторно не открывается.

## Повторная диагностика активной части записи

Добавлен `scripts/report_live_companion_behavior.py`. Отчёт:
`workspace/artifacts/live-companion-review-20261010/behavior.json`.
Чтение ограничено размером файла на старте и заканчивается после200
последовательных snapshots ожидания отсутствующего напарника. SHA относится
к реально прочитанному префиксу. Поздние переподключения после этой границы
не входят; это не заявление о просмотре всего продолжающего расти файла.
Считается первая строка каждого последовательного snapshot key.

В первой записи1119 provider snapshots,700 candidate attack,669 sent attack;
617 snapshots со скоростью XY<10,548 static_hull_blocked. Раннее наблюдение
188 provider frames/5 attack относится только к начальному срезу; запрещено
распространять его на всю активную запись.

В stochastic записи2353 provider snapshots,1936 candidate attack,
1382 sent attack;502 snapshots со скоростью XY<10. Сработали523
static_hull_blocked,209 component clipping,55 door stops,39 friendly line
blocks и **516 machinegun_partner_guard**. Команды не доказывают native
выстрелы или попадания, разные карты/условия не образуют причинный A/B.

В `internal/bot/combat_policy.go` guard автомата прекращает огонь при любом
наблюдаемом либо недавно виденном напарнике, независимо от геометрии луча.
Для shotgun есть аналогичный общий запрет. Это отдельная установленная
причина отмены предложенного огня в человеческом кооперативе.

В base2 spawncount1232907483 life1 provider window1060..1157 модель
предложила attack уже в frame1060, но первый sent attack только1131:
71 server frames,7.1 секунды при timescale1. До него59 предложений attack,
58 отмен machinegun_partner_guard. В life2 window2111..2191 задержка64
frames,6.4 секунды;31 предложение attack до первого sent,30 отмен этого
guard. Это установленные задержки команд в данной live записи, не замеры
первого native выстрела и не перенос вывода на все предыдущие демо.

Следующее исправление: заменить общий запрет проверкой попадания тела
напарника в возможный сектор hitscan выстрела, сохранив защиту от friendly
fire. Проверка должна учитывать native8192 range, muzzle offset, разброс,
случайную отдачу MG и накопленную отдачу в coop, усиление разброса при
переходе через воду, возраст позиции и перемещение до применения команды.
Нельзя просто удалить guard или ограничиться центральным лучом1024.
Native основания: `../yquake2/src/game/g_weapon.c` fire_lead,
`../yquake2/src/game/player/weapon.c` Machinegun_Fire,
`../yquake2/src/game/header/local.h` DEFAULT_BULLET/SHOTGUN spread.

Приёмка: напарник сзади/далеко вне сектора разрешает MG; внутри секторa
или возле muzzle запрещает; stale position и moving partner расширяют
опасную область; дробовик проверяется отдельно. Geometry tests не выполняют
нейросеть. Затем native сценарий с двумя клиентами подтверждает реальные
выстрелы и отсутствие damage напарнику для проверенных расположений.
До окончания текущих560 capture Go исходники не редактировать; сервер и
бот человеческой сессии не перезапускать без нового запроса пользователя.
