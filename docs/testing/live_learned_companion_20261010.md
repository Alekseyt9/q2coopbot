# Обученный бот в кооперативе с игроком — 10.10.2026

Живая сессия: `workspace/runtime/live-coop-20261010-133308`,
loopback UDP29110, coop1,skill1,cheats0,timescale1.
Сервер и клиент игрока продолжают работать; актуальные PID/пути в
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
Параллельно начат, но еще не протестирован и не запущен опыт reward v8
с выбранной целью; текущий бот использует прежние опубликованные веса,
v7 не принят. Продолжить v8 отдельно от человеческой сессии.
