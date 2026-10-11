# Полное обновление боевой политики, 11 октября 2026

Обе модели обучены на CUDA на собственных свежих боях всех 20 рецептов frozen registry: четыре train seeds на семейство, offset 32, 80 боёв на модель. Данные второй модели не переобозначаются как действия первой. В полном actor обучаются encoder, temporal attention, движение, выбор цели/оружия и spatial aim.

Native pool `combat-balanced-training-pool-v1-20261011`: 160/160 accepted, source unchanged, 16 слотов ×2. SHA-256 отчёта `068d229933d47f76d54e5fa1d5855c12f777517c9a078a04ad9b6b75b70921b8`. Оба closed processing receipt подтверждают CUDA и 80 train случаев. Веса и чекпоинты находятся под `workspace/artifacts/combat-balanced-training-analysis-v1-20261011`.

| Показатель | Parent | Shared target |
|---|---:|---:|
| Собственные учебные кадры | 9470 | 9230 |
| Actor steps | 10 | 10 |
| Новый update | 10 | 11 |
| Обучаемые параметры actor | 113100 | 129293 |
| Sequence retention KL до | 0 | 0 |
| Sequence retention KL после | 0,004896 | 0,005202 |
| Изменившиеся скаляры encoder | 25920 | 26035 |
| Изменившиеся скаляры movement outputs | 232 | 232 |
| Изменившиеся скаляры aim outputs | 2498 | 2534 |

Бюджет и config одинаковы: actor LR 0,00003, максимум 10 actor steps, 40 critic steps, sequence retention weight 1. Дообучение возобновляет собственные Adam/RNG и checkpoint counters, с явным fork прежнего decisions scope в all и fork sequence objective. Это не чистое сравнение архитектур с одинаковой историей: parent ранее имел девять обновлений, shared десять.

`checkpoint-parent.json` и `checkpoint-shared.json` в confirmation analysis подтверждают на CUDA: новый corpus append ровно один раз, приращение update/step counters, продолжение Adam clocks, привязку retention spec и фактическое изменение encoder/движения/прицела. Оба audit прошли. KL измеряет изменение политики на учебных историях, не win rate или качество прохождения карты.

Новые веса:

- `update-parent/weights.json`: `ecb2d42bd253b2e1ae4e848e92740739e9d4f6c0afd06c454a6392978f2bc269`.
- `update-shared/weights.json`: `edc01b498634b14efbce924ae23fc24713ed31dbe158054296819552d7da49c0`.

Запущена отдельная confirmation оценка: parent before/after, shared before/after, rules; 20 рецептов × четыре confirmation seeds, offset 0, 400 боёв, пул 16×2, порт 38600. Все пять планов проверены на точное совпадение сида, рецепта и сгенерированных условий. Pairwise before/after важнее сравнения разных архитектур с разным прошлым бюджетом.

Первоначальный coordinator остановился на rules plan: ему ошибочно передали model. Исправлен только вызов планировщика rules; четыре готовых learned plans и два passed checkpoint audits повторно не создавались. Сбор начат после создания корректного rules plan. Сохранены журналы первой ошибки; текущая последующая обработка ожидается в `combat-balanced-confirmation-quality-v2-20261011`.

Оценочные данные не участвуют в обучении. Действующий coop runtime и опубликованные веса не обновлены.

## Завершённая confirmation оценка

Пул завершён:400/400 usable captures, source unchanged,1047,62с,
report SHA-256 `4bc18fa3d4ec52a13fee8f433c1d7081751bd1357ce5bd63311005bbebb90678`.
Строгие native comparisons всех вариантов и отдельное shared own-update
сравнение прошли. CUDA verification320 learned боёв завершена:35692 кадров,
корпусы8364/9400/8969/8959. Оценка не выполняла обучение. Все четыре
ownership audit после release показывают0 rules frames при видимом враге,
0 неизвестных наблюдений/отсутствующих actions. Игровое повышение не выполнено.

| Вариант | Победы/80 | Kills | Deaths | Dealt | Received |
|---|---:|---:|---:|---:|---:|
| Parent до | 27 | 90 | 51 | 21479 | 5898 |
| Parent после | 31 | 111 | 45 | 22620 | 5670 |
| Shared до | 29 | 107 | 48 | 22536 | 5747 |
| Shared после | 35 | 115 | 42 | 23169 | 5530 |
| Rules | 29 | 123 | 42 | 24151 | 5240 |

Parent update улучшает10 проигрышей, но теряет6 прежних побед:net+4.
Этот прирост целиком на ware2:10→14; base3 5→5,city1 12→12,jail1 0→0.
Shared update улучшает10 проигрышей и теряет4 прежние победы:net+6.
Рост shared по картам неравномерен:base3 6→6, city1 10→15,
jail1 1→1, ware2 12→13. Jail1 по-прежнему проблемная геометрия/группа;
общий рост не доказывает исправления каждого вида боя. Rules дают больше
kills/damage и меньше received, хотя shared-after выигрывает больше полных
эпизодов. Архитектуры имеют разную прошлую историю; чистый вывод об их
превосходстве по этой таблице не делается.

Добавлен независимый аудит `audit_registered_weapon_choices.py` и пять
прошедших тестов. Он измеряет серии keep с неподдерживаемым текущим оружием
при видимом враге и задержку между applied intent и наблюдаемой экипировкой,
с цензурированием пропусков/смены владельца/superseded/конца записи.
Все четыре learned варианта и малый прежний recovery corpus проверены этим инструментом.

Weapon audits всех четырёх80-case learned выборок завершены.
Parent keep при видимом враге вне pilot fire set21→0; shared0→0.
Большая наблюдаемая задержка switch43–44 кадра остаётся. В одном общем
случае request Shotgun102 при непрерывном MG fire приводит к наблюдаемому
Shotgun145 после исчерпания Bullets. Native source SHA совпадает с capture;
WEAPON_FIRING препятствует началу смены. Нужны собственная request history
и обучение release fire, плюс контролируемая native проверка причины.
См. `combat_weapon_attack_coordination_20261011.md`.

## Поведение после release и ограничения результата

Исправлен audit v3: прежний v2 включал два provider-owned setup кадра на
каждый бой, хотя итоговый summary обещал исключать подготовку. Новая версия
явно фильтрует `frame < manifest.release_game_frame`. Все320 case provider
counts и trace SHA совпали с независимым ownership audit; задержки теперь
начинаются от первого provider кадра после release. Старые v2 отчёты оставлены
как исторические; используются `behavior-*-v3.json` и `behavior-summary-v3.json`.

| Показатель | Parent до | Parent после | Shared до | Shared после |
|---|---:|---:|---:|---:|
| Доля неподвижных provider кадров | 23,95% | 26,53% | 23,07% | 26,37% |
| Средняя задержка первого attack intent, кадров | 1,25 | 1,56 | 0,87 | 1,56 |
| Максимальная задержка attack intent | 34 | 38 | 22 | 47 |
| Бои без attack intent | 1 | 1 | 2 | 1 |
| Максимальная неподвижная серия при clear enemy | 144 | 140 | 144 | 95 |

Рост побед сопровождается более высокой общей долей стояния и более поздним
первым attack intent. У shared уменьшилась максимальная неподвижная серия,
но поведение в целом не исправлено. Attack intent не равен native выстрелу;
эти описательные показатели не доказывают бесполезности каждой остановки.
Новые веса не назначены live/published/reference по одному выигрышному набору.

После подтверждения закрытия всех exporter процессов добавлена own-request
history в Go observation: только после успешного локального очередирования
`use NAME`, со следующим кадром, immutable cached observation и сбросом
при map/spawncount/connection/actor/life/death/rewind. Фиксированные проверки
истории прошли; old v9 prefix остаётся неизменным. Это не скрытый server pending
state и не native ACK. Признаки нового поколения и миграция модели ещё впереди.
