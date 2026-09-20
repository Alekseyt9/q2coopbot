# План разработки кооперативного AI-напарника для Quake II

## 1. Цель проекта

Цель — создать AI-напарника, с которым интересно совместно проходить кампанию Quake II.

Бот не должен быть:

- обычным Deathmatch-ботом, которому разрешили атаковать монстров;
- турелью, следующей за игроком;
- идеальным суперсолдатом, самостоятельно уничтожающим весь уровень;
- пассивным NPC, которого приходится постоянно спасать;
- системой, требующей постоянных команд со стороны игрока.

Он должен ощущаться как **второй участник кооператива**.

Игрок должен периодически замечать:

> «Он понял, что я делаю.»

> «Он прикрыл меня.»

> «Он тоже отступил.»

> «Он занял другую сторону.»

> «Он добил именно того врага, который мешал мне.»

> «Он подождал меня, а не побежал дальше.»

> «Я понял, что он собирается сделать.»

Ключевая задача проекта — не максимизировать эффективность прохождения, а создавать **совместные игровые эпизоды**.

---

## 2. Основные принципы

### 2.1. Cooperation first

Приоритеты поведения:

```text
1. Не мешать игроку
2. Не ломать прохождение
3. Понимать действия игрока
4. Помогать игроку
5. Создавать совместные ситуации
6. Проявлять ограниченную самостоятельность
7. Быть эффективным
```

Эффективность специально стоит последней.

Бот с идеальной точностью и оптимальным маршрутом может быть хуже напарника, чем немного менее эффективный бот, который хорошо взаимодействует с человеком.

### 2.2. Optimal play != enjoyable coop play

Нельзя оптимизировать AI только по:

- damage;
- kills;
- accuracy;
- deaths;
- clear time.

Иначе оптимальный результат — бот, который самостоятельно проходит игру.

Нужно отдельно оптимизировать:

```text
cooperation
predictability
reciprocity
initiative balance
player agency
shared combat
```

### 2.3. Инициатива остаётся у игрока, но не всегда

Бот в основном реагирует на игрока.

Но он должен иметь право временно проявить инициативу:

- прикрыть;
- добить опасного врага;
- занять позицию;
- выйти немного вперед;
- спасти игрока;
- отступить первым;
- обратить внимание на угрозу.

То есть:

```text
не follower
не leader

а cooperative peer
```

---

## 3. Архитектура

Предлагаемая архитектура:

```text
                    GAME WORLD
                        |
                        v
              Sensors / Perception
                        |
                        v
                    Blackboard
                        |
          +-------------+-------------+
          |                           |
          v                           v
 Player Intent Model          World / Threat Model
          |                           |
          +-------------+-------------+
                        |
                        v
              Relationship Model
                        |
                        v
                Role Selection
                        |
                        v
              Tactical Utility AI
                        |
            +-----------+----------+
            |                      |
            v                      v
        Combat AI            Objective HTN
            |                      |
            +-----------+----------+
                        |
                        v
                Action Controller
                        |
                        v
             Gladiator / AAS botlib
                        |
                        v
                Quake II inputs
```

---

## 4. Ответственность уровней

### 4.1. Gladiator / botlib

Gladiator оставить низкоуровневым исполнительным слоем.

Он отвечает за:

- AAS pathfinding;
- движение;
- повороты;
- aiming;
- primitive combat actions;
- input-команды;
- обход препятствий;
- базовую навигацию.

Кооп-AI не должен без необходимости переписывать эти механизмы.

Он задаёт Gladiator:

```text
target
destination
allowed_area
movement_mode
aim_target
attack_permission
desired_range
desired_relative_position
```

---

## 5. Blackboard

Blackboard хранит агрегированное состояние сцены.

### 5.1. PlayerState

```text
position
velocity
view_direction

health
armor

weapon
ammo

current_target
recent_targets

damage_taken_recently
damage_source

movement_direction

distance_to_bot

is_firing
is_retreating
is_advancing
is_waiting
is_interacting

current_area
```

### 5.2. BotState

```text
position
velocity

health
armor

weapon
ammo

current_target

current_role
current_intention
current_action

initiative_budget

distance_to_player

recent_damage

current_area

commitment_time

last_safe_position
```

### 5.3. EnemyState

Для каждого известного врага:

```text
entity_id
class
position
health
visibility
last_seen_time
distance_to_player
distance_to_bot
threat_to_player
threat_to_bot
current_target
group_id
is_engaged
is_new_group
estimated_danger
```

### 5.4. WorldState

```text
current_area
known_areas
doors
buttons
elevators
safe_positions
danger_zones
choke_points
recent_pull_locations
visible_routes
known_enemy_groups
current_encounter
```

---

## 6. Кратковременная память

Не ограничиваться текущим кадром.

Хранить события последних примерно 10–30 секунд.

Например:

```text
player_was_attacked_by X
bot_was_attacked_by X
player_recently_retreating
player_saved_bot
bot_saved_player
dangerous_door
failed_path
recent_pull_location
recent_player_focus_target
recent_cover_position
last_safe_position
last_encounter_result
```

Это позволяет поведению иметь контекст.

Например: если из комнаты справа только что пришла опасная группа, бот некоторое время относится к этому направлению осторожнее.

---

## 7. Player Intent Model

Это один из центральных новых компонентов.

Бот должен оценивать не только:

> где игрок?

а:

> что игрок сейчас пытается сделать?

Начальные состояния:

```text
ADVANCE
HOLD
RETREAT
ENGAGE_TARGET
SEARCH
EXPLORE
INTERACT
LOOT
WAIT
UNKNOWN
```

### 7.1. Как определять намерение

Сначала исключительно эвристиками.

#### ADVANCE

```text
player_velocity направлена в unexplored area

+

игрок не ведёт интенсивный огонь

+

нет retreat behaviour
```

#### RETREAT

```text
игрок движется назад от угрозы

или

расстояние от текущей encounter-area увеличивается

или

получил значительный burst damage
```

#### ENGAGE_TARGET

```text
crosshair / view direction → enemy

+

weapon fire

+

target LOS
```

#### HOLD

```text
низкая скорость

+

есть активные враги

+

игрок удерживает направление
```

### 7.2. Confidence

Intent определяется вместе с уверенностью:

```text
intent = RETREAT
confidence = 0.82
```

При низкой уверенности бот становится более консервативным.

---

## 8. Relationship Model

Это небольшой слой, описывающий текущую динамику пары:

```text
player ↔ bot
```

Он может содержать:

```text
player_aggression
player_speed
player_accuracy
player_retreat_tendency
player_preferred_range
player_exploration_style
player_reliance_on_bot
bot_recent_helpfulness
recent_reciprocity
shared_focus_history
```

На первом этапе это обычные скользящие статистики.

Нейросеть для этого не нужна.

---

## 9. Динамические роли

Бот не должен постоянно вести себя одинаково.

Вводятся временные роли.

### FOLLOWER
Идёт с игроком. Используется между боями.

### SUPPORT
Поддерживает игрока с соседней позиции.

```text
player attacks A
bot attacks A or nearby threat
```

### ANCHOR
Удерживает позицию или проход. Полезно при отступлении игрока.

### COVER
Прикрывает перемещение игрока.

### RESCUER
Высокий приоритет угроз, атакующих игрока.

### FINISHER
Убирает тяжело раненую или особо опасную цель.

### VANGUARD
Очень ограниченный режим. Бот может пройти немного впереди игрока. Использовать редко.

### REGROUP
Немедленно сокращает дистанцию до игрока.

Роль не является строгим FSM-state.

Это модификатор Utility AI.

---

## 10. Initiative Budget

Вводится:

```text
initiative_budget = 0.0 .. 1.0
```

Он определяет, насколько бот сейчас может действовать самостоятельно.

Пример:

```text
0.0
только следует

0.2
может менять боевую позицию

0.4
может самостоятельно выбирать угрозы

0.6
может совершать короткие tactical moves

0.8
может временно взять инициативу

1.0
аварийная самостоятельность
```

### 10.1. Пример изменения

Игрок уверенно продвигается:

```text
initiative = 0.2
```

Игрок остановился:

```text
0.35
```

Игрок попал под сильный огонь:

```text
0.75
```

Игрок критически ранен:

```text
0.95
```

После стабилизации:

```text
0.3
```

---

## 11. Follow и Formation

Следование нельзя делать простым:

```text
distance(bot, player) < N
```

Нужно использовать относительные позиции.

### Возможные позиции

```text
LEFT_SUPPORT
RIGHT_SUPPORT
BEHIND
FRONT_SHORT
COVER_EXIT
COVER_RETREAT
HOLD_CHOKE
```

### 11.1. Выбор позиции

Учитывать:

- направление взгляда игрока;
- направление движения;
- угрозы;
- стены;
- проходы;
- LOS;
- позиции врагов;
- текущую роль.

---

## 12. Personal Space

Бот не должен мешать игроку.

Вводится пространственная cost-map вокруг игрока.

Пример:

```text
player crosshair cone
cost = extremely high

front 1-2 meters
cost = high

doorway
cost = very high

side
cost = normal

behind
cost = low
```

Особенно запрещать:

```text
пересекать линию player → target
```

без веской причины.

---

## 13. Regroup

Если расстояние слишком большое:

```text
state → REGROUP
```

При этом:

- прекратить chase;
- не начинать новый бой;
- не останавливаться ради слабого врага;
- искать безопасный путь;
- возвращаться к игроку.

Можно иметь два leash:

```text
soft_leash
hard_leash
```

Например:

```text
soft leash → штраф Utility
hard leash → жёсткий запрет
```

---

## 14. Encounter Model

Вместо отдельных монстров бот должен видеть **encounter**.

Пример:

```text
Encounter 17
    group A
    group B

player engaged: yes
bot engaged: yes

danger = 0.62

additional_pull_risk = 0.18
```

---

## 15. Enemy Groups

Монстры объединяются по:

- пространственной близости;
- visibility;
- AAS connectivity;
- совместной активации;
- комнате;
- encounter history.

Цель — понимать:

> эта группа уже участвует в бою

и:

> это новая группа, которую лучше не трогать.

---

## 16. Контроль pull

Вводятся:

```text
known_active_groups
allowed_active_groups
new_group_risk
```

Бот не атакует новую группу, если:

```text
player_intent != ADVANCE

и

нет непосредственной угрозы игроку

и

initiative_budget низкий
```

---

## 17. Utility Target Selection

Для каждой цели вычисляется полезность.

Пример:

```text
target_score =

    threat_to_player
  + threat_to_bot
  + player_focus_bonus
  + visibility
  + proximity
  + wounded_bonus
  + tactical_value

  - new_group_penalty
  - path_risk
  - leash_penalty
  - kill_steal_penalty
```

---

## 18. Kill Stealing

Отдельная механика.

Если игрок явно добивает слабого врага:

```text
player_ownership = high
```

тогда bot utility этой цели уменьшается.

Но если враг опасен:

```text
threat_override = true
```

и бот вмешивается.

---

## 19. Target hysteresis

Бот не меняет цель при каждом небольшом изменении score.

Например:

```text
switch_target only if

new_score >
current_score * 1.25
```

или произошёл interrupt:

```text
player threatened
current target dead
current target inaccessible
critical danger
```

---

## 20. Combat Utility AI

Боевой AI лучше сделать Utility-based.

Возможные действия:

```text
Attack
Suppress
CoverPlayer
Reposition
FinishTarget
ProtectPlayer
Advance
Hold
Retreat
SeekCover
Regroup
```

### 20.1. Utility

Для каждого:

```text
U(action | world, player, bot)
```

Пример:

```text
U(CoverPlayer)

= player_danger
* good_cover_position
* line_of_fire
* role_support
```

---

## 21. Commitment

После выбора действия бот не пересматривает решение каждый тик.

Каждое действие получает:

```text
minimum_commitment_time
maximum_commitment_time
interrupt_threshold
```

Например:

```text
CoverLeft
minimum = 1.2 sec
maximum = 4 sec
```

Это устраняет нервное поведение:

```text
лево
право
лево
target A
target B
назад
вперёд
```

---

## 22. Стрельба

Разделить:

```text
decision_to_attack

и

execution_of_attack
```

### 22.1. Перед выстрелом

Проверять:

```text
ammo
weapon ready
cooldown
LOS
aim angle
range
friendly fire / player crossing
tactical permission
```

---

## 23. Burst Control

Для автоматического оружия:

```text
start burst
hold attack
release
reaim
next burst
```

Не принимать решение `attack true/false` каждый AI tick.

---

## 24. Weapon Selection

Оценивать:

```text
distance
enemy class
ammo
splash risk
player proximity
number of enemies
available cover
```

---

## 25. Human-like limitations

Бот не должен иметь мгновенную идеальную реакцию.

Параметры:

```text
reaction_delay
target_acquisition_delay
aim_tracking_speed
aim_error
awareness_limit
decision_latency
memory_decay
```

### 25.1. Ошибки должны быть естественными

Не делать:

```text
random miss = 30%
```

Лучше:

новая цель появляется →

бот замечает её через 200–500 мс →

первоначальный aim неточный →

несколько кадров корректирует →

точность растёт.

---

## 26. Retreat

Retreat должен учитывать не только HP.

Danger Score:

```text
danger =

incoming_damage

+ enemy_count

+ flanking

+ low_health

+ low_armor

+ bad_position

+ low_ammo

+ distance_from_player
```

---

## 27. Совместное отступление

Особо важный coop event.

Если:

```text
player_intent = RETREAT
```

бот должен обычно:

```text
не продолжать chase
```

а выполнить:

```text
CoverRetreat
→ FallBack
→ Regroup
```

Иногда бот может ещё 1–2 секунды пострелять, прежде чем отходить.

---

## 28. Rescue Behaviour

Если игрок близок к смерти:

```text
player_danger > critical
```

приоритеты меняются.

Например:

```text
kill nearest threat
suppress dangerous enemy
body positioning
draw aggro
cover retreat
```

Даже если это немного опаснее для самого бота.

---

## 29. Intent Signaling

Игрок должен понимать действия AI.

### 29.1. Через движение

Перед манёвром:

```text
look
orient
pause briefly
move
```

Например бот хочет уйти налево:

```text
сначала смотрит туда
→ смещается
→ занимает позицию
```

### 29.2. Через короткие реплики

Минимальный набор:

```text
"Covering!"
"Back!"
"Wait."
"Got him."
"Going left."
"Low!"
"Clear."
```

Реплики должны:

- быть редкими;
- сообщать реальное намерение;
- не превращаться в постоянный chatter.

---

## 30. Coop Events

Это центральные игровые метрики.

Логировать:

```text
covered_player
covered_player_successfully
saved_player
player_saved_bot
waited_for_player
regrouped_with_player
retreated_with_player
focused_player_target
yielded_player_target
took_complementary_position
blocked_player_path
crossed_player_fireline
pulled_new_group
advanced_without_player
started_unwanted_encounter
```

---

## 31. Reciprocity

Ввести метрику:

```text
reciprocity_score
```

Примеры:

```text
bot helped player
player helped bot
bot responded to player action
player responded to bot action
```

Цель — чтобы эпизоды взаимной помощи возникали регулярно.

---

## 32. Player Style Model

После нескольких минут оценивать стиль человека.

Например:

```text
aggression
pace
preferred_range
accuracy
risk_tolerance
retreat_frequency
exploration_tendency
resource_usage
```

Использовать EMA:

```text
value =
old * 0.9
+
new_observation * 0.1
```

---

## 33. Adaptation

### Агрессивный игрок

Бот чаще:

```text
support
cover
anchor
```

и реже пытается лидировать.

### Осторожный игрок

Бот может немного чаще:

```text
vanguard
finish
take initiative
```

### Сильный игрок

Снизить:

```text
kill stealing
initiative
aim performance
```

### Слабый игрок

Повысить:

```text
protect player
threat removal
rescue behaviour
```

---

## 34. Модель уровня

После стабилизации базового взаимодействия добавить понимание мира.

Нужно выделять:

```text
areas
rooms
corridors
doors
buttons
elevators
transitions
safe zones
combat zones
```

---

## 35. Area State

Например:

```text
UNKNOWN
VISITED
ACTIVE_COMBAT
PARTIALLY_CLEARED
CLEARED
DANGEROUS
```

---

## 36. Продвижение

Бот не должен самостоятельно уходить в следующую область.

Логика:

```text
если current_area cleared

и player близко к exit

и player_intent == ADVANCE

→ AdvanceAfterClear
```

---

## 37. Objective HTN

HTN использовать прежде всего для длинных действий.

Например:

```text
OpenPath

├── FindDoor
├── DetectLocked
├── FindRelatedButton
├── NavigateToButton
├── WaitForPlayer
├── Activate
└── ReturnToPath
```

Для боя HTN не нужен.

---

## 38. Почему Combat = Utility, Objectives = HTN

Бой:

```text
быстро меняется
непредсказуем
требует continuous scoring
```

Поэтому:

```text
Utility AI
```

Навигационная задача:

```text
состоит из последовательности действий
```

Поэтому:

```text
HTN
```

---

## 39. Logging

Каждый episode:

```text
episode_id
map
seed
bot_version
config_version
model_version
difficulty
player_id/session
timestamp
```

---

## 40. Decision Log

Для каждого значимого решения:

```text
time
state
role
player_intent
initiative_budget
current_target
candidate_actions
utilities
chosen_action
reason
interrupt_reason
```

---

## 41. Combat telemetry

```text
shots
hits
damage
damage_received
kills
deaths
accuracy
engagement_duration
```

Но это вторичные метрики.

---

## 42. Coop telemetry

Основные:

```text
distance_to_player
time_near_player
shared_focus_duration
successful_cover
successful_rescue
unwanted_pull_count
player_block_count
fireline_cross_count
regroup_count
joint_retreat_count
cooperation_events_per_minute
```

---

## 43. Episode reconstruction

Логи должны позволять после игры восстановить:

```text
Player advanced

Bot followed right

Monster A appeared

Player attacked A

Bot chose cover-left

Player took damage

Bot switched to rescue

Bot killed B

Player retreated

Bot covered for 1.4 sec

Both regrouped
```

Это намного полезнее обычного dump состояния.

---

## 44. Automated scenarios

Начать с `base1`.

### Scenario 1 — Follow

Игрок идёт по пустому коридору.

Ожидание:

```text
bot follows
does not overtake unnecessarily
does not block
```

### Scenario 2 — Single enemy

Один враг.

Ожидание:

```text
player starts fight
bot supports
```

### Scenario 3 — Distant group

Видна дальняя группа.

Ожидание:

```text
bot does not pull
```

### Scenario 4 — Player retreat

Игрок резко отступает.

Ожидание:

```text
bot notices
covers
retreats
regroups
```

### Scenario 5 — Player critical HP

Ожидание:

```text
bot switches to rescue
```

### Scenario 6 — Player finishing enemy

Враг почти мёртв.

Игрок его атакует.

Ожидание:

```text
bot avoids unnecessary kill stealing
```

### Scenario 7 — Two directions

Враги появляются спереди и сбоку.

Ожидание:

```text
bot chooses complementary sector
```

### Scenario 8 — Doorway

Ожидание:

```text
bot never blocks player
```

### Scenario 9 — Lost LOS

Ожидание:

```text
bot does not chase endlessly
```

### Scenario 10 — Bad pull

Бот случайно активировал группу.

Ожидание:

```text
recognizes problem
retreats
does not pull more
```

---

## 45. Deterministic baseline

Каждый сценарий запускать:

```text
N >= 20
```

раз с фиксированными seed.

Сравнивать версии.

---

## 46. Acceptance criteria — технический уровень

P1 считается успешным, если:

- бот редко блокирует игрока;
- почти не тянет новые группы самостоятельно;
- держит leash;
- умеет нормально стрелять;
- умеет возвращаться;
- не продолжает бессмысленный chase;
- реагирует на низкое здоровье;
- бой воспроизводим.

---

## 47. Acceptance criteria — cooperative level

P2 считается успешным, если:

- бот замечает retreat игрока;
- поддерживает target focus;
- занимает соседние позиции;
- умеет прикрывать;
- возникают rescue events;
- kill stealing ограничен;
- существует reciprocity.

---

## 48. Acceptance criteria — companion level

P3 считается успешным, если игрок может после боя ответить:

```text
Что бот пытался делать?

В какой момент он помог?

Когда он проявил инициативу?

Приходилось ли его постоянно контролировать?
```

Если человек способен объяснить поведение AI — это хороший знак.

---

## 49. Главный UX-тест

После 10–20 минут прохождения спросить игрока оценить от 1 до 7:

```text
Мне было понятно, что делает бот

Бот реагировал на мои действия

Бот действительно помогал

Бот не мешал мне играть

Иногда мы действовали совместно

Бот казался самостоятельным

Мне пришлось мало заниматься управлением ботом

С ботом играть интереснее, чем одному

Я хотел бы продолжить кампанию с этим ботом
```

Последние два показателя — главные.

---

## 50. Этап P0 — инфраструктура экспериментов

Сделать:

1. `episode_id`
2. фиксированный seed
3. автоматический запуск карты
4. bot configuration
5. event log
6. decision log
7. combat telemetry
8. coop telemetry
9. scripted test scenarios
10. автоматический итоговый отчёт.

Цель:

```text
любое изменение можно сравнить с предыдущей версией
```

---

## 51. Этап P1 — бот перестаёт мешать

Реализовать:

### Follow
- soft leash;
- hard leash;
- regroup;
- ожидание игрока.

### Movement
- personal space;
- doorway avoidance;
- fireline avoidance.

### Aggro
- enemy groups;
- active encounter;
- new-group penalty.

### Combat
- нормальная стрельба;
- weapon selection;
- target utility;
- hysteresis.

### Survival
- danger score;
- retreat;
- basic cover.

Результат:

> бот уже пригоден для прохождения, хотя ещё не кажется настоящим товарищем.

---

## 52. Этап P2 — бот начинает помогать

Добавить:

```text
Player Intent Model
dynamic roles
CoverPlayer
ProtectPlayer
RescuePlayer
shared focus
complementary positioning
joint retreat
kill-steal control
```

Главная цель:

> действия игрока начинают непосредственно менять тактику бота.

---

## 53. Этап P3 — появляется ощущение личности

Добавить:

```text
initiative_budget
commitment
intent signaling
short-term memory
human reaction delays
limited awareness
human-like aiming
action persistence
```

Главная цель:

> бот перестаёт ощущаться как постоянно пересчитывающий utility алгоритм.

---

## 54. Этап P4 — адаптация к игроку

Добавить:

```text
Player Style Model
```

Накапливать:

```text
aggression
pace
skill
risk tolerance
preferred range
```

И адаптировать:

```text
role distribution
initiative
support level
aim strength
kill stealing
retreat threshold
```

Главная цель:

> бот начинает немного адаптироваться к конкретному человеку.

---

## 55. Этап P5 — понимание уровня

Добавить:

```text
room segmentation
area graph
doors
buttons
elevators
safe areas
objectives
```

Затем Objective HTN.

Главная цель:

> пройти существенную часть кампании без специальных костылей для каждой карты.

---

## 56. Этап P6 — ML

Только после появления качественного baseline.

### 56.1. Лучшие первые задачи

#### Threat prediction

```text
P(bad outcome within 3 seconds)
```

#### Target ranking

Модель оценивает:

```text
utility(target)
```

#### Action ranking

Например:

```text
attack
cover
retreat
reposition
```

#### Player intent

Если эвристик окажется недостаточно.

---

## 57. Что пока не отдавать ML

Жёсткими оставить:

```text
hard leash
critical retreat
new area permission
player blocking constraints
friendly fire rules
objective safety
fallback behaviour
```

---

## 58. Offline learning

Логи превращаются в dataset:

```text
state_t
action_t
result_t+N
```

Результат:

```text
damage dealt
damage received
player damage
pull happened
player saved
bot died
distance from player
encounter result
```

---

## 59. Imitation Learning

Позже можно дать человеку временно управлять ботом.

Собирать:

```text
world state
player state
bot state
human-selected action
```

И учить модель копировать хорошие решения.

Особенно полезно для:

```text
positioning
retreat
cover timing
initiative
```

---

## 60. Reinforcement Learning

RL имеет смысл использовать не для полного управления FPS, а для ограниченного tactical policy.

Например действия:

```text
FOLLOW
ATTACK
REPOSITION_LEFT
REPOSITION_RIGHT
COVER
RETREAT
REGROUP
```

Reward:

```text
+ player_survival
+ encounter_success
+ cooperation_event
+ useful_damage
- unwanted_pull
- blocking
- bot_death
- player_death
- excessive_distance
```

---

## 61. Human preference optimization

Особенно интересное позднее направление.

Записать два варианта одного эпизода:

```text
AI A
AI B
```

Человек выбирает:

> с каким было приятнее играть?

На этих preference pairs можно обучать:

```text
companion_quality_model
```

Это потенциально намного лучше оптимизации только по combat stats.

---

## 62. Personality

Когда базовый AI заработает, можно добавить характер.

Не отдельными скриптами, а параметрами.

Например:

```text
bravery
initiative
loyalty
caution
talkativeness
accuracy
risk_tolerance
```

### Осторожный

```text
high caution
low initiative
high regroup tendency
```

### Боевой товарищ

```text
medium-high initiative
high loyalty
medium risk
```

### Reckless

```text
high initiative
high aggression
```

Но даже aggressive profile подчиняется hard safety constraints.

---

## 63. Возможные дальнейшие механики

После основного AI:

### Ping system

Игрок может быстро указать:

```text
enemy
position
door
direction
```

без меню команд.

### Simple commands

```text
Follow
Wait
Attack
Cover
Come here
```

Но игра должна оставаться комфортной даже без них.

### Contextual dialogue

Позже можно добавить локальную LLM только для реплик.

LLM получает:

```text
game event
bot state
player state
recent history
```

и генерирует короткую фразу.

Но LLM не должна управлять combat loop.

---

## 64. Первый практический спринт

Первый спринт сделать максимально приземлённым.

### Шаг 1
Зафиксировать baseline текущего бота на `base1`.

Минимум:

```text
20 одинаковых прогонов
```

### Шаг 2
Добавить JSONL event log.

### Шаг 3
Добавить:

```text
player distance
bot state
current target
visible enemies
enemy groups
damage
shots
pull events
blocked player
```

### Шаг 4
Реализовать:

```text
soft leash
hard leash
regroup
```

### Шаг 5
Добавить Personal Space.

Особенно:

```text
crosshair cone avoidance
doorway avoidance
```

### Шаг 6
Исправить стрельбу.

### Шаг 7
Добавить target utility + hysteresis.

### Шаг 8
Добавить enemy groups и pull penalty.

### Шаг 9
Добавить retreat.

### Шаг 10
Пройти `base1` несколько раз вручную.

Цель первой версии:

> бот почти перестал раздражать.

Не пытаться пока сделать его умным.

---

## 65. Второй спринт

Добавить:

```text
Player Intent
CoverPlayer
RescuePlayer
joint retreat
dynamic positioning
roles
```

Главная цель:

> игрок впервые начинает замечать, что бот реагирует именно на него.

---

## 66. Третий спринт

Добавить:

```text
initiative_budget
commitment
intent signaling
short-term memory
human-like reaction
```

Главная цель:

> действия AI начинают выглядеть намеренными.

---

## 67. Четвёртый спринт

Добавить Player Style Model.

Главная цель:

> бот начинает немного адаптироваться к конкретному человеку.

---

## 68. Пятый спринт

Добавить понимание карты и HTN.

Главная цель:

> пройти существенную часть кампании без специальных костылей для каждой карты.

---

## 69. Главные риски

### Риск 1
Сделать слишком хорошего бойца.

Решение:

```text
optimize cooperation, not DPS
```

### Риск 2
Сделать слишком пассивного follower.

Решение:

```text
initiative_budget
```

### Риск 3
Слишком сложная архитектура раньше времени.

Решение:

```text
rules
→ utility
→ memory
→ adaptation
→ HTN
→ ML
```

### Риск 4
Behaviour oscillation.

Решение:

```text
hysteresis
commitment
cooldowns
```

### Риск 5
Игрок не понимает AI.

Решение:

```text
intent signaling
```

### Риск 6
ML делает поведение непредсказуемым.

Решение:

ML только внутри жёстких safety constraints.

---

## 70. Главный критерий проекта

Не:

```text
бот прошёл Quake II
```

И не:

```text
бот набрал много kills
```

А:

> человек добровольно предпочитает проходить Quake II вместе с этим ботом, а не в одиночку.

---

## 71. Итоговая стратегия

Последовательность проекта:

```text
1. Сделать AI наблюдаемым.
2. Сделать его безопасным.
3. Сделать его ненавязчивым.
4. Научить понимать игрока.
5. Научить помогать игроку.
6. Дать ограниченную инициативу.
7. Сделать намерения читаемыми.
8. Добавить память.
9. Добавить адаптацию.
10. Научить понимать уровень.
11. Только затем подключать обучение.
```

Главная эволюция поведения:

```text
Deathmatch bot
↓
Follower
↓
Useful teammate
↓
Cooperative teammate
↓
Adaptive companion
```

Именно последний переход должен быть основной целью проекта.

Не нужно стремиться создать максимально сильного AI.

Нужно создать AI, после совместного боя с которым у игрока появляется ощущение:

> «Мы это сделали вместе.»
