# SAC: ограничение отклонения прицела от behavior policy

Гипотеза после [неудачного fresh feedback](combat_conditional_sac_feedback_20261022.md):Q предпочитает неподтверждённые данными действия,поэтому actor нельзя свободно уводить далеко от поведения,собравшего corpus. Проверяется actor-only penalty `0.01 * KL(current aim || source aim)`. Это отдельный ограниченный SAC опыт;[behavior regularized RL](https://arxiv.org/abs/1911.11361) — связанное направление,не заявление о полном воспроизведении BRAC.

## Фиксированный протокол

Root: `workspace/artifacts/combat-sac-behavior-kl-pair-v1-20261022`. Обе ветки стартуют прежним `combat-sequence-hitscan-pair-v2-20261022/control-source`,а не collapsed SAC0/8. Те же32 закрытых TRAIN боя/3450 transitions,одинаковые начальные actor/std/Q/targets/Adam/RNG,seed20261022,256 Q warmup затем256 actor/256 Q updates на CUDA. Новых учебных симуляций не нужно. Только coefficient0 у контроля и0.01 у кандидата различается;reward/architecture/data/budget сохраняются. Actor scope остаётся conditional aim/std,10212 parameters.

Коэффициент выбран до fit по прежнему TRAIN diagnostic:предпочтение Q≈0.03 против действий,на32–50 std ушедших от behavior mean. При одинаковой дисперсии отклонение3 std даёт KL4.5,штраф0.045. Это выбор масштаба по учебным данным,без sweep/подбора на validation. Мягкий penalty не гарантирует жёсткую границу3 std.

KL считается аналитически по двухмерным Gaussian при одинаковых features,target/mode/pose и sampled movement. Общие invertible tanh/aim-scale transforms сокращаются. Source branch/std заморожены,градиент только к текущему прицелу. Дискретные решения и движение продолжают выдавать старые learned heads. В reward/Q target/inference новый штраф не добавляется;это ограничение update actor,не ручной контроллер боя.

## Проверки

CUDA unit test проверяет KL/аналитический градиент mean,нулевой градиент reference и отсутствие стимула компенсировать сдвиг mean увеличением std при равных дисперсиях. Existing tanh density/terminal twin-Q tests проходят. `--behavior-kl 0` должен точно воспроизвести веса прежнего unregularized warm fit. Оба fit сохраняют `warmup-boundary.pt` до первого actor update;independent audit требует точного совпадения actor/Q/targets/Adam/RNG boundary. Проверяются immutable anchor,protected parameters/variance,checkpoint readback иactual optimizer steps.

Resume сохраняет тот же initial reference anchor и coefficient. Самовольная смена coefficient отклоняется как objective fork. Это fixed-anchor опыт;изменение reference между новыми rounds потребовало бы отдельного протокола.

## Заранее закреплённая оценка

32 native/24 CUDA на8 новых matched условиях:initial,unregularized SAC,KL SAC,rules;offset652,порт46260,пул16 ×2. GPU audit закрывается до native запуска. Четыре карты/ordinary+recovery,никакого fitting по оценочным данным и automatic promotion. Главный исход — цели первой жизнью,дополнительно actual MG outcomes/damage/turn/standing. Возврат к качеству initial сам по себе ещё не является превосходством над ним.

Состояние перед запуском:оба fit/audits/оценка ещё должны завершиться. Ниже закрытый результат.

## Закрытый результат

Оба CUDA fit завершены:по256 actor/512 critic updates на тех же3450 rows. Warmup boundary actor/Q/targets/Adam/RNG совпадает точно,нулевой coefficient точно воспроизвёл прежние unregularized weights. Protected parameters/source anchor/Adam иcheckpoint read-back проверены. Никакого нового runtime controller не добавлено.

На logged TRAIN conditions mean `KL(current||source)`42.8104→0.001248,среднее изменение mean-action3.455°/ось→0.0141°/ось;доля current mean дальше3 behavior std93.45%→0. Это успешное сдерживание шага,ноне измерение точности прицела и не улучшение боя. При этом coefficient сделал update очень близким к initial behavior.

Все32 native/24 CUDA и пять bundle reports закрыты;входные SHA иactual MG report проверены.

| Вариант | Победы | Ordinary / recovery | Убийства | Полученный урон | Смерти | Yaw ≥30° | Стояние | Attack без цели |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial |5/8|2/4;3/4|13|419|2|7.48%|12.35%|22.50%|
| SAC без KL |5/8|2/4;3/4|12|476|2|6.05%|20.28%|17.57%|
| SAC + KL |4/8|2/4;2/4|10|271|1|4.42%|8.29%|31.66%|
| Rules |5/8|2/4;3/4|13|289|1|—|—|—|

К initial/control0 новых/1 потерянная победа;к rules1 новая/2 потерянные. Actual MG live-monster damage:initial127/213(59.62%),control125/200(62.50%),KL106/199(53.27%). У KL3 незавершённых живыми боя против1 у initial/control и2 у rules. Меньше damage/turn/standing не стало ростом завершённых целей.

Решение `behavior_kl_no_win_gain`,screen=false,promotion=false,default сохранён. Одно training seed/малый smoke не дают общего вывода о behavior regularization. На этой паре coefficient ограничил изменения слишком близко к source для заявления о существенном приобретённом навыке. Sweep coefficient по этим validation боям не назначен.

Важно:это не восстановление0/8→4/8.0/8 относилось к другому checkpoint после96 TRAIN случаев/768 actor updates;здесь сравниваются два равных32-case warm fit иконтроль без KL даёт5/8 на новых seeds.

## Следующий источник улучшений

Нужно добавить корректирующие рекомендации эксперта в реально посещаемых учеником состояниях (DAgger-подобный цикл),со связанными movement/aim/fire/mode,вместо дальнейшего изменения actor по ненадёжной Q поверхности. Правила используются как источник учебных рекомендаций;рабочий learned policy продолжает сам выдавать команды.

Первичная проверка формата текущего `bot.jsonl` подтвердила наличие actual observation/learner proposed+applied commands/geometry/enemies/movers/collision bodies,ноотдельной полной `snapshot` записи в этих command traces нет. Точный offline replay классического planner нельзя объявить готовым. Для первого опыта предпочтительна отдельная training-only expert query запись в текущем live состоянии,с тестом отсутствия влияния на learner и native off/on parity. Совет эксперта является альтернативным действием:его нельзя маркировать как уже исполненную native command. Existing teacher native-match labels не следует просто переименовывать в DAgger.

Этот следующий collector/query пока не реализован. Сначала проверить чистоту query иего соответствие реальному rules controller;затем общий fixed-budget CUDA BC на corrections и свежая native оценка. Полное управление боем остаётся целью;teacher не становится обязательным исполнителем.
