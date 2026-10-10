# Контекст навигации во время learned боя

Исходный случай — base2 learned185112 в закрытом batch
`natural-companion-ready-multiseed-v2-20261010`. На frames900..1080
goal=search_last_seen, AAS route ready (около747..900 units), но provider
контролирует движение и не приближается к remembered rendezvous.
На frame1094 age201 истекает20-second search budget. Затем741 consecutive
wait_for_teammate frames. Это не ошибка построения маршрута и не доказанная
смерть напарника: известная цель поиска отсутствует во входах модели.

Добавлен optional navigation_context в combat observation:

- goal/status из уже работающего Go planner;
- goal_relative только при активной цели, без старого cached goal после idle;
- waypoint_relative только из актуального известного маршрута;
- last_teammate_relative и last_teammate_age_frames из native наблюдений.

Последняя позиция не становится живым наблюдаемым игроком. Позиция второго
клиента из его отдельного trace не используется. Search age/route safety
ограничения сохранены, aim/movement не заменяются командами planner во
время provider-owned боя.

Feature versions v1..v7 игнорируют новый optional metadata. Их shape и
serialized feature values проверены без neural CPU inference. Старые веса
ещё не используют этот контекст; поведенческого улучшения пока нет.

`navigation-context-regressions.jsonl`: разделение observed/remembered,
отсутствие stale goal, route waypoint, unknown age; прежние bounded far
search, campaign leader и connect gate regressions прошли.

Native smoke `natural-companion-navigation-context-smoke-v1-20261010`:
4 scenes,600frames,timescale2,seeds185011/185111; старые attention64 weights.
Source unchanged; captures подтверждают наличие navigation_context,
включая реальные waypoint_relative. `navigation-context-verification.json`
сохраняет счётчики и SHA. Один base2 learned run завершён переходом назад
на base1 (`native_wrong_exit`), что считается неуспешным прохождением,
а не улучшением или успехом нового контекста.

Следующий этап: отдельная feature version с нормализованными nav channels,
миграция входного слоя без потери существующего baseline, GPU проверки
совместимости/градиентов и validated on-policy обучение в сценариях с
напарником. Natural captures по-прежнему evaluation-only; они не становятся
trainable только из-за наличия дополнительных полей. После этого сравнить
боевую эффективность и reacquisition с исходными весами на одинаковых seeds.
Human session и опубликованные веса не заменены.

## Feature v8 и проверенный перенос весов

Добавлен `combat_features_v8`: 881 вход вместо 854. Первые 854 значения —
прежний v7 без изменений. Хвост из 27 значений:

| Индексы | Значение |
|---|---|
| 854 | Наличие navigation_context |
| 855..862 | follow_teammate, cover_teammate, search_last_seen, probe_last_seen, wait_for_teammate, collect_item, recover_health, reach_level_exit |
| 863..866 | ready, direct_clear, unreachable, aas_missing |
| 867..870 | Маска и XYZ цели |
| 871..874 | Маска и XYZ следующей точки маршрута |
| 875..878 | Маска и XYZ последней известной позиции напарника |
| 879..880 | Маска возраста и min(age/200,1) |

XYZ поворачиваются относительно текущего yaw, делятся на512 и ограничиваются
диапазоном[-4,4]. Неизвестные названия дают нулевой one-hot; отсутствующие
позиции/возраст имеют нулевые маски. Отрицательный возраст и nonfinite позиции
отклоняются. Это наблюдения, а не готовые команды движения.

Go loader и Python target/precision/spatial validators поддерживают v8.
Spatial head использует прежний геометрический префикс854, а основной encoder
получает все881 вход. Размер GRU/attention и выходных голов сохранён.

`scripts/migrate_combat_navigation_cuda.py` переносит MLP64, GRU128,
attention64/128, добавляя по27 нулевых колонок входных матриц actor/value.
Свежие файлы сохранены только в
`workspace/artifacts/navigation-v8-migration-v1-20261010`.
`migration-verification.json` содержит SHA источников/результатов и GPU проверки
на RTX5070: actor/value output differences не больше2.87e-6, маски целей
совпали, новые колонки имеют конечные ненулевые градиенты. Проверка охватывает
последовательности3×12, но не заменяет оценку боевого поведения.

Это перенос весов, без обучения и без переноса optimizer checkpoint.
Старый checkpoint несовместим с новым первым слоем; нельзя объявлять такой
переход обычным `--resume` или молча сбрасывать optimizer history.

Проверки Go выполнялись только для feature serialization и planner context,
без CPU neural numerical tests. Native smoke новой attention64 версии:
`natural-companion-navigation-v8-smoke-v1-20261010`,4 scenes base1/base2,
600frames,×2,лимит пула16,stochastic seeds185011/185111.
Все4 infrastructure_ok=true, source_unchanged=true; combat friendly damage
и startup telefrag0. Learned provider реально работал89frames на base1 и
270frames на base2. Все сцены завершились по бюджету; улучшение прохождения
не доказано. На base1 остаётся265frames ожидания напарника.

Следующий обязательный этап — корректный перенос checkpoint/moments на GPU
и trainable on-policy сценарии с navigation_context и напарником. Текущие
natural traces остаются evaluation-only. После обучения нужны paired seeds
и оценка reacquisition/ожидания наряду с уроном, смертями и прохождением.
