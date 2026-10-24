# Уменьшение случайности движения: подтверждающая проверка, 24.10.2026

72 native боя / 48 CUDA replay закрыты без технических ошибок. 24 одинаковых условия на вариант, base3/city1/jail1/ware2, по две точки; поровну ordinary/recovery и двух/трёх противников. Validation offset44, одинаковые физические instances и seed у source/noise-only/rules, пул16, timescale2. Все source bindings сохранены. Это существующие геометрии с новыми сидами, не тест полной кампании или кооператива с человеком.

Noise-only уменьшает только std двух координат движения в sqrt(1-0,65²)=0,759934 раза. Исходные нейросетевые параметры, mean, angular std, reward и default policy сохранены. Fitting не выполнялся. Предыдущая проба8условий использовалась для выбора варианта и сюда не включена.

| Вариант | Победы / 24 | Смерти | Убийства | Нанесено HP | Получено HP |
|---|---:|---:|---:|---:|---:|
| Source | 13 | 11 | 34 | 6248 | 1461 |
| Меньше случайности | 15 | 8 | 42 | 6831 | 1363 |
| Rules | 12 | 7 | 32 | 5627 | 1078 |

| Карта | Source победы / 6 | Noise-only / 6 | Rules / 6 |
|---|---:|---:|---:|
| base3 | 3 | 3 | 4 |
| city1 | 4 | 3 | 3 |
| jail1 | 2 | 3 | 2 |
| ware2 | 4 | 6 | 3 |

Парные изменения: против source три новых победы и одна потерянная; против rules четыре новых и одна потерянная. Описательный exact paired sign test: p=0,625 и p=0,375 соответственно. Это не заранее заявленный статистический gate. Объём небольшой, явная прибавка сосредоточена в ware2; общего превосходства или устойчивого улучшения по всем картам нет. Rules получают меньше урона и реже погибают.

## Фактические действия

MG attribution взята из native execution windows до конца цели/первой жизни, только provider. У Source 559 выстрелов с уроном по живому монстру из740 (75,54%), solid contact138. У noise-only549/756 (72,62%), solid169. Это MG, не общая точность всех видов оружия; состояния и длины боёв различаются. Устранение стрельбы в стены не доказано.

Standing grounded-visible: source699/2058(33,97%), noise858/2393(35,85%). Большие повороты:135/2252provider кадров и162/2681. Эти наблюдения не подтверждают устранение стояния или вращений; не каждый такой кадр обязательно лишний. Ненулевая команда с перемещением меньше1единицы:89/1456 против144/1780. Нельзя объявлять шум единственной причиной проблем.

Ownership audits: у обеих моделей0 observed clear-enemy rules frames,0 unknown observation rows; у победных случаев нет rules attack commands. Post-goal/noncombat хвосты правил учитываются отдельно; это наблюдаемая ownership граница, не доказательство отсутствия любых незалогированных решений.

## Следующий шаг

Зафиксирован `../combat_movement_variance_replication_protocol_20261024.json`: повтор всех24условий с первыми общими неиспользованными validation seeds, offset45. Параметры и рецепты не подбираются заново; сравниваются source/noise-only/rules, ещё72native/48CUDA,16×2. Запущен `combat-movement-variance-replication-v1-20261024`. Затем отдельно показать повтор и объединённые48пар, с разбиением по map/site/loadout/composition. Промоушена и нового обучения пока нет.

## Свидетельства

- `workspace/artifacts/combat-movement-variance-confirmation-v2-20261024/complete.json`: native72, CUDA48, training=false, promotion=false.
- `combat-movement-variance-confirmation-v2-20261024/pool/report.json`: source_unchanged=true, usable72, errors0, wall_seconds204,26 для native части.
- `combat-movement-variance-quality-audit-v1-20261024/report.json`: реальные MG контакты и movement alignment.
- `combat-movement-variance-summary-v1-20261024/report.json`: парные изменения, страты, описательная статистика, хэши входов.
- Source weights SHA256:559ff0ad4297047cdc4a9ad531fa9159668b045b7f41bf8e32a6df975278bea6.
- Noise-only weights SHA256:4b1097038137bff824149836e951f3c6126bde234047a07bc0d1e1e03149d479.


24.10.2026. Независимый movement-variance повтор закрыт72native/48CUDA:source10/24побед,noise13/24,rules11/24. Вместе безdiscovery:23/48→28/48противrules23/48;смерти22→18противrules15. Превосходство пока не доказано;rulesменьшеурона2218HPпротивnoise2909. NativeMGдоля69,82%→71,38%,standing38,11%→35,69%,largeyaw6,42%→5,11%всумме двухоценок — частичноеописательноеулучшение,проблеманеустранена. ПодготовленCUDAпроверенныйcheckpointсexactоценённымиweights,сохранённойисторией/Adam/RNG,resetтолькомоментовmovementstd. Собран32freshTRAIN/32CUDA,4298rows,offset216..219,4картыsite02,всеseedsнепересекаютсясfullhistory. ЗапущенпервыйиздвухCUDAupdatesпофиксированномупротоколу;qualityпослеобученияещёнет,defaultнеизменён. См.testing/combat_movement_variance_replication_20261024.md.
