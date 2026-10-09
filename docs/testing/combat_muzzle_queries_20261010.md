# Дистанция, точка выстрела и blaster queries: 10.10.2026

Distance/512 и relative coordinates уже есть в восьми enemy slots. Новая абляция меняет геометрические метки, сохраняя feature contract854 и старые45 actor rows/encoder. Явный selected-target feature block пока не добавлен.

## Native доказательство геометрии

На sealed64-battle precision-v1 evaluation CUDA-аудит533 first-life blaster выстрелов сопоставил native projectile spawn с точным dispatch frame/actor/spawncount. Формула center-hand muzzle: observed position+24×native projectile direction+(0,0,viewheight−8). Максимальная ошибка позиции0.000817 units. Native velocity используется только в offline audit, не в model inputs.

**Исправление аудита:** прежний same-frame join был неверен: native event prefix F+1 относится к команде F внутри её begin/end окна. Исправленный `native_ordered_window_v2` связывает события по порядку строк и exact sequence. На первом cohort533shots, на втором658shots (terminal death orientation excluded): mismatch0, направление соответствует post-command observed angle, muzzle соответствует post-command position с ошибкой0.000817units. Средний vector error нового угла около0.0000005, прежнего около0.028. Предыдущий вывод о задержке прицела отменён. Старые JSON сохранены как `*-frame-join-v1-superseded.json`; победы и quality reports не менялись.

Исходники ClientThink подтверждают порядок: Pmove → position/viewangle → Think_Weapon при attack edge; обычный Think_Weapon в ClientBeginServerFrame происходит после ClientThink. Следовательно, instantaneous query от **предыдущей** own position не учитывает перемещение этой команды. В следующих synchronous harness cfg явно `set aimfix 0`; hand=2 задаёт ConnectRequest.

## Новые метки

Opt-in exporter `--center-muzzle`, query version `observed_center_blaster_muzzle_query_v2`. Старые eye-origin queries и артефакты сохранены. Для center-hand blaster forward muzzle offset параллелен projectile direction. Обозначив r как observed target AimPoint относительно origin+(viewheight−8)Z, v как observed enemy velocity, получаем:

`|r+v*t|²=(24+1000*t)²`.

Положительный root≤2s даёт unexecuted instantaneous intercept query. Targets внутри forward muzzle radius и за lifetime не получают known lead label. Mask зависит от observed bbox/velocity/weapon и native execution proof; machinegun fresh recoil неизвестна, соответствующие aim labels исключены. Не используется будущая player position, server target position или hit outcome. Не учтены shot delay, будущая собственная ходьба и ускорение монстра.

Повторно экспортированы80 train/16 validation own-policy captures:6268/1499 context frames,3844/1063 known pairs, seeds не пересекаются. CUDA serialized-vs-feature query error≤0.0000153°. Независимая подстановка в intercept equation: residual≤0.000064units; проверены stationary targets, inside-muzzle и lifetime masks.

| Observed enemy-origin distance | Train pairs | Validation pairs | Средняя абсолютная pitch-поправка validation |
| --- | ---: | ---: | ---: |
| ≤128 | 2285 | 597 | 7.05° |
| >128…512 | 1559 | 466 | 2.51° |
| >512 | 0 | 0 | нет данных |

Это различие eye/muzzle intercept queries, не промахи реальных выстрелов. Coverage дальних целей отсутствует; synthetic2048units только проверяет lifetime mask.

## CUDA обучение и native trial

От сильного FireBC target-before,100 epochs RTX5070, новые36 fine/mode rows; первые45 rows, encoder, value/std фиксированы. Fine validation query RMSE8.08→6.25° на411 fine labels; mode accuracy61.34→64.06%. Не сравнивать напрямую с eye-origin fine RMSE: метки и подмножества разные.

Завершено80/80 paired native боёв, native/source/binary proof принят,16 slots/x2: before81, muzzle-after81, прежний eye-after81, legacy FireBC и rules. Validation28,4 families×4seeds на вариант, final test отложен. Рабочий baseline не заменён.

| Вариант в новом cohort | Победы | Смерти | Firing applied-ray error | Native live-monster blaster hits |
| --- | ---: | ---: | ---: | ---: |
| Before81 | 7/16 | 9 | 11.371° | 100/142=70.42% |
| Muzzle after81 | 7/16 | 9 | 11.861° | 97/144=67.36% |
| Eye after81 | 7/16 | 9 | 13.223° | 97/142=68.31% |
| Legacy FireBC | 7/16 | 9 | explicit target не объявлен | 100/142=70.42% |
| Rules | 8/16 | 5 | explicit target не объявлен | 94/94=100% |

Unknown projectile outcomes0 у всех. Показатели projectile hits объединяют blaster выстрелы всех16 случаев, включая смену оружия; не machinegun hit rate. Победы всех learned вариантов совпадают по16 seeds. Разные длины траекторий и число выстрелов; pooled ratios не доказательство статистического превосходства. Ray metric без lead/recoil correction. **Muzzle after не принят как улучшение:** геометрический ray ближе, чем у eye-after, но фактических попаданий меньше и побед больше не стало.

Часть controls изменила урон/число shots относительно предыдущего cohort, несмотря на одинаковые веса: явно pin aimfix0 и новый harness/source fingerprint означают новый cohort. Причина межcohort различий отдельно не установлена. Сравнивать варианты внутри текущих пар, не выдавать разницу между cohort за чистый эффект новых меток.

Следующий приоритет уточнён после исправления join: учитывать own movement этой команды в corrective labels, не вводить ложную задержку угла. Проверять actual projectile hit metrics вместе с победами. Instantaneous labels от предыдущей позиции не учитывают Pmove этой команды. Postmove абляция также не дала улучшения; результаты в `combat_postmove_queries_20261010.md`. Явный selected-target distance block остаётся отдельной абляцией; текущая модель уже видит raw range во всех enemy slots.

## Настоящие projectile hits

Новый `report_combat_blaster_hits.py` использует native shot IDs, end events и damage contacts. Denominator — mod1 projectiles, выпущенные в first-life alive matched dispatch после frame100. Hit — подтверждённый положительный health damage живому monster. Unknown unresolved/freed остаются явными; итоговый hit fraction отсутствует при unknown. Это не machinegun hit rate и не попадание обязательно в выбранную цель. Server facts только для оценки.

Для предыдущего sealed precision-v1 cohort после исправления ordered-window join: before100/142=70.42%, eye-after106/149=71.14%, legacy FireBC112/151=74.17%, rules93/95=97.89%; unknown0 у всех. Разные траектории и число выстрелов, pooled ratios не доверительный интервал. Эта диагностика дополняет победы7/16,7/16,7/16,8/16. Подробности исправления и следующей абляции: `combat_postmove_queries_20261010.md`.

Артефакты: `precision-eval-v1-20261010/{blaster-muzzle-audit,blaster-projectile-hits}.json`, `muzzle-queries-v2-20261010/{report,cuda-query-audit,cuda-ballistic-audit}.json`, `precision-muzzle-v2-20261010/`, `precision-muzzle-eval-v2-20261010/` под `workspace/artifacts/`.
