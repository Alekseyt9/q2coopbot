# Native collision diagnosis, 2026-10-12

## Воспроизведение и причина

Повторены исходные validation478 fixtures с прежними весами:16 native/16 CUDA, затем targeted2 native/2 CUDA. Это диагностика с намеренно повторёнными seeds, не независимая оценка качества и не обучение. Guards, weights и actor feature columns не изменены.

Добавлены native PMove begin/end и PM_trace события с actor/call identity, game_frame и server network frame/spawncount, командами, состоянием до/после и blocker entity/class/model. Version2 дополнительно сохраняет реальные blocker origin/mins/maxs. Лог ограничен32 collision событиями на pmove; общий trace count сохранён. События trace включают ground probes и отвергнутые step paths: наличие blocker не доказывает, что именно эта плоскость определила конечное движение.

Native cvar g_test_motion по умолчанию0; лог разрешён только cheats+single-client combat barrier после release, для живого игрока. Харнес fixed-release включает1. Обычный coop не включает его. Серверный префикс связывает game telemetry с внешним network clock. Native CMake targets game/q2ded пересобраны; runtime snapshots старых прогонов не удалялись.

Первый повтор:2696 pmove calls;1913 пригодных next-frame пар,1855 однозначно связаны с native командами,58 неоднозначных исключены. Все29 крупных расхождений связались. Collision candidate classes: monster_parasite11, monster_infantry11, misc_explobox8, worldspawn2; классы пересекаются внутри кадров. Значительная часть прежних ошибок — тела врагов и бочки, отсутствовавшие в static/mover модели.

## Наблюдаемые корпуса

Decoder теперь сохраняет все видимые protocol34 SOLID_BBOX bodies, кроме собственного игрока;0 nonsolid и31 BSP marker исключены. Bounds декодируются по native CL_ClipMoveToEntities. Brush movers сохраняются отдельным списком. Это клиентские данные, не hidden native positions; actor observation/features пока не расширяются. Collision bodies добавлены в диагностические snapshot logs и используются физическим предиктором. Невидимые server entities и движение объектов между снимками не выдумываются.

29 small regression fixtures сначала содержали только enemies/barrels из JSONL:17/26 совпали в0.25. Полный набор packet bodies восстановлен из неизменных bot.dm2 через cmd/q2collisiondata: точное совпадение map/network frame/self; повторные demo frames свёрнуты только при равенстве полного body списка. Каждая выгрузка повторно выполнена и совпала побайтно; demo/output/fixture SHA сохранены. Это восстановление наблюдений, не пользовательских кнопок или labels для обучения.

## Остаточная неопределённость

С полными bodies24 primary траектории совпадают в0.25 после учёта сохранения crouch при заблокированном вставании. В двух случаях quarter-unit граница опоры допускает grounded/no-support переход перед ускорением, несмотря на прошлый ground flag. Предиктор теперь выдаёт явную дополнительную no-support траекторию с воздушным ускорением/гравитацией, не подменяет наблюдаемый исход выбранным ответом. Для26 из29 frames хотя бы одна заранее вычисленная допустимая траектория совпала с native в0.25 (max minimum error0.14744). Это покрытие альтернативами, не точность единственной основной траектории и не проверка безопасности.

Три ware2 случая остаются unsupported: квантованный packet body может выглядеть overlapping при реально свободной server pose, либо наоборот. Native telemetry показывает различие body pose с wire pose меньше1/8, но этого достаточно для смены contact branch. Slide collision-фаза сохраняет native startsolid escape semantics; применение единственной предсказанной overlap pose не разрешается автоматически.

Второй targeted native запуск с bodies:169 measured pairs,149 с командой движения, один residual >1 (city1 boundary),11 contact frames max position error0.1482. Этот запуск проверял v3, до явной альтернативы v4. v4 проверена по frozen native fixtures; свежая v4 live/guard acceptance ещё не выполнена.

## Следующий этап

Для guard проверить весь путь всех допустимых альтернатив: пол, двери, lasers/liquids; при отсутствии альтернативы либо неопределённом body overlap это незавершённая применимость, не доказательство безопасного движения. Затем включить экспериментальный wall-slide guard и провести matched малую оценку с неизменными весами. Не считать снижение количества blocks или minimum error по альтернативам выигрышем модели.

Проверки: native build; targeted Go protocol/hull/body/step/recorded physics tests;3 Python collision parser tests и4 next-frame filtering tests;18 CUDA replay cases. Go здесь не исполняет нейросеть. cmd/q2collisiondata не восстанавливает user input, не запускает демку и не исполняет policy.

Артефакты: combat-pmove-collision-reproduction-native-v1-20261012; combat-pmove-body-pose-native-v2-20261012; combat-pmove-body-demo-recovery-v1-20261012. Native code находится в ../yquake2/src/game/player/client.c и ../yquake2/src/server/sv_game.c, отдельно от q2coopbot checkout. Коммитов/push не выполнено.
