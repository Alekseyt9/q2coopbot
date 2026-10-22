# Четыре свежих раунда PPO: кривая обучения, 22.10.2026

Root `workspace/artifacts/combat-ppo-learning-curve-v1-20261022`.

Начальная модель — fixed `combat-motor-feedback-v1-20261022/round-2-full-update`. Её независимый [screen](combat_motor_feedback_20261022.md) закрыт72native/48CUDA:15/24побед против11/24собственного исходного веса и17/24rules;4gained/0lost кinitial,received1360→982,deaths6→2. Наbase3 остаётся0/6,все6падений192–264units. Это основание для дальнейшего опыта,не для promotion.

## Зафиксировано до сбора

- **4раунда ×32новых TRAIN боя =128**. Собственные текущие веса в каждом раунде,8семейств/base3+ware2/site01+02,ordinary иrecovery; fresh registered train seeds,16инстансов ×2,один native pool за раз.
- Full actor+motor,та же rewardv13,GAE/learning rates,retention strength1 иcaps10actor/40critic. Нет нового fire penalty,геометрического контроллера или архитектуры в этом опыте. Веса/actor/critic/std/Adam/RNG продолжаются от исходного checkpoint.
- Только CUDA для обучения/численного аудита. Go остаётся реальным игровым исполнителем.
- После каждого update — существующий CUDA boundary/resume/JSON-checkpoint audit,тот же,который прошёл все4предыдущих updates. Новый rollout потребляется один раз.
- **Нет промежуточной оценки или выбора лучшего checkpoint.** Сначала все4TRAIN раунда,потом фиксированные endpoints0/2/4.
- Финальная оценка: start/round2/round4/rules,по24одинаковых свежих validation условия на вариант: **96native/72CUDA**,12ordinary+12recovery,четыре карты,offset748–750,порт47280.
- Резерв перед native этапами20GiB;если нарушен,протокол останавливается. Данные автоматически не удаляет. Promotion/default replacement=false.

Бюджет одинаковый по боям/максимальным шагам,не по фактическим кадрам: живущая дольше модель собирает больше переходов. Endpoints — одна training lineage,а не независимые перезапуски обучения. Существующая multi-variant схема отчётов применяется к checkpoint curve;это не новая архитектурная абляция.

## Исполнение

`scripts/run_combat_ppo_learning_curve.py` — отдельный четырёхраундовый driver на основе проверенного feedback runner. Старый driver не изменён: закрытый прежний опыт сохраняет прежние hashes кода. Новый driver требует version/4rounds/128cases/endpoints0,2,4 иединственнуюfullветку;код/входы зафиксированы в `execution-source.json` и `protocol.json`.

Проверены syntax,source weight/checkpoint hashes,32×4budget,путь каждого endpoint,readiness training-complete ивсе1437code hashes. Это preflight протокола; прохождение нового полного live цикла подтверждается только его будущими seals.

Четыре TRAIN раунда завершены:128новых боёв,4GPU updates и4аудита закрыты. Веса/чекпоинты всех4раундов ивходы протокола повторно проверены после [очистки диска](disk_cleanup_100g_20261022.md). Endpoint evaluation также закрыта: **96native/72CUDA**,всеaudits complete,`complete.json` и`bundle/complete.json` опубликованы.

## Результат на одинаковых24свежих условиях

| Вариант | Победы | Убийства | Полученный урон | Смерти |
|---|---:|---:|---:|---:|
| Start | 10/24 | 27 | 1190 | 6 |
| После2новых раундов | 10/24 | 27 | 1420 | 10 |
| После4новых раундов | 10/24 | 30 | 1413 | 7 |
| Rules | 13/24 | 29 | 770 | 4 |

Midpoint vsstart2gained/2lost,final vsstart3/3,final vsrules4/7. Прибавка побед отсутствует,received/deaths кstart хуже. `decision.json`: больше не продолжать эту ветку вслепую;для следующего механистического теста сохранить **заранее заданный start**,не выбиратьmidpoint поудачной подгруппе. Promotion=false.

Это новый наборseeds. Нельзя сравнивать10/24здесь с15/24того жеstart на предыдущемscreen как падение качества весов:весstart не менялся,условия различаются. Кривая сравнивается внутри этой матрицы.

Native MG live damage282/670→301/644→299/628(42.09→46.74→47.61%),no-target/no-visible-bbox shots195→132→110. Largeyaw5.56→4.72→4.83%,attack безtarget31.52→23.70→25.39%. Отдельные показатели эффективности улучшились,ноне выполнение цели. Поворот/стояние не считаются ошибкой сами по себе.

Final по картам:base3 0/6,city1 6/6,jail1 1/6,ware2 3/6;start0/5/2/3 соответственно. Наbase3 final во всех6случаях падает на192–264units итеряет контакт. Это остаётся нерешённой задачей.

Следующий [conditional-fire core](combat_conditional_fire_prototype_20261022.md) создан и проверен отдельно;он ещё не подключён кruntime/likelihood/checkpoint,не обучен и не продвинут в игровойdefault.

```powershell
& F:/src/strat/.venv-gpu/Scripts/python.exe scripts/run_combat_ppo_learning_curve.py --protocol workspace/artifacts/combat-ppo-learning-curve-v1-20261022/protocol.json --resume
```

`--resume` допускается после остановки runner и переиспользует только закрытые фазы. Незакрытая фаза сохраняется для диагностики и отвергается. Текущий статус `progress.json`;128TRAIN подтверждает `training-complete.json`;весьопыт — `complete.json`.

Для запуска удалены12точно восстановимых merged duplicates двух закрытых quality evaluations11.10.2026:прирост свободного места≈4.82GiB,перед запуском24.35GiB. [Manifest/restore record](combat_cold_cache_retention_20261022.md);primary streams,TRAIN/dev имодели сохранены.
