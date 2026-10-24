# Исследование выбора оружия при бластере

24.10.2026. Наблюдение из собственного TRAIN случая city1/1468045009:экипированBlaster,inventory свежий и содержитShotgun+20Shells;mask разрешаетвыбор,ноpolicykeep≈99.8%,Shotgun≈0.16%. Предыдущие ранние movement варианты все проиграли. Это основание отдельной гипотезы,а не доказательство,что дробовик спасёт бой.

## Новый явно версионированный профиль

`combat_observed_blaster_grenades_weapon_temperature_v2` сохраняет прежнийtemperature2дляequippedGrenadesс прежним gate. Дополнительный`blaster_temperature` действует толькоприobservedBlasterи хотябыоднойдругойдоступнойопциивсвежейweapon mask. Наличие без ammo/freshness не включает gate. Доступность/минимальные патроны/argmax сохраняются;только12weaponlogitsделятсянатемпературу. Другие оружия/unknown/padding не получают новый коэффициент. Не выбирается конкретное оружие и не передаётся управление правилам.

Go v1 отвергает v2 поле; v2 требует явное значение1..4. Python проверяет те же версии/поля. Целевые Go metadata/scalar tests прошли(`^Test(WeaponExploration|BlasterWeaponExploration)`),безNNforward. Базовые и прочие политики без нового metadata не меняются.

## Проверки до native

`prepare_combat_blaster_exploration_cuda.py` использовал source neutral round02 weights `5dcf0d640d28394817bc105a58b379e4e6723eee747867d16a0cfbd87535ea0c` и39собственныхCUDA-verifiedTRAINстрокэтогослучая. Все tensor/sequence/gradient/distribution проверки наGPU. Actor/value state_dict exact,остальные serialized fields exact. Нулевойv2(BT1)целиком совпал с исходным. BT4 меняет толькоweaponlogits при новомgate,сохраняет остальные outputs/маски/argmax. На9syntheticCUDAсостоянияхпровереныстарыйGrenadesT2,недоступный/несвежийинвентарь,Blasterбезальтернативы,другиеweapon/paddingиanalyticgradient. Источник checkpoint не менялся;новыеexports толькоinference JSON,безresumecheckpoint и безfitting.

В реальных строках100..132BT4даётShotgun≈15.1..15.6%,keep≈75.1..76.0%. Это off-policy проверка распределения,не новая траектория/не качество.

## Фиксированный live опыт

Одна уже известная TRAIN сцена,тотжеengine/policy seed1468045009,тотжеfixture/оружие/reward. Триprofiles:source,zero(BT1),blaster4(BT4);по2coldповтора=6native/6CUDA,пул≤16,×2. Исследование действует с первого подходящего наблюдения,не привязано кframe132. ПослесменыweaponсBlasterновыйgateвыключается; прежнийgrenades режим остаётся.

Технические gates:zero/sourcefullbehavior exact;всеcoldrepeats exact;candidateprefixдо первой изменённойweaponselection истейтпереднейсовпадают;видимыйбойпринадлежитмодели. Нормализуетсятолькоspawncount,entityIDсохраняются. Candidatelogprobестественноменяетсядоизменениядействия,этотscoreнеявляетсяbehaviorprefix;остальныесостояния/выборы/памятьсравниваются. Отдельно фиксируются первыеShotgunrequest/observed-equipped,полныйfirst-life исход ифактически использованные оружия. Не выдавать6повторовза6независимыхусловий и не продвигать профиль по одному случаю.

Протокол `docs/combat_blaster_weapon_exploration_probe_protocol_20261024.json`,SHA `757627209af0bc1b9b4f68350aacd499ae1a75d459300d0d71375551d17051c5`. GPUpreparation: `workspace/artifacts/combat-blaster-exploration-preflight-v1-20261024`. Native output: `workspace/artifacts/combat-blaster-exploration-native-v1-20261024`. Runner `scripts/run_combat_blaster_exploration_probe.py` фиксируетsource/config/input/registry/reward/model SHA до игр и проверяет в конце. Всеtrajectoryздесьdiagnostic,неподлежатобычномуTRAINmerge;weights-onlyexportsнельзяпредставлятькакдообученные/resumable. Default/promotion отсутствуют, общее превосходство ещё не установлено.

24.10.2026, 22:44 МСК. Blaster exploration v2: диагностические 6 native / 6 CUDA закрыты, все 16 технических проверок прошли. В двух одинаковых TRAIN повторах исходная/нулевая версия погибает на кадре137 с0 kills/60 damage, BT4 на221 с2 kills/341 damage; обе проигрывают. Запрос Shotgun105, наблюдаемая экипировка112 (canonical alias v_shotg; исправленный производный отчёт сохранён отдельно, исходный не перезаписан). Это inference calibration, не fitting и не resumable checkpoint. Подготовлено свежее сравнение source/candidate/rules:8 validation условий offset1,4карты/2точки/duo+trio,24native/16CUDA,×2. Gate до игр: candidate минимум+2цели source, не меньше rules, deaths не выше обоих, received≤110%обоих; promotion отсутствует. Временно8инстансов из-за реальных memory-commit ошибок при16. Первый запуск отклонён до игр жёстким16 в evaluator; исправление уважает spec4/8/16, настоящий pool DryRun всех трёх прошёл, invalidcaps отклонены. Pre-game source revision явно записана в selection.json, original сохранён; модели/seed/spec/protocol прежние. Далее запуск и независимая оценка, общего преимущества пока нет.
