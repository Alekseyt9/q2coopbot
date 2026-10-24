# Следующая гипотеза: научиться смене оружия, а не только чаще пробовать её

24.10.2026. Это план следующего опыта, не реализованное обучение и не положительный результат.

## Основание

На собственной TRAIN сцене city1 модель почти всегда оставляет бластер при доступном дробовике с патронами. Фиксированное исследование BT4 дало два убийства вместо нуля, но не победу. Независимый свежий screen: 5/8 против 2/8 source и 4/8 rules. Расширенная confirmation пока проходит техническое закрытие; native итоги 13/24 против 12/24 source и 15/24 rules. На blaster 2/8 против 1/8 source и 3/8 rules, остальные loadout исходы с source совпали. Следовательно, одного изменения распределения выбора недостаточно; этот профиль не становится default.

## Проверяемая гипотеза

Собственные новые TRAIN бои с более частой пробой доступных оружий дадут полезные переходы для обучения выбора. После обучения модель должна улучшать исходы боёв, а не просто чаще переключать оружие. Эта гипотеза остаётся непроверенной. Оценочные captures не попадают в fitting.

## Порядок и ограниченный бюджет

1. Закрыть текущие 72native/48CUDA, проверить source fence, ownership и отрицательный критерий расширения. Не запускать ещё один широкий confirmation того же неизменного BT4.
2. От исходного resumable neutral round02 создать две явно объявленные ветки BT1/BT4. Сохранить tensors, Adam, RNG и consumed history, отдельно архивировать несовместимый retention при смене распределения. Техническая CUDA проверка миграции и настоящего возобновления обязательна. Старый weights-only export нельзя объявлять resumable checkpoint.
3. Изолировать обучение выбора оружия: заморозить движение, прицел, память и остальные головы; value обучать отдельно. Новый scope должен иметь явный контракт и отклонять несовместимые checkpoints. Если такой scope ещё не реализован, не выдавать существующее обучение всех решений за этот опыт.
4. Начальный pilot: по одному собственному корпусу из 24 TRAIN условий на ветку, общий бюджет 48 новых боёв, одинаковые reward и число шагов, отдельные own-policy траектории. Обработка и fitting только CUDA. Реестр: четыре карты, две точки, duo/trio, blaster/machinegun/weapons. Пул временно8, ×2; после каждого обновления проверять frozen outputs и checkpoint/Adam continuation.
5. До новых оценочных боёв зафиксировать отдельные protocol/spec, свежий seed offset и критерии качества. Сравнить source, BT1-trained, BT4-untrained, BT4-trained и rules на восьми парных условиях (40native/32CUDA). Контроль BT1-trained нужен для отделения эффекта дополнительных боёв от исследования оружия. Не выбирать лучший промежуточный checkpoint по оценке.
6. Расширять только после заранее заданного screen gate; проверить исходы, смерти, полученный урон, реальные применённые оружия, задержку первого выстрела, пустые выстрелы и неподвижность. Полное прохождение и кооператив с человеком остаются отдельными этапами приёмки.

Этот pilot не гарантирует преимущества над rules. Если миграция или frozen-head контракт не проходят проверку, обучение не запускается; исправляется конкретный технический разрыв. Если качество отрицательное, сохраняется результат и выбирается другая гипотеза по TRAIN диагностике.

## Реализованная подготовка — 24.10.2026, 23:08 МСК

Текущая confirmation полностью закрыта:72native/48CUDA,gate=false,source fence иownership прошли;13/24 candidate против12/24 source и15/24 rules. Этот отрицательный итог остаётся основанием отказа от автоматического продвижения BT4.

Реализованы `scripts/combat_weapon_only_scope.py` и `scripts/initialize_combat_weapon_only_cuda.py`. Новый явный checkpoint scope `combat_observed_weapon_residual_only_ppo_v1` разрешает обучение только `weapon_state_branch`:12396параметров. Весь остальной actor, temporal memory, std и унаследованные optimizer scopes заморожены. Активный weapon Adam наследует именно моменты текущего projectile-training optimizer, а не устаревший legacy anchor. Critic получает отдельную копию текущего critic Adam.

Initializer protocol: `docs/combat_weapon_only_initializer_protocol_20261024.json`. Новые resumable ветки `workspace/artifacts/combat-weapon-only-initial-v1-20261024/control` (BT1) и `explore` (BT4), обе от neutral round02. Источник и старые weights-only exports сохранены. Непустой sequence retention требует отдельной миграции и отвергается; у данного source он отсутствует. Старые trainer scopes отвергают новый checkpoint. Существующие projectile/body optimizers, RNG, consumed history и tensors сохранены точно.

CUDA проверка39ранее собранных собственныхTRAIN переходов seed1468045009 подтверждена pinned train-plan, не оценочными боями. Нулевая ветка совпадает по outputs с source, у explore меняются только weapon logits. Проверены фактические disposable actor/critic Adam steps до/после сериализации, exact frozen tensors и nonweapon outputs после шага, изменения обучаемой ветки, reject несовместимых scopes, целостность source и итоговых seals. Все проверки прошли. В checkpoint сохранено0новых optimizer steps; эти технические шаги отброшены и не считаются fitting или новым опытом.

Следующее незавершённое действие: dedicated PPO trainer для этого scope с полным native credit/advantage admission, далее collector/runner равного24+24own-policyTRAIN бюджета и заранее фиксированныйfresh screen. Не подавать новые checkpoints существующему projectile trainer и не объявлять initializer обученной моделью.

## Pilot реализован, протокол до игр — 24.10.2026, 23:14 МСК

Добавлены dedicated trainer `train_combat_weapon_only_cuda.py`, проверка реальных возможностей/выборов `audit_combat_weapon_only_exposure.py`, runner `run_combat_weapon_only_training.py`, итоговый reporter `report_combat_weapon_only_training.py`. Trainer использует существующие native credit, advantage, CUDA replay и PPO KL guards. Он отдельно проверяет точное сохранение всех nonweapon outputs и условных movement/aim means после обучения; родительские projectile/body/legacy optimizers остаются anchors. Actor и critic получают собственные resume proofs после настоящего обновления. Само обновление пока не проверено живым запуском.

Фиксированный бюджет:24own-policyTRAIN на каждую ветку, одно обновление,48native/48CUDA; затем source/control-trained/explore-static/explore-trained/rules на8новыхусловиях,40native/32CUDA. TRAIN24рецепта:4карты/2точки/duo+trio,по8оружейныхloadout. Screen4duosblaster(поодномунакарте)+2triosmachinegun+2triosweapons. TRAIN offset≥20 с автоматическим поиском неиспользованных seed; validation offset2 с отклонением любого исторического повтора. Пул8×2,storage reserve32GiB.

До fitting нужны native credit/advantage,нулевойliveinfighting и projectile exposure как в предыдущемпротоколе;дополнительно≥8наблюденийBlasterсдоступнойальтернативойминимумв2бояхкаждойветки,аexploreдолжнафактическипопробоватьальтернативумиинимумв2боях. Недостаточнаяэкспозицияостанавливаетобучение;отрицательныебоевыеисходынепересэмплируются.

Expansion gate до оценки:explore-trained≥control-trained+2победы,≥explore-static+1,не меньшеrules;deaths≤control-trained/rules,received≤110%обоих;ownershipвсех4моделей. Никакойавтопромоции. Protocol SHA74bc160b555ca17c90df18e4a7c87105dd7d0bc925a7e403671b1fa21f092ec0;spec SHA fcbebdc8642ed097fd94ebcb2d17265263a097b2c67493433da124d399d097a0. Все4новыхentrypointпрошлиPythoncompile/CLIimport;полныйliveTRAINещёнеокончен. После запуска source fence запрещает Go/Python/PS/native edits до закрытия/явной новой admission.

24.10.2026. Первый live pilot собрал48/48validnative,control24CUDA/3029rows закрыты. Pre-fit weapon auditor остановился до любого fitting на KeyError weapon: nativeJSON штатно опускает weapon=0(keep),как уже учитывают существующие trainers. Исправлено толькочтение.get(weapon,0),добавленучётomitted_keep_rows;realcorpus3029прошёл(2995omitted,1182legalBlasteralternative rows,4alternative samples). Originalrun/failure/log/source snapshots сохранены вpre-fit-keep-default-recovery. Явнаяpre-fitrevisiond29fd010e661ee53afd796a93fefcc6f899e74e94fd103e222e5512c13a1968d допускаеттолькоauditor+runnerrecoveryvalidation;source/model/reward/protocol/gates/seedsнеизменны. Runnerresumeпроверяетoriginalfence,двеwhitelistзамены,poolSHA,отсутствиеupdateдиректорийпередпервымприменением. Все48боёвсохранены,новыхnative0,фиттингапередrevision0. Возобновлениезапущенопослеterminalfailure;эффективнаяfenceновая,original_source_unchangedбудетfalseиотдельнозаписано. Следующее—первоеCUDAобновлениеcontrol,затемобработкаexplore.

24.10.2026. Control weapon-only update1 ЗАКРЫТ наGPU:24собственныхTRAINбоя/3029переходов,30actor/40criticsteps,weightsbbf64c0ef60241bbd0ec259b7b45325dbaea4f9e7da10be2ff431ef2bd2370a6. Реальноизмениласьweapon residual;всеостальныеactoroutputs и movement/aim conditional means exact,KLкоординат0. Checkpoint/source/legacyиостальныеparentoptimizeranchorsсохранены,export/readbackиdisposableactor+criticAdamcontinuationпрошли. Обновлениенеозначаеткачество;freshscreenещёнебыл. Runner10228жив,finalize-explore;все48исходныхnativeбоёвиспользуютсябезповтора. Sourcefence/revisiond29fd010... активны;Go/Python/PS/nativeнеправитьдозакрытия. Дальше24exploreCUDA/credit/exposure/update,затем40freshnative/32CUDAfivevariantcomparison.

## Отрицательный admission v1 и отдельное продолжение v2

Оригинальный pilot v1 остановился послеcontrol update,доexplorefitting:weapon coverage прошла(116legalalternative rows/8cases,13выборовальтернативыв8боях),но унаследованный projectile-learning порог скорости enemybolt дал7knownrowsв4casesвместо≥8. No infighting,grenade presence,velocity bound,native credit и advantage прошли. V1 остаётсяfailed,егоprotocol/gates/source/data/controlupdateне переписаны. Это остановкадопуска,неотрицательнаяоценкакачестваweapontraining.

Дляновогопродолжения v2 ДО explorefitting явнообъявлендругойдопуск:enemy velocityfrequencyописательная,посколькуprojectile/body/memoryиостальныеactorparameters/outputsзаморожены;обязательныCUDAfrozenproofinitializerитренера,полныйnativecredit/advantage,noinfighting,velocityerrorbound,grenadepresenceиоригинальныйweaponcoverage. Это изменение экспериментального допуска,а не объявление старого порога пройденным. Число7не заменяетсявыдуманными8и исходныйпровалсохранён.

Все48ownpolicyTRAINcapturesсохраняютсябезновыхигр/ресэмплинга. Control update v1 сохраняетсякакссылкабезновогообученияилиперемаркировки:егоoriginaltrainingprotocolSHAпроверяетсяотдельно. V2обучаеттолькоexploreоттого жеinitializercheckpointнаегоисходных24собственныхбоях,новыеnativeTRAIN0. Reward/KL/шаги/weaponfloors/8невиденныхоценочныхусловий/qualitygateнеизменны. Тажеfivevariantоценка40native/32CUDAпослеобновления;не заявляетсяmatchedv2fittingдляcontrol.

Protocol `combat_weapon_only_training_v2_protocol_20261024.json` SHA1b9a0f7e63f06b876420b42c0e42f89caa9ffbf8a930c5b8b144e8bb0a1a37bd;specSHA b9de21b4e125e7bd0ad49b56d4de2c132c79853d6b6b75f88b8057380ca7dde0. Новыйrunner/reporterдопускаюттолькоэтотобъявленныйfailedv1reference,проверяютнеизменностьегоsource/data/controlcheckpointиотсутствиеexploreupdateвстаромroot. Новыйrootcombat-weapon-only-training-v2-20261024. CUDAcompile/importподготовлены;результатовкачестваещёнет.

V2 explore update1 ЗАКРЫТCUDA:2135own-policyTRAINпереходов,30actor/40criticsteps,weightseb5221c4016d049f65e5f8a9ff9a6f50ddd7280fff4fa4c4b22d364191639497. Nonweaponoutputs exact,movement/aimKL0,checkpoint/export/sourceanchorиdisposableactor+criticAdamcontinuationпрошли,все3sealsпроверены. Вместеcontrolv1(3029rows/30+40)иexplorev2(2135/30+40)использовалипо24своихисходныхбоёв,новыхnativeTRAIN0. Старыйv1failedсохранён;этообъявленноепродолжениеv2,непереписанныйadmissionv1. Запущенfixedfreshscreen40native/32CUDA(source/control-trained/explore-static/explore-trained/rules),actualrunner24980/evaluationchild24384. Покаидётпроверкаисториисидов,результатовкачестваещёнет. Всеruntime/trainingGo/Python/PS/nativeисходникизаморожены. Defaultнеизменён,общее превосходство не установлено.

## Закрытый результат weapon-only pilot v2

40freshnative/32CUDA оценка завершена;все4modelownershipgates прошли,sourcefence/retainedinputs проверены. Source/control-trained/explore-static/explore-trained/rules:3/3/4/4/5целей из8,4/4/3/3/3смертей,13/13/15/15/14kills,571/571/529/529/477HPполучено. Gate=false,promotionfalse. Уtrainedexploreнетдополнительныхпобеднадstaticилиcontrol;receivedхуже110%rules. ResultSHA8875d04fc9ad67efaa1e7ad9ea226f36186ba8350aaa34b88d7faebb426002d9. Новыйдлинныйtraining/confirmationтехжевесовнезапускаетсяпоэтомуscreen.

ДополнительнаядиагностикаТОЛЬКОownTRAIN,CUDAnumericalinference:в1182legalBlasteropportunitiescontrolpKeep99.715→99.794%,pShotgun0.188→0.122%;в116exploreopportunitiespKeep74.769→76.291%,pShotgun15.480→15.271%. Этоодинраунднасвоихразныхтраекториях,некаузальнаяоценка. Файлыtrain-weapon-distributions.json/.complete.jsonвv2rootсодержатpins,безfitting/новыхигр.

СкалярныйGAEразборсовпалснезависимымadvantagecredit:explore13Shotgunвыборов/8cases,mean normalizedadvantage+.446;90keepвыборов+.526. Положительныесигналыестьуобоихдействий,состоянияразные;этичисланедоказываютправильность/ошибкуконкретнойсмены. Вcontrolвсего4alternativeвыбора,3изнихShotgun. Файлtrain-weapon-advantage-diagnostic.json/.complete.json. РеальногоулучшениябояотPPOupdateвэтомpilotнет.

Следующаяконкретнаяподготовка:выбраны4ownTRAINкорнядляпричинногосравненияkeep/Shotgun,первыйадmissibleпоseed/frame/indexкаждойкарты,безпросмотрабудущегоreward/outcome. Base3seed1404045020,city11468045020,jail11414045020,ware21412045020;всеframe98/index1. Sourcecontrolinitializer7b47e0a5... . Selectionrootcombat-weapon-choice-counterfactual-selection-v1-20261024,cases4,native_games_run0. ЕстьBlaster,legalShotgunmask,sourcekeep,selectedtarget. Runtime диагностического weapon intervention ещё не реализован;этиrootsнеобъявляютсяготовымэкспериментомилиметкамидляобучения. Планtesting/combat_weapon_choice_counterfactual_20261024.md.
