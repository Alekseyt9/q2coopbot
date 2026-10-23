# Supervised head: fixed endpoint на17native контекстах

23.10.2026. Фиксируется до fitting и до исходов ещё трёх направлений one-frame extension. V1 admission на16состояниях остаётся false. Новый способ доопределённого вмешательства в кадре110 уже дал независимый второй TRAIN seed с полезной поправкой. Проверочные состояния не переносились.

Новый dataset v2: прежние16контекстов без изменений +полный5-way TRAIN контекст seed1172065112/frame110. Всего17состояний/85вариантов; expected12train/5development. Старые4development seed фиксированы и все17состояний разделены по целым seed. Максимум2состояния на seed. Источники neural body baseline S4 и action/feature contracts одинаковы. Dataset сборка должна доказать неизменность старых16записей и split по их прежнему seal.

Сохраняются критерии поддержки: не меньше2TRAIN состояний с native улучшением, теперь также на2разных TRAIN seed; не меньше1такого development состояния; минимум3принятых альтернативы на каждый контекст. Метки только native, непроверенные направления неизвестны. Приёмка всех необходимых baseline/source prefixes обязательна.

Алгоритм/бюджет/контроль/gate из `combat_native_aim_supervised_protocol_20261023.md` сохраняются: исходная S4 head1359→64→25; заморожены actor/value/std/fire/predictor и их Adam/RNG/history. Target0,1source+0,9preference, baseline при не худшем результате, иначе равномерно по лучшим наблюдённым HP альтернативам. Adam LR0,0003,weight decay0,0001,clip0,5,ровно200полных train steps, CUDA. Старый head optimizer сбрасывается явно. Нет early stopping или checkpoint selection по development. Оценивается fixed endpoint.

Constant control совпадает со средним финальной head распределением наTRAIN, включая направления. На development показываются вероятность на проверенных улучшающих/ухудшающих действиях, покрытие известных действий и условная expected native HP на них. Никакого присвоения меток неизвестным направлениям.

Native screen допускается только при росте средней development вероятности известных улучшающих действий ≥0,05vs source и≥0,02vs constant, при росте вредной массы ≤0,05vs source. Только после проверки fixed endpoint. Screen sourceS4/trained/constant/rules,8fresh matched validation условий на вариант,32native/24CUDA,пул16×2. Первый свободный validation offset≥33 по доступной истории. Default автоматически не менять. Эти17targeted состояний не доказывают перенос на кампанию даже при хорошем loss/gate.

Probe модели никогда не подаются как train source. Обучается обычная S4; one-frame данные используются только как supervised условные native метки. Новые checkpoints имеют отдельный scope, несовместимый с прежним B/G/S budget trainer; PPO counters/consumed rollouts сохраняются как история, не увеличиваются этим fit.
